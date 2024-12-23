#!/usr/bin/env python3

import sys
from lightglue_feature_tracker.lightglue import LightGlue, SuperPoint, DISK
from lightglue_feature_tracker.lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue_feature_tracker.lightglue import viz2d
import yaml
import rclpy
from sensor_msgs.msg import CompressedImage, Image, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scipy.spatial import cKDTree
import numpy as np
import torch
import time
from rclpy.node import Node
import message_filters

class FeatureTracker(Node):
    def __init__(self):
        super().__init__('feature_tracker_node')
        
        self.declare_parameter('cam_config_file')
        cfg_path = self.get_parameter('cam_config_file').get_parameter_value().string_value
        self.cfg = self.load_camera_config(cfg_path)
        self.K = self.cfg["K"]
        self.dist_coeffs = self.cfg["dist_coeffs"]

        # Configuration des publishers
        self.matches_pub = self.create_publisher(Image, '/feature_tracker/feature_img', 10)
        self.image_pub0 = self.create_publisher(Image, "/feature_tracker/feature_img0", 1000)
        self.image_pub1 = self.create_publisher(Image, "/feature_tracker/feature_img1", 1000)
        self.pub_features0 = self.create_publisher(PointCloud, self.cfg["topic_features0"], 1000)
        self.pub_features1 = self.create_publisher(PointCloud, self.cfg["topic_features1"], 1000)

        # Configuration des subscribers
        # self.subscriber0 = self.create_subscription(Image, self.cfg["topic_images0"], self.callback0, 1000)
        # self.subscriber1 = self.create_subscription(Image, self.cfg["topic_images1"], self.callback1, 1000)
        self.subscriber0 = message_filters.Subscriber(self, Image, self.cfg["topic_images0"])
        self.subscriber1 = message_filters.Subscriber(self, Image, self.cfg["topic_images1"])

        # Create time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.subscriber0, self.subscriber1],  # List of subscribers
            queue_size=10,           # Queue size
            slop=0.1                 # Time tolerance in seconds
        )

        self.ts.registerCallback(self.sync_callback)

        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info("the device connected is %s" % self.device)

        # Configuration de l'extracteur et du matcher
        self.extractor_max_num_keypoints = 300
        self.extractor = SuperPoint(max_num_keypoints=self.extractor_max_num_keypoints, nms_radius=4).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        self.target_n_features = 300
        self.img_h = -1
        self.img_w = -1

    def load_camera_config(self, cfg_path):
        self.get_logger().info("[feature_tracker] loading config from %s" % cfg_path)
        fs = cv2.FileStorage(cfg_path, cv2.FILE_STORAGE_READ)
        
        k1 = fs.getNode("distortion_parameters").getNode("k1").real()
        k2 = fs.getNode("distortion_parameters").getNode("k2").real()
        p1 = fs.getNode("distortion_parameters").getNode("p1").real()
        p2 = fs.getNode("distortion_parameters").getNode("p2").real()
        k3 = 0.0
        
        fx = fs.getNode("projection_parameters").getNode("fx").real()
        fy = fs.getNode("projection_parameters").getNode("fy").real()
        cx = fs.getNode("projection_parameters").getNode("cx").real()
        cy = fs.getNode("projection_parameters").getNode("cy").real()
        
        return {
            'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
            'dist_coeffs': np.array([k1, k2, p1, p2, k3]),
            'topic_images0': fs.getNode("topic_images0").string(),
            'topic_images1': fs.getNode("topic_images1").string(),
            'topic_features0': fs.getNode("topic_features0").string(),
            'topic_features1': fs.getNode("topic_features1").string()
        }

    def np_image_to_torch(self, image):
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))
        elif image.ndim == 2:
            image = image[None]
        else:
            raise ValueError(f'Not an image: {image.shape}')
        return torch.tensor(image / 255., dtype=torch.float)

    def undistort_keypoints(self, keypoints):
        points = cv2.undistortPoints(keypoints, self.K, self.dist_coeffs, None, None)
        return points

    def publish_features(self, kpts_data, header, is_cam0=True):
        pc_msg = PointCloud()
        pc_msg.header = header

        channels = {
            'id_of_point': ChannelFloat32(name='id_of_point'),
            'u_of_point': ChannelFloat32(name='u_of_point'),
            'v_of_point': ChannelFloat32(name='v_of_point'),
            'velocity_x_of_point': ChannelFloat32(name='velocity_x_of_point'),
            'velocity_y_of_point': ChannelFloat32(name='velocity_y_of_point'),
            'score_of_point': ChannelFloat32(name='score_of_point')
        }

        kpts = np.array(kpts_data[:, 1:3]).astype(np.float64)
        kpts_undistorted = self.undistort_keypoints(kpts)[:, 0, :]

        for i, pt in enumerate(kpts_undistorted):
            point = Point32()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = 1.0
            pc_msg.points.append(point)
            
            channels['id_of_point'].values.append(float(kpts_data[i, 0]))
            channels['u_of_point'].values.append(float(kpts_data[i, 1]))
            channels['v_of_point'].values.append(float(kpts_data[i, 2]))
            channels['velocity_x_of_point'].values.append(0.0)
            channels['velocity_y_of_point'].values.append(0.0)
            channels['score_of_point'].values.append(float(kpts_data[i, 3]))

        pc_msg.channels = list(channels.values())
        
        if is_cam0:
            self.pub_features0.publish(pc_msg)
        else:
            self.pub_features1.publish(pc_msg)

    def process_image(self, cv_image, header, is_cam0):
        if self.img_h == -1:
            self.img_h = cv_image.shape[0]
            self.img_w = cv_image.shape[1]

        img_torch = self.np_image_to_torch(cv_image).to(self.device)
        features = self.extractor.extract(img_torch)
        
        points = features['keypoints'][0].detach().cpu().numpy()
        scores = features['keypoint_scores'][0].detach().cpu().numpy()
        
        points_data = np.zeros((len(points), 4))
        for i, (point, score) in enumerate(zip(points, scores)):
            points_data[i] = [i, point[0], point[1], score]
            
        self.publish_features(points_data, header, is_cam0)

        # Publication de l'image avec les features
        img_with_features = cv_image.copy()
        if len(img_with_features.shape) == 2:
            img_with_features = cv2.cvtColor(img_with_features, cv2.COLOR_GRAY2BGR)
            
        for point, score in zip(points, scores):
            cv2.circle(img_with_features, (int(point[0]), int(point[1])), 2, (0, 255, 0), 2)
            
        if is_cam0:
            self.image_pub0.publish(self.bridge.cv2_to_imgmsg(img_with_features, encoding="bgr8"))
        else:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(img_with_features, encoding="bgr8"))


    

    # def callback0(self, ros_data):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(ros_data)
    #         self.process_image(cv_image, ros_data.header, True)
    #     except CvBridgeError as e:
    #         self.get_logger().error(f"Error processing image from camera 0: {str(e)}")

    # def callback1(self, ros_data):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(ros_data)
    #         self.process_image(cv_image, ros_data.header, False)
    #     except CvBridgeError as e:
    #         self.get_logger().error(f"Error processing image from camera 1: {str(e)}")


    def sync_callback(self, msg0, msg1):
        try:
            # Convert ROS messages to OpenCV images
            cv_image0 = self.bridge.imgmsg_to_cv2(msg0)
            cv_image1 = self.bridge.imgmsg_to_cv2(msg1)

            self.process_image(cv_image0, msg0.header, True)
            self.process_image(cv_image1, msg1.header, False)

            # Convert to torch tensors
            img_torch0 = self.np_image_to_torch(cv_image0).to(self.device)
            img_torch1 = self.np_image_to_torch(cv_image1).to(self.device)

            # Extract features from both images
            feats0 = self.extractor.extract(img_torch0)
            feats1 = self.extractor.extract(img_torch1)

            # Match features
            matches01 = self.matcher({"image0": feats0, "image1": feats1})

            # Remove batch dimension using rbd
            feats0, feats1, matches01 = [
                rbd(x) for x in [feats0, feats1, matches01]
            ]

            # Get keypoints and matches
            kpts0 = feats0["keypoints"].detach().cpu().numpy()
            kpts1 = feats1["keypoints"].detach().cpu().numpy()
            scores0 = feats0["keypoint_scores"].detach().cpu().numpy()
            scores1 = feats1["keypoint_scores"].detach().cpu().numpy()
            matches = matches01["matches"].detach().cpu().numpy()

            # Créer l'image de visualisation
            img0_vis = cv_image0.copy()
            img1_vis = cv_image1.copy()
            
            # Convertir en RGB si nécessaire
            if len(img0_vis.shape) == 2:
                img0_vis = cv2.cvtColor(img0_vis, cv2.COLOR_GRAY2BGR)
                img1_vis = cv2.cvtColor(img1_vis, cv2.COLOR_GRAY2BGR)

            # Concatener les images horizontalement
            H, W = img0_vis.shape[:2]
            matched_img = np.hstack([img0_vis, img1_vis])

            # Dessiner les matches
            for (idx0, idx1) in matches:
                pt0 = (int(kpts0[idx0][0]), int(kpts0[idx0][1]))
                pt1 = (int(kpts1[idx1][0]) + W, int(kpts1[idx1][1]))
                cv2.line(matched_img, pt0, pt1, (0, 255, 0), 1)
                cv2.circle(matched_img, pt0, 3, (0, 255, 0), -1)
                cv2.circle(matched_img, pt1, 3, (0, 255, 0), -1)

            self.matches_pub.publish(self.bridge.cv2_to_imgmsg(matched_img, encoding="bgr8"))

            # Process and publish features for each image
            points_data0 = np.zeros((len(kpts0), 4))
            points_data1 = np.zeros((len(kpts1), 4))
            
            for i, (point, score) in enumerate(zip(kpts0, scores0)):
                points_data0[i] = [i, point[0], point[1], score]
            
            for i, (point, score) in enumerate(zip(kpts1, scores1)):
                points_data1[i] = [i, point[0], point[1], score]

            # Publish features
            self.publish_features(points_data0, msg0.header, True)
            self.publish_features(points_data1, msg1.header, False)

        except Exception as e:
            self.get_logger().error(f'Error in sync_callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    feature_tracker = FeatureTracker()
    rclpy.spin(feature_tracker)
    feature_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
