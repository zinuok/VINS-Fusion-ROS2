#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import message_filters

from lightglue_feature_tracker.lightglue import LightGlue, SuperPoint


class FeatureTracker(Node):
    def __init__(self):
        super().__init__('feature_tracker_node')
        self.declare_parameter('cam_config_file')
        cfg_path = self.get_parameter('cam_config_file').get_parameter_value().string_value
        self.cfg = self.load_camera_config(cfg_path)
        self.K = self.cfg["K"]  # Parameters for the left camera
        self.dist_coeffs = self.cfg["dist_coeffs"]
        self.K1 = self.cfg["K1"]  # Parameters for the right camera
        self.dist_coeffs1 = self.cfg["dist_coeffs1"]

        self.image_pub0 = self.create_publisher(Image, "/feature_tracker/feature_img0", 10)
        self.image_pub1 = self.create_publisher(Image, "/feature_tracker/feature_img1", 10)
        self.pub_features0 = self.create_publisher(PointCloud, self.cfg["topic_features0"], 10)
        self.pub_features1 = self.create_publisher(PointCloud, self.cfg["topic_features1"], 10)

        self.subscriber0 = message_filters.Subscriber(self, Image, self.cfg["topic_images0"])
        self.subscriber1 = message_filters.Subscriber(self, Image, self.cfg["topic_images1"])

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.subscriber0, self.subscriber1],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        self.bridge = CvBridge()

        self.skip_n = 1
        self.skip_n_curr = 0
        self.img_curr0 = None
        self.img_prev0 = None
        self.img_curr1 = None
        self.feat_curr0 = None
        self.feat_prev0 = None
        self.feat_curr1 = None
        self.feat_prev_order_to_id0 = []
        self.feat_curr_order_to_id0 = []
        self.feat_obs_cnt0 = [0] * 10000
        self.cnt = 0
        self.cnt_id0 = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor_max_num_keypoints = 1000
        self.extractor = SuperPoint(max_num_keypoints=self.extractor_max_num_keypoints, nms_radius=4).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        self.target_n_features = 1000
        self.img_h = -1
        self.img_w = -1

    def load_camera_config(self, cfg_path):
        fs = cv2.FileStorage(cfg_path, cv2.FILE_STORAGE_READ)
        
        # Left parameters
        k1 = fs.getNode("distortion_parameters").getNode("k1").real()
        k2 = fs.getNode("distortion_parameters").getNode("k2").real()
        p1 = fs.getNode("distortion_parameters").getNode("p1").real()
        p2 = fs.getNode("distortion_parameters").getNode("p2").real()
        k3 = 0.0
        fx = fs.getNode("projection_parameters").getNode("fx").real()
        fy = fs.getNode("projection_parameters").getNode("fy").real()
        cx = fs.getNode("projection_parameters").getNode("cx").real()
        cy = fs.getNode("projection_parameters").getNode("cy").real()
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        
        # Right parameters
        k1_1 = fs.getNode("distortion_parameters1").getNode("k1").real()
        k2_1 = fs.getNode("distortion_parameters1").getNode("k2").real()
        p1_1 = fs.getNode("distortion_parameters1").getNode("p1").real()
        p2_1 = fs.getNode("distortion_parameters1").getNode("p2").real()
        k3_1 = 0.0
        fx1 = fs.getNode("projection_parameters1").getNode("fx").real()
        fy1 = fs.getNode("projection_parameters1").getNode("fy").real()
        cx1 = fs.getNode("projection_parameters1").getNode("cx").real()
        cy1 = fs.getNode("projection_parameters1").getNode("cy").real()
        K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1.0]])
        dist_coeffs1 = np.array([k1_1, k2_1, p1_1, p2_1, k3_1])

        return {
            'K': K,
            'dist_coeffs': dist_coeffs,
            'K1': K1,
            'dist_coeffs1': dist_coeffs1,
            'topic_images0': fs.getNode("topic_images0").string(),
            'topic_images1': fs.getNode("topic_images1").string(),
            'topic_features0': fs.getNode("topic_features0").string(),
            'topic_features1': fs.getNode("topic_features1").string()
        }

    def np_image_to_torch(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))
        elif image.ndim == 2:
            image = image[None]
        else:
            raise ValueError(f'Not an image: {image.shape}')
        return torch.tensor(image / 255., dtype=torch.float)

    def undistort_keypoints(self, keypoints, K, dist_coeffs):
        points = cv2.undistortPoints(keypoints, K, dist_coeffs, None, None)
        return points

    def publish_features(self, kpts_data, header, pub_features, topic):
        kpts_ids = np.array(kpts_data[:, 0]).astype(int)
        kpts = np.array(kpts_data[:, 1:3]).astype(np.float64)
        
        if topic == self.cfg["topic_features1"]:
            current_K = self.K1
            current_dist_coeffs = self.dist_coeffs1
        else:
            current_K = self.K
            current_dist_coeffs = self.dist_coeffs
        
        kpts_undistorted = self.undistort_keypoints(kpts, current_K, current_dist_coeffs)[:, 0, :]
        kpts_n_obs = kpts_data[:, 3]

        pc_msg = PointCloud()
        pc_msg.header = header

        id_of_point = ChannelFloat32()
        u_of_point = ChannelFloat32()
        v_of_point = ChannelFloat32()
        velocity_x_of_point = ChannelFloat32()
        velocity_y_of_point = ChannelFloat32()
        score_of_point = ChannelFloat32()
        id_of_point.name = 'id_of_point'
        u_of_point.name = 'u_of_point'
        v_of_point.name = 'v_of_point'
        velocity_x_of_point.name = 'velocity_x_of_point'
        velocity_y_of_point.name = 'velocity_y_of_point'
        score_of_point.name = 'score_of_point'

        for i_p, pt in enumerate(kpts_undistorted):
            point = Point32()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = 1.0
            pc_msg.points.append(point)
            id_of_point.values.append(kpts_ids[i_p])
            u_of_point.values.append(kpts[i_p, 0])
            v_of_point.values.append(kpts[i_p, 1])
            velocity_x_of_point.values.append(0.0)
            velocity_y_of_point.values.append(0.0)
            score_of_point.values.append(kpts_n_obs[i_p])

        pc_msg.channels.append(id_of_point)
        pc_msg.channels.append(u_of_point)
        pc_msg.channels.append(v_of_point)
        pc_msg.channels.append(velocity_x_of_point)
        pc_msg.channels.append(velocity_y_of_point)
        pc_msg.channels.append(score_of_point)
        pub_features.publish(pc_msg)

    def sort_matches(self, matches, scores):
        indices_match = torch.argsort(scores)
        return matches[indices_match]

    def sync_callback(self, ros_data0, ros_data1):
        self.skip_n_curr += 1
        if (self.skip_n_curr - 1) % self.skip_n != 0:
            return
        try:
            cv_image0 = self.bridge.imgmsg_to_cv2(ros_data0)
            cv_image1 = self.bridge.imgmsg_to_cv2(ros_data1)
        except:
            return

        if self.cnt == 0:
            self.img_h, self.img_w = cv_image0.shape[:2]

        self.img_curr0 = self.np_image_to_torch(cv_image0).to(self.device)
        self.img_curr1 = self.np_image_to_torch(cv_image1).to(self.device)
        self.feat_curr0 = self.extractor.extract(self.img_curr0)
        self.feat_curr1 = self.extractor.extract(self.img_curr1)

        if self.cnt == 1:
            self.feat_prev_order_to_id0 = np.zeros(self.extractor_max_num_keypoints) - 1
            self.cnt_id0 = 0

        if self.cnt > 0:
            matches_data = self.matcher({'image0': self.feat_prev0, 'image1': self.feat_curr0})
            scores = matches_data['scores'][0]
            matches = matches_data['matches'][0]
            points_curr0 = self.feat_curr0['keypoints'][0].detach().cpu().numpy()
            self.feat_curr_order_to_id0 = np.zeros(self.extractor_max_num_keypoints) - 1

            matches = self.sort_matches(matches, scores)
            cnt_plot0 = 0
            points_to_plot0 = np.zeros((2000, 4)).astype(np.float32)

            cnt_matches = 0
            for i_m, match in enumerate(matches):
                coords_curr0 = points_curr0[match[1]].astype(float)
                feat_id = int(self.feat_prev_order_to_id0[match[0]])
                if feat_id > -1:
                    self.feat_curr_order_to_id0[match[1]] = feat_id
                    self.feat_obs_cnt0[feat_id] += 1
                    if self.feat_obs_cnt0[feat_id] > 2:
                        score_match = scores[i_m].detach().cpu().numpy()
                        points_to_plot0[cnt_plot0] = [feat_id, coords_curr0[0], coords_curr0[1], score_match]
                        cnt_plot0 += 1
                else:
                    self.cnt_id0 += 1
                    if self.cnt_id0 > len(self.feat_obs_cnt0) - 1:
                        self.feat_obs_cnt0 += [0] * 1000
                    feat_id = self.cnt_id0
                    self.feat_curr_order_to_id0[match[1]] = feat_id
                    self.feat_obs_cnt0[feat_id] = 1
                cnt_matches += 1
                if cnt_matches >= self.target_n_features:
                    break

            if cnt_plot0 > 1:
                img_matches0 = cv2.normalize(cv_image0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_matches0 = cv2.cvtColor(img_matches0, cv2.COLOR_GRAY2BGR) if len(img_matches0.shape) == 2 else img_matches0
                for i in range(cnt_plot0):
                    feat_id = int(points_to_plot0[i, 0])
                    coords = (int(points_to_plot0[i, 1]), int(points_to_plot0[i, 2]))
                    color = (0, 0, 255) if self.feat_obs_cnt0[feat_id] > 16 else (0, 255, 0)
                    cv2.circle(img_matches0, coords, 2, color, 2)
                self.image_pub0.publish(self.bridge.cv2_to_imgmsg(img_matches0, encoding="bgr8"))

    def sync_callback(self, ros_data0, ros_data1):
        self.skip_n_curr += 1
        if (self.skip_n_curr - 1) % self.skip_n != 0:
            return
        try:
            cv_image0 = self.bridge.imgmsg_to_cv2(ros_data0)
            cv_image1 = self.bridge.imgmsg_to_cv2(ros_data1)
        except Exception as e:
            self.get_logger().error(f"Conversion image error: {e}")
            return

        if self.cnt == 0:
            self.img_h, self.img_w = cv_image0.shape[:2]

        self.img_curr0 = self.np_image_to_torch(cv_image0).to(self.device)
        self.img_curr1 = self.np_image_to_torch(cv_image1).to(self.device)
        self.feat_curr0 = self.extractor.extract(self.img_curr0)
        self.feat_curr1 = self.extractor.extract(self.img_curr1)

        if self.cnt == 0:
            # Initialisation
            self.feat_prev0 = self.feat_curr0
            n_prev = self.feat_prev0['keypoints'][0].shape[0]
            self.feat_prev_order_to_id0 = np.full(n_prev, -1, dtype=int)
            self.cnt_id0 = 0
        else:
            n_curr = self.feat_curr0['keypoints'][0].shape[0]
            self.feat_curr_order_to_id0 = np.full(n_curr, -1, dtype=int)

            # Temporal matching (only left)
            matches_data = self.matcher({'image0': self.feat_prev0, 'image1': self.feat_curr0})
            scores = matches_data['scores'][0]
            matches = matches_data['matches'][0]
            points_curr0 = self.feat_curr0['keypoints'][0].detach().cpu().numpy()

            for i_m, match in enumerate(matches):
                prev_idx = match[0]
                curr_idx = match[1]
                if prev_idx >= len(self.feat_prev_order_to_id0):
                    continue
                prev_feat_id = int(self.feat_prev_order_to_id0[prev_idx])
                if prev_feat_id > -1:
                    self.feat_curr_order_to_id0[curr_idx] = prev_feat_id
                    self.feat_obs_cnt0[prev_feat_id] += 1
                else:
                    self.cnt_id0 += 1
                    self.feat_curr_order_to_id0[curr_idx] = self.cnt_id0
                    if self.cnt_id0 >= len(self.feat_obs_cnt0):
                        self.feat_obs_cnt0 += [0] * 1000
                    self.feat_obs_cnt0[self.cnt_id0] = 1

            # Stereo matching (left -> right)
            matches_data_lr = self.matcher({'image0': self.feat_curr0, 'image1': self.feat_curr1})
            scores_lr = matches_data_lr['scores'][0]
            matches_lr = matches_data_lr['matches'][0]
            points_curr1 = self.feat_curr1['keypoints'][0].detach().cpu().numpy()

            stereo_points_left = []
            stereo_points_right = []

            for i_m, match_lr in enumerate(matches_lr):
                left_idx = match_lr[0]
                right_idx = match_lr[1]
                feat_id = int(self.feat_curr_order_to_id0[left_idx])
                if feat_id > -1:
                    left_coord = points_curr0[left_idx]
                    right_coord = points_curr1[right_idx]
                    score = scores_lr[i_m].detach().cpu().numpy()
                    stereo_points_left.append([feat_id, left_coord[0], left_coord[1], score])
                    stereo_points_right.append([feat_id, right_coord[0], right_coord[1], score])

            if len(stereo_points_left) > 1:
                stereo_points_left = np.array(stereo_points_left, dtype=np.float32)
                stereo_points_right = np.array(stereo_points_right, dtype=np.float32)

                self.publish_features(stereo_points_left, ros_data0.header, self.pub_features0, self.cfg["topic_features0"])
                self.publish_features(stereo_points_right, ros_data1.header, self.pub_features1, self.cfg["topic_features1"])

                # Visualisation
                img_matches0 = cv2.normalize(cv_image0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if len(img_matches0.shape) == 2:
                    img_matches0 = cv2.cvtColor(img_matches0, cv2.COLOR_GRAY2BGR)
                for pt in stereo_points_left:
                    feat_id, u, v, _ = pt
                    color = (0, 0, 255) if self.feat_obs_cnt0[int(feat_id)] > 16 else (0, 255, 0)
                    cv2.circle(img_matches0, (int(u), int(v)), 2, color, 2)
                self.image_pub0.publish(self.bridge.cv2_to_imgmsg(img_matches0, encoding="bgr8"))

                img_matches1 = cv2.normalize(cv_image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if len(img_matches1.shape) == 2:
                    img_matches1 = cv2.cvtColor(img_matches1, cv2.COLOR_GRAY2BGR)
                for pt in stereo_points_right:
                    feat_id, u, v, _ = pt
                    color = (0, 0, 255) if self.feat_obs_cnt0[int(feat_id)] > 16 else (0, 255, 0)
                    cv2.circle(img_matches1, (int(u), int(v)), 2, color, 2)
                self.image_pub1.publish(self.bridge.cv2_to_imgmsg(img_matches1, encoding="bgr8"))

        self.img_prev0 = self.img_curr0
        self.feat_prev0 = self.feat_curr0
        self.feat_prev_order_to_id0 = self.feat_curr_order_to_id0.copy()
        self.cnt += 1


def main(args=None):
    rclpy.init(args=args)
    feature_tracker = FeatureTracker()
    rclpy.spin(feature_tracker)
    feature_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()