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
        self.K = self.cfg["K"]
        self.dist_coeffs = self.cfg["dist_coeffs"]

        self.image_pub = self.create_publisher(Image, "/feature_tracker/feature_img0", 1000)
        self.pub_features = self.create_publisher(PointCloud, self.cfg["topic_features0"], 1000)

        self.use_compressed_input = False
        self.subscriber = message_filters.Subscriber(self, Image, self.cfg["topic_images0"])
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.subscriber],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        self.bridge = CvBridge()

        self.skip_n = 1
        self.skip_n_curr = 0
        self.img_curr = None
        self.img_prev = None
        self.feat_curr = None
        self.feat_prev = None
        self.feat_prev_order_to_id = []
        self.feat_curr_order_to_id = []
        self.feat_obs_cnt = [0] * 10000
        self.cnt = 0
        self.cnt_id = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor_max_num_keypoints = 700
        self.extractor = SuperPoint(max_num_keypoints=self.extractor_max_num_keypoints, nms_radius=4).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        self.target_n_features = 700
        self.img_h = -1
        self.img_w = -1

    def load_camera_config(self, cfg_path):
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
            'K': np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1.0]]),
            'dist_coeffs': np.array([k1, k2, p1, p2, k3]),
            'topic_images0': fs.getNode("topic_images0").string(),
            'topic_features0': fs.getNode("topic_features0").string()
        }

    def np_image_to_torch(self, image: np.ndarray) -> torch.Tensor:
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

    def publish_features(self, kpts_data, header):
        kpts_ids = np.array(kpts_data[:, 0]).astype(int)
        kpts = np.array(kpts_data[:, 1:3]).astype(np.float64)
        kpts_undistorted = self.undistort_keypoints(kpts)[:, 0, :]
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
        self.pub_features.publish(pc_msg)

    def sort_matches(self, matches, scores):
        indices_match = torch.argsort(scores)
        return matches[indices_match]

    def sync_callback(self, ros_data):
        self.skip_n_curr += 1
        if (self.skip_n_curr - 1) % self.skip_n != 0:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_data)
        except:
            return

        if self.cnt == 0:
            self.img_h, self.img_w = cv_image.shape[:2]

        self.img_curr = self.np_image_to_torch(cv_image).to(self.device)
        self.feat_curr = self.extractor.extract(self.img_curr)

        if self.cnt == 1:
            self.feat_prev_order_to_id = np.zeros(self.extractor_max_num_keypoints) - 1
            self.cnt_id = 0

        if self.cnt > 0:
            matches_data = self.matcher({'image0': self.feat_prev, 'image1': self.feat_curr})
            scores = matches_data['scores'][0]
            matches = matches_data['matches'][0]
            points_curr = self.feat_curr['keypoints'][0].detach().cpu().numpy()
            self.feat_curr_order_to_id = np.zeros(self.extractor_max_num_keypoints) - 1

            matches = self.sort_matches(matches, scores)
            cnt_plot = 0
            points_to_plot = np.zeros((2000, 4)).astype(np.float32)

            cnt_matches = 0
            for i_m, match in enumerate(matches):
                coords_curr = points_curr[match[1]].astype(float)
                feat_id = int(self.feat_prev_order_to_id[match[0]])
                if feat_id > -1:
                    self.feat_curr_order_to_id[match[1]] = feat_id
                    self.feat_obs_cnt[feat_id] += 1
                    if self.feat_obs_cnt[feat_id] > 2:
                        score_match = scores[i_m].detach().cpu().numpy()
                        points_to_plot[cnt_plot] = [feat_id, coords_curr[0], coords_curr[1], score_match]
                        cnt_plot += 1
                else:
                    self.cnt_id += 1
                    if self.cnt_id > len(self.feat_obs_cnt) - 1:
                        self.feat_obs_cnt += [0] * 1000
                    feat_id = self.cnt_id
                    self.feat_curr_order_to_id[match[1]] = feat_id
                    self.feat_obs_cnt[feat_id] = 1
                cnt_matches += 1
                if cnt_matches >= self.target_n_features:
                    break

            if cnt_plot > 1:
                self.publish_features(points_to_plot[:cnt_plot], ros_data.header)
                img_matches = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_matches = cv2.cvtColor(img_matches, cv2.COLOR_GRAY2BGR) if len(img_matches.shape) == 2 else img_matches
                for i in range(cnt_plot):
                    feat_id = int(points_to_plot[i, 0])
                    coords = (int(points_to_plot[i, 1]), int(points_to_plot[i, 2]))
                    color = (0, 0, 255) if self.feat_obs_cnt[feat_id] > 16 else (0, 255, 0)
                    cv2.circle(img_matches, coords, 2, color, 2)
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(img_matches, encoding="bgr8"))

        self.img_prev = self.img_curr
        self.feat_prev = self.feat_curr
        self.feat_prev_order_to_id = self.feat_curr_order_to_id
        self.cnt += 1


def main(args=None):
    rclpy.init(args=args)
    feature_tracker = FeatureTracker()
    rclpy.spin(feature_tracker)
    feature_tracker.destroy_node()
    rclpy.shutdown()
