#!/usr/bin/env python3

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


class StereoFeatureTracker(Node):
    def __init__(self):
        super().__init__('stereo_feature_tracker_node')
        self.declare_parameter('cam_config_file')
        cfg_path = self.get_parameter('cam_config_file').get_parameter_value().string_value
        self.cfg = self.load_camera_config(cfg_path)

        # Camera intrinsics/distortion for both cameras (assumed same model for simplicity)
        # If they differ, you can extend the config file to hold separate intrinsics
        self.K = self.cfg["K"]
        self.dist_coeffs = self.cfg["dist_coeffs"]

        # Publishers
        self.image_pub0 = self.create_publisher(Image, "/feature_tracker/feature_img0", 10)
        self.image_pub1 = self.create_publisher(Image, "/feature_tracker/feature_img1", 10)
        self.pub_features0 = self.create_publisher(PointCloud, self.cfg["topic_features0"], 10)
        self.pub_features1 = self.create_publisher(PointCloud, self.cfg["topic_features1"], 10)

        # Subscribers
        self.subscriber0 = message_filters.Subscriber(self, Image, self.cfg["topic_images0"])
        self.subscriber1 = message_filters.Subscriber(self, Image, self.cfg["topic_images1"])

        # Synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.subscriber0, self.subscriber1],
            queue_size=100,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        self.bridge = CvBridge()

        # Just to optionally skip frames
        self.skip_n = 1
        self.skip_n_curr = 0

        # Image and feature info for camera 0 (left)
        self.img_prev0 = None
        self.feat_prev0 = None
        self.feat_prev_order_to_id0 = []
        self.feat_obs_cnt0 = [0] * 10000
        self.cnt_id0 = 0

        # Image and feature info for camera 1 (right)
        self.img_prev1 = None
        self.feat_prev1 = None
        self.feat_prev_order_to_id1 = []
        self.feat_obs_cnt1 = [0] * 10000
        self.cnt_id1 = 0

        # Counters
        self.cnt = 0
        self.img_h = -1
        self.img_w = -1

        # Device and feature extractors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor_max_num_keypoints = 10000
        self.extractor = SuperPoint(max_num_keypoints=self.extractor_max_num_keypoints, nms_radius=4).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        # Target number of features to keep from temporal matching
        self.target_n_features = 1000

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

        # You can extend your config file with separate topics for camera1 if needed
        return {
            'K': np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1.0]]),
            'dist_coeffs': np.array([k1, k2, p1, p2, k3]),
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

    def undistort_keypoints(self, keypoints):
        points = cv2.undistortPoints(keypoints, self.K, self.dist_coeffs, None, None)
        return points

    def publish_features(self, kpts_data, header, pc_pub):
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
        pc_pub.publish(pc_msg)

    def sort_matches(self, matches, scores):
        indices_match = torch.argsort(scores, descending=True)
        return matches[indices_match], scores[indices_match]

    def match_and_assign_ids(
        self,
        feat_prev,
        feat_curr,
        feat_prev_id_map,
        feat_obs_cnt,
        cnt_id,
        target_n_features
    ):
        """
        Temporal matching from feat_prev -> feat_curr. Return:
        - updated feat_curr_id_map
        - updated feat_obs_cnt
        - updated cnt_id
        - points_to_plot: shape Nx5 => [ feat_id, x, y, match_score, curr_kp_idx ]
        """
        if feat_prev is None:
            # First frame => initialize everything as -1
            feat_curr_id_map = np.zeros(self.extractor_max_num_keypoints) - 1
            return feat_curr_id_map, feat_obs_cnt, cnt_id, np.empty((0, 5))

        matches_data = self.matcher({'image0': feat_prev, 'image1': feat_curr})
        scores = matches_data['scores'][0]
        matches = matches_data['matches'][0]
        matches, scores = self.sort_matches(matches, scores)

        points_curr = feat_curr['keypoints'][0].detach().cpu().numpy()
        feat_curr_id_map = np.zeros(self.extractor_max_num_keypoints) - 1

        points_to_plot = []
        cnt_kept = 0

        for i_m, match in enumerate(matches):
            idx_curr = match[1].item()
            idx_prev = match[0].item()

            # We can define a threshold if we want
            # if scores[i_m] < 0.8:  # Example threshold
            #     continue

            feat_id = int(feat_prev_id_map[idx_prev])
            coords_curr = points_curr[idx_curr].astype(float)

            # If we already know this feature
            if feat_id > -1:
                feat_curr_id_map[idx_curr] = feat_id
                feat_obs_cnt[feat_id] += 1
                if feat_obs_cnt[feat_id] > 2:
                    points_to_plot.append([feat_id, coords_curr[0], coords_curr[1], scores[i_m].item(), idx_curr])
            else:
                # New feature
                cnt_id += 1
                if cnt_id >= len(feat_obs_cnt):
                    feat_obs_cnt += [0] * 1000
                feat_id = cnt_id
                feat_curr_id_map[idx_curr] = feat_id
                feat_obs_cnt[feat_id] = 1

            cnt_kept += 1
            if cnt_kept >= target_n_features:
                break

        return (
            feat_curr_id_map,
            feat_obs_cnt,
            cnt_id,
            np.array(points_to_plot, dtype=np.float32)
        )

    def keep_only_stereo_matches(
        self,
        feat_curr0,
        feat_curr1,
        points_to_plot0,
        points_to_plot1
    ):
        """
        Cross-match between feat_curr0 and feat_curr1, keep only the features
        that match between left and right. points_to_plotX is Nx5:
         [feat_id, x, y, score, kp_idx].
        Returns filtered points_to_plot0, points_to_plot1.
        """

        # Cross matching
        stereo_data = self.matcher({'image0': feat_curr0, 'image1': feat_curr1})
        stereo_scores = stereo_data['scores'][0]
        stereo_matches = stereo_data['matches'][0]
        stereo_matches, stereo_scores = self.sort_matches(stereo_matches, stereo_scores)

        # Build sets from cross-matches
        matched_idx0 = set()
        matched_idx1 = set()
        for i_m, match in enumerate(stereo_matches):
            # Optionally threshold stereo_scores[i_m]
            idx0 = match[0].item()  # kp index in feat_curr0
            idx1 = match[1].item()  # kp index in feat_curr1
            matched_idx0.add(idx0)
            matched_idx1.add(idx1)

        # Filter points_to_plot0
        filtered_pts0 = []
        for row in points_to_plot0:
            kp_idx = int(row[4])
            if kp_idx in matched_idx0:
                filtered_pts0.append(row)

        # Filter points_to_plot1
        filtered_pts1 = []
        for row in points_to_plot1:
            kp_idx = int(row[4])
            if kp_idx in matched_idx1:
                filtered_pts1.append(row)

        return np.array(filtered_pts0), np.array(filtered_pts1)

    def sync_callback(self, ros_data0, ros_data1):
        self.get_logger().info("sync_callback a été appelé")

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

        # Convert and extract features for each camera
        img_curr0 = self.np_image_to_torch(cv_image0).to(self.device)
        feat_curr0 = self.extractor.extract(img_curr0)
        img_curr1 = self.np_image_to_torch(cv_image1).to(self.device)
        feat_curr1 = self.extractor.extract(img_curr1)

        # Temporal matching for camera 0
        feat_curr_order_to_id0, self.feat_obs_cnt0, self.cnt_id0, pts0 = self.match_and_assign_ids(
            self.feat_prev0,
            feat_curr0,
            self.feat_prev_order_to_id0 if self.feat_prev_order_to_id0 != [] else None,
            self.feat_obs_cnt0,
            self.cnt_id0,
            self.target_n_features
        )

        # Temporal matching for camera 1
        feat_curr_order_to_id1, self.feat_obs_cnt1, self.cnt_id1, pts1 = self.match_and_assign_ids(
            self.feat_prev1,
            feat_curr1,
            self.feat_prev_order_to_id1 if self.feat_prev_order_to_id1 != [] else None,
            self.feat_obs_cnt1,
            self.cnt_id1,
            self.target_n_features
        )

        # Keep only stereo matches for current frame
        pts0_stereo, pts1_stereo = self.keep_only_stereo_matches(
            feat_curr0,
            feat_curr1,
            pts0,
            pts1
        )

        # Publish feature pointclouds
        if pts0_stereo.shape[0] > 0:
            self.publish_features(pts0_stereo[:, :4], ros_data0.header, self.pub_features0)
        if pts1_stereo.shape[0] > 0:
            self.publish_features(pts1_stereo[:, :4], ros_data1.header, self.pub_features1)

        # Draw features on images to display
        img_disp0 = cv2.normalize(cv_image0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_disp0 = cv2.cvtColor(img_disp0, cv2.COLOR_GRAY2BGR) if len(img_disp0.shape) == 2 else img_disp0
        for row in pts0_stereo:
            feat_id = int(row[0])
            x, y = int(row[1]), int(row[2])
            color = (0, 255, 0) if self.feat_obs_cnt0[feat_id] < 16 else (0, 0, 255)
            cv2.circle(img_disp0, (x, y), 2, color, 2)

        img_disp1 = cv2.normalize(cv_image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_disp1 = cv2.cvtColor(img_disp1, cv2.COLOR_GRAY2BGR) if len(img_disp1.shape) == 2 else img_disp1
        for row in pts1_stereo:
            feat_id = int(row[0])
            x, y = int(row[1]), int(row[2])
            color = (0, 255, 0) if self.feat_obs_cnt1[feat_id] < 16 else (0, 0, 255)
            cv2.circle(img_disp1, (x, y), 2, color, 2)

        self.image_pub0.publish(self.bridge.cv2_to_imgmsg(img_disp0, encoding="bgr8"))
        self.image_pub1.publish(self.bridge.cv2_to_imgmsg(img_disp1, encoding="bgr8"))

        # Prepare for next iteration
        self.feat_prev0 = feat_curr0
        self.feat_prev_order_to_id0 = feat_curr_order_to_id0
        self.feat_prev1 = feat_curr1
        self.feat_prev_order_to_id1 = feat_curr_order_to_id1
        self.cnt += 1

    def main(self, args=None):
        rclpy.spin(self)
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = StereoFeatureTracker()
    node.main(args)
