#!/usr/bin/env python3

import sys
from lightglue_feature_tracker.lightglue import LightGlue, SuperPoint
from lightglue_feature_tracker.lightglue.utils import rbd, numpy_image_to_torch
import rclpy
from sensor_msgs.msg import Image, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge, CvBridgeError
import cv2
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

        self.image_pub0 = self.create_publisher(Image, "/feature_tracker/feature_img0", 1000)
        self.image_pub1 = self.create_publisher(Image, "/feature_tracker/feature_img1", 1000)
        self.pub_features0 = self.create_publisher(PointCloud, self.cfg["topic_features0"], 1000)
        self.pub_features1 = self.create_publisher(PointCloud, self.cfg["topic_features1"], 1000)
        self.matches_pub = self.create_publisher(Image, '/feature_tracker/stereo_matches_img', 1000)

        self.subscriber0 = message_filters.Subscriber(self, Image, self.cfg["topic_images0"])
        self.subscriber1 = message_filters.Subscriber(self, Image, self.cfg["topic_images1"])
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.subscriber0, self.subscriber1],
            queue_size=100,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info("the device connected is %s" % self.device)

        self.extractor_max_num_keypoints = 1000
        self.extractor = SuperPoint(max_num_keypoints=self.extractor_max_num_keypoints, nms_radius=4).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        self.target_n_features = 1000
        self.img_h0 = -1
        self.img_w0 = -1
        self.img_h1 = -1
        self.img_w1 = -1

        self.skip_n = 1
        self.skip_n_curr = 0
        self.cnt = 0

        self.img_prev0 = None
        self.feat_prev0 = None
        self.feat_prev_order_to_id0 = np.zeros(self.extractor_max_num_keypoints) - 1

        self.img_prev1 = None
        self.feat_prev1 = None
        self.feat_prev_order_to_id1 = np.zeros(self.extractor_max_num_keypoints) - 1

        self.next_feature_id = 0  
        self.feat_obs_cnt = [0] * 10000

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
        points_undist = cv2.undistortPoints(keypoints, self.K, self.dist_coeffs, None, None)
        return points_undist

    def publish_features(self, kpts_data, header, is_cam0=True):
        pc_msg = PointCloud()
        pc_msg.header = header

        id_of_point = ChannelFloat32(name='id_of_point')
        u_of_point = ChannelFloat32(name='u_of_point')
        v_of_point = ChannelFloat32(name='v_of_point')
        vx_of_point = ChannelFloat32(name='velocity_x_of_point')
        vy_of_point = ChannelFloat32(name='velocity_y_of_point')
        score_of_point = ChannelFloat32(name='score_of_point')

        kpts = np.array(kpts_data[:, 1:3]).astype(np.float64)
        kpts_undist = self.undistort_keypoints(kpts)
        kpts_undist = kpts_undist[:, 0, :]

        for i_p, pt in enumerate(kpts_undist):
            point = Point32()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = 1.0
            pc_msg.points.append(point)

            feat_id = kpts_data[i_p, 0]
            id_of_point.values.append(feat_id)
            u_of_point.values.append(kpts_data[i_p, 1])
            v_of_point.values.append(kpts_data[i_p, 2])
            vx_of_point.values.append(0.0)
            vy_of_point.values.append(0.0)
            score_of_point.values.append(kpts_data[i_p, 3])

        pc_msg.channels.append(id_of_point)
        pc_msg.channels.append(u_of_point)
        pc_msg.channels.append(v_of_point)
        pc_msg.channels.append(vx_of_point)
        pc_msg.channels.append(vy_of_point)
        pc_msg.channels.append(score_of_point)

        if is_cam0:
            self.pub_features0.publish(pc_msg)
        else:
            self.pub_features1.publish(pc_msg)

    def sort_matches_by_score(self, matches, scores):
        indices = torch.argsort(scores, descending=True)
        return matches[indices]

    def unify_stereo_ids(self, matches, order_id_left, order_id_right):
        matches_np = matches.detach().cpu().numpy()
        for (idx_left, idx_right) in matches_np:
            id_l = order_id_left[idx_left]
            id_r = order_id_right[idx_right]
            if id_l < 0 and id_r < 0:
                new_id = self.next_feature_id
                self.next_feature_id += 1
                if new_id >= len(self.feat_obs_cnt):
                    self.feat_obs_cnt += [0]*2000
                order_id_left[idx_left] = new_id
                order_id_right[idx_right] = new_id
                self.feat_obs_cnt[new_id] = 1
            elif id_l >= 0 and id_r < 0:
                order_id_right[idx_right] = id_l
                self.feat_obs_cnt[int(id_l)] += 1
            elif id_l < 0 and id_r >= 0:
                order_id_left[idx_left] = id_r
                self.feat_obs_cnt[int(id_r)] += 1
            else:
                if id_l != id_r:
                    pass
        return order_id_left, order_id_right

    def sync_callback(self, msg0, msg1):
        self.skip_n_curr += 1
        if (self.skip_n_curr - 1) % self.skip_n != 0:
            return

        try:
            cv_image0 = self.bridge.imgmsg_to_cv2(msg0)
            cv_image1 = self.bridge.imgmsg_to_cv2(msg1)
        except CvBridgeError as e:
            self.get_logger().error(f"[sync_callback] CV Bridge error: {str(e)}")
            return
        
        if len(cv_image0.shape) == 2:
            cv_image0 = cv2.cvtColor(cv_image0, cv2.COLOR_GRAY2BGR)
        if len(cv_image1.shape) == 2:
            cv_image1 = cv2.cvtColor(cv_image1, cv2.COLOR_GRAY2BGR)

        if self.img_h0 == -1:
            self.img_h0, self.img_w0 = cv_image0.shape[:2]
            self.img_h1, self.img_w1 = cv_image1.shape[:2]

        img_torch0 = self.np_image_to_torch(cv_image0).to(self.device)
        img_torch1 = self.np_image_to_torch(cv_image1).to(self.device)

        feat_curr0 = self.extractor.extract(img_torch0)
        feat_curr1 = self.extractor.extract(img_torch1)

        if self.cnt > 0:
            # Cam0
            matches_data0 = self.matcher({'image0': self.feat_prev0, 'image1': feat_curr0})
            scores0 = matches_data0['scores'][0]
            matches0 = matches_data0['matches'][0]
            matches0 = self.sort_matches_by_score(matches0, scores0)

            points_prev0 = self.feat_prev0['keypoints'][0].detach().cpu().numpy()
            points_curr0 = feat_curr0['keypoints'][0].detach().cpu().numpy()
            order_id_curr0 = np.zeros(self.extractor_max_num_keypoints) - 1

            cnt_used0 = 0
            for i_m, match in enumerate(matches0):
                idx_prev = match[0].item()
                idx_curr = match[1].item()
                old_id = self.feat_prev_order_to_id0[idx_prev]
                if old_id >= 0:
                    order_id_curr0[idx_curr] = old_id
                    self.feat_obs_cnt[int(old_id)] += 1
                else:
                    new_id = self.next_feature_id
                    self.next_feature_id += 1
                    if new_id >= len(self.feat_obs_cnt):
                        self.feat_obs_cnt += [0]*2000
                    order_id_curr0[idx_curr] = new_id
                    self.feat_obs_cnt[int(new_id)] = 1

                cnt_used0 += 1
                if cnt_used0 >= self.target_n_features:
                    break

            # Cam1
            matches_data1 = self.matcher({'image0': self.feat_prev1, 'image1': feat_curr1})
            scores1 = matches_data1['scores'][0]
            matches1 = matches_data1['matches'][0]
            matches1 = self.sort_matches_by_score(matches1, scores1)

            points_prev1 = self.feat_prev1['keypoints'][0].detach().cpu().numpy()
            points_curr1 = feat_curr1['keypoints'][0].detach().cpu().numpy()
            order_id_curr1 = np.zeros(self.extractor_max_num_keypoints) - 1

            cnt_used1 = 0
            for i_m, match in enumerate(matches1):
                idx_prev = match[0].item()
                idx_curr = match[1].item()
                old_id = self.feat_prev_order_to_id1[idx_prev]
                if old_id >= 0:
                    order_id_curr1[idx_curr] = old_id
                    self.feat_obs_cnt[int(old_id)] += 1
                else:
                    new_id = self.next_feature_id
                    self.next_feature_id += 1
                    if new_id >= len(self.feat_obs_cnt):
                        self.feat_obs_cnt += [0]*2000
                    order_id_curr1[idx_curr] = new_id
                    self.feat_obs_cnt[int(new_id)] = 1

                cnt_used1 += 1
                if cnt_used1 >= self.target_n_features:
                    break

            overlay0 = self.draw_features(cv_image0.copy(), points_curr0, order_id_curr0, self.feat_obs_cnt)
            overlay1 = self.draw_features(cv_image1.copy(), points_curr1, order_id_curr1, self.feat_obs_cnt)
            self.image_pub0.publish(self.bridge.cv2_to_imgmsg(overlay0, encoding="bgr8"))
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(overlay1, encoding="bgr8"))

            kpts_data0 = []
            for i_pt in range(len(points_curr0)):
                feat_id = order_id_curr0[i_pt]
                if feat_id >= 0 and self.feat_obs_cnt[int(feat_id)] > 2:
                    x, y = points_curr0[i_pt]
                    score = float(self.feat_obs_cnt[int(feat_id)]) 
                    kpts_data0.append([feat_id, x, y, score])
            if len(kpts_data0) > 0:
                kpts_data0 = np.array(kpts_data0)
                self.publish_features(kpts_data0, msg0.header, is_cam0=True)

            kpts_data1 = []
            for i_pt in range(len(points_curr1)):
                feat_id = order_id_curr1[i_pt]
                if feat_id >= 0 and self.feat_obs_cnt[int(feat_id)] > 2:
                    x, y = points_curr1[i_pt]
                    score = float(self.feat_obs_cnt[int(feat_id)])
                    kpts_data1.append([feat_id, x, y, score])
            if len(kpts_data1) > 0:
                kpts_data1 = np.array(kpts_data1)
                self.publish_features(kpts_data1, msg1.header, is_cam0=False)

            stereo_matches_data = self.matcher({'image0': feat_curr0, 'image1': feat_curr1})
            s_scores = stereo_matches_data['scores'][0]
            s_matches = stereo_matches_data['matches'][0]

            order_id_curr0, order_id_curr1 = self.unify_stereo_ids(s_matches, order_id_curr0, order_id_curr1)

            matched_img = self.draw_stereo_matches(
                cv_image0, cv_image1,
                feat_curr0['keypoints'][0].detach().cpu().numpy(),
                feat_curr1['keypoints'][0].detach().cpu().numpy(),
                s_matches
            )
            self.matches_pub.publish(self.bridge.cv2_to_imgmsg(matched_img, encoding="bgr8"))

            self.feat_prev0 = feat_curr0
            self.feat_prev1 = feat_curr1
            self.feat_prev_order_to_id0 = order_id_curr0
            self.feat_prev_order_to_id1 = order_id_curr1

        else:
            self.feat_prev0 = feat_curr0
            self.feat_prev1 = feat_curr1

        self.cnt += 1

    def draw_features(self, img_bgr, keypoints, feat_order_id, feat_obs_cnt):
        for i_pt, (u, v) in enumerate(keypoints):
            feat_id = int(feat_order_id[i_pt])
            if feat_id >= 0:
                obs_count = feat_obs_cnt[feat_id]
                if obs_count < 5:
                    color = (0, 0, 255)
                elif obs_count < 15:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.circle(img_bgr, (int(u), int(v)), 3, color, -1)
        return img_bgr

    def draw_stereo_matches(self, cv_image0, cv_image1, kpts0, kpts1, matches):
        H0, W0 = cv_image0.shape[:2]
        H1, W1 = cv_image1.shape[:2]
        Hc = max(H0, H1)
        out = np.zeros((Hc, W0 + W1, 3), dtype=np.uint8)
        out[:H0, :W0] = cv_image0
        out[:H1, W0:] = cv_image1

        matches_np = matches.detach().cpu().numpy()
        for (idx0, idx1) in matches_np:
            pt0 = (int(kpts0[idx0][0]), int(kpts0[idx0][1]))
            pt1 = (int(kpts1[idx1][0]) + W0, int(kpts1[idx1][1]))
            color = (0, 255, 255)
            cv2.line(out, pt0, pt1, color, 1)
            cv2.circle(out, pt0, 2, (0, 0, 255), -1)
            cv2.circle(out, pt1, 2, (0, 0, 255), -1)
        return out

def main(args=None):
    rclpy.init(args=args)
    node = FeatureTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
