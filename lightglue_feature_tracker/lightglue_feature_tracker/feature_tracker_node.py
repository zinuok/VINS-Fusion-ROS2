#!/usr/bin/env python3

import sys
print(sys.path)
from lightglue_feature_tracker.lightglue import LightGlue, SuperPoint, DISK
from lightglue_feature_tracker.lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue_feature_tracker.lightglue import viz2d
import sys
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
from multiprocessing import Lock
from multiprocessing.pool import Pool
from multiprocessing import Manager
from rclpy.node import Node

class FeatureTracker(Node):
    def __init__(self):
        # Assurez-vous d'appeler le constructeur de la classe parente correctement
        super().__init__('feature_tracker_node') 
        # self.node = rclpy.create_node('feature_tracker_node')
        self.declare_parameter('cam_config_file')
        cfg_path = self.get_parameter('cam_config_file').get_parameter_value().string_value
        self.cfg = self.load_camera_config(cfg_path)
        self.K = self.cfg["K"]
        self.dist_coeffs = self.cfg["dist_coeffs"]
        # modes
        self.MODE_FILTER_KEYPOINTS = False
        self.MODE_FILTER_MATCHES = False
        self.MODE_ADAPTATIVE_INFO = True
        self.MODE_MATCH_MULTIPLE_FRAMES = False

        # topics
        self.image_pub = self.create_publisher(Image, "/feature_tracker/feature_img", 1000)
        self.pub_features = self.create_publisher(PointCloud, self.cfg["topic_features"], 1000)
        self.use_compressed_input = False
        if self.use_compressed_input:
            self.subscriber = self.create_subscription(CompressedImage, self.cfg["topic_images"], self.callback, 1000)
        self.subscriber = self.create_subscription(Image, self.cfg["topic_images"], self.callback, 1000)
        self.bridge = CvBridge()

        # params
        self.skip_n = 1
        self.skip_n_curr = 0
        self.img_curr = None
        self.img_prev = None
        self.feat_curr = None
        self.feat_prev = None
        self.feat_ids_curr = []
        self.feat_prev_order_to_id = []
        self.feat_curr_order_to_id = []
        self.feat_id_to_track_cnt = {}
        self.feat_out_prev = {} 
        self.feat_out_curr = {} # array of [feat_id, track_cnt, x, y, v_x, v_y]
        self.cnt = 0
        self.ids = []
        self.cnt_id = 0
        self.feat_obs_cnt = [0] * 10000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info("the device connected is %s" % self.device)


        # mode multiple frames
        self.feat_prev_window = []
        self.feat_prev_order_to_id_window = []

        # extractor and matcher
        self.extractor_max_num_keypoints = 700
        self.extractor = SuperPoint(max_num_keypoints=self.extractor_max_num_keypoints, nms_radius=4).eval().to(self.device)  # load the extractor
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)  # load the matcher

        self.target_n_features = 700
        self.img_h = -1
        self.img_w = -1

        # file writer
        self.file_log = open("./log.txt", "a")

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
            'K': np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1.0]
            ]),
            'dist_coeffs': np.array([k1, k2, p1, p2, k3]),
            'topic_images': fs.getNode("topic_images").string(),
            'topic_features': fs.getNode("topic_features").string()
        }

    def np_image_to_torch(self, image: np.ndarray) -> torch.Tensor:
        """Normalize the image tensor and reorder the dimensions."""
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        elif image.ndim == 2:
            # image = image.transpose().repeat(2,axis=0).repeat(2,axis=1)
            image = image[None]  # add channel axis
            '''image = np.stack((image.transpose(),)*3, axis=-1)
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW'''
            # print(image.shape)
            # print(np.amin(image), np.amax(image)) # 1593 3404
            # if thermal image, need to do something like: image = image / 40.
        else:
            raise ValueError(f'Not an image: {image.shape}')
        return torch.tensor(image / 255., dtype=torch.float)

    def filter_keypoints_score(self, features):
        self.get_logger().info("[feature_tracker] filtering keypoints...")
        features_out = {
            'keypoints': None,
            'keypoint_scores': None,
            'descriptors': None,
            'image_size': torch.clone(features['image_size']).detach().to(self.device),
        }
        for _iv in [
            ("keypoints", features['keypoints'][0]), 
            ("keypoint_scores", features['keypoint_scores'][0]), 
            ("descriptors", features['descriptors'][0])
            ]:
            _i, _vector = _iv
                
            _vector_out = []
            _vector_np = _vector.cpu().detach().numpy()
            for i in range(len(_vector_np)):
                if features['keypoint_scores'][0][i] >= 0.1:
                    _vector_out.append(_vector_np[i])

            _vector_out = torch.from_numpy(np.array(_vector_out))
            _vector = _vector_out
            
            features_out[_i] = torch.unsqueeze(_vector_out, 0).to(self.device)
        print("begin size: %d" % self.get_length(features))
        print("final size: %d" % self.get_length(features_out))
        return features_out

    def filter_keypoints_voxelgrid(self, features):
        points_2d = features['keypoints'][0].detach().cpu().numpy()
        # print(features['keypoints'][0][:10])
        scores = features['keypoint_scores'][0].detach().cpu().numpy()
        tree = cKDTree(points_2d)
        cell_size = 30
        p_ids_out = []
        for x in range(0, self.img_w, cell_size):
            for y in range(0, self.img_h, cell_size):
                cell_center = [x+cell_size / 2, y+cell_size / 2]
                # print("center", cell_center)
                p_ids = tree.query_ball_point(cell_center, cell_size * 1.41, p=2)
                # print("candidates:")
                # print(points_2d[p_ids], p_ids)
                score_max = -1
                p_id_max = -1
                # print(p_ids)
                for p_id in p_ids:
                    score = scores[p_id]
                    if score > score_max:
                        score_max = score
                        p_id_max = p_id
                if p_id_max > -1:
                    p_ids_out.append(p_id_max)
                    # print("using", p_id_max)


        p_ids_out = np.unique(p_ids_out)

        features_out = {
            'keypoints': None,
            'keypoint_scores': None,
            'descriptors': None,
            'image_size': torch.clone(features['image_size']).detach().to(self.device),
        }
        for _iv in [
            ("keypoints", features['keypoints'][0]), 
            ("keypoint_scores", features['keypoint_scores'][0]), 
            ("descriptors", features['descriptors'][0])
            ]:
            _i, _vector = _iv
                
            _vector_out = []
            _vector_np = _vector.cpu().detach().numpy()
            for p_id in p_ids_out:
                _vector_out.append(_vector_np[p_id])

            _vector_out = torch.from_numpy(np.array(_vector_out))
            _vector = _vector_out
            
            features_out[_i] = torch.unsqueeze(_vector_out, 0).to(self.device)
        return features_out
    
    def filter_matches_voxelgrid(self, matches, scores, points_curr_detected):
        matches_np = matches.detach().cpu().numpy()
        points_curr = points_curr_detected[matches_np[..., 1]]
        tree = cKDTree(points_curr)
        cell_size = 40
        p_ids_out = []
        cnt_plot = 0
        points_to_plot = np.zeros((2000,3)).astype(int)
        for x in range(0, self.img_w, cell_size):
            for y in range(0, self.img_h, cell_size):
                cell_center = [x+cell_size / 2, y+cell_size / 2]
                # print("center", cell_center)
                p_ids = tree.query_ball_point(cell_center, cell_size * 1.41, p=2)
                # print("candidates:")
                # print(points_2d[p_ids], p_ids)
                score_max = -1
                p_id_max = -1
                # print(p_ids)
                for p_id in p_ids:
                    feat_id = int(self.feat_prev_order_to_id[matches[p_id][0]])
                    match_score = scores[p_id]
                    if match_score < 0.8:
                        continue

                    '''
                    # if feature is known from previous frame(s)
                    if feat_id > -1:
                        self.feat_curr_order_to_id[matches[p_id][1]] = feat_id
                        self.feat_obs_cnt[feat_id] += 1
                    # else, add new feature
                    else: 
                        self.cnt_id += 1
                        if self.cnt_id > len(self.feat_obs_cnt) - 1:
                            self.feat_obs_cnt += [0] * 1000
                        feat_id = self.cnt_id
                        self.feat_curr_order_to_id[matches[p_id][1]] = feat_id
                        self.feat_obs_cnt[feat_id] = 1
                    '''


                    score = self.feat_obs_cnt[feat_id]
                    if score > score_max:
                        score_max = score
                        p_id_max = p_id
                if p_id_max > -1:
                    p_ids_out.append(p_id_max)
                    # print("using", p_id_max)


        matches_out = matches_np[p_ids_out]
        return matches_out

    def get_length(self, vector):
        if vector is None:
            return 0
        else:
            return vector['keypoints'][0].shape[0]

    def undistort_keypoints(self, keypoints):
        # transform keypoints from image frame to camera frame
        points = cv2.undistortPoints(keypoints, self.K, self.dist_coeffs, None, None)  
        return points 

    def publish_features(self, kpts_data, header):
        kpts_ids = np.array(kpts_data[:, 0]).astype(int) # 1, n
        kpts = np.array(kpts_data[:, 1:3]).astype(np.float64)
        # undistort keypoints
        kpts_undistorted = self.undistort_keypoints(kpts)[:, 0, :] # 2, n 
        kpts_n_obs = kpts_data[:, 3] # (kpts_data[:, 3] - 0.9) * 10.0 # CHEXP2 kpts_data[:, 3] # CHEXP3 (kpts_data[:, 3] - 0.8) * 6 + 0.8

        pc_msg = PointCloud()
        pc_msg.header = header
        
        # add channels
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

        # Populate custom channels
        for i_p, pt in enumerate(kpts_undistorted):
            point = Point32()
            point.x = float(pt[0])  # Ensure x is a float
            point.y = float(pt[1])  # Ensure y is a float
            point.z = float(1)      # Ensure z is a float
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

        
        # Create a ROS publisher for the PointCloud message
        self.pub_features.publish(pc_msg)
            
    def sort_matches(self, matches, scores):
        matches_out = []
        indices_match = torch.argsort(scores)
        matches_out = matches[indices_match]
        return matches_out

    
    def callback(self, ros_data):
        self.skip_n_curr += 1
        # self.get_logger().info("[feature_tracker] image received")
        if (self.skip_n_curr-1) % self.skip_n != 0:
            print("[feature_tracker]: skip")
            return True
        
        if self.cnt % 100 == 0:
            print("[feature_tracker] frame\t %d" % self.cnt)

        # Record the start time
        time_s = time.time()

        # self.get_logger().info("[feature_tracker] received image message")
        try:
            if self.use_compressed_input:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(ros_data)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(ros_data)
                # clahe
                if False:
                    clahe = cv2.createCLAHE()
                    cv_image = clahe.apply(cv_image)
        except CvBridgeError:
            self.get_logger().info("[feature_tracker] error in reading image")
            return True

        time_bridge = time.time()

        # if first image, extract parameters
        if self.cnt == 0:
            self.img_h = cv_image.shape[0]
            self.img_w = cv_image.shape[1]

        # for all images: extract points
        self.img_curr = self.np_image_to_torch(cv_image).to(self.device)
        self.feat_curr = self.extractor.extract(self.img_curr)
        if False: #  self.MODE_FILTER_KEYPOINTS:
            self.get_logger().info("[feature_tracker] filtering keypoints")
            self.feat_curr = self.filter_keypoints_voxelgrid(self.feat_curr)
            self.get_logger().info("[feature_tracker] filtered keypoints")

        time_extract = time.time()

        # if second image (meaning the first will be matched), init feature ids with range 0 to 2047
        if self.cnt == 1:
            self.feat_prev_order_to_id = np.zeros(self.extractor_max_num_keypoints) - 1 # np.array([*range(2048)])
            self.cnt_id = 0 # np.max(self.feat_prev_order_to_id)         

        # if we have at least two images
        self.prev_frame_n = -4
        if self.cnt > 0:
            time_match = time.time()
            if self.MODE_MATCH_MULTIPLE_FRAMES:
                # match prev and curr
                matches_data = self.matcher({'image0': self.feat_prev, 'image1': self.feat_curr})
                scores = matches_data['scores'][0]
                matches = matches_data['matches'][0]

                # match prev-5 and curr
                if len(self.feat_prev_window) > 5:
                    matches_data_p5 = self.matcher({'image0': self.feat_prev_window[self.prev_frame_n], 'image1': self.feat_curr})
                    scores_p5 = matches_data_p5['scores'][0]
                    matches_p5 = matches_data_p5['matches'][0]
            else:
                matches_data = self.matcher({'image0': self.feat_prev, 'image1': self.feat_curr})
                time_match = time.time()
                scores = matches_data['scores'][0]
                matches = matches_data['matches'][0]
            # points_prev  = self.feat_prev['keypoints'][0].detach().cpu().numpy()
            points_curr  = self.feat_curr['keypoints'][0].detach().cpu().numpy() # n, 2
            cnt_plot = 0
            points_to_plot = np.zeros((2000,4)).astype(np.float32)

            # filter matches and manage feature ids
            img_matches = torch.clone(self.img_curr).permute(1, 2, 0).detach().cpu().numpy()
            time_detach = time.time()
            cnt_known_feats = 0
            self.feat_curr_order_to_id = np.zeros(self.extractor_max_num_keypoints) - 1

            # matches = self.filter_matches(matches, scores, 0.9, points_prev, points_curr)
            time_filter = time.time()
            if self.MODE_FILTER_MATCHES:
                self.get_logger().info("[feature_tracker] filtering matches")
                matches = self.filter_matches_voxelgrid(matches, scores, points_curr)
                self.get_logger().info("[feature_tracker] filtered matches")

            # sort matches 
            matches = self.sort_matches(matches, scores)

            # add features from current and previous frame pair
            cnt_matches = 0
            for i_m, match in enumerate(matches):
                '''
                if scores[i_m] < 0.9: # CHEXP was 0.8 for DF-VIO*
                    continue'''
                coords_curr = np.array(points_curr[match[1]]).astype(float)

                feat_id = int(self.feat_prev_order_to_id[match[0]])

                # if feature is known from previous frame(s)
                if feat_id > -1:
                    self.feat_curr_order_to_id[match[1]] = feat_id
                    self.feat_obs_cnt[feat_id] += 1

                    # if feature has been observed long enough
                    if self.feat_obs_cnt[feat_id] > 2: # CHEXP2 was 6
                        score_match =scores[i_m].detach().cpu().numpy()
                        points_to_plot[cnt_plot, :] = [feat_id, coords_curr[0], coords_curr[1], score_match]
                        cnt_plot += 1

                # else, add new feature
                else: 
                    self.cnt_id += 1
                    if self.cnt_id > len(self.feat_obs_cnt) - 1:
                        self.feat_obs_cnt += [0] * 1000
                    feat_id = self.cnt_id
                    self.feat_curr_order_to_id[match[1]] = feat_id
                    self.feat_obs_cnt[feat_id] = 1

                cnt_known_feats += 1
                cnt_matches += 1
                if cnt_matches >= self.target_n_features:
                    break 

            if self.cnt % 1000 == 0:
                print("used matches:", cnt_matches)

            # add feature from past frames if need be           
            if self.MODE_MATCH_MULTIPLE_FRAMES:
                if len(self.feat_prev_window) > 5:
                    # loop in all matches between frame n and frame n-5
                    for i_m, match in enumerate(matches_p5):

                        # check if feat is already observed by (n,n-1) matches, if so: discard
                        '''feat_id = int(self.feat_prev_order_to_id_window[self.prev_frame_n][match[0]])
                        if feat_id in feat_ids_obs_curr:
                            continue'''
                        
                        # check if kp order_1 in matches_p5 is in   ll kps_1 of matches
                        # in other words, if this feature point has been observed in the current frame already
                        if match[1] in matches[..., 1]:
                            continue

                        # check if score is below threshold, if so: discard
                        if scores_p5[i_m] < 0.9:
                            continue
                        coords_curr = np.array(points_curr[match[1]])

                        # if we're here, it means it's worth it to add this feature
                        # if feature is known from previous frame(s) and it's a good one
                        if feat_id > -1 and self.feat_obs_cnt[feat_id] > 10:
                            self.feat_curr_order_to_id[match[1]] = feat_id
                            self.feat_obs_cnt[feat_id] += 1

                        # else, add new feature
                        else: 
                            self.cnt_id += 1
                            if self.cnt_id > len(self.feat_obs_cnt) - 1:
                                self.feat_obs_cnt += [0] * 1000
                            feat_id = self.cnt_id
                            self.feat_curr_order_to_id[match[1]] = feat_id
                            self.feat_obs_cnt[feat_id] = 1

                        # show all
                        if self.feat_obs_cnt[feat_id] > 10:
                            points_to_plot[cnt_plot, :] = [feat_id, coords_curr[0], coords_curr[1], self.feat_obs_cnt[feat_id]]
                            cnt_plot += 1
                        cnt_known_feats += 1
                print("n, n-5 and n,n-1 matches: %d" % cnt_plot)
            
        
            time_filtered = time.time()

            time_manage_ids = time.time()
            img_matches = np.stack((img_matches, img_matches, img_matches), axis=2)[:,:,:,0]

            if cnt_plot > 1:
                self.publish_features(points_to_plot[:cnt_plot, :], ros_data.header)
                for i in range(cnt_plot):
                    el = points_to_plot[i]
                    feat_id = int(el[0])
                    coords_curr = [int(el[1]), int(el[2])]
                    origin = el[3]
                    color = (0, 0 ,1)
                    size = 2
                    if self.feat_obs_cnt[feat_id] < 8:
                        color =  color = (0, 0 ,1)
                    elif self.feat_obs_cnt[feat_id] < 16:
                        color =  color = (0, 1 ,0)
                    else:
                        color = (1, 0.05 ,0.05)
                    '''if origin > 4:
                        color = (1, 0 , 1)
                        size = 10'''

                    cv2.circle(img_matches, coords_curr, size, color, size)



            # print("[feature_tracker] cnt_known_feats:    %d" % cnt_known_feats)
            # print("[feature_tracker] cnt_found_matches:  %d" % len(matches))
            img_matches = cv2.normalize(img_matches, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img_matches, encoding="8UC3"))
            time_plot = time.time()
            
            if False:
                # time display
                duration_bridge = time_bridge - time_s
                duration_extract = time_extract - time_bridge
                duration_match = time_match - time_extract
                duration_detach = time_detach - time_match
                duration_filter = time_filter - time_detach
                duration_filtered = time_filtered - time_filter
                duration_manage_ids = time_manage_ids - time_filter
                duration_plot = time_plot - time_manage_ids
                duration_total = time_plot - time_s
                print("times:")
                print("duration_bridge    :  %.5f" % duration_bridge)
                print("duration_extract   :  %.5f" % duration_extract)
                print("duration_match     :  %.5f" % duration_match)
                print("duration_detach    :  %.5f" % duration_detach)
                print("duration_filter    :  %.5f" % duration_filter)
                print("duration_filtered  :  %.5f" % duration_filtered)
                print("duration_manage_ids:  %.5f" % duration_manage_ids)
                print("duration_plot      :  %.5f" % duration_plot)
                print("duration_total     :  %.5f" % duration_total)
             

        # prepare next callback
        if self.MODE_MATCH_MULTIPLE_FRAMES:
            self.slide_feat_window()
        self.img_prev = self.img_curr
        self.feat_prev = self.feat_curr
        self.feat_prev_order_to_id = self.feat_curr_order_to_id
        self.cnt += 1

    def slide_feat_window(self):
        # add feat_prev at end of window
        self.feat_prev_window.append(self.feat_prev)
        self.feat_prev_order_to_id_window.append(self.feat_curr_order_to_id)

        # if len is bigger than 10, chop head
        if len(self.feat_prev_window) > 10:
            self.feat_prev_window.pop(0)
            self.feat_prev_order_to_id_window.pop(0) 


def main(args=None):
    rclpy.init(args=args)
    feature_tracker = FeatureTracker()
    rclpy.spin(feature_tracker)
    feature_tracker.destroy_node()
    rclpy.shutdown()

# if __name__ == '__main__':
#     main(sys.argv)