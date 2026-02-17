/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/image.hpp>
// #include <sensor_msgs/image_encodings.h>
#include "image_encodings.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point_stamped.h>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
#include "../estimator/estimator.h"
#include "../estimator/parameters.h"
#include <fstream>

extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry;
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
// extern ros::Publisher pub_cloud, pub_map;
extern rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
// extern ros::Publisher pub_ref_pose, pub_cur_pose;
// extern ros::Publisher pub_key;
extern nav_msgs::msg::Path path;
extern rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_pose_graph; // maybe not used ??
extern int IMAGE_ROW, IMAGE_COL;

void registerPub(rclcpp::Node::SharedPtr n);

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t);

void pubTrackImage(const cv::Mat &imgTrack, const double t);

void printStatistics(const Estimator &estimator, double t);

void pubOdometry(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubInitialGuess(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubKeyPoses(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubCameraPose(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubPointCloud(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubKeyframe(const Estimator &estimator);

void pubRelocalization(const Estimator &estimator);

void pubCar(const Estimator & estimator, const std_msgs::msg::Header &header);
