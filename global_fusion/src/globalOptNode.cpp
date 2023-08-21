/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <rclcpp/rclcpp.hpp>
#include "globalOpt.h"
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <stdio.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <fstream>
#include <queue>
#include <mutex>
#include <functional>

GlobalOptimization globalEstimator;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_global_odometry;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_global_path;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_car;
nav_msgs::msg::Path *global_path;
double last_vio_t = -1;
std::queue<sensor_msgs::msg::NavSatFix::SharedPtr> gpsQueue;
std::mutex m_buf;

void publish_car_model(double t, Eigen::Vector3d t_w_car, Eigen::Quaterniond q_w_car)
{
    visualization_msgs::msg::MarkerArray markerArray_msg;
    visualization_msgs::msg::Marker car_mesh;

    int sec_ts = (int)t;
    uint nsec_ts = (uint)((t - sec_ts) * 1e9);
    car_mesh.header.stamp.sec = sec_ts;
    car_mesh.header.stamp.nanosec = nsec_ts;

    car_mesh.header.frame_id = "world";
    car_mesh.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
    car_mesh.action = visualization_msgs::msg::Marker::ADD;
    car_mesh.id = 0;

    car_mesh.mesh_resource = "package://global_fusion/models/car.dae";

    Eigen::Matrix3d rot;
    rot << 0, 0, -1, 0, -1, 0, -1, 0, 0;
    
    Eigen::Quaterniond Q;
    Q = q_w_car * rot; 
    car_mesh.pose.position.x    = t_w_car.x();
    car_mesh.pose.position.y    = t_w_car.y();
    car_mesh.pose.position.z    = t_w_car.z();
    car_mesh.pose.orientation.w = Q.w();
    car_mesh.pose.orientation.x = Q.x();
    car_mesh.pose.orientation.y = Q.y();
    car_mesh.pose.orientation.z = Q.z();

    car_mesh.color.a = 1.0;
    car_mesh.color.r = 1.0;
    car_mesh.color.g = 0.0;
    car_mesh.color.b = 0.0;

    float major_scale = 2.0;

    car_mesh.scale.x = major_scale;
    car_mesh.scale.y = major_scale;
    car_mesh.scale.z = major_scale;
    markerArray_msg.markers.push_back(car_mesh);
    pub_car->publish(markerArray_msg);
}


void vio_callback(const nav_msgs::msg::Odometry::SharedPtr pose_msg)
{
    //printf("vio_callback! \n");
    double t = pose_msg->header.stamp.sec;
    last_vio_t = t;
    Eigen::Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Eigen::Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;
    globalEstimator.inputOdom(t, vio_t, vio_q);


    m_buf.lock();
    while(!gpsQueue.empty())
    {
        sensor_msgs::msg::NavSatFix::ConstPtr GPS_msg = gpsQueue.front();
        double gps_t = GPS_msg->header.stamp.sec;
        printf("vio t: %f, gps t: %f \n", t, gps_t);
        // 10ms sync tolerance
        if(gps_t >= t - 0.01 && gps_t <= t + 0.01)
        {
            //printf("receive GPS with timestamp %f\n", GPS_msg->header.stamp.sec);
            double latitude = GPS_msg->latitude;
            double longitude = GPS_msg->longitude;
            double altitude = GPS_msg->altitude;
            //int numSats = GPS_msg->status.service;
            double pos_accuracy = GPS_msg->position_covariance[0];
            if(pos_accuracy <= 0)
                pos_accuracy = 1;
            //printf("receive covariance %lf \n", pos_accuracy);
            //if(GPS_msg->status.status > 8)
                globalEstimator.inputGPS(t, latitude, longitude, altitude, pos_accuracy);
            gpsQueue.pop();
            break;
        }
        else if(gps_t < t - 0.01)
            gpsQueue.pop();
        else if(gps_t > t + 0.01)
            break;
    }
    m_buf.unlock();

    Eigen::Vector3d global_t;
    Eigen:: Quaterniond global_q;
    globalEstimator.getGlobalOdom(global_t, global_q);

    nav_msgs::msg::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";
    odometry.pose.pose.position.x = global_t.x();
    odometry.pose.pose.position.y = global_t.y();
    odometry.pose.pose.position.z = global_t.z();
    odometry.pose.pose.orientation.x = global_q.x();
    odometry.pose.pose.orientation.y = global_q.y();
    odometry.pose.pose.orientation.z = global_q.z();
    odometry.pose.pose.orientation.w = global_q.w();
    pub_global_odometry->publish(odometry);
    pub_global_path->publish(*global_path);
    publish_car_model(t, global_t, global_q);


    // write result to file
    std::ofstream foutC("/home/tony-ws1/output/vio_global.csv", ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(0);
    foutC << pose_msg->header.stamp.sec * 1e9 << ",";
    foutC.precision(5);
    foutC << global_t.x() << ","
            << global_t.y() << ","
            << global_t.z() << ","
            << global_q.w() << ","
            << global_q.x() << ","
            << global_q.y() << ","
            << global_q.z() << endl;
    foutC.close();
}


void GPS_callback(const sensor_msgs::msg::NavSatFix::SharedPtr GPS_msg)
{
    m_buf.lock();
    gpsQueue.push(GPS_msg);
    m_buf.unlock();
}




int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto n = rclcpp::Node::make_shared("globalEstimator");
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(100));


    auto sub_GPS = n->create_subscription<sensor_msgs::msg::NavSatFix>("/gps", rclcpp::QoS(rclcpp::KeepLast(100)), GPS_callback);

    auto sub_vio = n->create_subscription<nav_msgs::msg::Odometry>("/vins_estimator/odometry", rclcpp::QoS(rclcpp::KeepLast(100)), vio_callback);





    pub_global_path = n->create_publisher<nav_msgs::msg::Path>("global_path", 100);
    pub_global_odometry = n->create_publisher<nav_msgs::msg::Odometry>("global_odometry", 100);
    pub_car = n->create_publisher<visualization_msgs::msg::MarkerArray>("car_model", 1000);


    global_path = &(globalEstimator.global_path);
    rclcpp::spin(n);
    return 0;
}

