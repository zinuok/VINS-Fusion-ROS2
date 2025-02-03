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

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

Estimator estimator;

queue<sensor_msgs::msg::Imu::ConstPtr> imu_buf;
queue<sensor_msgs::msg::PointCloud::ConstPtr> feature_buf;
queue<sensor_msgs::msg::PointCloud::ConstPtr> feature1_buf;
queue<sensor_msgs::msg::PointCloud::ConstPtr> feature0_buf;
queue<sensor_msgs::msg::Image::ConstPtr> img0_buf;
queue<sensor_msgs::msg::Image::ConstPtr> img1_buf;
std::mutex m_buf;

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> convertBuffersToFeatureFrame(
    const sensor_msgs::msg::PointCloud::ConstPtr &cloud_msg0,
    const sensor_msgs::msg::PointCloud::ConstPtr &cloud_msg1)
{

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

    // Traitement de la caméra gauche (camera_id = 0)
    for (size_t i = 0; i < cloud_msg0->points.size(); i++)
    {
        int feature_id = static_cast<int>(cloud_msg0->channels[0].values[i]); // id_of_point

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << cloud_msg0->points[i].x, // x
            cloud_msg0->points[i].y,                // y
            cloud_msg0->points[i].z,                // z
            cloud_msg0->channels[1].values[i],      // u_of_point
            cloud_msg0->channels[2].values[i],      // v_of_point
            cloud_msg0->channels[3].values[i],      // velocity_x
            cloud_msg0->channels[4].values[i];      // velocity_y

        featureFrame[feature_id].emplace_back(0, xyz_uv_velocity);
    }

    // Traitement de la caméra droite (camera_id = 1)
    for (size_t i = 0; i < cloud_msg1->points.size(); i++)
    {
        int feature_id = static_cast<int>(cloud_msg1->channels[0].values[i]); // id_of_point

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << cloud_msg1->points[i].x, // x
            cloud_msg1->points[i].y,                // y
            cloud_msg1->points[i].z,                // z
            cloud_msg1->channels[1].values[i],      // u_of_point
            cloud_msg1->channels[2].values[i],      // v_of_point
            cloud_msg1->channels[3].values[i],      // velocity_x
            cloud_msg1->channels[4].values[i];      // velocity_y

        featureFrame[feature_id].emplace_back(1, xyz_uv_velocity);
    }
    return featureFrame;
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> convertBufferToFeatureFrame2(
    const sensor_msgs::msg::PointCloud::ConstPtr &cloud_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

    for (size_t i = 0; i < cloud_msg->points.size(); i++)
    {
        // Récupérer l'ID de la feature
        int feature_id = static_cast<int>(cloud_msg->channels[0].values[i]);

        // Créer le vecteur 7D
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << cloud_msg->points[i].x, // x
            cloud_msg->points[i].y,                // y
            cloud_msg->points[i].z,                // z
            cloud_msg->channels[1].values[i],      // u_of_point
            cloud_msg->channels[2].values[i],      // v_of_point
            cloud_msg->channels[3].values[i],      // velocity_x
            cloud_msg->channels[4].values[i];      // velocity_y

        // Associer à la caméra principale (ID = 0)
        featureFrame[feature_id].emplace_back(0, xyz_uv_velocity);
    }

    return featureFrame;
}

// header: 1403715278
void img0_callback(const sensor_msgs::msg::Image::SharedPtr img_msg)
{
    m_buf.lock();
    // std::cout << "Left : " << img_msg->header.stamp.sec << "." << img_msg->header.stamp.nanosec << endl;
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::msg::Image::SharedPtr img_msg)
{
    m_buf.lock();
    // std::cout << "Right: " << img_msg->header.stamp.sec << "." << img_msg->header.stamp.nanosec << endl;
    img1_buf.push(img_msg);
    m_buf.unlock();
}

void feature0_callback(const sensor_msgs::msg::PointCloud::SharedPtr feature_msg)
{
    m_buf.lock();
    feature0_buf.push(feature_msg);
    m_buf.unlock();
}

void feature1_callback(const sensor_msgs::msg::PointCloud::SharedPtr feature_msg)
{
    m_buf.lock();
    feature1_buf.push(feature_msg);
    m_buf.unlock();
}

// cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::SharedPtr img_msg)
cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::ConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::msg::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
// void sync_process()
// {
//     while (1)
//     {
//         if (STEREO)
//         {
//             cv::Mat image0, image1;
//             std_msgs::msg::Header header;
//             double time = 0;
//             m_buf.lock();
//             if (!img0_buf.empty() && !img1_buf.empty())
//             {
//                 double time0 = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
//                 double time1 = img1_buf.front()->header.stamp.sec + img1_buf.front()->header.stamp.nanosec * (1e-9);
// //
//                 // 0.003s sync tolerance
//                 if (time0 < time1 - 0.003)
//                 {
//                     img0_buf.pop();
//                     printf("throw img0\n");
//                 }
//                 else if (time0 > time1 + 0.003)
//                 {
//                     img1_buf.pop();
//                     printf("throw img1\n");
//                 }
//                 else
//                 {
//                     time = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
//                     header = img0_buf.front()->header;
//                     image0 = getImageFromMsg(img0_buf.front());
//                     img0_buf.pop();
//                     image1 = getImageFromMsg(img1_buf.front());
//                     img1_buf.pop();
//                     // printf("find img0 and img1\n");
// //
//                     // std::cout << std::fixed << img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9) << std::endl;
//                     // assert(0);
//                 }
//             }
//             m_buf.unlock();
//             if (!image0.empty())
//                 estimator.inputImage(time, image0, image1);
//         }
//         else
//         {
//             cv::Mat image;
//             std_msgs::msg::Header header;
//             double time = 0;
//             m_buf.lock();
//             if (!img0_buf.empty())
//             {
//                 time = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
//                 header = img0_buf.front()->header;
//                 image = getImageFromMsg(img0_buf.front());
//                 img0_buf.pop();
//             }
//             m_buf.unlock();
//             if (!image.empty())
//                 estimator.inputImage(time, image);
//         }
// //
//         std::chrono::milliseconds dura(2);
//         std::this_thread::sleep_for(dura);
//     }
// }

// void sync_process()
// {
//     while (1)
//     {
//         cv::Mat image;
//         std_msgs::msg::Header header;
//         double time = 0;
//         //
//         // Accès au buffer pour récupérer une image monoculaire
//         m_buf.lock();
//         if (!img0_buf.empty())
//         {
//             time = img0_buf.front()->header.stamp.sec + img0_buf.front()->header.stamp.nanosec * (1e-9);
//             header = img0_buf.front()->header;
//             image = getImageFromMsg(img0_buf.front());
//             img0_buf.pop();
//         }
//         m_buf.unlock();
//         //
//         // Si une image est disponible, la transmettre à l'estimateur
//         if (!image.empty())
//         {
//             estimator.inputImage(time, image);
//         }
//         //
//         // Pause pour éviter une boucle trop rapide
//         std::chrono::milliseconds dura(2);
//         std::this_thread::sleep_for(dura);
//     }
// }

// void sync_process()
// {
//     while (1)
//     {
//         if (STEREO)
//         {
//             double time = 0;
//             m_buf.lock();
//             if (!feature0_buf.empty() && !feature1_buf.empty())
//             {
//                 double time0 = feature0_buf.front()->header.stamp.sec +
//                                feature0_buf.front()->header.stamp.nanosec * (1e-9);
//                 double time1 = feature1_buf.front()->header.stamp.sec +
//                                feature1_buf.front()->header.stamp.nanosec * (1e-9);

//                 if (abs(time0 - time1) < 0.06)
//                 {
//                     time = time0;
//                     auto frame = convertBuffersToFeatureFrame(feature0_buf.front(), feature1_buf.front());
//                     // std::cout << "Frame : ";
//                     // for (const auto& pair : frame) {
//                     //     std::cout << "Key: " << pair.first << " Values: ";
//                     //     for (const auto& value : pair.second) {
//                     //         std::cout << "(" << value.first << ", " << value.second.transpose() << ") "; // Affichez les valeurs selon vos besoins
//                     //     }
//                     // }
//                     // std::cout << std::endl;
//                     feature0_buf.pop();
//                     feature1_buf.pop();
//                     m_buf.unlock();

//                     if (!frame.empty())
//                     {
//                         std::cout << "inputFrame2 a été déclenché avec le temps " << time << std::endl;
//                         estimator.inputFrame2(time, frame);
//                     }
//                     continue;
//                 }
//             }
//             m_buf.unlock();
//         }
//         std::chrono::milliseconds dura(2);
//         std::this_thread::sleep_for(dura);
//     }
// }

// void sync_process()
// {
//     while (1)
//     {
//         double time = 0;
//         m_buf.lock();
//         try
//         {
//             if (!feature0_buf.empty() && !feature1_buf.empty())
//             {
//                 double time0 = feature0_buf.front()->header.stamp.sec +
//                                feature0_buf.front()->header.stamp.nanosec * (1e-9);
//                 double time1 = feature1_buf.front()->header.stamp.sec +
//                                feature1_buf.front()->header.stamp.nanosec * (1e-9);

//                 time = time0;
//                 auto frame = convertBuffersToFeatureFrame(feature0_buf.front(), feature1_buf.front());
//                 feature0_buf.pop();
//                 feature1_buf.pop();
//                 m_buf.unlock();

//                 if (!frame.empty())
//                 {
//                     estimator.inputFrame2(time, frame);
//                 }
//                 else
//                 {
//                     m_buf.unlock();
//                 }
//             }
//             else
//             {
//                 m_buf.unlock();
//             }
//         }
//         catch (const std::exception &e)
//         {
//             m_buf.unlock();
//         }

//         std::chrono::milliseconds dura(2);
//         std::this_thread::sleep_for(dura);
//     }
// }

// void sync_process()
// {
//     while (1)
//     {
//         double time = 0;
//         m_buf.lock();
//         if (!feature0_buf.empty())
//         {
//             try
//             {
//                 double time0 = feature0_buf.front()->header.stamp.sec +
//                                feature0_buf.front()->header.stamp.nanosec * (1e-9);
//                 time = time0;
//                 //
//                 auto framedelightglue = convertBufferToFeatureFrame2(feature0_buf.front());
//                 //
//                 feature0_buf.pop();
//                 m_buf.unlock();
//                 //
//                 if (!framedelightglue.empty())
//                 {
//                     estimator.inputFrame2(time, framedelightglue);
//                 }
//             }
//             catch (const std::exception &e)
//             {
//                 m_buf.unlock();
//             }
//         }
//         else
//         {
//             m_buf.unlock();
//         }
//         //
//         std::chrono::milliseconds dura(2);
//         std::this_thread::sleep_for(dura);
//     }
// }

void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
{
    // std::cout << "IMU cb" << std::endl;

    double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * (1e-9);
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);

    // std::cout << "got t_imu: " << std::fixed << t << endl;
    estimator.inputIMU(t, acc, gyr);
    return;
}

void feature_callback(const sensor_msgs::msg::PointCloud::SharedPtr feature_msg)
{
    std::cout << "feature cb" << std::endl;
    std::cout << "Feature: " << feature_msg->points.size() << std::endl;

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if (feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            // printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        assert(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.sec + feature_msg->header.stamp.nanosec * (1e-9);
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::msg::Bool::SharedPtr restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::msg::Bool::SharedPtr switch_msg)
{
    if (switch_msg->data == true)
    {
        // ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        // ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::msg::Bool::SharedPtr switch_msg)
{
    if (switch_msg->data == true)
    {
        // ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        // ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto n = rclcpp::Node::make_shared("vins_estimator");
    // ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if (argc != 2)
    {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu = NULL;
    if (USE_IMU)
    {
        sub_imu = n->create_subscription<sensor_msgs::msg::Imu>(IMU_TOPIC, rclcpp::QoS(rclcpp::KeepLast(2000)), imu_callback);
    }
    auto sub_feature = n->create_subscription<sensor_msgs::msg::PointCloud>("/feature_tracker/feature", rclcpp::QoS(rclcpp::KeepLast(2000)), feature_callback);

    auto sub_feature0 = n->create_subscription<sensor_msgs::msg::PointCloud>(
        "/feature_tracker/feature0",
        rclcpp::QoS(rclcpp::KeepLast(2000)),
        feature0_callback);

    auto sub_feature1 = n->create_subscription<sensor_msgs::msg::PointCloud>(
        "/feature_tracker/feature1",
        rclcpp::QoS(rclcpp::KeepLast(2000)),
        feature1_callback);

    auto sub_img0 = n->create_subscription<sensor_msgs::msg::Image>(IMAGE0_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), img0_callback);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img1 = NULL;
    if (STEREO)
    {
        sub_img1 = n->create_subscription<sensor_msgs::msg::Image>(IMAGE1_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), img1_callback);
    }

    auto sub_restart = n->create_subscription<std_msgs::msg::Bool>("/vins_restart", rclcpp::QoS(rclcpp::KeepLast(100)), restart_callback);
    auto sub_imu_switch = n->create_subscription<std_msgs::msg::Bool>("/vins_imu_switch", rclcpp::QoS(rclcpp::KeepLast(100)), imu_switch_callback);
    auto sub_cam_switch = n->create_subscription<std_msgs::msg::Bool>("/vins_cam_switch", rclcpp::QoS(rclcpp::KeepLast(100)), cam_switch_callback);

    std::thread sync_thread{sync_process};
    rclcpp::spin(n);

    return 0;
}
