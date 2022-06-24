cmake_minimum_required(VERSION 3.5)
project(vins)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wextra -Wpedantic)

set(CMAKE_BUILD_TYPE Release)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcpputils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_models REQUIRED)
find_package(image_transport REQUIRED)


find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

# include camera_models
include_directories("../camera_models/include")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
)


add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp)
target_link_libraries(vins_lib  ${OpenCV_LIBS} ${CERES_LIBRARIES})

ament_target_dependencies(vins_lib rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge camera_models image_transport)

add_executable(vins_node src/rosNodeTest.cpp)
ament_target_dependencies(vins_node rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge camera_models image_transport)
target_link_libraries(vins_node vins_lib) 

add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
ament_target_dependencies(kitti_odom_test rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge camera_models image_transport)
target_link_libraries(kitti_odom_test vins_lib) 

add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
ament_target_dependencies(kitti_gps_test rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge camera_models image_transport)
target_link_libraries(kitti_gps_test vins_lib) 

# added from DEBUG

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install nodes
install(
  TARGETS vins_node
  DESTINATION lib/${PROJECT_NAME}
)



# Install binaries
#install(TARGETS vins_node kitti_odom_test kitti_gps_test vins_lib
#    ARCHIVE DESTINATION lib
#    LIBRARY DESTINATION lib
#    RUNTIME DESTINATION bin
#)

# Install launch files
#install(DIRECTORY
#    launch
#    # config
#    # rviz
#    DESTINATION share/${PROJECT_NAME}
#    )


ament_package()
