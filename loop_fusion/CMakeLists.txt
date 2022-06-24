cmake_minimum_required(VERSION 3.5)
project(loop_fusion)

set(CMAKE_CXX_STANDARD 14)
add_compile_options( -Wextra -Wpedantic) # -Wall : print warning
set(CMAKE_BUILD_TYPE Release)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_models REQUIRED)

find_package(OpenCV)
find_package(Ceres REQUIRED)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories( ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR}) #${catkin_INCLUDE_DIRS})



add_executable(loop_fusion_node
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
    )

    
# added from DEBUG
ament_target_dependencies(loop_fusion_node camera_models rclcpp ament_index_cpp std_msgs sensor_msgs visualization_msgs nav_msgs cv_bridge OpenCV) # roslib) # does 'roslib' really need ??

target_link_libraries(loop_fusion_node ${OpenCV_LIBS} ${CERES_LIBRARIES} ) #${catkin_LIBRARIES}  )



# Install nodes
install(
  TARGETS loop_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)



#install(TARGETS
#    loop_fusion_node
#    DESTINATION lib/${PROJECT_NAME}
#)


#install(TARGETS loop_fusion_node
#    ARCHIVE DESTINATION lib
#    LIBRARY DESTINATION lib
#    RUNTIME DESTINATION bin
#)

#install(TARGETS loop_fusion_node
#    RUNTIME DESTINATION lib/${PROJECT_NAME}
#)





ament_package()
