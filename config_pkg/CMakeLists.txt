cmake_minimum_required(VERSION 3.5)
project(config_pkg)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcpputils REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

install(DIRECTORY
  config
  support_files
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()