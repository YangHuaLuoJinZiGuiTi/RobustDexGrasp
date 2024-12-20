cmake_minimum_required(VERSION 3.0.2)
project(test_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(test_ros test_ros.cpp)

target_link_libraries(test_ros
  ${catkin_LIBRARIES}
)