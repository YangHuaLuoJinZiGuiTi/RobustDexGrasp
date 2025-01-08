cmake_minimum_required(VERSION 3.0.2)
project(step_test_ros)


list(APPEND CMAKE_PREFIX_PATH "/opt/ros/noetic")
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
# roslaunch  # Disabled due to testing
)
catkin_package(
  CATKIN_DEPENDS
  roscpp
  sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable (step_test_ros canAPI.cpp step_test_ros.cpp)
target_link_libraries (step_test_ros BHand pcanbasic pthread ${catkin_LIBRARIES})
install (TARGETS step_test_ros DESTINATION ${PROJECT_BINARY_DIR}/bin)
