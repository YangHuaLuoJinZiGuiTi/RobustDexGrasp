cmake_minimum_required(VERSION 3.5)
project(testcmake)

# for raisim
list(APPEND CMAKE_PREFIX_PATH "/home/ubuntu/hand/github/vision_dex/raisim/linux/lib/cmake/raisim/")
find_package(raisim CONFIG REQUIRED)

# for real ur5 
find_package(ur_rtde REQUIRED)

# for pinocchio
list(APPEND CMAKE_PREFIX_PATH "/opt/openrobots/")
find_package(pinocchio REQUIRED)

# for ROS real allegro
list(APPEND CMAKE_PREFIX_PATH "/opt/ros/noetic")
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
)

# for ur5
find_package(ur_rtde REQUIRED)

set(Dependencies 
    # raisim
    raisim::raisim
    # IKFast
    -llapack -lblas 
    # UR5 Real
    ur_rtde::rtde
    # pinocchio
    pinocchio::pinocchio 
    # ROS Allegro Real
    ${catkin_LIBRARIES}
    )

set(Includes 
    # ROS Allegro Real
    ${catkin_INCLUDE_DIRS}
)


add_executable(testcmake cmake.cpp)
target_link_libraries(testcmake PRIVATE ${Dependencies})

include_directories(
  ${Includes}
)
