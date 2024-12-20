cmake_minimum_required(VERSION 3.10)
project(test_ur5_control)

set(CMAKE_PREFIX_PATH /home/ubuntu/hand/github/vision_dex/raisim/linux/lib/cmake/raisim)
find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
find_package(ur_rtde REQUIRED)
add_executable(test_ur5_control test_ur5_control.cpp)
target_link_libraries(test_ur5_control PRIVATE ur_rtde::rtde Eigen3::Eigen raisim::raisim Threads::Threads)