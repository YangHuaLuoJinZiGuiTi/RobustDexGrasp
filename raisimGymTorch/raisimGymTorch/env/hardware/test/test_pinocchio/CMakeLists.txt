CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
project(overview_simple)

set(CMAKE_PREFIX_PATH /home/ubuntu/hand/github/artigrasp/raisim/linux/lib/cmake/raisim)
find_package(raisim CONFIG REQUIRED)


find_package(Eigen3 REQUIRED NO_MODULE)
list(APPEND CMAKE_PREFIX_PATH "/opt/openrobots/")
find_package(pinocchio REQUIRED)

include_directories(
  ${raisim_INCLUDE_DIRS}
)
add_executable(fd FD.cpp)
target_link_libraries(fd PRIVATE Eigen3::Eigen pinocchio::pinocchio raisim::raisim)
