CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
project(overview_simple)

set(CMAKE_PREFIX_PATH /home/ubuntu/hand/github/vision_dex/raisim/linux/lib/cmake/raisim)
find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

include_directories(
  ${raisim_INCLUDE_DIRS}
)
add_executable(test_raisim_allegro test_raisim_allegro.cpp)
target_link_libraries(test_raisim_allegro PRIVATE Eigen3::Eigen raisim::raisim Threads::Threads)
