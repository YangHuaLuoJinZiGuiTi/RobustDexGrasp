cmake_minimum_required(VERSION 3.10)
project(UR5_IK LANGUAGES CXX)

set(CMAKE_PREFIX_PATH /home/ubuntu/hand/dgrasp-main/raisim/linux/lib/cmake/raisim)

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

#find_package(pinocchio REQUIRED PATHS "/home/ubuntu/hand/pinocchio/build/cmake")

include_directories(
  ${raisim_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} src/test.cpp)
#target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen raisim::raisim pinocchio::pinocchio pthread ${EIGEN3_LIBRARIES} -llapack -lblas)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen raisim::raisim pthread ${EIGEN3_LIBRARIES} -llapack -lblas)