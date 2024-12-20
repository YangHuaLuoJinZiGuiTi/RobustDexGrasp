
#include <thread>
#include <chrono>

// pinochio library
#include "pinocchio/parsers/urdf.hpp"
 
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
 
//#include "pinocchio/multibody/sample-models.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/jacobian.hpp"

// ur5 library
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <ur_rtde/rtde_io_interface.h>

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

// ros library
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <thread>
#include <chrono>

int main(int argc, char* argv[])
{
    std::unique_ptr<pinocchio::Model> model_;
    std::unique_ptr<pinocchio::Data> data_;

    std::unique_ptr<ur_rtde::RTDEControlInterface> rtde_control_;
    std::unique_ptr<ur_rtde::RTDEReceiveInterface> rtde_receive_;

    raisim::ArticulatedSystem *platform_;

    ros::NodeHandle* nh_;
    ros::Publisher pub_tar_joints;
    ros::Subscriber sub_cur_joints;
    sensor_msgs::JointState cur_joint_state_;
    sensor_msgs::JointState tar_joint_state_;
    std::thread subscribe_thread_;

    std::cout << "Control interrupted!" << std::endl;

  return 0;
}