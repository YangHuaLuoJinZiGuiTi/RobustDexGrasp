#include "pinocchio/parsers/urdf.hpp"
 
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
 
#include <iostream>

std::string link_name[] =  {"Flange2hand_fixed_joint", "Flange_base_link", "Allegro_base_link",
"link_1.0", "link_2.0", "link_3.0", "link_3.0_tip", "joint_3.0_tip",
"link_5.0", "link_6.0", "link_7.0", "link_7.0_tip","joint_7.0_tip",
"link_9.0", "link_10.0", "link_11.0", "link_11.0_tip","joint_11.0_tip",
"link_13.0", "link_14.0", "link_15.0", "link_15.0_tip", "joint_15.0_tip", 
"shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link",
"base_link"
};

std::string parts_name[] =  {"Flange2hand_fixed_joint",
    "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
    "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
    "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
    "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip", 
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};

int main(int argc, char ** argv)
{
  pinocchio::Model model;
  pinocchio::urdf::buildModel("/home/ubuntu/hand/github/vision_dex/rsc/ur5_allegro/ur5_allegro_fk.urdf", model);
  pinocchio::Data data(model);

  //pinocchio::forwardKinematics(model, data, q); // do not calculate fix joint

  
  auto starttime = std::chrono::system_clock::now();
  //for (int i = 0; i < 1000; i++) {
  //Eigen::VectorXd q = pinocchio::randomConfiguration(model);  // get random rad for each joints
  //std::cout << "random q(" << q.size() << "): " << q.transpose() << std::endl;
  Eigen::VectorXd q(6+16);
  q << 0.241097 ,  -1.45322 ,    1.7305, -0.0448583  ,  2.31736,  -1.51723    ,    0.2  ,      0.8,        0.2   ,     0.2     ,   0.2       , 0.8   ,     0.2    ,    0.2      ,  0.2      ,  0.8  ,      0.2   ,     0.2    ,   1.57      ,    0   ,    -0.5  ,      0.2;
  pinocchio::framesForwardKinematics(model,data, q); // = forwardKinematics + updateFramePlacements
  //}
   auto diff_time = std::chrono::system_clock::now() - starttime;
  std::cout << "time consume: " << diff_time.count() / 1e6 << "ms" << std::endl;

  // data.oMi: Vector of absolute joint placements (wrt the world). x,y,z of each joint
  // 
  std::cout << "-----------movable joint number = " << (pinocchio::JointIndex)model.njoints << std::endl;
  for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex)model.njoints; ++joint_id) {
    Eigen::Matrix3d rot = data.oMi[joint_id].rotation();
    Eigen::Vector3d eul = rot.eulerAngles(0,1,2);
    std::cout << std::setw(24) << std::left << model.names[joint_id]  << " (id:" <<  joint_id << "), " << std::fixed 
              << std::setprecision(2) << data.oMi[joint_id].translation().transpose() << " || "
              << std::setprecision(2) << eul.transpose() << std::endl;
  }

  std::cout << "-----------user joint name = " << sizeof(parts_name) / sizeof(parts_name[0]) << std::endl;
  for (int i = 0; i < sizeof(parts_name) / sizeof(parts_name[0]); i++) {
    pinocchio::JointIndex fid = model.getJointId(parts_name[i]);
    Eigen::Matrix3d rot = data.oMi[fid].rotation();
    Eigen::Vector3d eul = rot.eulerAngles(0,1,2);
    std::cout << std::setw(24) << std::left << parts_name[i] << " (id:" <<  fid << ", "<< model.names[fid] << "), " << std::fixed 
              << std::setprecision(2) << data.oMi[fid].translation().transpose() << " || "
              << std::setprecision(2) << eul.transpose() << std::endl;
  }

  std::cout << "-----------link number = " << data.oMf.size() << std::endl;
  std::cout << "-----------user link name = " << sizeof(link_name) / sizeof(link_name[0]) << std::endl;
  for (int i = 0; i < sizeof(link_name) / sizeof(link_name[0]); i++) {
    pinocchio::FrameIndex fid = model.getBodyId(link_name[i]);
    Eigen::Matrix3d frame_rot = data.oMf[fid].rotation();
    Eigen::Vector3d frame_eul = frame_rot.eulerAngles(0,1,2);
    std::cout << std::setw(18) << std::left << link_name[i] << " (id:" <<  fid << "). " << std::fixed
              << std::setprecision(2) << data.oMf[fid].translation().transpose() << " || "
              << std::setprecision(2) << frame_eul.transpose()  << std::endl;
  }

}
