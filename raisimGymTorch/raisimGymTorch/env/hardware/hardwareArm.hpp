#ifndef HARDWARE_ARM_HPP
#define HARDWARE_ARM_HPP

#include "Yaml.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

class HardwareArm {
public:
    virtual void init(const std::string &rsc_pth, const Yaml::Node &cfg) = 0;
    virtual void setSimPlatform(raisim::ArticulatedSystem *platform) = 0; // only use in simulation mode

    virtual void updateArmState() = 0;
    virtual void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget) const = 0;

    virtual void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int head_shift) const = 0;
    virtual const int getDim() const = 0;
    virtual Eigen::VectorXd & getSimBasePose() = 0;
    virtual Eigen::VectorXd & getJointPosition() = 0;
    virtual Eigen::VectorXd & getJointVelocity() = 0;
    virtual Eigen::VectorXd & getEefPose() = 0;
    virtual Eigen::VectorXd & getEefVelocity() = 0;
    virtual Eigen::VectorXd & getEefAngleVelocity() = 0;
    virtual void getBodyParts(std::vector<std::string> & get_vec) const = 0;
    virtual void getContactBodies(std::vector<std::string> & get_vec) const = 0;

    virtual ~HardwareArm() = default;

    // value updated after using updateArmState()
    Eigen::VectorXd end_effector_pose_;
    Eigen::VectorXd end_effector_velocity_;
    Eigen::VectorXd end_effector_angle_velocity_;
    Eigen::VectorXd arm_joint_position_;
    Eigen::VectorXd arm_joint_velocity_;
    Eigen::VectorXd arm_init_base_pose_;
};

#endif //HARDWARE_ARM_HPP
