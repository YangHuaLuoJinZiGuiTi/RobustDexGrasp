#ifndef UR5_REAL_HPP
#define UR5_REAL_HPP

#include "../hardwareArm.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

class UR5Real : public HardwareArm {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        arm_joint_position_.setZero(num_joint_);
        arm_joint_velocity_.setZero(num_joint_);
        end_effector_pose_.setZero(6);
        end_effector_velocity_.setZero(3);
        end_effector_angle_velocity_.setZero(3);
        arm_init_base_pose_.setZero(6);
        arm_init_base_pose_ << 0.55, 0.75152, 0.0, 0.0, 0.0, 0.0;
    }
    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }

    void updateArmState() final override {
        std::cout << "updateArmState in real UR5: TBD" << std::endl;
    }

    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget) const final override {
        std::cout << "setPdTarget in real UR5: TBD" << std::endl;
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int head_shift) const final override {
        pgain.head(head_shift).setConstant(Pgain);
        dgain.head(head_shift).setConstant(Dgain);
    }
    
    void getBodyParts(std::vector<std::string> & get_vec) const final override {
        for (int i = 0; i < 6; i++) {
            get_vec.push_back(body_parts_[i]);
        }
    }
    void getContactBodies(std::vector<std::string> & get_vec) const final override {
        for (int i = 0; i < 6; i++) {
            get_vec.push_back(contact_bodies_[i]);
        }
    }

    const int getDim() const final override {
        return num_joint_;
    }

    Eigen::VectorXd & getSimBasePose() final override {
        return arm_init_base_pose_;
    }

    Eigen::VectorXd & getJointVelocity() final override {
        return arm_joint_velocity_;
    }
    Eigen::VectorXd & getJointPosition() final override {
        return arm_joint_position_;
    }
    Eigen::VectorXd & getEefPose() final override {
        return end_effector_pose_;
    }
    Eigen::VectorXd & getEefVelocity() final override {
        return end_effector_velocity_;
    }
    Eigen::VectorXd & getEefAngleVelocity() final override {
        return end_effector_angle_velocity_;
    }

private:
    raisim::ArticulatedSystem *platform_;

    const double Pgain = 3000.0;
    const double Dgain = 150.0;

    const int num_joint_ = 6;

    const std::string body_parts_[6] =  {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
    const std::string contact_bodies_[6] =  {"shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"};
};

#endif //UR5_REAL_HPP
