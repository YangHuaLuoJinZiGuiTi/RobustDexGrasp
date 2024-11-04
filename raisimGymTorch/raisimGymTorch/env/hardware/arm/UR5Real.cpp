#include "../hardwareArm.hpp"

#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <ur_rtde/rtde_io_interface.h>

#include <thread>
#include <chrono>

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

        std::string robot_ip = cfg["ur5_real"]["ip"].As<std::string>();
        double rtde_frequency = cfg["ur5_real"]["freq_hz"].As<double>();
        double dt = 1.0 / rtde_frequency; // 2ms
        uint16_t flags = ur_rtde::RTDEControlInterface::FLAG_USE_EXT_UR_CAP;

        rtde_control_ = std::make_unique<ur_rtde::RTDEControlInterface>(robot_ip, rtde_frequency, flags);
        rtde_receive_ = std::make_unique<ur_rtde::RTDEReceiveInterface>(robot_ip, rtde_frequency);

        move_vel_ = cfg["ur5_real"]["move_vel"].As<double>();
        move_acc_ = cfg["ur5_real"]["move_acc"].As<double>();
    }
    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }

    void updateArmState() final override {
        std::vector<double> actual_tcp_pose = rtde_receive_->getActualTCPPose();
        std::vector<double> joint_positions = rtde_receive_->getActualQ();
        std::vector<double> actual_tcp_speed = rtde_receive_->getActualTCPSpeed();
    }

    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget) const final override {
        std::cout << "setPdTarget in real UR5: TBD" << std::endl;
        std::vector<double> tar_joint_pos;
        for (int i = 0; i < 6; i++) {
            tar_joint_pos.push_back(posTarget[i]);
        }
        rtde_control_->moveL_FK(tar_joint_pos, move_vel_, move_acc_);
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int head_shift) const final override {
        pgain.head(head_shift).setConstant(Pgain);
        dgain.head(head_shift).setConstant(Dgain);
    }
    int getBodies(std::vector<std::string> & get_vec, bool contact_flag) const final override {
        if (contact_flag) {
            for (int i = 0; i < 6; i++) {
                get_vec.push_back(contact_bodies_[i]);
            }
            return 6;
        } else {
            for (int i = 0; i < 6; i++) {
                get_vec.push_back(body_parts_[i]);
            }
            return 6;
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

    double move_vel_ = 0.0;
    double move_acc_ = 0.0;

    std::unique_ptr<ur_rtde::RTDEControlInterface> rtde_control_;
    std::unique_ptr<ur_rtde::RTDEReceiveInterface> rtde_receive_;
};

extern "C" std::unique_ptr<HardwareArm> createUR5Real() {
    return std::make_unique<UR5Real>();
}
