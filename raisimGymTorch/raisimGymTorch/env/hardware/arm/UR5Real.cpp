#include "../hardwareArm.hpp"

#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <ur_rtde/rtde_io_interface.h>

#include <thread>
#include <chrono>
#include <time.h>

class UR5Real : public HardwareArm {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        arm_joint_position_.setZero(num_joint_);
        arm_joint_velocity_.setZero(num_joint_);
        end_effector_pose_.setZero(6);
        end_effector_velocity_.setZero(3);
        end_effector_angle_velocity_.setZero(3);
        arm_init_base_pose_.setZero(6);
        last_end_effector_pose_.setZero(6);
        last_arm_joint_position_.setZero(num_joint_);
        arm_init_base_pose_ << 0.55, 0.75152, 0.0, 0.0, 0.0, 0.0;

        std::string robot_ip = cfg["arm_real"]["ip"].As<std::string>();
        double rtde_frequency = cfg["arm_real"]["freq_hz"].As<double>();
        double dt = 1.0 / rtde_frequency; // 2ms
        uint16_t flags = ur_rtde::RTDEControlInterface::FLAG_USE_EXT_UR_CAP;

        rtde_control_ = std::make_unique<ur_rtde::RTDEControlInterface>(robot_ip, rtde_frequency, flags);
        rtde_receive_ = std::make_unique<ur_rtde::RTDEReceiveInterface>(robot_ip, rtde_frequency);

        move_vel_ = cfg["arm_real"]["move_vel"].As<double>();
        move_acc_ = cfg["arm_real"]["move_acc"].As<double>();
        velocity_dt_s_ = cfg["real_velocity_dt_s"].As<double>();
    }
    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }
    void updateArmState() final override {
        auto now_time = std::chrono::system_clock::now();
        double diff_time_s = ((now_time - last_time_).count() / 1e9);

        // Actual Cartesian coordinates of the tool: (x,y,z,rx,ry,rz), in m and rad
        // where rx, ry and rz is a rotation vector representation of the tool orientation
        std::vector<double> actual_tcp_pose = rtde_receive_->getActualTCPPose();
        Eigen::Vector3d vec(actual_tcp_pose[3], actual_tcp_pose[4], actual_tcp_pose[5]);
        Eigen::AngleAxisd rotation_vector (vec.norm(), vec.normalized());
        Eigen::Vector3d eulerAngle = rotation_vector.matrix().eulerAngles(0,1,2);
        for (int i = 0; i < 3; i++) {
            end_effector_pose_[i] = actual_tcp_pose[i];
            end_effector_pose_[i+3] = eulerAngle[i];
        }

        // Actual joint positions in rad
        std::vector<double> joint_positions = rtde_receive_->getActualQ();
        for (int i = 0; i < 6; i++) {
            arm_joint_position_[i] = joint_positions[i];
        }
        // Actual speed of the tool given in Cartesian coordinates
        std::vector<double> actual_tcp_speed = rtde_receive_->getActualTCPSpeed();
        for (int i = 0; i < 3; i++) {
            end_effector_velocity_[i] = actual_tcp_speed[i];
        }

        // calculate average velocity
        if (diff_time_s > velocity_dt_s_) {
            if (diff_time_s < 1.0) {
                for (int i = 0; i < num_joint_; i++) {
                    arm_joint_velocity_[i] = const_angle(arm_joint_position_[i] - last_arm_joint_position_[i]) / diff_time_s;
                }
                for (int i = 0; i < 3; i++) {
                    end_effector_velocity_[i] = (end_effector_pose_[i] - last_end_effector_pose_[i]) / diff_time_s;
                    end_effector_angle_velocity_[i] = const_angle(end_effector_pose_[i + 3] - last_end_effector_pose_[i + 3]) / diff_time_s;
                }
            } else {
                std::cout << "first init or sth. block" << std::endl;
            }

            last_time_ = now_time;
            last_end_effector_pose_ = end_effector_pose_;
            last_arm_joint_position_ = arm_joint_position_;
        }
    }
    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) const final override {
        std::vector<double> tar_joint_pos;
        for (int i = 0; i < 6; i++) {
            tar_joint_pos.push_back(posTarget[i]);
        }
        if (async == true) {
            rtde_control_->moveJ(tar_joint_pos, move_vel_, move_acc_, true);
        } else {
            rtde_control_->moveJ(tar_joint_pos, move_vel_, move_acc_, false);
        }
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
    double const_angle(double in) {
        double out = in;
        while (in > M_PI) {
            out -= 2*M_PI;
        }
        while (in < -M_PI) {
            out += 2*M_PI;
        }
        return out;
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
    double velocity_dt_s_ = 0.0;

    std::unique_ptr<ur_rtde::RTDEControlInterface> rtde_control_;
    std::unique_ptr<ur_rtde::RTDEReceiveInterface> rtde_receive_;

    Eigen::VectorXd last_end_effector_pose_;
    Eigen::VectorXd last_arm_joint_position_;
    std::chrono::system_clock::time_point last_time_;
};

extern "C" std::unique_ptr<HardwareArm> createUR5Real() {
    return std::make_unique<UR5Real>();
}
