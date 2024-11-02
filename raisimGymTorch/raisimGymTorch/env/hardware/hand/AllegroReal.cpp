#include "../hardwareHand.hpp"

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <thread>
#include <chrono>

class AllegroReal : public HardwareHand {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        wrist_pose_.setZero(6);
        wrist_velocity_.setZero(6);
        hand_joint_position_.setZero(num_joint_);
        hand_joint_velocity_.setZero(num_joint_);
        
        flying_hand_mode_ = cfg["flying_hand_mode"].As<bool>();
        freq_hz_ = cfg["allegro_real"]["freq_hz"].As<double>();
       
        const char* name = "test_node";
        char* argv[] = { const_cast<char*>(name) }; 
        int argc = 1; 
        ros::init(argc, argv, "joint_state_publisher");
        for (int i = 0; i < DOF_JOINTS; i++) {
            cur_joint_state_.name.push_back(joint_names[i]);
            cur_joint_state_.position.push_back(0.0);
            cur_joint_state_.velocity.push_back(0.0);
            cur_joint_state_.effort.push_back(0.0);
            tar_joint_state_.name.push_back(joint_names[i]);
            tar_joint_state_.position.push_back(0.0);
        }
        
        nh_ = new ros::NodeHandle();
        pub_tar_joints = nh_->advertise<sensor_msgs::JointState>("/allegroHand/joint_cmd", 1);
        sub_cur_joints = nh_->subscribe("/allegroHand/joint_states", 1, &AllegroReal::jointStateCallback, this);
        subscribe_thread_ = std::thread(&AllegroReal::subscribeLoop, this);

        std::cout << "init finish all !!!" << std::endl;
    }

    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }

    void updateHandState(const Eigen::VectorXd &eef_pos) final override {
        std::cout << "updateHandState in real Allegro: TBD" << std::endl;

    }
    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget) const final override {
        std::cout << "setPdTarget in real Allegro: TBD" << std::endl;
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int tail_shift) const final override {
        pgain.tail(tail_shift).setConstant(Pgain);
        dgain.tail(tail_shift).setConstant(Dgain);
    }

    void getFrameOrientation(const std::string &frameName, raisim::Mat<3, 3> &orientation_W) final override {
        std::cout << "getFrameOrientation in real Allegro: TBD" << std::endl;
    }
    void getFramePosition(const std::string &frameName, raisim::Vec<3> &point_W) final override {
        std::cout << "getFramePosition in real Allegro: TBD" << std::endl;
    }
    void getFrameAngularVelocity(const std::string &frameName, raisim::Vec<3> &angVel_W) final override {
        std::cout << "getFrameAngularVelocity in real Allegro: TBD" << std::endl;
    }
    void getFrameVelocity(const std::string &frameName, raisim::Vec<3> &vel_W) final override {
        std::cout << "getFrameVelocity in real Allegro: TBD" << std::endl;
    }
    Eigen::VectorXd & getJointVelocity() final override {
        return hand_joint_velocity_;
    }
    Eigen::VectorXd & getJointPosition() final override {
        return hand_joint_position_;
    }
    const int getDim() const final override {
        return num_joint_;
    }
    const int getNumContacts() const final override {
        return num_contacts_;
    }
    const int getNumBodies() const final override {
        return num_bodies_;
    }
    const int getNumFinger() const final override {
        return num_finger_;
    }
    void getBodyParts(std::vector<std::string> & get_vec) const final override {
        for (int i = 0; i < num_bodies_; i++) {
            if (flying_hand_mode_) {
                get_vec.push_back(body_parts_flying_[i]);
            } else {
                get_vec.push_back(body_parts_[i]);
            }
        }
    }
    void getContactBodies(std::vector<std::string> & get_vec) const final override {
        for (int i = 0; i < num_contacts_; i++) {
            get_vec.push_back(contact_bodies_[i]);
        }
    }


    void publishJointStates(std::vector<double> &tar_pos) {
        for (int i = 0; i < DOF_JOINTS; i++) {
            tar_joint_state_.position[i] = tar_pos[i];
        }
        pub_tar_joints.publish(tar_joint_state_);
    }

    void getCurrentStates(std::vector<double> &cur_pos, std::vector<double> &cur_vel) {
        for (int i = 0; i < DOF_JOINTS; i++) {
            cur_pos[i] = cur_joint_state_.position[i];
            cur_vel[i] = cur_joint_state_.velocity[i];
        }
    }

    static const int DOF_JOINTS = 16;
    bool start_get_flag_ = false;
    double min_limit[DOF_JOINTS] = {
        -0.47, -0.196, -0.174, -0.227, 
        -0.47, -0.196, -0.174, -0.227, 
        -0.47, -0.196, -0.174, -0.227, 
        0.263, -0.105, -0.189, -0.162
    };
    double max_limit[DOF_JOINTS] = {
        0.47, 1.61, 1.709, 1.618, 
        0.47, 1.61, 1.709, 1.618, 
        0.47, 1.61, 1.709, 1.618, 
        1.396, 1.163, 1.644, 1.719
    };

    std::string joint_names[DOF_JOINTS] = {
        "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
        "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
        "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
        "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"
    };

private:

    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg) {
        for (int i = 0; i < DOF_JOINTS; i++) {
            cur_joint_state_.position[i] = msg->position[i];
            cur_joint_state_.velocity[i] = msg->velocity[i];
            cur_joint_state_.effort[i] = msg->effort[i];
        }
    }

    void subscribeLoop() {
        ros::Rate rate(freq_hz_);
        while (ros::ok()) {
            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    raisim::ArticulatedSystem *platform_;

    bool flying_hand_mode_ = false;

    const static int num_contacts_ = 13;
    const static int num_bodies_ = 17;
    const static int num_finger_ = 4;
    const static int num_joint_ = 16;

    const double Pgain = 60.0;
    const double Dgain = 0.2;

    const std::string body_parts_flying_[num_bodies_] =  {"z_rotation_joint",
    "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
    "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
    "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
    "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"};

    const std::string body_parts_[num_bodies_] =  {"Flange2hand_fixed_joint",
    "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
    "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
    "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
    "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"};

    // for raisim contact check
    const std::string contact_bodies_[num_contacts_] =  {"wrist_3_link",
    "link_1.0", "link_2.0", "link_3.0",
    "link_5.0", "link_6.0", "link_7.0",
    "link_9.0", "link_10.0", "link_11.0",
    "link_13.0", "link_14.0", "link_15.0"};

    ros::NodeHandle* nh_;
    ros::Publisher pub_tar_joints;
    ros::Subscriber sub_cur_joints;
    sensor_msgs::JointState cur_joint_state_;
    sensor_msgs::JointState tar_joint_state_;
    std::thread subscribe_thread_;

    double freq_hz_ = 0.0;
};

extern "C" std::unique_ptr<HardwareHand> createAllegroReal() {
    return std::make_unique<AllegroReal>();
}
