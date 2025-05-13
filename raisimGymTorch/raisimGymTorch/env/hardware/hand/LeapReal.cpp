#include "../hardwareHand.hpp"

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <thread>
#include <chrono>
#include <time.h>
#include <mutex>

class LeapReal : public HardwareHand {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        wrist_pose_.setZero(6);
        wrist_velocity_.setZero(6);
        hand_joint_position_.setZero(num_joint_);
        hand_joint_velocity_.setZero(num_joint_);
        hand_joint_effort_.setZero(num_joint_);
        
        flying_hand_mode_ = cfg["flying_hand_mode"].As<bool>();
        freq_hz_ = cfg["hand_real"]["freq_hz"].As<double>();
        std::string pd_file = cfg["hand_pd_file"].As<std::string>();
       
        const char* name = "test_node";
        char* argv[] = { const_cast<char*>(name) }; 
        int argc = 1; 
        ros::init(argc, argv, "joint_state_publisher");
        for (int i = 0; i < num_joint_; i++) {
            cur_joint_state_.name.push_back(joint_names[i]);
            cur_joint_state_.position.push_back(0.0);
            cur_joint_state_.velocity.push_back(0.0);
            cur_joint_state_.effort.push_back(0.0);
            tar_joint_state_.name.push_back(joint_names[i]);
            tar_joint_state_.position.push_back(0.0);
            tar_tor_joint_state_.name.push_back(joint_names[i]);
            tar_tor_joint_state_.effort.push_back(0.0);
        }
        
        nh_ = new ros::NodeHandle();
        pub_tar_joints = nh_->advertise<sensor_msgs::JointState>("/leaphand_node/cmd_allegro", 1);
        sub_cur_joints = nh_->subscribe("/leaphand_node/leap_joint", 1, &LeapReal::jointStateCallback, this);
        subscribe_thread_ = std::thread(&LeapReal::subscribeLoop, this);

        std::ifstream pd_txt;
        pd_txt.open(rsc_pth+"/../raisimGymTorch/raisimGymTorch/env/hardware/hand/"+pd_file);
        if (pd_txt) {
            std::string line;
            int line_cnt = 0;
            while (getline(pd_txt, line)) {
                std::stringstream ss(line); 
                if (line_cnt < num_joint_) {
                    ss >> Pgain[line_cnt];
                } else {
                    ss >> Dgain[line_cnt - num_joint_];
                }
                line_cnt++;
            }
            if (line_cnt != (num_joint_*2)) {
                std::cout << "error txt line:" << line_cnt << std::endl;
                pd_txt.close();
                exit(0);
            }
            pd_txt.close();
        } else {
            for (int i = 0; i < num_joint_; i++) {
                Pgain[i] = Pgain[0];
                Dgain[i] = Dgain[0];
            }
        }
        std::cout << "------------- leaphand real init finish !!!!!" << std::endl;
    }

    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }

    void updateHandState(const Eigen::VectorXd &eef_pos) final override {
    }

    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) final override {
        for (int i = 0; i < num_joint_; i++) {
            tar_joint_state_.position[i] = posTarget[i];
        }
        pub_tar_joints.publish(tar_joint_state_);
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int tail_shift) const final override {
        for (int i = 0; i < num_joint_; i++) {
            pgain.tail(tail_shift)[i] = Pgain[i];
            dgain.tail(tail_shift)[i] = Dgain[i];
        }
    }

    Eigen::VectorXd & getJointEffort() final override {
        std::chrono::milliseconds timeout(10);
        //std::lock_guard<std::mutex> lock(cb_mutex);
        if (cb_mutex.try_lock_for(timeout)){
            cb_mutex.unlock();
        } else {
            printf("++++++++++++++++++++++++++++lock timeout get joint effort");
        }
        return hand_joint_effort_;
    }
    Eigen::VectorXd & getJointVelocity() final override {
        std::chrono::milliseconds timeout(10);
        //std::lock_guard<std::mutex> lock(cb_mutex);
        if (cb_mutex.try_lock_for(timeout)){
            cb_mutex.unlock();
        } else {
            printf("++++++++++++++++++++++++++++lock timeout get joint vel");
        }
        return hand_joint_velocity_;
    }
    Eigen::VectorXd & getJointPosition() final override {
        std::chrono::milliseconds timeout(10);
        //std::lock_guard<std::mutex> lock(cb_mutex);
        if (cb_mutex.try_lock_for(timeout)){
            cb_mutex.unlock();
        } else {
            printf("++++++++++++++++++++++++++lock timeout get joint pos");
        }
        return hand_joint_position_;
    }
    const int getDim() const final override {
        return num_joint_;
    }
    const int getNumFinger() const final override {
        return num_finger_;
    }
    int getBodies(std::vector<std::string> & get_vec, bool contact_flag) const final override {
        if (contact_flag) {
            for (int i = 0; i < num_contacts_; i++) {
                get_vec.push_back(contact_bodies_[i]);
            }
            return num_contacts_;
        } else {
            for (int i = 0; i < num_bodies_; i++) {
                get_vec.push_back(body_parts_[i]);
            }
            return num_bodies_;
        }
    }

    std::string changeJointToLinkName(std::string frameName) const final override {
        if (!frameName.compare("tool02Flange_fixed_joint")) {
            return std::string("tool0");
        } else if (!frameName.compare("leap_joint3_tip")) {
            return std::string("leap_link3_tip");
        } else if (!frameName.compare("leap_joint7_tip")) {
            return std::string("leap_link7_tip");
        } else if (!frameName.compare("leap_joint11_tip")) {
            return std::string("leap_link11_tip");
        } else if (!frameName.compare("leap_joint15_tip")) {
            return std::string("leap_link15_tip");
        } else {
            return frameName;
        }
    }

private:
    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg) {

        std::chrono::milliseconds timeout(10);
        //std::lock_guard<std::mutex> lock(cb_mutex);
        if (cb_mutex.try_lock_for(timeout)){
            cb_mutex.unlock();
        } else {
            printf("+++++++++++++++++++lock timeout cb");
        }

        for (int i = 0; i < num_joint_; i++) {
            hand_joint_position_[i] = msg->position[i];
        }
    }

    void subscribeLoop() {
        ros::Rate rate(freq_hz_);
        while (ros::ok()) {
            ros::spinOnce();
            rate.sleep();
        }
    }

public:
    std::string joint_names[16] = {
        "leap_joint0", "leap_joint1", "leap_joint2", "leap_joint3",
        "leap_joint4", "leap_joint5", "leap_joint6", "leap_joint7",
        "leap_joint8", "leap_joint9", "leap_joint10", "leap_joint11",
        "leap_joint12", "leap_joint13", "leap_joint14", "leap_joint15"
    };


private:
    raisim::ArticulatedSystem *platform_;

    bool flying_hand_mode_ = false;

    const static int num_contacts_ = 13;
    const static int num_bodies_ = 17;
    const static int num_finger_ = 4;
    const static int num_joint_ = 16;

    double Pgain[num_joint_] = {60.0};
    double Dgain[num_joint_] = {0.2};

    const std::string body_parts_flying_[num_bodies_] = {"TBD"};

    const std::string body_parts_[num_bodies_] =  {"tool02Flange_fixed_joint",
    "leap_joint0", "leap_joint2", "leap_joint3", "leap_joint3_tip",
    "leap_joint4", "leap_joint6", "leap_joint7", "leap_joint7_tip",
    "leap_joint8", "leap_joint10", "leap_joint11", "leap_joint11_tip",
    "leap_joint13", "leap_joint14", "leap_joint15", "leap_joint15_tip"};

    // for raisim contact check
    const std::string contact_bodies_[num_contacts_] =   {"wrist_3_link",
    "pip", "dip", "fingertip",
    "pip_2", "dip_2", "fingertip_2",
    "pip_3", "dip_3", "fingertip_3",
    "thumb_pip", "thumb_dip", "thumb_fingertip"};

    ros::NodeHandle* nh_;
    ros::Publisher pub_tar_joints;
    ros::Subscriber sub_cur_joints;
    sensor_msgs::JointState cur_joint_state_;
    sensor_msgs::JointState tar_joint_state_;
    sensor_msgs::JointState tar_tor_joint_state_;
    std::thread subscribe_thread_;
    std::timed_mutex cb_mutex;

    double freq_hz_ = 0.0;
};

extern "C" std::unique_ptr<HardwareHand> createLeapReal() {
    return std::make_unique<LeapReal>();
}
