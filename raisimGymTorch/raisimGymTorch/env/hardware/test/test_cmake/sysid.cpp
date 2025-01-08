#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <thread>
#include <chrono>
#include <mutex>
/* cpp library */
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <functional>
#include <time.h>
#include <stack>

#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <ur_rtde/rtde_io_interface.h>


// "allegroHand/joint_cmd" allegro节点监听等待我们设置的joint目标
// "allegroHand/lib_cmd" allegro节点监听等待我们设置一些指定的命令

// "/allegroHand/commanded_joint_states"   allegro节点发布他收到的命令？
// "allegroHand/joint_states"    allegro节点发布当前joint state？

#define SIM_DT 0.005
#define CONTROL_DT 0.05

class AllegroReal {
public:
    void init() {
        std::string robot_ip = "192.168.56.101";
        double rtde_frequency = 1.0/SIM_DT;
        uint16_t flags = ur_rtde::RTDEControlInterface::FLAG_USE_EXT_UR_CAP;

        rtde_control = std::make_unique<ur_rtde::RTDEControlInterface>(robot_ip, rtde_frequency, flags);
        rtde_receive = std::make_unique<ur_rtde::RTDEReceiveInterface>(robot_ip, rtde_frequency);
        reset_arm_pos.resize(6);

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
        std::srand(time(NULL));
    }

    void init_step(int mode) {

        double updown = 0.0, leftright = -1.57;
        if (mode >= 4 && mode < 8) {
            leftright -= 0.5;
        } else if (mode >= 8 && mode < 12) {
            leftright += 0.5;
        } else if (mode >= 12 && mode < 16) {
            updown += 0.25;
        } else if (mode >= 16 && mode < 20) {
            updown -= 0.25;
        } else if (mode >= 20 && mode < 24) {
            leftright += 0.5; updown += 0.25;
        } else if (mode >= 24 && mode < 28) {
            leftright += 0.5; updown -= 0.25;
        } else if (mode >= 28 && mode < 32) {
            leftright -= 0.5; updown -= 0.25;
        } else if (mode >= 32 && mode < 36) {
            leftright -= 0.5; updown += 0.25;
        }

        reset_arm_pos[0] = 0.0;
        reset_arm_pos[1] = -1.57;
        reset_arm_pos[2] = 1.57;
        reset_arm_pos[3] = updown;
        reset_arm_pos[4] =  1.57;
        reset_arm_pos[5] = leftright;
        printf("mode:%d will move to %f %f %f %f %f %f\n",mode, reset_arm_pos[0], reset_arm_pos[1], reset_arm_pos[2], reset_arm_pos[3], reset_arm_pos[4], reset_arm_pos[5]);
        rtde_control->moveJ(reset_arm_pos, 0.5, 1.0);
        usleep(500000);

        std::string name = "./" + std::to_string(mode) + ".csv";
        std::cout << "will open file : " << name << std::endl;
        csv_file_.open(name.c_str(), std::ios::out);
        if (!csv_file_) {
            std::cout << "open file fail" << std::endl;
            exit(0);
        }
        for (int i = 0; i < 6; i++) {
            csv_file_ << i << "_arm_tar," << i << "_arm_cur," << i << "_arm_vel," << i << "_arm_eff,,";
        }
        for (int i = 0; i < 16; i++) {
            csv_file_ << i << "_tar," << i << "_cur," << i << "_vel," << i << "_eff,,";
        }
        csv_file_ << "\n";

    }

    void publishJointStates(std::vector<double> &tar_pos) {
        for (int i = 0; i < DOF_JOINTS; i++) {
            tar_joint_state_.position[i] = tar_pos[i];
        }
        pub_tar_joints.publish(tar_joint_state_);
    }

    bool getArmCurrentStates(std::vector<double> &cur_pos, std::vector<double> &cur_vel) {
        cur_vel = rtde_receive->getActualQd();
        cur_pos = rtde_receive->getActualQ();
        return true;
    }

    bool getCurrentStates(std::vector<double> &cur_pos, std::vector<double> &cur_vel, std::vector<double> &cur_eff) {
        std::chrono::milliseconds timeout(2);
        if (cb_mutex.try_lock_for(timeout)){
            for (int i = 0; i < DOF_JOINTS; i++) {
                cur_pos[i] = cur_joint_state_.position[i];
                cur_vel[i] = cur_joint_state_.velocity[i];
                cur_eff[i] = cur_joint_state_.effort[i];
            }
            cb_mutex.unlock();
            return true;
        } else {
            std::cout << "joint pos is writing" << std::endl;
            return false;
        }
    }

    void sysid_log_once(std::vector<double> &tar) {
        std::vector<double> get_arm_pos(6), get_arm_vel(6), get_arm_eff(6);
        getArmCurrentStates(get_arm_pos, get_arm_vel);
        for (int i = 0; i < 6; i++) {
            csv_file_ << reset_arm_pos[i] << "," << get_arm_pos[i] << "," << get_arm_vel[i] << "," << get_arm_eff[i] << ",,";
        }

        std::vector<double> get_pos(16), get_vel(16), get_eff(16);
        getCurrentStates(get_pos, get_vel, get_eff);
        for (int i = 0; i < 16; i++) {
            csv_file_ << tar[i] << "," << get_pos[i] << "," << get_vel[i] << "," << get_eff[i] << ",,";
        }
        csv_file_ << "\n";
        csv_file_.flush();
    }

    void end() {
        csv_file_.flush();
        csv_file_.close();
    }

    void test_once(int mode) {
        double max_step = 0.04, min_step = 0.005;
        bool add_mode[16] = {false};
        int test_step = 150;

        if (mode % 4 < 2) {
            max_step = 0.05;
        }

        std::vector<double> set_pos(16);
        double tmp[] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5, 0.9, 0.0, 0.0, 0.0};
        set_pos.insert(set_pos.begin(), tmp,  tmp+16);
        publishJointStates(set_pos);
        usleep(2000000);
        sysid_log_once(set_pos);
        std::vector<int> indices;
        if (mode % 2 == 0) {
            int indices_tmp[] = {1,2,3, 5,6,7, 9,10,11, 15};
            for (int t = 0; t < sizeof(indices_tmp)/sizeof(indices_tmp[0]); t++) indices.push_back(indices_tmp[t]);
        } else {
            int indices_tmp[] = {0, 4, 8, 12,13,14};
            for (int t = 0; t < sizeof(indices_tmp)/sizeof(indices_tmp[0]); t++) indices.push_back(indices_tmp[t]);
        }
        // test 1 
        while (ros::ok() && test_step-- > 0) {
            double rand_step = get_random_pulse(max_step, min_step);
            for (int idx = 0; idx < indices.size(); idx++) {
                int setidx = indices[idx];
                if (add_mode[setidx]) {
                    set_pos[setidx] += rand_step;
                    if (set_pos[setidx] > max_limit[setidx]) {
                        set_pos[setidx] = max_limit[setidx];
                        add_mode[setidx] = false;
                    }
                } else {
                    set_pos[setidx] -= rand_step;
                    if (set_pos[setidx] < min_limit[setidx]) {
                        set_pos[setidx] = min_limit[setidx];
                        add_mode[setidx] = true;
                    }
                }
            }
            
            if (test_step % 20 == 0) printf("-- %d\n",test_step);

            publishJointStates(set_pos);
            usleep(CONTROL_DT*1e6);
            sysid_log_once(set_pos);
        }

    }
    double get_random_pulse(double max, double min) {
        return min + rand() / (double)RAND_MAX * (max - min);
    }

    static const int DOF_JOINTS = 16;
    double min_limit[DOF_JOINTS] = {
        -0.4, -0.15, -0.16, -0.2, 
        -0.4, -0.15, -0.16, -0.2, 
        -0.4, -0.15, -0.16, -0.2, 
        0.4, -0.1, 0.2, -0.15
    };
    double max_limit[DOF_JOINTS] = {
        0.4, 1.5, 1.6, 1.6, 
        0.4, 1.5, 1.6, 1.6, 
        0.4, 1.5, 1.6, 1.6, 
        1.1, 1.16, 1.0, 1.7
    };

    std::string joint_names[DOF_JOINTS] = {
        "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
        "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
        "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
        "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"
    };

private:

    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg) {
        std::chrono::milliseconds timeout(2);
        if (cb_mutex.try_lock_for(timeout)){
            cb_mutex.unlock();
            for (int i = 0; i < DOF_JOINTS; i++) {
                cur_joint_state_.position[i] = msg->position[i];
                cur_joint_state_.velocity[i] = msg->velocity[i];
                cur_joint_state_.effort[i] = msg->effort[i];
            }
        } else {
            std::cout << "joint is reading" << std::endl;
        }
    }

    void subscribeLoop()
    {
        ros::Rate rate(1.0/SIM_DT);
        while (ros::ok())
        {
            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    std::ofstream csv_file_;
    ros::NodeHandle* nh_;
    ros::Publisher pub_tar_joints;
    ros::Subscriber sub_cur_joints;
    sensor_msgs::JointState cur_joint_state_;
    sensor_msgs::JointState tar_joint_state_;
    std::thread subscribe_thread_;
    //std::mutex mutex_;
    std::timed_mutex cb_mutex;

    std::unique_ptr<ur_rtde::RTDEControlInterface> rtde_control;
    std::unique_ptr<ur_rtde::RTDEReceiveInterface> rtde_receive;
    std::vector<double> reset_arm_pos;
};


int sysid_control()
{
    AllegroReal test;


    test.init();
    usleep(1000000);

    for(int i = 0; i < 36; i++) {
        if (ros::ok()) {
            test.init_step(i);
            test.test_once(i);
            test.end();
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    return sysid_control();
}