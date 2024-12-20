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


// "allegroHand/joint_cmd" allegro节点监听等待我们设置的joint目标
// "allegroHand/lib_cmd" allegro节点监听等待我们设置一些指定的命令

// "/allegroHand/commanded_joint_states"   allegro节点发布他收到的命令？
// "allegroHand/joint_states"    allegro节点发布当前joint state？

class AllegroReal {
public:
    void init() {
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
        hand_joint_position_last_.resize(16);
        hand_joint_velocity_last_.resize(16);

        csv_file_.open("./log.csv", std::ios::out);
        for (int i = 0; i < 16; i++) {
            csv_file_ << i << "_tar========," << i << "_cur,,";
        }
        csv_file_ << "\n";

        nh_ = new ros::NodeHandle();
        pub_tar_joints = nh_->advertise<sensor_msgs::JointState>("/allegroHand/joint_cmd", 1);
        sub_cur_joints = nh_->subscribe("/allegroHand/joint_states", 1, &AllegroReal::jointStateCallback, this);
        subscribe_thread_ = std::thread(&AllegroReal::subscribeLoop, this);

        std::cout << "init finish all !!!" << std::endl;
        std::srand(time(NULL));
    }

    void publishJointStates(std::vector<double> &tar_pos) {
        for (int i = 0; i < DOF_JOINTS; i++) {
            tar_joint_state_.position[i] = tar_pos[i];
        }
        pub_tar_joints.publish(tar_joint_state_);
    }

    bool getCurrentStates(std::vector<double> &cur_pos, std::vector<double> &cur_vel) {
            std::chrono::milliseconds timeout(2);
            if (cb_mutex.try_lock_for(timeout)){
                cb_mutex.unlock();
            for (int i = 0; i < DOF_JOINTS; i++) {
                cur_pos[i] = cur_joint_state_.position[i];
                //cur_vel[i] = cur_joint_state_.velocity[i];
                //hand_joint_position_last_[i] = cur_pos[i];
                //hand_joint_velocity_last_[i] = cur_vel[i];
            }
            return true;
        } else {
            std::cout << "joint pos is writing" << std::endl;
            return false;
        }
    }

    void control_delay(std::vector<double> &tar, double delay_s) {
        std::vector<double> get_pos(16), get_vel(16);
        int wait_cnt = int(delay_s / 0.005);
        publishJointStates(tar);
        while (ros::ok() && wait_cnt > 0) {
            wait_cnt--;
            getCurrentStates(get_pos, get_vel);
            for (int i = 0; i < 16; i++) {
                csv_file_ << tar[i] << "," << get_pos[i] << ",,";
            }
            csv_file_ << "\n";
            usleep(5000);
        }
    }

    double set_random(double max, double min) {
        return min + rand() / (double)RAND_MAX * (max - min);
    }

    void end() {
        csv_file_.flush();
    }

    static const int DOF_JOINTS = 16;
    bool start_get_flag_ = false;
    double min_limit[DOF_JOINTS] = {
        -0.45, -0.196, -0.174, -0.227, 
        -0.46, -0.196, -0.174, -0.227, 
        -0.46, -0.196, -0.174, -0.227, 
        0.35, -0.105, 0.15, -0.162
    };
    double max_limit[DOF_JOINTS] = {
        0.44, 1.61, 1.709, 1.618, 
        0.44, 1.61, 1.709, 1.618, 
        0.44, 1.61, 1.709, 1.618, 
        1.38, 1.163, 1.2, 1.719
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
        ros::Rate rate(500.0);
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
    std::vector<double> hand_joint_position_last_, hand_joint_velocity_last_;

};
#if 0
int control_test()
{
    AllegroReal test;
    test.init();
    usleep(2000000);

    std::vector<double> init_pos(16), init_vel(16), get_pos(16), get_vel(16), set_pos(16);
    double tmp[] = {0.0, 1.6, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 1.35, 0.0, 0.0, 0.0};
    init_pos.insert(init_pos.begin(), tmp,  tmp+16);
    set_pos.insert(set_pos.begin(), tmp,  tmp+16);
    test.publishJointStates(set_pos);
    usleep(2000000);

    double step_length_rad = 0.1;
    double step_dt_s = 0.2;

    for (int j = 1; j <= int(1.8/step_length_rad); j++) {
        set_pos[1] = 1.6 - j * step_length_rad;
        set_pos[5] = 1.6 - j * step_length_rad;
        set_pos[9] = 1.6 - j * step_length_rad;
        test.control_delay(set_pos, step_dt_s);
    }
    for (int j = 1; j <= int(1.8/step_length_rad); j++) {
        set_pos[1] = -0.2 + j * step_length_rad;
        set_pos[5] = -0.2 + j * step_length_rad;
        set_pos[9] = -0.2 + j * step_length_rad;
        test.control_delay(set_pos, step_dt_s);
    }
    test.end();
}
#else 
void control_test()
{
    AllegroReal test;
    test.init();
    usleep(2000000);
    std::vector<double> set_pos(16);
    double tmp[] = {0.0, 0.0, 0.0, 0., 0.0, 0.0, 0.0, 0., 0.0, 0.0, 0.0, 0., 1.3, 0.0, 0.0, 0.0};
    set_pos.insert(set_pos.begin(), tmp,  tmp+16);
    test.publishJointStates(set_pos);
    usleep(2000000);
    test.control_delay(set_pos, 0.2);
    bool flag[16] = {false};
    while (ros::ok()) {
        for (int i = 0; i < 4; i++) {
            int finger = 4*i+2;
            if (flag[finger]) {
                set_pos[finger] -= test.set_random(0.05, 0.1);
            } else {
                set_pos[finger] += test.set_random(0.05, 0.1);
            }

            if (set_pos[finger] > test.max_limit[finger]) {
                set_pos[finger] = test.max_limit[finger];
                flag[finger] = true;
            } else if (set_pos[finger] < test.min_limit[finger]) {
                set_pos[finger] = test.min_limit[finger];
                flag[finger] = false;
            }
        }
        test.control_delay(set_pos, 0.2);
        test.end();
    }
}
#endif
int step_test()
{
    AllegroReal test;
    test.init();
    usleep(5000000);

    std::vector<double> init_pos(16), init_vel(16), get_pos(16), get_vel(16), set_pos(16);
    //double tmp[] = {0.2, 0.8, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 1.57, 0, -0.05, 0.2};
    double tmp[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    set_pos.insert(set_pos.begin(), tmp,  tmp+16);
    init_pos.insert(init_pos.begin(), tmp,  tmp+16);

    while (ros::ok()) {
        std::cout << "current pos=";
        for (int t = 0; t < 16; t++) 
            std::cout << get_pos[t] << ", ";
        std::cout << std::endl;

        test.publishJointStates(set_pos);
        usleep(5000000);
        test.getCurrentStates(get_pos, get_vel);
        for (int t = 0; t < 16; t++) {
            if (abs(set_pos[t] - get_pos[t]) > 0.04) {
                printf("hand joint[%d] has a large gap: %f --- %f\n", t, set_pos[t], get_pos[t]);
                continue;
            } 
        }
        usleep(1000000);
        break;
    }

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            set_pos[j] = init_pos[j];
        }

        test.getCurrentStates(get_pos, get_vel);

        while (ros::ok() && set_pos[i] < (test.max_limit[i] - 0.01)) {
            set_pos[i] += 0.04;
            if (set_pos[i] > test.max_limit[i]) {
                set_pos[i] = test.max_limit[i];
            }
            test.publishJointStates(set_pos);
            usleep(500000);
            test.getCurrentStates(get_pos, get_vel);
            if (abs(get_pos[i] - set_pos[i]) > 0.04) {
                std::cout << "large gap when ++++++ joint[" << i << "]: set pos = " << set_pos[i] << ", get cur pos=" << get_pos[i] << std::endl;
            }
        }

        while (ros::ok() && set_pos[i] > (test.min_limit[i] + 0.01)) {
            set_pos[i] -= 0.04;
            if (set_pos[i] < test.min_limit[i]) {
                set_pos[i] = test.min_limit[i];
            }
            test.publishJointStates(set_pos);
            usleep(500000);
            test.getCurrentStates(get_pos, get_vel);

            if (abs(get_pos[i] - set_pos[i]) > 0.04) {
                std::cout << "large gap when ----- joint[" << i << "]: set pos = " << set_pos[i] << ", get cur pos=" << get_pos[i] << std::endl;
            }
        }

    
    }

    return 0;
}


int main(int argc, char **argv)
{
    control_test();
}