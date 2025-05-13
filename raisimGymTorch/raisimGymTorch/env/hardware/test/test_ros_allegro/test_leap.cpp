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
        pub_tar_joints = nh_->advertise<sensor_msgs::JointState>("/leaphand_node/cmd_allegro", 1);
        sub_cur_joints = nh_->subscribe("/leaphand_node/leap_joint", 1, &AllegroReal::jointStateCallback, this);
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
        double obs_dt = 0.01;
        int wait_cnt = int(delay_s / obs_dt);
        publishJointStates(tar);
        while (ros::ok() && wait_cnt > 0) {
            wait_cnt--;
            getCurrentStates(get_pos, get_vel);
            /*for (int i = 0; i < 16; i++) {
                csv_file_ << tar[i] << "," << get_pos[i] << ",,";
            }
            csv_file_ << "\n";*/
            usleep(obs_dt*1000000.0);
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

         double min_limit[16] = {
    -0.6, -0.20, -0.50, -0.30, 
    -0.6, -0.20, -0.50, -0.30, 
    -0.6, -0.20, -0.50, -0.30, 
    -2.00, -2.00, -1.20, -1.30
};
         double max_limit[16] = {
    1.0, 0.20, 1.80, 2.0, 
    1.0, 0.20, 1.80, 2.0, 
    1.0, 0.20, 1.80, 2.0, 
    0.40, 0.30, 1.90, 1.80
};


    std::string joint_names[DOF_JOINTS] = {
        "leap_joint0", "leap_joint1", "leap_joint2", "leap_joint3",
        "leap_joint4", "leap_joint5", "leap_joint6", "leap_joint7",
        "leap_joint8", "leap_joint9", "leap_joint10", "leap_joint11",
        "leap_joint12", "leap_joint13", "leap_joint14", "leap_joint15"
    };


private:

    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg) {
        std::chrono::milliseconds timeout(2);
        if (cb_mutex.try_lock_for(timeout)){
            cb_mutex.unlock();
            for (int i = 0; i < DOF_JOINTS; i++) {
                cur_joint_state_.position[i] = msg->position[i];
                // cur_joint_state_.velocity[i] = msg->velocity[i];
                // cur_joint_state_.effort[i] = msg->effort[i];
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
        test.control_delay(set_pos, step_dt_s);
    }
    for (int j = 1; j <= int(1.8/step_length_rad); j++) {
        set_pos[1] = -0.2 + j * step_length_rad;
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
    double tmp[16] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    set_pos.insert(set_pos.begin(), tmp,  tmp+16);
    test.publishJointStates(set_pos);
    usleep(2000000);
    test.control_delay(set_pos, 0.2);
    bool flag[16] = {false};
    while (ros::ok()) {
        for (int i = 0; i < 1; i++) {
            int finger = 14;
            if (flag[finger]) {
                set_pos[finger] -= test.set_random(0.1, 0.1);
            } else {
                set_pos[finger] += test.set_random(0.1, 0.1);
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
    usleep(1500000);

    std::vector<double> init_pos(16), init_vel(16), get_pos(16), get_vel(16), set_pos(16);
    //double tmp[] = {0.2, 0.8, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 1.57, 0, -0.05, 0.2};
    double tmp[] = {0.5, 0., 0.3, 0.5, 0.5, 0., 0.3, 0.5, 0.5, 0., 0.3, 0.5, -1.5, 0.0, -0.1, 0.2};
    set_pos.insert(set_pos.begin(), tmp,  tmp+16);
    init_pos.insert(init_pos.begin(), tmp,  tmp+16);

    while (ros::ok()) {
        test.publishJointStates(set_pos);
        usleep(2000000);
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
            set_pos[i] += 0.05;
            if (set_pos[i] > test.max_limit[i]) {
                set_pos[i] = test.max_limit[i];
            }
            
            test.publishJointStates(set_pos);
            usleep(200000);
            test.getCurrentStates(get_pos, get_vel);

            std::cout << "+++ set joint[" << i << "]: set pos = " << set_pos[i] << ", get cur pos=" << get_pos[i] << std::endl;

            if (abs(get_pos[i] - set_pos[i]) > 0.06) {
                std::cout << "large gap when ++++++ joint[" << i << "]: set pos = " << set_pos[i] << ", get cur pos=" << get_pos[i] << std::endl;
            }
        }

        while (ros::ok() && set_pos[i] > (test.min_limit[i] + 0.01)) {
            set_pos[i] -= 0.05;
            if (set_pos[i] < test.min_limit[i]) {
                set_pos[i] = test.min_limit[i];
            }
            test.publishJointStates(set_pos);
            usleep(200000);
            test.getCurrentStates(get_pos, get_vel);

            std::cout << "--- set joint[" << i << "]: set pos = " << set_pos[i] << ", get cur pos=" << get_pos[i] << std::endl;

            if (abs(get_pos[i] - set_pos[i]) > 0.06) {
                std::cout << "large gap when ----- joint[" << i << "]: set pos = " << set_pos[i] << ", get cur pos=" << get_pos[i] << std::endl;
            }
        }

    
    }

    return 0;
}


int main(int argc, char **argv)
{
    step_test();
}