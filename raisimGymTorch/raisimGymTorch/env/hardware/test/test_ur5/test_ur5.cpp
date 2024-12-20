#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <ur_rtde/rtde_io_interface.h>
#include <thread>
#include <chrono>
#include <time.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace ur_rtde;
using namespace std::chrono;

#define RAD2DEG(n) ((n)*180.0/M_PI)
#define DEG2RAD(n) ((n)*M_PI/180.0)

// Setup parameters
std::string robot_ip = "192.168.56.101";
double dt = 0.005; // 5ms
double rtde_frequency = 200.0;
uint16_t flags = ur_rtde::RTDEControlInterface::FLAG_USE_EXT_UR_CAP;

double lookahead_time = 0.2;
double gain = 100;

std::unique_ptr<ur_rtde::RTDEControlInterface> rtde_control;
std::unique_ptr<ur_rtde::RTDEReceiveInterface> rtde_receive;

void set_target1(std::vector<double> &tar_joint) {
    while (1) {
        std::vector<double> cur_tcp_speed = rtde_receive->getActualTCPSpeed();
        std::vector<double> cur_speed = rtde_receive->getActualQd();
        std::vector<double> cur_joint = rtde_receive->getActualQ();
        int arrive_cnt = 0;
        std::cout << "get: ";
        for (int i = 0; i < 6; i++) {
            double diff = std::abs(cur_joint[i] - tar_joint[i]);
            if (diff< 0.01) {
                arrive_cnt++;
            }
            //std::cout << diff << ", ";
            std::cout << cur_speed[i] << ", ";
        }
        std::cout << cur_tcp_speed[0] << ", " << cur_tcp_speed[1] << ", " << cur_tcp_speed[2];
        std::cout << std::endl;
        if (arrive_cnt == 6) {
            std::cout << "move to target pose success after async !!!!" << std::endl << std::endl;
            break;
        }
        rtde_control->servoJ(tar_joint, 0, 0, dt, lookahead_time, gain);

        usleep(50000);
    }

    usleep(1000000);
    rtde_control->servoStop();
    usleep(1000000);
    rtde_control->stopScript();
    usleep(500000);
}

void set_target2(std::vector<double> &tar_joint, bool asyn) {
    while (1) {
        std::vector<double> cur_tcp_speed = rtde_receive->getActualTCPSpeed();
        std::vector<double> cur_speed = rtde_receive->getActualQd();
        std::vector<double> cur_joint = rtde_receive->getActualQ();
        int arrive_cnt = 0;
        std::cout << "get: ";
        for (int i = 0; i < 6; i++) {
            double diff = std::abs(cur_joint[i] - tar_joint[i]);
            if (diff< 0.01) {
                arrive_cnt++;
            }
            //std::cout << diff << ", ";
            std::cout << cur_speed[i] << ", ";
        }
        std::cout << cur_tcp_speed[0] << ", " << cur_tcp_speed[1] << ", " << cur_tcp_speed[2];
        std::cout << std::endl;
        if (arrive_cnt == 6) {
            std::cout << "move to target pose success after async !!!!" << std::endl << std::endl;
            break;
        }
        rtde_control->moveJ(tar_joint, 0.5, 1.0, asyn);

        usleep(50000);
    }

    usleep(1000000);
    rtde_control->stopJ();
    rtde_control->stopScript();
    usleep(500000);
}

int test_pose()
{
    rtde_control = std::make_unique<ur_rtde::RTDEControlInterface>(robot_ip, rtde_frequency, flags);
    rtde_receive = std::make_unique<ur_rtde::RTDEReceiveInterface>(robot_ip, rtde_frequency);
    
    double init_tmp[] = {DEG2RAD(-20.), DEG2RAD(-100.), DEG2RAD(100.), DEG2RAD(0), DEG2RAD(100.), DEG2RAD(-120.)};
    std::vector<double> target_pose(init_tmp, init_tmp + 6);
    double init_tmp2[] = {DEG2RAD(0.), DEG2RAD(-60.), DEG2RAD(90.), DEG2RAD(-30.), DEG2RAD(50.), DEG2RAD(-60.)};
    std::vector<double> target_pose2(init_tmp2, init_tmp2 + 6);

    set_target2(target_pose, false);
    set_target2(target_pose2, false);
    set_target1(target_pose);
    set_target1(target_pose2);
    return 0;
}


int test_torque()
{
    rtde_control = std::make_unique<ur_rtde::RTDEControlInterface>(robot_ip, rtde_frequency, flags);
    rtde_receive = std::make_unique<ur_rtde::RTDEReceiveInterface>(robot_ip, rtde_frequency);

    // Parameters
    std::vector<double> task_frame = {0, 0, 0, 0, 0, 0};
    std::vector<int> selection_vector = {0, 0, 0, 0, 0, 1};
    std::vector<double> wrench_down = {0, 0, 0, 0, 0, 30};
    std::vector<double> wrench_up = {0, 0, 0, 0, 0, -30};
    int force_type = 2;
    std::vector<double> limits = {2, 2, 1.5, 1, 1, 1};

    // Move to initial joint position with a regular moveJ
    /*
    double init_tmp[] = {DEG2RAD(-20.), DEG2RAD(-100.), DEG2RAD(100.), DEG2RAD(0), DEG2RAD(100.), DEG2RAD(-120.)};
    std::vector<double> target_pose(init_tmp, init_tmp + 6);
    set_target(target_pose);
    usleep(2000000);
    rtde_control->stopJ();
    rtde_control->servoStop();
    rtde_control->speedStop();
    */

    std::cout << "start......." << std::endl;


    for (unsigned int i=0; i<200; i++) {
        std::vector<double> cur_torque = rtde_control->getJointTorques();
        std::cout << "torque: " << cur_torque[0] << ", " << cur_torque[1] << ", " << cur_torque[2] << ", " << cur_torque[3] << std::endl;

        if (i > 100) {
            rtde_control->forceMode(task_frame, selection_vector, wrench_up, force_type, limits);
            std::cout << "move up" << std::endl;
        }
        else {
            rtde_control->forceMode(task_frame, selection_vector, wrench_down, force_type, limits);
            std::cout << "move down" << std::endl;
        }
        usleep(100000);
    }

    std::cout << "end......." << std::endl;
    rtde_control->forceModeStop();
    rtde_control->stopScript();

    return 0;
}

int test_speedJ() {
    rtde_control = std::make_unique<ur_rtde::RTDEControlInterface>(robot_ip, rtde_frequency, flags);
    rtde_receive = std::make_unique<ur_rtde::RTDEReceiveInterface>(robot_ip, rtde_frequency);

    for (unsigned int i=0; i<3000; i++) {
        if (i < 1000) {
            double rand_speed = 0.3 + (i % 100) * 0.01;
            std::vector<double> tar_speed = {0, 0, 0, 0, rand_speed, 0.0};
            rtde_control->speedJ(tar_speed, 0.05, 0.0);
            usleep(5000);
            std::cout << "+++++" << i << std::endl;
        } else {
            double rand_speed = -0.3 - (i % 100) * 0.01;
            std::vector<double> tar_speed = {0, 0, 0, 0, rand_speed, 0.0};
            rtde_control->speedJ(tar_speed, 0.05, 0.0);
            usleep(5000);
            std::cout << "-----" << i << std::endl;
        }
    }
    rtde_control->speedStop();
    rtde_control->stopScript();
}

class PIDController {
public:
    void setParam(double setp, double seti, double setd, double max_v, double setdt) {
        kp_ = setp; ki_ = seti; kd_ = setd; dt_ = setdt, max_v_ = max_v; 
    }

    void reset_state() {
        integral_ = 0.0, previous_error_ = 0.0;
    }

    double compute(double target_position, double current_position) {
        double error = target_position - current_position;
        integral_ += error * dt_;
        double derivative = (error - previous_error_) / dt_;

        double P = kp_ * error;
        double I = ki_ * integral_;
        double D = kd_ * derivative;

        double control_output = P + I + D;
        if (control_output > max_v_) control_output = max_v_;
        if (control_output < -max_v_) control_output = -max_v_;

        previous_error_ = error;

        return control_output;
    }
private:
    double kp_, ki_, kd_, dt_, max_v_, integral_ = 0.0, previous_error_ = 0.0;
};

class PosController {
public:
    void init(double set_dt, std::string param_pth) {
        dt_ms_ = int(set_dt * 1000.0);
        pid_.resize(6);
        std::ifstream file(param_pth);
        if (!file.is_open()) {
            std::cerr << "open file error!" << std::endl;
            exit(0);
        }

        std::string line;
        int line_cnt = 0;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            std::vector<double> row;
            while (std::getline(ss, item, ',')) {
                row.push_back(std::stod(item));
            }
            if (row.size() == 4) { 
                pid_[line_cnt].setParam(row[0], row[1], row[2], row[3], set_dt);
                line_cnt++;

                printf("read line (%d) set data %f, %f, %f, %f , %f \n", line_cnt, row[0], row[1], row[2], row[3], set_dt);
            } else {
                std::cerr << "read line(" << line_cnt << ") error: " << line << std::endl;
                exit(0);
            }
        }

        file.close();


        rtde_control_ = std::make_unique<ur_rtde::RTDEControlInterface>(robot_ip, rtde_frequency, flags);
        rtde_receive_ = std::make_unique<ur_rtde::RTDEReceiveInterface>(robot_ip, rtde_frequency);


        double init_tmp[] = {-0.262358,	-1.50493,	1.60241,	-0.112123,	1.33796,	-1.92271};
        std::vector<double> reset_pos(init_tmp, init_tmp + 6);
        sync_reset(reset_pos, false);

        std::vector<double> cur_joint = rtde_receive_->getActualQ();
        tar_joint_.resize(6);
        std::cout << "current joint: ";
        for (int i = 0; i < 6; i++) {
            tar_joint_[i] = cur_joint[i];
            std::cout << cur_joint[i] << ", ";
        }
        std::cout << std::endl;

        pid_thread_= std::thread(&PosController::control_loop, this);
    }

    void set_servoJ_tar_joint(std::vector<double> &tar_joint) {
        for (int i = 0; i < 6; i++) {
            tar_joint_[i] = tar_joint[i];
        }
        rtde_control_->servoJ(tar_joint_, 0., 0., dt_ms_/1000.0, ahead_time, gain);
    }

    void set_tar_joint(std::vector<double> &tar_joint) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < 6; i++) {
            tar_joint_[i] = tar_joint[i];
            pid_[i].reset_state();
        }
    }
    std::vector<double> get_tar_joint() {
        std::lock_guard<std::mutex> lock(mutex_);
        return tar_joint_;
    }

    void control_loop() {

        std::ofstream csv_file_("/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/log_data/other/pidlog.csv");
        csv_file_ << "0tar,0cur,0out,1tar,1cur,1out,2tar,2cur,2out,3tar,3cur,3out,4tar,4cur,4out,5tar,5cur,5out\n";
        std::vector<double> set_v(6);
        while (1) {
            std::vector<double> cur_tcp_speed = rtde_receive_->getActualTCPSpeed();
            std::vector<double> cur_speed = rtde_receive_->getActualQd();
            std::vector<double> cur_joint = rtde_receive_->getActualQ();
            std::vector<double> tar_joint = get_tar_joint();
            for (int i = 0; i < 6; i++) {
                set_v[i] = pid_[i].compute(tar_joint[i], cur_joint[i]);
                csv_file_ << tar_joint[i] << "," << cur_joint[i] << "," << set_v[i] << ",";
            }
            csv_file_ << "\n";
            //rtde_control_->speedJ(set_v, max_a, 0.0);
            std::this_thread::sleep_for(std::chrono::milliseconds(dt_ms_));
        }
    }

    void sync_reset(std::vector<double> &tar_joint, bool asyn) {
        rtde_control_->moveJ(tar_joint, 0.5, 1.0, asyn);
        usleep(1000000);
        rtde_control_->stopJ();
        rtde_control_->stopScript();
        usleep(500000);
    }


private:
    std::unique_ptr<ur_rtde::RTDEControlInterface> rtde_control_;
    std::unique_ptr<ur_rtde::RTDEReceiveInterface> rtde_receive_;
    std::vector<PIDController> pid_;
    std::vector<double> tar_joint_;
    std::thread pid_thread_;
    std::mutex mutex_;
    int dt_ms_;
    double max_a = 2.0;

    double gain = 500;
    double ahead_time = 0.05;
};

int test_speed_pid() {

    PosController c;
    c.init(dt, "/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/arm/UR5PIDReal.csv");

    usleep(1000000);
    std::ifstream file("/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/log_data/other/test.csv");
    if (!file.is_open()) {
        std::cerr << "open file error!" << std::endl;
        exit(0);
    }
    std::string line;
    int line_cnt = 0;
    
    while (std::getline(file, line)) {
        line_cnt++;
        std::stringstream ss(line);
        std::string item;
        std::vector<double> row;
        while (std::getline(ss, item, ',')) {
            row.push_back(std::stod(item));
        }
        std::cout << "read and set tar: ";
        for (int i = 0; i < row.size(); i++) {
            std::cout << row[i] << ", ";
        }
        std::cout << std::endl;

        c.set_servoJ_tar_joint(row);
        usleep(50000);
    }

    usleep(200000);
    file.close();

}

int main(int argc, char* argv[])
{
    test_speed_pid();
}