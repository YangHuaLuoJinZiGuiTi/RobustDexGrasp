#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <ur_rtde/rtde_io_interface.h>
#include <thread>
#include <chrono>
#include <time.h>

#include <math.h>
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
double rtde_frequency = 500.0;
uint16_t flags = ur_rtde::RTDEControlInterface::FLAG_USE_EXT_UR_CAP;

double lookahead_time = 0.05;
double gain = 1400;

std::ofstream csv_file_;

std::unique_ptr<ur_rtde::RTDEControlInterface> rtde_control;
std::unique_ptr<ur_rtde::RTDEReceiveInterface> rtde_receive;

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

void control_delay(std::vector<double> &tar, double delay_s) {
    int wait_cnt = int(delay_s / 0.005);
    rtde_control->servoJ(tar, 0, 0, dt, lookahead_time, gain);
    while (wait_cnt > 0) {
        wait_cnt--;
        std::vector<double> cur_joint = rtde_receive->getActualQ();
        for (int i = 0; i < 6; i++) {
            csv_file_ << tar[i] << "," << cur_joint[i] << ",,";
        }
        csv_file_ << "\n";
        usleep(5000);
    }
    csv_file_.flush();
}


double set_random(double max, double min) {
    return min + rand() / (double)RAND_MAX * (max - min);
}

int test_pd()
{
    rtde_control = std::make_unique<ur_rtde::RTDEControlInterface>(robot_ip, rtde_frequency, flags);
    rtde_receive = std::make_unique<ur_rtde::RTDEReceiveInterface>(robot_ip, rtde_frequency);
    
    csv_file_.open("./log.csv", std::ios::out);
    for (int i = 0; i < 16; i++) {
        csv_file_ << i << "_tar========," << i << "_cur,,";
    }
    csv_file_ << "\n";


    double init_tmp[] =  {DEG2RAD(-10.0), DEG2RAD(-100.0),  DEG2RAD(80.0),  DEG2RAD(-10.0), DEG2RAD(80.0),   DEG2RAD(-100.0)};
    double end_pos[]   = {DEG2RAD(10.0),  DEG2RAD(-80.0),  DEG2RAD(100.0), DEG2RAD(10.0),  DEG2RAD(100.0),  DEG2RAD(-80.0)};
    std::vector<double> target_pose(init_tmp, init_tmp + 6);
    set_target2(target_pose, false);

    int test_joint = 3;
    double step_length_rad = 0.01;
    double step_dt_s = 0.2;
    int step_len = int((abs(end_pos[test_joint] - init_tmp[test_joint]))/step_length_rad);

    for (int j = 1; j <= step_len; j++) {
        for (int k = 0; k < 6; k++) {
            target_pose[k] = init_tmp[k] + j * step_length_rad;
        }
        control_delay(target_pose, step_dt_s);
    }
    for (int j = 1; j <= step_len; j++) {
        for (int k = 0; k < 6; k++) {
            target_pose[k] = end_pos[k] - j * step_length_rad;
        }
        control_delay(target_pose, step_dt_s);
    }

    usleep(100000);
    rtde_control->servoStop();
    usleep(1000000);
    rtde_control->stopScript();
    return 0;
}


int main(int argc, char* argv[])
{
    test_pd();
}