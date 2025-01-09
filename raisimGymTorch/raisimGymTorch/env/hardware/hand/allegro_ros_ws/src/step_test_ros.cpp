#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include "canAPI.h"
#include "rDeviceAllegroHandCANDef.h"
#include <BHand/BHand.h>

/* cpp library */
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <functional>
#include <time.h>
#include <stack>
#include <thread>
#include <chrono>
#include <time.h>
#include <mutex>

// ros
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Float32MultiArray.h"
#include "sensor_msgs/JointState.h"

using namespace std;

// for ros
const std::string JOINT_STATE_TOPIC = "allegroHand/joint_states";
const std::string DESIRED_STATE_TOPIC = "allegroHand/joint_cmd";
ros::Publisher joint_state_pub;
ros::Subscriber joint_cmd_sub;
std::chrono::milliseconds timeout(10);

// Store the current and desired joint states.
sensor_msgs::JointState current_joint_state;
sensor_msgs::JointState desired_joint_state;


// for CAN communication
const double delT = 0.003;
int CAN_Ch = 18;
bool ioThreadRun = false;
std::thread        hThread;
AllegroHand_DeviceMemory_t vars;
std::timed_mutex get_target_mutex;
std::timed_mutex pub_value_mutex;

// for BHand library 
BHand* pBHand = NULL;
double q[MAX_DOF]; // joint position
double q_des[MAX_DOF]; // desired joint position used in joint pd control motion
double tau_des[MAX_DOF]; // desired joint torque
double cur_des[MAX_DOF]; // current joint torque  

const double tau_cov_const_v4 = 1200.0; // 1200.0 for SAH040xxxxx 

static double min_limit[] = {
    -0.47, -0.196, -0.174, -0.227, 
    -0.47, -0.196, -0.174, -0.227, 
    -0.47, -0.196, -0.174, -0.227, 
    0.263, -0.105, -0.189, -0.162
};
static double max_limit[] = {
    0.47, 1.61, 1.709, 1.618, 
    0.47, 1.61, 1.709, 1.618, 
    0.47, 1.61, 1.709, 1.618, 
    1.396, 1.163, 1.644, 1.719
};

std::ofstream csv_file_;

// CAN communication thread
void ioThreadProc(void)
{
    int id;
    int len;
    unsigned char data[8];
    unsigned char data_return = 0;
    int i;

    long kezitest=0;

    static int cnt = 0;

    auto last_time = std::chrono::system_clock::now();
    while (ioThreadRun)
    {
        /* wait for the event */
        while (0 == get_message(CAN_Ch, &id, &len, data, FALSE))
        {
            switch (id)
            {
            case ID_RTR_FINGER_POSE_1:
            case ID_RTR_FINGER_POSE_2:
            case ID_RTR_FINGER_POSE_3:
            case ID_RTR_FINGER_POSE_4:
            {
                int findex = (id & 0x00000007);

                vars.enc_actual[findex*4 + 0] = (short)(data[0] | (data[1] << 8));
                vars.enc_actual[findex*4 + 1] = (short)(data[2] | (data[3] << 8));
                vars.enc_actual[findex*4 + 2] = (short)(data[4] | (data[5] << 8));
                vars.enc_actual[findex*4 + 3] = (short)(data[6] | (data[7] << 8));
                data_return |= (0x01 << (findex));

                if (data_return == (0x01 | 0x02 | 0x04 | 0x08))
                {
                    if (pub_value_mutex.try_lock_for(timeout)){
                        // convert encoder count to joint angle
                        for (i=0; i<MAX_DOF; i++)
                        {
                            q[i] = (double)(vars.enc_actual[i])*(333.3/65536.0)*3.141592f/180.0f;
                        }
                        pub_value_mutex.unlock();
                    } else {
                        printf("++++++++++++++++++++++++++++lock timeout set q");
                    }

                    pBHand->SetJointPosition(q); // tell BHand library the current joint positions

                    if (get_target_mutex.try_lock_for(timeout)){
                        pBHand->SetJointDesiredPosition(q_des);
                        get_target_mutex.unlock();
                    } else {
                        printf("++++++++++++++++++++++++++++lock timeout set q_des");
                    }
                    pBHand->UpdateControl(0);
                    pBHand->GetJointTorque(tau_des);
                    // convert desired torque to desired current and PWM count

                    if (pub_value_mutex.try_lock_for(timeout)){
                        for (int i=0; i<MAX_DOF; i++)
                        {
                            cur_des[i] = tau_des[i];
                            // set limit
                            if (cur_des[i] > 0.7) cur_des[i] = 0.7;
                            else if (cur_des[i] < -0.7) cur_des[i] = -0.7;
                        }
                        pub_value_mutex.unlock();
                    } else {
                        printf("++++++++++++++++++++++++++++lock timeout set eff");
                    }


                    // auto now_time = std::chrono::system_clock::now();
                    // auto diff_time = now_time - last_time;
                    // double dt = diff_time.count() / 1e6;
                    // last_time = now_time;
                    // cnt++;
                    // if (cnt % 2 == 0) {
                    //     for (int i = 0; i < 16; i++) {
                    //         csv_file_ << q_des[i] << "," << q[i] << "," << cur_des[i] << "," << dt << ",,";
                    //     }
                    //     csv_file_ << "\n";
                    //     csv_file_.flush();
                    // }



                    for (int i=0; i<4;i++)
                    {
                        vars.pwm_demand[i*4+0] = (short)(cur_des[i*4+0]*tau_cov_const_v4);
                        vars.pwm_demand[i*4+1] = (short)(cur_des[i*4+1]*tau_cov_const_v4);
                        vars.pwm_demand[i*4+2] = (short)(cur_des[i*4+2]*tau_cov_const_v4);
                        vars.pwm_demand[i*4+3] = (short)(cur_des[i*4+3]*tau_cov_const_v4);

                        command_set_torque(CAN_Ch, i, &vars.pwm_demand[4*i]);
                    }
                    data_return = 0;
                }
                break;
            }
            
            default:
                printf(">CAN(%d): unknown command %d, len %d\n", CAN_Ch, id, len);
            }
        }
    }
}


// Open a CAN data channel
bool OpenCAN()
{
    printf(">CAN(%d): open\n", CAN_Ch);

    int ret = command_can_open(CAN_Ch);
    if(ret < 0)
    {
        printf("ERROR command_can_open !!! \n");
        exit(0);
    }

    // initialize CAN I/O thread
    ioThreadRun = true;
    hThread = std::thread(&ioThreadProc);
    printf(">CAN: starts listening CAN frames\n");

    //query h/w information
    printf(">CAN: query system information\n");
    ret = request_hand_information(CAN_Ch);
    if(ret < 0)
    {
        printf("ERROR request_hand_information !!! \n");
        command_can_close(CAN_Ch);
        exit(0);
    }
    ret = request_hand_serial(CAN_Ch);
    if(ret < 0)
    {
        printf("ERROR request_hand_serial !!! \n");
        command_can_close(CAN_Ch);
        exit(0);
    }

    // set periodic communication parameters(period)
    printf(">CAN: Comm period set\n");
    short comm_period[3] = {3, 0, 0}; // millisecond {position, imu, temperature}
    ret = command_set_period(CAN_Ch, comm_period);
    if(ret < 0)
    {
        printf("ERROR command_set_period !!! \n");
        command_can_close(CAN_Ch);
        exit(0);
    }

    // servo on
    printf(">CAN: servo on\n");
    ret = command_servo_on(CAN_Ch);
    if(ret < 0)
    {
        printf("ERROR command_servo_on !!! \n");
        command_set_period(CAN_Ch, 0);
        command_can_close(CAN_Ch);
        exit(0);
    }

    return true;
}


void setJointCallback(const sensor_msgs::JointState &msg) {
    if (get_target_mutex.try_lock_for(timeout)) {
        for (int i = 0; i < 16; i++)
            q_des[i] = msg.position[i];
        get_target_mutex.unlock();
    } else {
        std::cout << "set jcb timeout" << std::endl;
    }
}


// Program main
int main(int argc, char **argv) 
{
    // ROS init
    ros::init(argc, argv, "test_ros");
    ros::NodeHandle nh;
    joint_cmd_sub = nh.subscribe(DESIRED_STATE_TOPIC, 1, setJointCallback);
    joint_state_pub = nh.advertise<sensor_msgs::JointState>(JOINT_STATE_TOPIC, 1);
    current_joint_state.position.resize(MAX_DOF);
    current_joint_state.velocity.resize(MAX_DOF);
    current_joint_state.effort.resize(MAX_DOF);
    current_joint_state.name.resize(MAX_DOF);


    csv_file_.open("./test.csv", std::ios::out);
    if (!csv_file_) {
        std::cout << "open file fail" << std::endl;
        exit(0);
    }
    for (int i = 0; i < 16; i++) {
        csv_file_ << i << "_tar," << i << "_cur," << i << "_eff," << i << "_dt,,";
    }
    csv_file_ << "\n";

    memset(&vars, 0, sizeof(vars));
    memset(q, 0, sizeof(q));
    memset(tau_des, 0, sizeof(tau_des)); 
    memset(cur_des, 0, sizeof(cur_des)); 
    double tmp[16] = {0.0, 1.6, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 1.3, 0.0, 0.0, 0.0};
    for (int i=0; i<16; i++) q_des[i] = tmp[i];

    pBHand = bhCreateRightHand();
    if (!pBHand) {
        std::cout << "start error "<< std::endl;
        exit(0);
    }
    pBHand->SetMotionType(eMotionType_NONE);
    pBHand->SetTimeInterval(delT);
    OpenCAN();

    pBHand->SetMotionType(eMotionType_JOINT_PD);
	// double kp[] = {
	// 		2000,  3000,  2000,  1600,
	// 		2000,  3000,  2000,  2000,
	// 		2000,  3000,  2000,  2000,
	// 		3600,  2500,  3000,  2000
	// 	};
	// double kd[] = {
	// 		80, 100, 80, 60,
	// 		80, 100, 80, 80,
	// 		80, 100, 80, 80,
	// 		120, 80, 100, 80
	// 	};
	double kp[] = {
			3800,  4800,  4600,  4800,
			3800,  4800,  4600,  4800,
			3800,  4600,  4600,  4800,
			4500,  4500,  4500, 4500
		};
	double kd[] = {
			100, 110, 110, 110,
			100, 110, 110, 110,
			100, 100, 100, 110,
			110, 110, 110, 100
		};
	pBHand->SetGainsEx(kp, kd);


    ros::Rate rate(500.0);
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
        if (pub_value_mutex.try_lock_for(timeout)){
            for (int i = 0; i < MAX_DOF; i++) {
                current_joint_state.position[i] = q[i];
                current_joint_state.effort[i] = cur_des[i];
            }
            pub_value_mutex.unlock();
        } else {
            std::cout << "pub timeout" << std::endl;
        }
        joint_state_pub.publish(current_joint_state);
    }
    return 0;
}


