#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "../include/Kinematics.hpp"

#include <cstdlib>
#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#define RAD2DEG(n) ((n)*180.0/EIGEN_PI)
#define DEG2RAD(n) ((n)*EIGEN_PI/180.0)

#define PRINT_VEC(str, vec, a, b) \
    { \
    std::cout << str; \
    for (int i = a; i < b; i++) std::cout << (vec) << ", "; \
    std::cout << std::endl; \
    }
void print_vec(const char* str, std::vector<double> vec, int start, int end)
{
    std::cout << str ;
    for (int i = start; i < end; i++) std::cout << vec[i] << ", ";
    std::cout << std::endl;
}

void print_vec(const char* str, Eigen::VectorXd vec, int start, int end)
{
    std::cout << str ;
    for (int i = start; i < end; i++) std::cout << vec[i] << ", ";
    std::cout << std::endl;
}

void T2Eulert(std::vector<double>& vec, Eigen::Vector3d& ang, Eigen::Vector3d& t)
{
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; i++)
    {
        Eigen::Vector3d v;
        for (int j = 0; j < 4; j++)
        {
            if (j < 3) v(j) = vec[4*i+ j];
            else t(i) = vec[4*i+ j];
        }
        R.row(i) = v;
    }
    ang = R.eulerAngles(0,1,2);
}

std::vector<double> Eulert2T(Eigen::Vector3d ang, Eigen::Vector3d t)
{
    Eigen::Matrix3d rot;
    rot = Eigen::AngleAxisd(ang[0], Eigen::Vector3d::UnitX()) * 
                       Eigen::AngleAxisd(ang[1], Eigen::Vector3d::UnitY()) * 
                       Eigen::AngleAxisd(ang[2], Eigen::Vector3d::UnitZ());
    std::vector<double> ret;
    for (unsigned int i=0; i<3; i++) {
        ret.push_back(rot(i,0));
        ret.push_back(rot(i,1));
        ret.push_back(rot(i,2));
        ret.push_back(t[i]);
    }

    return ret;
}

void T2Quatt(std::vector<double>& vec, Eigen::Vector4d& q, Eigen::Vector3d& t)
{
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; i++)
    {
        Eigen::Vector3d v;
        for (int j = 0; j < 4; j++)
        {
            if (j < 3) v(j) = vec[4*i+ j];
            else t(i) = vec[4*i+ j];
        }
        R.row(i) = v;
    }
    Eigen::Quaterniond q_(R);
    q[0] = q_.w();
    q[1] = q_.x();
    q[2] = q_.y();
    q[3] = q_.z();
}


void Eulert2Quat(double roll, double pitch, double yaw, std::vector<double>& vec)
{
    Eigen::Quaterniond q =  Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    vec[0] = q.w();
    vec[1] = q.x();
    vec[2] = q.y();
    vec[3] = q.z();
}

std::vector<double> random_ret(std::vector<double> ret_in) {
    Eigen::Vector3d pos, angle;
#if 0
    T2Eulert(ret_in, angle, pos);
    for (int i = 0; i < 3; i++) {
        pos[i] = pos[i] + (rand()/double(RAND_MAX) - 0.5) * 2.0 * 0.1;
        angle[i] = angle[i] + (rand()/double(RAND_MAX) - 0.5) * 2.0 * 0.1;
    }
#else
    #define MIN_R   0.15
    #define MAX_R   0.7

    while (1) { // from 0.1 to 0.75
      for (int i = 0; i < 3; i++) {
          // [0,1]   [0, 0.65]   [0.1, 0.75]
          double sign = rand()/double(RAND_MAX);
          if (sign > 0.5) pos[i] = (rand()/double(RAND_MAX) * (MAX_R - MIN_R) + MIN_R);
          else pos[i] = -(rand()/double(RAND_MAX) * (MAX_R - MIN_R) + MIN_R);
      }
      double d = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2];
      if (d > MAX_R*MAX_R || d < MIN_R*MIN_R)
        continue;
      
      // [0,1]   [-0.5, 0.5]   [-PI, PI]
      for (int i = 0; i < 3; i++) {
          angle[i] = angle[i] + (rand()/double(RAND_MAX) - 0.5) * 2 * M_PI;
      }
      break;
    }
#endif
    std::vector<double> ret = Eulert2T(angle, pos);
    return ret;
}

void target_switch(Eigen::VectorXd& cur, std::vector<double>& tar, Eigen::VectorXd& out)
{
    double loss = -1.0;
    int tar_i = 0;

    if (0 == tar.size())
    {
        std::cout << "IK ERROR" << std::endl;
        return;
    }

    for (int i = 0; i < tar.size() / 6; i++)
    {
        double loss_tmp = 0.0;
        for (int j = 0; j < 6; j++)
        {
            loss_tmp += abs(cur[j] - tar[6 * i + j]);
        }
        if (loss < 0.0 || loss_tmp < loss)
        {
            loss = loss_tmp;
            tar_i = i;
        }
    }

    for (int j = 0; j < 6; j++)
    {
        out[j] = tar[6 * tar_i + j];
    }
}

int FK_IK_test() {
    Kinematics ur5;
    double diff_time_sum = 0.0;
    int test_cnt = 0;
    int err_cnt = 0;
    #define START_ANGLE     25
    #define END_ANLGE       65
    #define STEP            5
    int j0=START_ANGLE, j1=START_ANGLE, j2=START_ANGLE, j3=START_ANGLE, j4=START_ANGLE, j5=START_ANGLE; // from 0 to 90
    
    srand((unsigned)time(NULL));
    while (1) {
        std::vector<double> ret;
        Eigen::VectorXd tar_sc_(6);
        while (1) {
            //Eigen::VectorXd tar_sc_(6); tar_sc_ << DEG2RAD(j0), DEG2RAD(j1 - 90),DEG2RAD(j2),DEG2RAD(j3 - 90),DEG2RAD(j4),DEG2RAD(j5);
            for (int i = 0; i < 6; i++) {
                tar_sc_[i] = (rand()/double(RAND_MAX) - 0.5) * 2.0 * EIGEN_PI;
            }
            
            std::vector<double> ret_fk = ur5.forward(tar_sc_); // 3X4 matrix
            ret = random_ret(ret_fk);
            Eigen::Vector3d t, eulerAngle;
            T2Eulert(ret, eulerAngle, t);

            break;
        }
        //std::cout << "FK:\t" << RAD2DEG(eulerAngle.transpose()) << ", " << t.transpose() << std::endl;

        //Eulert2Quat(0.1, 0.1, 0.1, ret);
        //ret[4]=rand_xy(e);ret[5]=rand_xy(e);ret[6]=rand_z(e);
        // set target end-point position

        auto starttime = std::chrono::system_clock::now();
        std::vector<double> IK_joint = ur5.inverse(ret); 
        auto diff_time = std::chrono::system_clock::now() - starttime;
        diff_time_sum += (diff_time.count() / 1e9);
        if (IK_joint.size() == 0) {
            //if (RAD2DEG(eulerAngle.transpose()[0]) < 88.0 || RAD2DEG(eulerAngle.transpose()[0]) > 92.0) {
            err_cnt++;
            //std::cout << RAD2DEG(eulerAngle.transpose()) <<  std::endl;
            //}
        }
        test_cnt++;
#if 0
        j0 += STEP;
        if (j0 > END_ANLGE) {
            j0 = START_ANGLE;
            j1 += STEP;
            if (j1 > END_ANLGE) {
                j1 = START_ANGLE;
                j2 += STEP;
                if (j2 > END_ANLGE) {
                    j2 = START_ANGLE;
                    j3 += STEP;
                    if (j3 > END_ANLGE) {
                        j3 = START_ANGLE;
                        j4 += STEP;
                        if (j4 > END_ANLGE) {
                            j4 = START_ANGLE;
                            j5 += STEP;
                            if (j5 > END_ANLGE) {
                                break;
                            }
                        }
                    }
                }
            }
        }
#endif
            
        if (test_cnt % 100000 == 0) {
            std::cout << "error rate: " << err_cnt << "/" << test_cnt << ", time cost: " << diff_time_sum << std::endl;
        }

    }

    std::cout << "time consume: " << diff_time_sum << "s" << std::endl;
    std::cout << "error rate =  " << err_cnt << "/" << test_cnt << std::endl;
    //Eigen::VectorXd gc_(6); gc_.setZero();
    //target_switch(gc_, IK_joint, tar_sc_);
    //print_vec("IK target joint POS:\t", IK_joint, 0, IK_joint.size());

}


int IK_test()
{
    Kinematics ur5;

  double diff_time_sum = 0.0;
  int test_cnt = 0;
  int err_cnt = 0;
    #define MIN_R     0.41
    #define MAX_R     0.42
    #define STEP_POS  0.005
    #define MIN_RAD -0.05
    #define MAX_RAD   0.05
    #define STEP_RAD  0.01

  double x=MIN_R, y=MIN_R, z=MIN_R, roll=MIN_RAD, pitch=MIN_RAD, yaw=MIN_RAD;
  Eigen::VectorXd ik_q(6);

  while (1) {

        Eigen::Matrix3d rot;
        rot = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) * 
                            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * 
                            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
        double pos[3] = {x, y, z};
        std::vector<double> ret;
        for (unsigned int i=0; i<3; i++) {
            ret.push_back(rot(i,0));
            ret.push_back(rot(i,1));
            ret.push_back(rot(i,2));
            ret.push_back(pos[i]);
        }

        auto starttime = std::chrono::system_clock::now();
        std::vector<double> IK_joint = ur5.inverse(ret); 
        auto diff_time = std::chrono::system_clock::now() - starttime;
        diff_time_sum += (diff_time.count() / 1e9);
        if (IK_joint.size() == 0) {
            std::cout << "err: " << x << "," << y << ", " << z << "," << roll << ", " << pitch << ", " << yaw << std::endl;
            err_cnt++;
        }
        test_cnt++;

        bool end_flag = false;
        while (1) {
            x += STEP_POS;
            if (x > MAX_R) {
                x = MIN_R;
                y += STEP_POS;
                if (y > MAX_R) {
                    y = MIN_R;
                    z += STEP_POS;
                    if (z > MAX_R) {
                        z = MIN_R;
                        roll += STEP_RAD;
                        if (roll > MAX_RAD) {
                            roll = MIN_RAD;
                            pitch += STEP_RAD;
                            if (pitch > MAX_RAD) {
                                pitch = MIN_RAD;
                                yaw += STEP_RAD;
                                if (yaw > MAX_RAD) {
                                    end_flag = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if ( abs(roll) < 0.00001 && abs(pitch) < 0.00001 ) {
                continue;
            }

            break;
        }

        if (end_flag == true) {
            break;
        }
    
    if (test_cnt % 50000 == 0) {
        std::cout << "error rate: " << err_cnt << "/" << test_cnt << ", time cost: " << diff_time_sum << std::endl;
    }
  }

  std::cout << "time consume: " << diff_time_sum << "s" << std::endl;
  std::cout << "error rate =  " << err_cnt << "/" << test_cnt << std::endl;
  //std::cout << "\nresult: " << ik_q.transpose() << std::endl;
}


int test_single_pose()
{
    Kinematics ur5;

  double diff_time_sum = 0.0;
  int test_cnt = 0;
  int err_cnt = 0;
  
  double x=0.47608, y=0.11604, z=0.598022, roll=DEG2RAD(1.01745), pitch=DEG2RAD(-0.982241), yaw=DEG2RAD(-177.983);

    Eigen::VectorXd tar_sc_(6); tar_sc_ << DEG2RAD(1), DEG2RAD(-91),DEG2RAD(91),DEG2RAD(-91),DEG2RAD(91),DEG2RAD(91);
    std::vector<double> ret_fk = ur5.forward(tar_sc_); // 3X4 matrix
    print_vec("FK-RET = \t", ret_fk, 0, ret_fk.size());

    Eigen::Matrix3d rot;
    rot = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) * 
                        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * 
                        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    double pos[3] = {x, y, z};
    std::vector<double> ret;
    for (unsigned int i=0; i<3; i++) {
        ret.push_back(rot(i,0));
        ret.push_back(rot(i,1));
        ret.push_back(rot(i,2));
        ret.push_back(pos[i]);
    }
    print_vec("set-RET = \t", ret, 0, ret.size());

    auto starttime = std::chrono::system_clock::now();
    std::vector<double> IK_joint = ur5.inverse(ret); 
    auto diff_time = std::chrono::system_clock::now() - starttime;
    diff_time_sum += (diff_time.count() / 1e9);
    if (IK_joint.size() == 0) {
        std::cout << "error joint IK: " << std::endl;
    } else {
        std::cout << "time consume: " << diff_time_sum << "s" << std::endl;
        print_vec("IK target joint POS:\t", IK_joint, 0, IK_joint.size());
    }
}

int main(int argc, char ** argv)
{
  FK_IK_test();
}

