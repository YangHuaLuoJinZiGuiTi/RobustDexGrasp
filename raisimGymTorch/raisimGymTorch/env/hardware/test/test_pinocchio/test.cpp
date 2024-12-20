// ref to https://github.com/stack-of-tasks/pinocchio/issues/802 

#include "pinocchio/parsers/urdf.hpp"
 
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
 
#include "pinocchio/multibody/sample-models.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/jacobian.hpp"


#include <cstdlib>
#include <time.h>
#include <iostream>
#include<algorithm>

#define RAD2DEG(n) ((n)*180.0/EIGEN_PI)
#define DEG2RAD(n) ((n)*EIGEN_PI/180.0)

class pinocchio_ur5_test {
public:

  pinocchio_ur5_test() {
    model_ = std::make_unique<pinocchio::Model>();
    pinocchio::urdf::buildModel("/home/ubuntu/hand/UR5_IK/rsc/ur5.urdf", *model_);
    data_ = std::make_unique<pinocchio::Data>(*model_);

    std::cout << "dim configure q = " << model_->nq << std::endl;       // 6
    std::cout << "dim velocity = " << model_->nv << std::endl;          // 6
    std::cout << "number joints = " << model_->njoints << std::endl;    // 7
    std::cout << "number bodies = " << model_->nbodies << std::endl;    // 7
    std::cout << "number frames = " << model_->nframes << std::endl;    // 23

  }

  pinocchio::SE3 get_fk(Eigen::VectorXd set_q) {
    pinocchio::framesForwardKinematics(*model_, *data_, set_q);
    return data_->oMf[model_->getFrameId("tool0")];
  }

  int get_ik(pinocchio::SE3 target_SE3, Eigen::VectorXd &current_q, Eigen::VectorXd &get_q) {
    Eigen::VectorXd q = current_q; 
    pinocchio::FrameIndex frame_id = model_->getFrameId("tool0");

    pinocchio::Data::Matrix6x J(6, model_->nv);
    J.setZero();

    Eigen::Matrix<double, 6, 1> err;
    Eigen::VectorXd v(model_->nv);

    int timeout_cnt = 0;
    int step = 0;

    while (1) {
      pinocchio::framesForwardKinematics(*model_, *data_, q);
      const pinocchio::SE3 iMd = data_->oMf[frame_id].actInv(target_SE3); // A X B^-1
      err = pinocchio::log6(iMd).toVector(); // in joint frame
      if (err.norm() < eps) {
        for (int i = 0; i < 3; i++) {
          diff_SE3[i] = abs(target_SE3.translation()[i] - data_->oMf[frame_id].translation()[i]);
          max_save[i] = std::max(diff_SE3[i], max_save[i]);
        }

        if ( max_save[0] > 0.02 || max_save[1] > 0.02 || max_save[2] > 0.02 || false == target_SE3.rotation().isApprox(data_->oMf[frame_id].rotation(), 0.02476) ) {
          std::cout << "target:  "  << target_SE3.translation().transpose() << " || " << RAD2DEG(target_SE3.rotation().eulerAngles(0,1,2).transpose()) << std::endl;
          std::cout << "current: "  << data_->oMf[frame_id].translation().transpose() << " || " << RAD2DEG(data_->oMf[frame_id].rotation().eulerAngles(0,1,2).transpose()) << std::endl;
          std::cout << "angle: " << RAD2DEG(data_->oMf[frame_id].rotation().eulerAngles(0,2,1).transpose()) << std::endl;
          std::cout << "angle: " << RAD2DEG(data_->oMf[frame_id].rotation().eulerAngles(1,0,2).transpose()) << std::endl;
          std::cout << "angle: " << RAD2DEG(data_->oMf[frame_id].rotation().eulerAngles(1,2,0).transpose()) << std::endl;
          std::cout << "angle: " << RAD2DEG(data_->oMf[frame_id].rotation().eulerAngles(2,1,0).transpose()) << std::endl;
          std::cout << "angle: " << RAD2DEG(data_->oMf[frame_id].rotation().eulerAngles(2,0,1).transpose()) << std::endl;
          //std::cout << "imd: "  << iMd.translation().transpose() << " || " << (iMd.rotation().eulerAngles(0,1,2).transpose()) << std::endl;
          //std::cout << "error: "  << err.transpose() << std::endl;
    
          std::cout << "max=" << max_save[0] << ", " << max_save[1] <<  ", " << max_save[2] << std::endl << std::endl << std::endl;
        }
        break;

      }
      if (step >= IT_MAX) {
        timeout_cnt++;
        step = 0;
        if (timeout_cnt < TMO_MAX / 2) {
          q = pinocchio::randomConfiguration(*model_);
        } else {
          for (int i = 0; i < 6; i++) {
              q[i] = (rand()/double(RAND_MAX) - 0.5) * 2.0 * EIGEN_PI;
          }
        }
        continue;
      }
      if (timeout_cnt >= TMO_MAX) {
        timeout_cnt = -1;
        //std::cout << "err = " << err.transpose() << std::endl;
        break;
      }
      
      pinocchio::computeFrameJacobian(*model_, *data_, q, frame_id, J); // J in joint frame
      pinocchio::Data::Matrix6 Jlog;
      pinocchio::Jlog6(iMd.inverse(), Jlog);
      J = -Jlog * J;
      pinocchio::Data::Matrix6 JJt;
      JJt.noalias() = J * J.transpose();
      JJt.diagonal().array() += damp;
      v.noalias() = -J.transpose() * JJt.ldlt().solve(err);
      q = pinocchio::integrate(*model_, q, v * DT);

      step++;
    }

    for(int i = 0; i < 6; i++) {
        while(q(i) < -M_PI) {
            q(i) += 2 * M_PI;
        }
        while(q(i) > M_PI) {
            q(i) -= 2 * M_PI;
        }
    }

    get_q = q;
    return timeout_cnt;
  }

  double max_save[6] = {0};
  double diff_SE3[6] = {0};

private:
  std::unique_ptr<pinocchio::Model> model_;
  std::unique_ptr<pinocchio::Data> data_;

  const double eps =  0.02;  // desired position precision  0.01m 3degree
  const int IT_MAX = 1000;  // maximum number of iterations 
  const double DT = 0.1; // convergence rate (smaller may have higher resolution?)
  const double damp = 1e-6; // damping factor for the pseudoinversion
  const int TMO_MAX = 30; // maximum number of the cnt of over iterations

};

int single_test()
{
  class pinocchio_ur5_test test;
  Eigen::VectorXd ik_q(6);
  Eigen::VectorXd q(6);

   q << DEG2RAD(-47.1584), DEG2RAD(-67.6464),DEG2RAD(-32.848),DEG2RAD(-140.93),DEG2RAD(133.052),DEG2RAD(-90.2594);
    pinocchio::SE3 gfk = test.get_fk(q);
    std::cout << "fk: "  << gfk.translation().transpose() << " || " << RAD2DEG(gfk.rotation().eulerAngles(0,1,2).transpose()) << std::endl;


    auto starttime = std::chrono::system_clock::now();
    int success = test.get_ik(gfk, ik_q, ik_q);
    auto diff_time = std::chrono::system_clock::now() - starttime;
    std::cout << "time consume: " << diff_time.count() / 1e6 << "ms" << std::endl;
    std::cout << "result: " << RAD2DEG(ik_q.transpose()) << " and success number is " << success << std::endl;

}

int FK_IK_test()
{
  class pinocchio_ur5_test test;

  double diff_time_sum = 0.0;
  int test_cnt = 0;
  int err_cnt = 0;
  int success_cnt = 0;

    #define START_ANGLE     25
    #define END_ANLGE       65
    #define STEP            5
  int j0=START_ANGLE, j1=START_ANGLE, j2=START_ANGLE, j3=START_ANGLE, j4=START_ANGLE, j5=START_ANGLE; // from 0 to 90
  
  Eigen::VectorXd q(6);
  Eigen::VectorXd ik_q(6);


  srand((unsigned)time(NULL));

  while (1) {

    Eigen::Vector3d pos;
    Eigen::Vector3d angle;

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
    
    while (0) {
      //q << DEG2RAD(j0), DEG2RAD(j1 - 90),DEG2RAD(j2),DEG2RAD(j3 - 90),DEG2RAD(j4),DEG2RAD(j5);
      for (int i = 0; i < 6; i++) {
          q[i] = (rand()/double(RAND_MAX) - 0.5) * 2.0 * EIGEN_PI;
      }

      pinocchio::SE3 gfk_fl = test.get_fk(q);
      //std::cout << "fk: "  << gfk.translation().transpose() << " || " << RAD2DEG(gfk.rotation().eulerAngles(0,1,2).transpose()) << std::endl;
      
      pos = gfk_fl.translation();
      for (int i = 0; i < 3; i++) {
          pos[i] = pos[i] + (rand()/double(RAND_MAX) - 0.5) * 2.0 * 0.2;
      }

      double d = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2];
      if (d > 0.7*0.7 || d < 0.25*0.25)
        continue;

      angle = gfk_fl.rotation().eulerAngles(0,1,2);
      for (int i = 0; i < 3; i++) {
          angle[i] = angle[i] + (rand()/double(RAND_MAX) - 0.5) * 2.0 * 0.2;
      }

      break;
    }
    
    Eigen::Matrix3d rot;
    rot = Eigen::AngleAxisd(angle[0], Eigen::Vector3d::UnitX()) * 
                       Eigen::AngleAxisd(angle[1], Eigen::Vector3d::UnitY()) * 
                       Eigen::AngleAxisd(angle[2], Eigen::Vector3d::UnitZ());
    pinocchio::SE3 gfk(rot, pos);

    

    auto starttime = std::chrono::system_clock::now();
    int success = test.get_ik(gfk, ik_q, ik_q);
    auto diff_time = std::chrono::system_clock::now() - starttime;
    diff_time_sum += (diff_time.count() / 1e9);
    if (success == -1) {
        //std::cout << "error fk: "  << gfk.translation().transpose() << " || " << RAD2DEG(gfk.rotation().eulerAngles(0,1,2).transpose()) << std::endl;
        //std::cout << "error joint: "  << RAD2DEG(q.transpose()) << std::endl;
        err_cnt++;
    } else if (success > 0) {
      success_cnt += success;
    }
    test_cnt++;

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

    if (test_cnt % 1000 == 0) {
        std::cout << "error rate: " << err_cnt << "/" << test_cnt << ", time cost: " << diff_time_sum << ", cnt=" << success_cnt << ", max=" << test.max_save[0] << ", " << test.max_save[1] <<  ", " << test.max_save[2] <<  ", " << test.max_save[3] <<  ", " << test.max_save[4] <<  ", " << test.max_save[5] << std::endl;
    }

  //std::cout << "time consume: " << diff_time.count() / 1e6 << "us" << std::endl;
  }

  std::cout << "time consume: " << diff_time_sum << "s" << std::endl;
  std::cout << "error rate =  " << err_cnt << "/" << test_cnt << std::endl;
  //std::cout << "\nresult: " << ik_q.transpose() << std::endl;
}


int IK_test()
{
  class pinocchio_ur5_test test;

  double diff_time_sum = 0.0;
  int test_cnt = 0;
  int err_cnt = 0;
  int success_cnt = 0;
    #define MIN_R     0.3
    #define MAX_R     0.5
    #define STEP_POS  0.03
    #define MIN_RAD -3.1
    #define MAX_RAD   3.1
    #define STEP_RAD  0.1

  double x=MIN_R, y=MIN_R, z=MIN_R, roll=MIN_RAD, pitch=MIN_RAD, yaw=MIN_RAD;
  Eigen::VectorXd ik_q(6);

  while (1) {

    Eigen::Matrix3d rot;
    rot = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) * 
                       Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * 
                       Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    pinocchio::SE3 gfk(rot, Eigen::Vector3d(x, y, z));

    //std::cout << "fk: "  << gfk.translation().transpose() << " || " << RAD2DEG(gfk.rotation().eulerAngles(0,1,2).transpose()) << std::endl;

    auto starttime = std::chrono::system_clock::now();
    int success = test.get_ik(gfk, ik_q, ik_q);
    auto diff_time = std::chrono::system_clock::now() - starttime;
    diff_time_sum += (diff_time.count() / 1e9);
    if (success == -1) {
        err_cnt++;
    } else if (success > 0) {
      success_cnt += success;
    }
    test_cnt++;
  //std::cout << "time consume: " << diff_time.count() / 1e6 << "us" << std::endl;

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
                                break;
                            }
                        }
                    }
                }
            }
        }

  
    if (test_cnt % 5000 == 0) {
        std::cout << "error rate: " << err_cnt << "/" << test_cnt << ", time cost: " << diff_time_sum << ", more time success cnt = " << success_cnt << std::endl;
    }
  
  }

  std::cout << "time consume: " << diff_time_sum << "s" << std::endl;
  std::cout << "error rate =  " << err_cnt << "/" << test_cnt << std::endl;
  //std::cout << "\nresult: " << ik_q.transpose() << std::endl;
}


int main(int argc, char ** argv)
{
  FK_IK_test();
}
