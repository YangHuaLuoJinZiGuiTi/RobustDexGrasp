#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"


#include <iostream>

int main(int argc, char ** argv)
{
  using namespace pinocchio;
  // Load the URDF model
  Model model;
  pinocchio::urdf::buildModel("/home/ubuntu/hand/github/vision_dex/rsc/ur5_allegro/ur5_allegro_fk.urdf", model);

  // Build a data related to model
  Data data(model);

  // Sample a random joint configuration as well as random joint velocity and torque
  Eigen::VectorXd q = randomConfiguration(model);
  Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(model.nv);

  // Allocate result container
  Eigen::MatrixXd djoint_acc_dq = Eigen::MatrixXd::Zero(model.nv, model.nv);
  Eigen::MatrixXd djoint_acc_dv = Eigen::MatrixXd::Zero(model.nv, model.nv);
  Eigen::MatrixXd djoint_acc_dtau = Eigen::MatrixXd::Zero(model.nv, model.nv);


  auto starttime = std::chrono::system_clock::now();

  // Computes the forward dynamics (ABA) derivatives for all the joints of the robot
  computeABADerivatives(model, data, q, v, tau, djoint_acc_dq, djoint_acc_dv, djoint_acc_dtau);

  auto diff_time = std::chrono::system_clock::now() - starttime;
  std::cout << "collision cost time: " <<diff_time.count() / 1e9 << std::endl;

  // Get access to the joint acceleration
  std::cout << "Joint acceleration: " << data.ddq.transpose() << std::endl;

}
