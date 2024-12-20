#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/sample-models.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include <iostream>

int main(int /* argc */, char ** /* argv */)
{
  pinocchio::Model model;
  pinocchio::urdf::buildModel("/home/ubuntu/hand/UR5_IK/rsc/ur5.urdf", model);
  pinocchio::Data data(model);

  const int JOINT_ID = 6;

  // set desired pose
  const pinocchio::SE3 oMdes(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.6, 0.0, 0.4));

  // use neutral as init pose?
  //Eigen::VectorXd q = pinocchio::neutral(model); 
  Eigen::VectorXd q(6);
  q << -0.18,  -1.08,   1.00, 0.076, -0.18,   -3.14;
  std::cout << "init q = " << q.transpose() << std::endl;
  
  const double eps = 1e-4;  // desired position precision
  const int IT_MAX = 1000;  // maximum number of iterations 
  const double DT = 1e-1; // convergence rate
  const double damp = 1e-6; // damping factor for the pseudoinversion

  pinocchio::Data::Matrix6x J(6, model.nv);
  J.setZero();

  bool success = false;
  Eigen::Matrix<double, 6, 1> err;
  Eigen::VectorXd v(model.nv);

  auto starttime = std::chrono::system_clock::now();
  for (int i = 0;; i++)
  {
    pinocchio::forwardKinematics(model, data, q);
    const pinocchio::SE3 iMd = data.oMi[JOINT_ID].actInv(oMdes);
    err = pinocchio::log6(iMd).toVector(); // in joint frame
    if (err.norm() < eps)
    {
      success = true;
      break;
    }
    if (i >= IT_MAX)
    {
      success = false;
      break;
    }
    pinocchio::computeJointJacobian(model, data, q, JOINT_ID, J); // J in joint frame
    pinocchio::Data::Matrix6 Jlog;
    pinocchio::Jlog6(iMd.inverse(), Jlog);
    J = -Jlog * J;
    pinocchio::Data::Matrix6 JJt;
    JJt.noalias() = J * J.transpose();
    JJt.diagonal().array() += damp;
    v.noalias() = -J.transpose() * JJt.ldlt().solve(err);
    q = pinocchio::integrate(model, q, v * DT);
  }
  auto diff_time = std::chrono::system_clock::now() - starttime;
  std::cout << "time consume: " << diff_time.count() / 1e6 << "us" << std::endl;

  if (success)
  {
    std::cout << "Convergence achieved!" << std::endl;
  }
  else
  {
    std::cout
      << "\nWarning: the iterative algorithm has not reached convergence to the desired precision"
      << std::endl;
  }

  std::cout << "\nresult: " << q.transpose() << std::endl;
  std::cout << "\nfinal error: " << err.transpose() << std::endl;
}
