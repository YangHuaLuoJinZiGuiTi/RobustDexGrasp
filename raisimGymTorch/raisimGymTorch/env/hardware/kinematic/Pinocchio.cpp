#include "../hardwareKinematic.hpp"

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/srdf.hpp"
 
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
 
#include "pinocchio/multibody/sample-models.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/jacobian.hpp"

#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/collision/collision.hpp"

// cpp library
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

class Pinocchio : public HardwareKinematic {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) override {
        const std::string pth = rsc_pth + "/" + cfg["rsc_model"].As<std::string>();
        const std::string fk_hand_urdf_pth = pth + "/" + cfg["hand_type"].As<std::string>() + ".urdf";
        const std::string ik_arm_urdf_pth = pth + "/" + cfg["arm_type"].As<std::string>() + ".urdf";
        const std::string ik_arm_srdf_pth = pth + "/" + cfg["arm_type"].As<std::string>() + ".srdf";
        flying_mode_ = cfg["flying_hand_mode"].As<bool>();

        hand_fk_model_ = std::make_unique<pinocchio::Model>();
        pinocchio::urdf::buildModel(fk_hand_urdf_pth, *hand_fk_model_);
        hand_fk_data_ = std::make_unique<pinocchio::Data>(*hand_fk_model_);

        if (flying_mode_ == false) {
            arm_ik_model_ = std::make_unique<pinocchio::Model>();
            pinocchio::urdf::buildModel(ik_arm_urdf_pth, *arm_ik_model_);
            arm_ik_data_ = std::make_unique<pinocchio::Data>(*arm_ik_model_);
    
            arm_ik_geom_model_ = std::make_unique<pinocchio::GeometryModel>();
            pinocchio::urdf::buildGeom(*arm_ik_model_, ik_arm_urdf_pth, pinocchio::COLLISION, *arm_ik_geom_model_, pth);
            arm_ik_geom_model_->addAllCollisionPairs();
            pinocchio::srdf::removeCollisionPairs(*arm_ik_model_, *arm_ik_geom_model_, ik_arm_srdf_pth);
            arm_ik_geom_data_ = std::make_unique<pinocchio::GeometryData>(*arm_ik_geom_model_);
            pinocchio::srdf::loadReferenceConfigurations(*arm_ik_model_, ik_arm_srdf_pth); 
        }
    }

    // eef is x, y, z(m), rx, ry, rz(rad)
    // current_q is current joint position
    IK_ERRCODE getArmIKSolve(const Eigen::VectorXd eef, const Eigen::VectorXd current_q, Eigen::VectorXd &solved_q) const override {
        IK_ERRCODE ik_flag = IK_FAIL;
        if (flying_mode_) {
            return IK_FAIL;
        }

        Eigen::Matrix3d rot;
        rot = Eigen::AngleAxisd(eef[3], Eigen::Vector3d::UnitX()) * 
                        Eigen::AngleAxisd(eef[4], Eigen::Vector3d::UnitY()) * 
                        Eigen::AngleAxisd(eef[5], Eigen::Vector3d::UnitZ());
        pinocchio::SE3 target_SE3(rot, Eigen::Vector3d(eef[0], eef[1], eef[2]));

        Eigen::VectorXd q = current_q; 
        pinocchio::FrameIndex frame_id = arm_ik_model_->getFrameId("tool0");

        pinocchio::Data::Matrix6x J(6, arm_ik_model_->nv);
        J.setZero();

        Eigen::Matrix<double, 6, 1> err;
        Eigen::VectorXd v(arm_ik_model_->nv);

        int timeout_cnt = 0;
        int step = 0;

        while (1) {
            pinocchio::framesForwardKinematics(*arm_ik_model_, *arm_ik_data_, q);
            const pinocchio::SE3 iMd = arm_ik_data_->oMf[frame_id].actInv(target_SE3); // A X B^-1
            err = pinocchio::log6(iMd).toVector(); // in joint frame

            // reach error range and break
            if (err.norm() < eps) {
                ik_flag = IK_OK;
                break;
            }

            // The number of iterations exceeded the maximum, will init the q again from random_cfg or random_num
            if (step >= IT_MAX) {
                timeout_cnt++;
                step = 0;
                if (timeout_cnt < TMO_MAX / 2) {
                    q = pinocchio::randomConfiguration(*arm_ik_model_);
                } else {
                    for (int i = 0; i < 6; i++) {
                        q[i] = (rand()/double(RAND_MAX) - 0.5) * 2.0 * EIGEN_PI;
                    }
                }
                continue;
            }

            // try more then max times.
            if (timeout_cnt >= TMO_MAX) {
                timeout_cnt = -1;
                //std::cout << "err = " << err.transpose() << std::endl;
                ik_flag = IK_TIMEOUT;
                break;
            }
            
            pinocchio::computeFrameJacobian(*arm_ik_model_, *arm_ik_data_, q, frame_id, J); // J in joint frame
            pinocchio::Data::Matrix6 Jlog;
            pinocchio::Jlog6(iMd.inverse(), Jlog);
            J = -Jlog * J;
            pinocchio::Data::Matrix6 JJt;
            JJt.noalias() = J * J.transpose();
            JJt.diagonal().array() += damp;
            v.noalias() = -J.transpose() * JJt.ldlt().solve(err);
            q = pinocchio::integrate(*arm_ik_model_, q, v * DT);

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

        solved_q = q;

        if (ik_flag != IK_OK) {
            return ik_flag;
        }

        // check self collision
        if (false == pinocchio::computeCollisions(*arm_ik_model_, *arm_ik_data_, *arm_ik_geom_model_, *arm_ik_geom_data_, solved_q, true)) {
            return IK_OK;
        }

        for (size_t k = 0; k < arm_ik_geom_model_->collisionPairs.size(); ++k) {
            const pinocchio::CollisionPair & cp = arm_ik_geom_model_->collisionPairs[k];
            const hpp::fcl::CollisionResult & cr = arm_ik_geom_data_->collisionResults[k];
            if (cr.isCollision()) {
                std::cout << "collision pair: " << cp.first << " , " << cp.second << " - collision: " << std::endl;
                return IK_SELF_COLLISION;
            }
        }
        
        std::cout << "!!!!!! UNKNOW collision !!!!!!!!!" << std::endl;
        return IK_SELF_COLLISION;
    }

    void updateHandFK(const Eigen::VectorXd &hand_q, const Eigen::VectorXd &eef_pos) const override {
        Eigen::VectorXd set_q(hand_q.size() + eef_pos.size());
        set_q.head(hand_q.size()) = hand_q;
        set_q.tail(eef_pos.size()) = eef_pos;
        pinocchio::framesForwardKinematics(*hand_fk_model_, *hand_fk_data_, set_q);
    }

    // set joint position
    void getFKOri(const std::string &frameName, raisim::Mat<3, 3> &orientation_W) const override {
        orientation_W.e() = hand_fk_data_->oMf[hand_fk_model_->getBodyId(frameName)].rotation();
    }

    void getFKPos(const std::string &frameName, raisim::Vec<3> &point_W) const override {
        point_W.e() = hand_fk_data_->oMf[hand_fk_model_->getBodyId(frameName)].translation();
    }

private:
  std::unique_ptr<pinocchio::Model> hand_fk_model_;
  std::unique_ptr<pinocchio::Data> hand_fk_data_;

  std::unique_ptr<pinocchio::Model> arm_ik_model_;
  std::unique_ptr<pinocchio::Data> arm_ik_data_;
  std::unique_ptr<pinocchio::GeometryModel> arm_ik_geom_model_;
  std::unique_ptr<pinocchio::GeometryData> arm_ik_geom_data_;

  bool flying_mode_ = false;

  const double eps =  0.005;  // desired position precision  0.01m 3degree
  const int IT_MAX = 1000;  // maximum number of iterations 
  const double DT = 0.1; // convergence rate (smaller may have higher resolution?)
  const double damp = 1e-6; // damping factor for the pseudoinversion
  const int TMO_MAX = 30; // maximum number of the cnt of over iterations
};

extern "C" std::unique_ptr<HardwareKinematic> createPinocchio() {
    return std::make_unique<Pinocchio>();
}
