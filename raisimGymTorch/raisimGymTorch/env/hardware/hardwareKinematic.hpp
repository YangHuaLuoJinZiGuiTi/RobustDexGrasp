#ifndef HARDWARE_KINEMATIC_HPP
#define HARDWARE_KINEMATIC_HPP

#include "Yaml.hpp"

class HardwareKinematic {
public:
    typedef enum ik_err_code {
        IK_OK = 0,
        IK_FAIL = 1,
        IK_TIMEOUT = 2,
        IK_SELF_COLLISION = 3
    } IK_ERRCODE;

    virtual void init(const std::string &rsc_pth, const Yaml::Node &cfg) = 0;
    virtual IK_ERRCODE getArmIKSolve(const Eigen::VectorXd eef, const Eigen::VectorXd current_q, Eigen::VectorXd &solved_q) const = 0;
    virtual void updateHandFK(const Eigen::VectorXd &hand_q, const Eigen::VectorXd &eef_pos) const = 0;
    virtual void getFKOri(const std::string &frameName, raisim::Mat<3, 3> &orientation_W) const = 0;
    virtual void getFKPos(const std::string &frameName, raisim::Vec<3> &point_W) const = 0;
};


#endif //HARDWARE_KINEMATIC_HPP
