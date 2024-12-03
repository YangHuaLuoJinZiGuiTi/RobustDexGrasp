#include "../hardwareKinematic.hpp"

// cpp library
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <time.h>

class SimFK : public HardwareKinematic {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) override {
    }

    void setFrameVelocityNames(std::vector<std::string> &names) override {
    }

    IK_ERRCODE getArmIKSolve(const Eigen::VectorXd eef, const Eigen::VectorXd current_q, Eigen::VectorXd &solved_q) const override {
    }

    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }

    void updateHandFK(const Eigen::VectorXd &hand_q, const Eigen::VectorXd &eef_pos) const override {
    }

    void updateURDFFK(const Eigen::VectorXd &joint) override {
    }

    void getFrameOrientation(const std::string &frameName, raisim::Mat<3, 3> &orientation_W) final override {
        platform_->getFrameOrientation(frameName, orientation_W);
    }
    void getFramePosition(const std::string &frameName, raisim::Vec<3> &point_W) final override {
        platform_->getFramePosition(frameName, point_W);
    }
    void getFrameAngularVelocity(const std::string &frameName, raisim::Vec<3> &angVel_W) final override {
        platform_->getFrameAngularVelocity(frameName, angVel_W);
    }
    void getFrameVelocity(const std::string &frameName, raisim::Vec<3> &vel_W) final override {
        platform_->getFrameVelocity(frameName, vel_W);
    }

private:
    raisim::ArticulatedSystem *platform_;
};

extern "C" std::unique_ptr<HardwareKinematic> createSimFK() {
    return std::make_unique<SimFK>();
}
