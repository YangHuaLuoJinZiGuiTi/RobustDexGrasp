#include "../hardwareHand.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

class AllegroSim : public HardwareHand {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        wrist_pose_.setZero(6);
        wrist_velocity_.setZero(6);
        hand_joint_position_.setZero(num_joint_);
        hand_joint_velocity_.setZero(num_joint_);
        
        flying_hand_mode_ = cfg["flying_hand_mode"].As<bool>();
    }
    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
    }

    void updateHandState(const Eigen::VectorXd &eef_pos) final override {
        wrist_pose_ = eef_pos;
        Eigen::VectorXd gc(platform_->getGeneralizedCoordinateDim()), gv(platform_->getDOF());
        platform_->getState(gc, gv);
        hand_joint_position_ = gc.tail(num_joint_);
        hand_joint_velocity_ = gv.tail(num_joint_);

    }
    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) final override {
        platform_->setPdTarget(posTarget, velTarget);
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int tail_shift) const final override {
        pgain.tail(tail_shift).setConstant(Pgain);
        dgain.tail(tail_shift).setConstant(Dgain);
    }

    Eigen::VectorXd & getJointVelocity() final override {
        return hand_joint_velocity_;
    }
    Eigen::VectorXd & getJointPosition() final override {
        return hand_joint_position_;
    }
    const int getDim() const final override {
        return num_joint_;
    }
    const int getNumFinger() const final override {
        return num_finger_;
    }
    
    int getBodies(std::vector<std::string> & get_vec, bool contact_flag) const final override {
        if (contact_flag) {
            for (int i = 0; i < num_contacts_; i++) {
                get_vec.push_back(contact_bodies_[i]);
            }
            return num_contacts_;
        } else {
            for (int i = 0; i < num_bodies_; i++) {
                if (flying_hand_mode_) {
                    get_vec.push_back(body_parts_flying_[i]);
                } else {
                    get_vec.push_back(body_parts_[i]);
                }
            }
            return num_bodies_;
        }
    }

    std::string changeLinkToJointName(std::string frameName) const final override {
        if (!frameName.compare("Flange_base_link")) {
            return std::string("Flange2hand_fixed_joint");
        } else if (!frameName.compare("link_3.0_tip")) {
            return std::string("joint_3.0_tip");
        } else if (!frameName.compare("link_7.0_tip")) {
            return std::string("joint_7.0_tip");
        } else if (!frameName.compare("link_11.0_tip")) {
            return std::string("joint_11.0_tip");
        } else if (!frameName.compare("link_15.0_tip")) {
            return std::string("joint_15.0_tip");
        } else {
            return frameName;
        }
    }

private:
    raisim::ArticulatedSystem *platform_;

    bool flying_hand_mode_ = false;

    const static int num_contacts_ = 13;
    const static int num_bodies_ = 17;
    const static int num_finger_ = 4;
    const static int num_joint_ = 16;

    const double Pgain = 60.0;
    const double Dgain = 0.2;

    const std::string body_parts_flying_[num_bodies_] =  {"z_rotation_joint",
    "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
    "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
    "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
    "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"};

    const std::string body_parts_[num_bodies_] =  {"Flange_base_link",
    "joint_1.0", "joint_2.0", "joint_3.0", "link_3.0_tip",
    "joint_5.0", "joint_6.0", "joint_7.0", "link_7.0_tip",
    "joint_9.0", "joint_10.0", "joint_11.0", "link_11.0_tip",
    "joint_13.0", "joint_14.0", "joint_15.0", "link_15.0_tip"};

    // for raisim contact check
    const std::string contact_bodies_[num_contacts_] =  {"wrist_3_link",
    "link_1.0", "link_2.0", "link_3.0",
    "link_5.0", "link_6.0", "link_7.0",
    "link_9.0", "link_10.0", "link_11.0",
    "link_13.0", "link_14.0", "link_15.0"};
};

extern "C" std::unique_ptr<HardwareHand> createAllegroSim() {
    return std::make_unique<AllegroSim>();
}
