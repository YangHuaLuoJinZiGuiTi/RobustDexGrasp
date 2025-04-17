#include "../hardwareHand.hpp"

// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"

class LeapSim : public HardwareHand {
public:
    void init(const std::string &rsc_pth, const Yaml::Node &cfg) final override {
        wrist_pose_.setZero(6);
        wrist_velocity_.setZero(6);
        hand_joint_position_.setZero(num_joint_);
        hand_joint_velocity_.setZero(num_joint_);
        
        flying_hand_mode_ = cfg["flying_hand_mode"].As<bool>();
        if (!cfg["randomize_gc_hand"].IsNone()) {
            randomize_gc_ = cfg["randomize_gc_hand"].As<double>();
        }
        std::string pd_file = cfg["hand_pd_file"].As<std::string>();

        std::ifstream pd_txt;
        pd_txt.open(rsc_pth+"/../raisimGymTorch/raisimGymTorch/env/hardware/hand/"+pd_file);
        if (pd_txt) {
            std::string line;
            int line_cnt = 0;
            while (getline(pd_txt, line)) {
                std::stringstream ss(line); 
                if (line_cnt < num_joint_) {
                    ss >> Pgain[line_cnt];
                } else {
                    ss >> Dgain[line_cnt - num_joint_];
                }
                line_cnt++;
            }
            if (line_cnt != (num_joint_*2)) {
                std::cout << "error txt line:" << line_cnt << std::endl;
                pd_txt.close();
                exit(0);
            }
            pd_txt.close();
        } else {
            for (int i = 0; i < num_joint_; i++) {
                Pgain[i] = Pgain[0];
                Dgain[i] = Dgain[0];
            }
        }

        srand(time(0));
    }
    void setSimPlatform(raisim::ArticulatedSystem *platform) final override {
        platform_ = platform;
        std::vector<raisim::Vec<2>> joint_limits = platform_->getJointLimits();
        joint_limit_high_.setZero(num_joint_); joint_limit_low_.setZero(num_joint_);
        int all_joint_num = joint_limits.size();
        for(int i = 0; i < num_joint_; i++){
            joint_limit_low_[i] = joint_limits[all_joint_num - num_joint_ + i][0];
            joint_limit_high_[i] = joint_limits[all_joint_num - num_joint_ + i][1];
        }
    }

    void updateHandState(const Eigen::VectorXd &eef_pos) final override {
        wrist_pose_ = eef_pos;
        Eigen::VectorXd gc(platform_->getGeneralizedCoordinateDim()), gv(platform_->getDOF());
        platform_->getState(gc, gv);
        hand_joint_position_ = gc.tail(num_joint_);
        hand_joint_velocity_ = gv.tail(num_joint_);
        hand_joint_effort_ = platform_->getGeneralizedForce().e().tail(num_joint_);
        if (randomize_gc_ > 1e-9) {
            hand_joint_position_ += Eigen::VectorXd::Random(num_joint_) * randomize_gc_;
            hand_joint_position_ = hand_joint_position_.cwiseMax(joint_limit_low_).cwiseMin(joint_limit_high_);
        }

    }
    void setPdTarget(const Eigen::VectorXd &posTarget, const Eigen::VectorXd &velTarget, bool async = true) final override {
        platform_->setPdTarget(posTarget, velTarget);
    }

    void getPdgains(Eigen::VectorXd &pgain, Eigen::VectorXd &dgain, int tail_shift) const final override {
        for (int i = 0; i < num_joint_; i++) {
            pgain.tail(tail_shift)[i] = Pgain[i];
            dgain.tail(tail_shift)[i] = Dgain[i];
        }
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

    std::string changeJointToLinkName(std::string frameName) const final override {
        if (!frameName.compare("tool02Flange_fixed_joint")) {
            return std::string("tool0");
        } else if (!frameName.compare("leap_joint3_tip")) {
            return std::string("leap_link3_tip");
        } else if (!frameName.compare("leap_joint7_tip")) {
            return std::string("leap_link7_tip");
        } else if (!frameName.compare("leap_joint11_tip")) {
            return std::string("leap_link11_tip");
        } else if (!frameName.compare("leap_joint15_tip")) {
            return std::string("leap_link15_tip");
        } else {
            return frameName;
        }
    }

private:
    raisim::ArticulatedSystem *platform_;

    bool flying_hand_mode_ = false;
    double randomize_gc_ = 0.0;
    Eigen::VectorXd joint_limit_high_;
    Eigen::VectorXd joint_limit_low_;

    const static int num_contacts_ = 13;
    const static int num_bodies_ = 17;
    const static int num_finger_ = 4;
    const static int num_joint_ = 16;

    double Pgain[num_joint_] = {60.0};
    double Dgain[num_joint_] = {0.2};

    const std::string body_parts_flying_[num_bodies_] = {"z_rotation_joint",
    "leap_joint1", "leap_joint2", "leap_joint3", "leap_joint3_tip",
    "leap_joint5", "leap_joint6", "leap_joint7", "leap_joint7_tip",
    "leap_joint9", "leap_joint10", "leap_joint11", "leap_joint11_tip",
    "leap_joint13", "leap_joint14", "leap_joint15", "leap_joint15_tip"};

    const std::string body_parts_[num_bodies_] =  {"tool02Flange_fixed_joint",
    "leap_joint0", "leap_joint2", "leap_joint3", "leap_joint3_tip",
    "leap_joint4", "leap_joint6", "leap_joint7", "leap_joint7_tip",
    "leap_joint8", "leap_joint10", "leap_joint11", "leap_joint11_tip",
    "leap_joint13", "leap_joint14", "leap_joint15", "leap_joint15_tip"};

    // for raisim contact check
    const std::string contact_bodies_[num_contacts_] =   {"wrist_3_link",
    "pip", "dip", "fingertip",
    "pip_2", "dip_2", "fingertip_2",
    "pip_3", "dip_3", "fingertip_3",
    "thumb_pip", "thumb_dip", "thumb_fingertip"};
};

extern "C" std::unique_ptr<HardwareHand> createLeapSim() {
    return std::make_unique<LeapSim>();
}
