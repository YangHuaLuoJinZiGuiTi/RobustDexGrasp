// raisim library
#include "raisim/World.hpp"
#include "raisim/math.hpp"
#include "raisim/RaisimServer.hpp"

// cpp library
#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

#include <stack>

std::ofstream csv_file_;
std::unique_ptr<raisim::RaisimServer> server_;
raisim::World world_;
raisim::ArticulatedSystem* robot;

double simulation_dt_ = 0.005;

static const int DOF_JOINTS = 16;
double min_limit[DOF_JOINTS] = {
    -0.47, -0.196, -0.174, -0.227, 
    -0.47, -0.196, -0.174, -0.227, 
    -0.47, -0.196, -0.174, -0.227, 
    0.263, -0.105, -0.189, -0.162
};
double max_limit[DOF_JOINTS] = {
    0.47, 1.61, 1.709, 1.618, 
    0.47, 1.61, 1.709, 1.618, 
    0.47, 1.61, 1.709, 1.618, 
    1.396, 1.163, 1.644, 1.719
};

void log_test(Eigen::VectorXd &gc, double delay_s) {
    Eigen::VectorXd get_pos = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd get_vel = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd gv = Eigen::VectorXd::Zero(22);

    robot->setPdTarget(gc, gv);
    int step_cnt = 0;
    while (step_cnt < int(delay_s / simulation_dt_ + 1e-10)) {
        step_cnt++;

        robot->getState(get_pos, get_vel);
        raisim::VecDyn tor = robot->getGeneralizedForce();
        for (int i = 0; i < 22; i++) {
            csv_file_ << gc[i] << "," << get_pos[i] << "," << tor[i] << ",,";
        }
        csv_file_ << "\n";
        csv_file_.flush();

        if(server_) server_->lockVisualizationServerMutex();
        world_.integrate();
        if(server_) server_->unlockVisualizationServerMutex();
    }
    
}

void test()
{
    Eigen::VectorXd set_pos = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd gv_r_ = Eigen::VectorXd::Zero(22);
    set_pos << 0.,  -1.57,  1.57,  0.,  1.57,  -1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0;
    robot->setState(set_pos, gv_r_);

    usleep(3000000);
    while (1) {
        while (1) {
            set_pos[14+6] -= 0.1;
            if (set_pos[14+6] < min_limit[14]) {
                set_pos[14+6] = min_limit[14];
                log_test(set_pos, 0.5);
                break;
            }
            log_test(set_pos, 0.5);
        }

        while (1) {
            set_pos[14+6] += 0.1;
            if (set_pos[14+6] > max_limit[14]) {
                set_pos[14+6] = max_limit[14];
                log_test(set_pos, 0.5);
                break;
            }
            log_test(set_pos, 0.5);
        }
    }

}


int main(int argc, char ** argv)
{
    csv_file_.open("./log.csv", std::ios::out);
    for (int i = 0; i < 22; i++) {
        csv_file_ << i << "_tar,cur,torque,,";
    }
    csv_file_ << "\n";

    /// create raisim world_
    world_.setTimeStep(simulation_dt_);
    world_.setGravity(raisim::Vec<3>{0,0,9.8});

    server_ = std::make_unique<raisim::RaisimServer>(&world_);
    server_->launchServer();

    /// create objects
    auto ground = world_.addGround(0, "gnd");
    robot = world_.addArticulatedSystem("/home/ubuntu/hand/github/vision_dex/rsc/ur5_allegro/ur5_allegro_ori.urdf","",{},raisim::COLLISION(0),raisim::COLLISION(63)); 

    Eigen::VectorXd pgain = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd dgain = Eigen::VectorXd::Zero(22);
    pgain << 16000, 16000, 16000, 16000, 16000, 16000, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5;
    dgain << 800, 800, 800, 800, 800, 800, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2;
    robot->setPdGains(pgain, dgain);

    test();
}
