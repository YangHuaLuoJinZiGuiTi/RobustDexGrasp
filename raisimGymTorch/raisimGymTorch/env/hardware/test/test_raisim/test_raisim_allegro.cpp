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

void log_test(Eigen::VectorXd &gc, double delay_s) {
    Eigen::VectorXd get_pos = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd get_vel = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd gv = Eigen::VectorXd::Zero(22);

    robot->setPdTarget(gc, gv);
    int step_cnt = 0;
    while (step_cnt < int(delay_s / simulation_dt_ + 1e-10)) {
        step_cnt++;

        robot->getState(get_pos, get_vel);
        for (int i = 0; i < 22; i++) {
            csv_file_ << gc[i] << "," << get_pos[i] << ",,";
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
    set_pos << -0.0, -0.0, 0.0, -0.0, 0.0, 1.57, 0.3, 1.5, 0.3, 0.3, 0.3, 1.5, 0.3, 0.3, 0.3, 1.5, 0.3, 0.3, 1.57, 0, -0.1, 0.3;
    robot->setState(set_pos, gv_r_);

    for (int j = 1; j <= 16; j++) {
        set_pos[1+6] = 1.5 - j * 0.1;
        set_pos[5+6] = 1.5 - j * 0.1;
        set_pos[9+6] = 1.5 - j * 0.1;
        log_test(set_pos, 0.2);
    }
    for (int j = 1; j <= 16; j++) {
        set_pos[1+6] = -0.1 + j * 0.1;
        set_pos[5+6] = -0.1 + j * 0.1;
        set_pos[9+6] = -0.1 + j * 0.1;
        log_test(set_pos, 0.2);
    }
}


int main(int argc, char ** argv)
{
    csv_file_.open("./log.csv", std::ios::out);
    for (int i = 0; i < 22; i++) {
        csv_file_ << i << "_tar========," << i << "_cur,,";
    }
    csv_file_ << "\n";

    /// create raisim world_
    world_.setTimeStep(simulation_dt_);
    world_.setGravity(raisim::Vec<3>{0,0,0});

    server_ = std::make_unique<raisim::RaisimServer>(&world_);
    server_->launchServer();

    /// create objects
    auto ground = world_.addGround(0, "gnd");
    robot = world_.addArticulatedSystem("/home/ubuntu/hand/github/vision_dex/rsc/ur5_allegro/ur5_allegro.urdf","",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(63)); 

    Eigen::VectorXd pgain = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd dgain = Eigen::VectorXd::Zero(22);
    pgain << 16000, 16000, 16000, 16000, 16000, 16000, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 250;
    dgain << 800, 800, 800, 800, 800, 800, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30;
    robot->setPdGains(pgain, dgain);

    test();
}
