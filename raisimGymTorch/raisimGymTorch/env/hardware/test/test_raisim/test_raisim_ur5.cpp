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

#define RAD2DEG(n) ((n)*180.0/EIGEN_PI)
#define DEG2RAD(n) ((n)*EIGEN_PI/180.0)

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

        if(server_) server_->lockVisualizationServerMutex();
        world_.integrate();
        if(server_) server_->unlockVisualizationServerMutex();
    }
    csv_file_.flush();
}

void test()
{
    Eigen::VectorXd set_pos = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd init_pos = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd end_pos = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd gv_r_ = Eigen::VectorXd::Zero(22);
    init_pos << DEG2RAD(-10.0), DEG2RAD(-100.0),  DEG2RAD(80.0),  DEG2RAD(-10.0), DEG2RAD(80.0),   DEG2RAD(-100.0), 0.3, 1.5, 0.3, 0.3, 0.3, 1.5, 0.3, 0.3, 0.3, 1.5, 0.3, 0.3, 1.57, 0, -0.1, 0.3;
    set_pos = init_pos;
    end_pos << DEG2RAD(10.0),  DEG2RAD(-80.0),  DEG2RAD(100.0), DEG2RAD(10.0),  DEG2RAD(100.0),  DEG2RAD(-80.0), 0.3, 1.5, 0.3, 0.3, 0.3, 1.5, 0.3, 0.3, 0.3, 1.5, 0.3, 0.3, 1.57, 0, -0.1, 0.3;
    robot->setState(init_pos, gv_r_);

    double step_length_rad = 0.01;
    double step_dt_s = 0.2;
    int step_len = int((abs(end_pos[0] - init_pos[0]))/step_length_rad);

    for (int j = 1; j <= step_len; j++) {
        for (int k = 0; k < 6; k++) {
            set_pos[k] = init_pos[k] + j * step_length_rad;
        }
        log_test(set_pos, step_dt_s);
    }
    for (int j = 1; j <= step_len; j++) {
        for (int k = 0; k < 6; k++) {
            set_pos[k] = end_pos[k] - j * step_length_rad;
        }
        log_test(set_pos, step_dt_s);
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
    robot = world_.addArticulatedSystem("/home/ubuntu/hand/github/vision_dex/rsc/ur5_allegro/ur5_allegro_ori.urdf","",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(63)); 

    Eigen::VectorXd pgain = Eigen::VectorXd::Zero(22);
    Eigen::VectorXd dgain = Eigen::VectorXd::Zero(22);
    pgain << 16000, 16000, 16000, 16000, 16000, 16000, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 250;
    dgain << 600, 600, 600, 600, 600, 600, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30;
    robot->setPdGains(pgain, dgain);

    test();
}
