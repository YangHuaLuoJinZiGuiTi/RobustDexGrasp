#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include<iostream>
#include <fstream>
#include <string>
#include "Kinematics.hpp"


#define MAX_ENDPOINT_ERR    0.001
#define MAX_MOVE_TIMEOUT_STEP    500000
#define WAIT_TIME_AFTER_ARRIVE    3000
#define MAX_JOINT_SETP_POSITION     0.1  // 0.0001
#define MAX_JOINT_SETP_VELOCITY     2.0


#define RAD2DEG(n) ((n)*180.0/EIGEN_PI)
#define DEG2RAD(n) ((n)*EIGEN_PI/180.0)

#define PRINT_VEC(str, vec, a, b) \
    { \
    std::cout << str; \
    for (int i = a; i < b; i++) std::cout << (vec) << ", "; \
    std::cout << std::endl; \
    }
void print_vec(const char* str, std::vector<double> vec, int start, int end)
{
    std::cout << str ;
    for (int i = start; i < end; i++) std::cout << vec[i] << ", ";
    std::cout << std::endl;
}

void print_vec(const char* str, Eigen::VectorXd vec, int start, int end)
{
    std::cout << str ;
    for (int i = start; i < end; i++) std::cout << vec[i] << ", ";
    std::cout << std::endl;
}

void T2Eulert(std::vector<double>& vec, Eigen::Vector3d& ang, Eigen::Vector3d& t)
{
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; i++)
    {
        Eigen::Vector3d v;
        for (int j = 0; j < 4; j++)
        {
            if (j < 3) v(j) = vec[4*i+ j];
            else t(i) = vec[4*i+ j];
        }
        R.row(i) = v;
    }
    ang = R.eulerAngles(0,1,2);
}

void T2Quatt(std::vector<double>& vec, Eigen::Vector4d& q, Eigen::Vector3d& t)
{
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; i++)
    {
        Eigen::Vector3d v;
        for (int j = 0; j < 4; j++)
        {
            if (j < 3) v(j) = vec[4*i+ j];
            else t(i) = vec[4*i+ j];
        }
        R.row(i) = v;
    }
    Eigen::Quaterniond q_(R);
    q[0] = q_.w();
    q[1] = q_.x();
    q[2] = q_.y();
    q[3] = q_.z();
}

void Eulert2Quat(double roll, double pitch, double yaw, std::vector<double>& vec)
{
    Eigen::Quaterniond q =  Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    vec[0] = q.w();
    vec[1] = q.x();
    vec[2] = q.y();
    vec[3] = q.z();
}

void target_switch(Eigen::VectorXd& cur, std::vector<double>& tar, Eigen::VectorXd& out)
{
    double loss = -1.0;
    int tar_i = 0;

    if (0 == tar.size())
    {
        std::cout << "IK ERROR" << std::endl;
        return;
    }

    for (int i = 0; i < tar.size() / 6; i++)
    {
        double loss_tmp = 0.0;
        for (int j = 0; j < 6; j++)
        {
            loss_tmp += abs(cur[j] - tar[6 * i + j]);
        }
        if (loss < 0.0 || loss_tmp < loss)
        {
            loss = loss_tmp;
            tar_i = i;
        }
    }

    for (int j = 0; j < 6; j++)
    {
        out[j] = tar[6 * tar_i + j];
    }
}

inline void sleep_ms(raisim::RaisimServer& srv, int ms)
{
    for (int i = 0; i < ms*20; i++) srv.integrateWorldThreadSafe();
}

static auto last_time = clock();
static int cnt = 0;
void step_log(raisim::RaisimServer& srv, raisim::World& world)
{
    cnt++;
    if (cnt % 200000 == 0)
    {
        auto now = clock();
        std::cout << "cost time = " << (now - last_time) / 200000 << "us for each step" << std::endl ;
        last_time = now;
    }
    world.integrate();
    srv.integrateWorldThreadSafe();
}

// tp = target position, cp = current position, np = next position
// tv = target velocity, cv = current velocity, nv = next velocity
// return if arrive the target position?
bool calculate_tar_joint_pos(Eigen::VectorXd &tp, Eigen::VectorXd& cp, Eigen::VectorXd& np, Eigen::VectorXd &tv, Eigen::VectorXd& cv, Eigen::VectorXd& nv)
{
    double diff_sum = 0.0;
    for (int i = 0; i < 6; i++) {
        double diff_p = tp[i] - cp[i];

        // when joint limit, do not check the 2 PI
        //if (diff_p > EIGEN_PI) diff_p -= (2*EIGEN_PI);
        //else if (diff_p < -EIGEN_PI) diff_p += (2*EIGEN_PI);
        diff_sum += abs(diff_p);

        if (diff_p > MAX_ENDPOINT_ERR) {
            np[i] = cp[i] + MAX_JOINT_SETP_POSITION;
        } else if (diff_p < MAX_ENDPOINT_ERR) {
            np[i] = cp[i] - MAX_JOINT_SETP_POSITION;
        } else {
            np[i] = cp[i];
        }
    }

    if (diff_sum < MAX_ENDPOINT_ERR * 10) return true;
    else return false;
}

int main(int argc, char* argv[]) {
    Kinematics ur5;

    // create random number
    std::uniform_real_distribution<double> rand_pry(-EIGEN_PI, EIGEN_PI);
    std::uniform_real_distribution<double> rand_xy(-0.8, 0.8);
    std::uniform_real_distribution<double> rand_z(0.4, 0.8);    // rand_z(0.4, 0.8)→(1.051, 1.451), ly test: add the height of UR5 platform z = 0.651m
    std::default_random_engine e(time(NULL));

    raisim::World::setActivationKey("/home/wzj/.raisim/activation.raisim");
    raisim::RaiSimMsg::setFatalCallback([](){throw;});

    /// create raisim world
    raisim::World world;
    world.setTimeStep(0.001);
    world.setGravity(raisim::Vec<3>{0,0,0});

    /// create objects
    auto ground = world.addGround(0, "gnd");
    ground->setAppearance("hidden");
    auto robot = world.addArticulatedSystem("/home/leech/raisim/dgrasp_base/rsc/ur5_allegro_dgrasp/ur5_dgrasp.urdf","",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(63)); 
    //auto ball = world.addSphere(0.1, 0);

    // set robot state
    Eigen::VectorXd jointPgain(robot->getDOF()), jointDgain(robot->getDOF());
    jointPgain.head(6).setConstant(1000.0); jointDgain.head(6).setConstant(150);
    jointPgain.tail(robot->getDOF()-6).setConstant(50.0); jointDgain.tail(robot->getDOF()-6).setConstant(0.2);
    Eigen::VectorXd gc_(robot->getGeneralizedCoordinateDim()), gv_(robot->getDOF());
    Eigen::VectorXd sc_(robot->getGeneralizedCoordinateDim()), sv_(robot->getDOF());
    Eigen::VectorXd tar_sc_(robot->getGeneralizedCoordinateDim()), tar_sv_(robot->getDOF());
    gc_.setZero();gv_.setZero();sc_.setZero();sv_.setZero();tar_sc_.setZero();tar_sv_.setZero();
    robot->setGeneralizedCoordinate(sc_);
    robot->setGeneralizedForce(Eigen::VectorXd::Zero(robot->getDOF()));
    robot->setPdGains(jointPgain, jointDgain);
    robot->setPdTarget(sc_, sv_);
    robot->setName("robot_ur5");

    /// launch raisim server
    raisim::RaisimServer server(&world);
    server.setMap("wheat");
    server.launchServer();
    server.focusOn(robot);
    auto ball = server.addVisualSphere("v_sphere", 0.05, 255, 255, 255, 0);
    while (!server.isConnected()) RS_TIMED_LOOP(int(world.getTimeStep()*1e6));

    /// set graphs
    std::vector<std::string> jointNames = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
    auto jcGraph = server.addTimeSeriesGraph("joint position", jointNames, "time", "position");
    auto jvGraph = server.addTimeSeriesGraph("joint velocity", jointNames, "time", "velocity");
    auto jfGraph = server.addTimeSeriesGraph("joint torque", jointNames, "time", "torque");

    std::cout << "INIT ALL OK" << std::endl ;


    int switch_flag = 0;
    while (1) {
        RS_TIMED_LOOP(int(world.getTimeStep()*1e6))

//#define USE_FK
#ifdef USE_FK
        // target position change
        switch (switch_flag)
        {
            case 0: tar_sc_ << DEG2RAD(88.0), DEG2RAD(-20.0),DEG2RAD(5.0),DEG2RAD(1.0),DEG2RAD(1.0),DEG2RAD(5.0); switch_flag = 1; std::cout << "will switch to flag 1" << std::endl; break;
            case 1: tar_sc_ << DEG2RAD(0.0), DEG2RAD(-96.0),DEG2RAD(3.0),DEG2RAD(2.0),DEG2RAD(1.0),DEG2RAD(5.0); switch_flag = 2; std::cout << "will switch to flag 2" << std::endl; break;
            case 2: tar_sc_ << DEG2RAD(12.0), DEG2RAD(-170.0),DEG2RAD(2.0),DEG2RAD(10.0),DEG2RAD(20.0),DEG2RAD(-6.0); switch_flag = 3; std::cout << "will switch to flag 3" << std::endl; break;
            case 3: tar_sc_ << DEG2RAD(20.0), DEG2RAD(-60.0),DEG2RAD(-20.0),DEG2RAD(10.0),DEG2RAD(-4.0),DEG2RAD(1.0); switch_flag = 4; std::cout << "will switch to flag 4" << std::endl; break;
            case 4: tar_sc_ << DEG2RAD(-38.0), DEG2RAD(-120.0),DEG2RAD(-10.0),DEG2RAD(-20.0),DEG2RAD(30.0),DEG2RAD(-89.0); switch_flag = 5; std::cout << "will switch to flag 5" << std::endl; break;
            case 5: tar_sc_ << DEG2RAD(-3.0), DEG2RAD(-15.0),DEG2RAD(5.0),DEG2RAD(8.0),DEG2RAD(2.0),DEG2RAD(1.0); switch_flag = 0; std::cout << "will switch to flag 0" << std::endl; break;
        }
    
        // get target end-point position and move ball to it
        std::vector<double> ret = ur5.forward(tar_sc_); // 3X4 matrix
        Eigen::Vector3d t, eulerAngle;
        T2Eulert(ret, eulerAngle, t);
        ball->setPosition(t[0], t[1], t[2]);
        step_log(server, world);
        std::cout << "FK:\t" << RAD2DEG(eulerAngle.transpose()) << ", " << t.transpose() << std::endl;
        sleep_ms(server, 500);
#else
        std::vector<double> ret(7);
        Eulert2Quat(rand_pry(e), rand_pry(e), rand_pry(e), ret);
        ret[4]=rand_xy(e);ret[5]=rand_xy(e);ret[6]=rand_z(e);
        ball->setPosition(ret[4], ret[5], ret[6]);
        step_log(server, world);
        print_vec("rand target:\t", ret, 0, 7);
        sleep_ms(server, 500);

#endif

        // set target end-point position
        std::vector<double> IK_joint = ur5.inverse(ret); 
        if (IK_joint.size() == 0) {
            std::cout << "!!!!! IK ERROR !!!!!" << std::endl;
            // sleep_ms(server, WAIT_TIME_AFTER_ARRIVE);
            continue;
        }
        robot->getState(gc_, gv_);
        target_switch(gc_, IK_joint, tar_sc_);
        print_vec("IK current joint POS:\t", gc_, 0, 6);
        print_vec("IK target joint POS:\t", tar_sc_, 0, 6);

        // move UR5 to the target endpoint
        int wait_cnt = 0;
        while (1) { 
            robot->getState(gc_, gv_);
            bool ret = calculate_tar_joint_pos(tar_sc_, gc_, sc_, tar_sv_, gv_, sv_);
            if (ret) {
                std::cout << "***** achieve success *****" << std::endl;
                sleep_ms(server, WAIT_TIME_AFTER_ARRIVE);
                break;
            }
            if (wait_cnt++ > MAX_MOVE_TIMEOUT_STEP) {
                std::cout << "!!!!! timeout !!!!!:\t";
                for (int i = 0; i < 6; i++) std::cout << abs(tar_sc_[i] - gc_[i]) << ", ";
                std::cout << std::endl;
                sleep_ms(server, WAIT_TIME_AFTER_ARRIVE);
                break;
            }
            std::cout << "***** start setPdTarget *****" << std::endl;
            // // ly test: hand joint can run normally or not
            // sc_(9) = 1.57;
            // sc_(15) = 1.57;

            robot->setPdTarget(sc_, sv_);
            step_log(server, world);
        }
    }

    server.killServer();
}
