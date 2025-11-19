//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "RaisimGymEnv.hpp"
#include "omp.h"
#include "Yaml.hpp"

namespace raisim {

int THREAD_COUNT;

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg)
      : resourceDir_(resourceDir), cfgString_(cfg) {
    Yaml::Parse(cfg_, cfg);
	raisim::World::setActivationKey(raisim::Path(resourceDir + "/activation.raisim").getString());
    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
    init();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  const std::string& getResourceDir() const { return resourceDir_; }
  const std::string& getCfgString() const { return cfgString_; }

  void init() {
//    omp_set_num_threads(cfg_["num_threads"].template As<int>());
    THREAD_COUNT = cfg_["num_threads"].template As<int>();
    omp_set_num_threads(THREAD_COUNT);
    num_envs_ = cfg_["num_envs"].template As<int>();
    forceClosureCosts_.resize(num_envs_);
    frictionConeCosts_.resize(num_envs_);
    forceClosureCosts_.setZero();
    frictionConeCosts_.setZero();
    environments_.reserve(num_envs_);
    rewardInformation_r_.reserve(num_envs_);
    rewardInformation_r_.reserve(num_envs_);
    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0)); //visualize given env (可视化)
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
      rewardInformation_r_.push_back(environments_.back()->getRewardsRight().getStdMap());
      rewardInformation_l_.push_back(environments_.back()->getRewardsLeft().getStdMap());
    }

    setSeed(0);

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->reset();
    }
    obDim_r_ = environments_[0]->getRightObDim();
    obDim_l_ = environments_[0]->getLeftObDim();
    actionDim_ = environments_[0]->getActionDim();
    gsDim_ = environments_[0]->getGSDim();
    RSFATAL_IF(obDim_r_ == 0 || obDim_l_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")
  }

  // resets all environments and returns observation
  void reset() {
    for (auto env: environments_)
      env->reset();
  }
  void getForceClosureCosts(Eigen::Ref<EigenVec> costs) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++) {
            costs[i] = environments_[i]->getForceClosureCost();
        }
    }
    
  void getFrictionConeCosts(Eigen::Ref<EigenVec> costs) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++) {
            costs[i] = environments_[i]->getFrictionConeCost();
        }
    }
  void set_pd_wrist() {
    for (auto env: environments_)
      env->set_pd_wrist();
  }

  void switch_arctic(int idx){
      for (auto env: environments_)
          env->switch_arctic(idx);
  }

  void add_stage(Eigen::Ref<EigenRowMajorMat> stage_dim,
                 Eigen::Ref<EigenRowMajorMat> stage_pos) {
#pragma omp parallel for
      for (int i = 0; i < num_envs_; i++)
          environments_[i]->add_stage(stage_dim.row(i), stage_pos.row(i));
  }
    void load_object(Eigen::Ref<EigenRowMajorMatInt> &obj_idx, Eigen::Ref<EigenRowMajorMat> &obj_weight, Eigen::Ref<EigenRowMajorMat>& obj_dim, Eigen::Ref<EigenRowMajorMatInt>& obj_type) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->load_object(obj_idx.row(i), obj_weight.row(i), obj_dim.row(i), obj_type.row(i));
    }

    void load_articulated(const std::string& obj_model) {
  #pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->load_articulated(obj_model);
    }

    void load_multi_articulated(const std::vector<std::string>& obj_models) {
  #pragma omp parallel for
        for (int i = 0; i < num_envs_; i++){
            std::srand(time(NULL));
            environments_[i]->load_articulated(obj_models[i]);
        }
    }

  void reset_state(Eigen::Ref<EigenRowMajorMat> &init_state_r, 
                   Eigen::Ref<EigenRowMajorMat> &init_state_l, 
                   Eigen::Ref<EigenRowMajorMat> &init_vel_r, 
                   Eigen::Ref<EigenRowMajorMat> &init_vel_l, 
                   Eigen::Ref<EigenRowMajorMat> &obj_pose) {
#pragma omp parallel for
      for (int i = 0; i < num_envs_; i++)
          environments_[i]->reset_state(init_state_r.row(i), init_state_l.row(i), init_vel_r.row(i), init_vel_l.row(i), obj_pose.row(i));
  }

  void reset_state2(Eigen::Ref<EigenRowMajorMat> &init_state_r, 
                   Eigen::Ref<EigenRowMajorMat> &init_state_l, 
                   Eigen::Ref<EigenRowMajorMat> &init_vel_r, 
                   Eigen::Ref<EigenRowMajorMat> &init_vel_l, 
                   Eigen::Ref<EigenRowMajorMat> &obj_pose) {
#pragma omp parallel for
      for (int i = 0; i < num_envs_; i++)
          environments_[i]->reset_state2(init_state_r.row(i), init_state_l.row(i), init_vel_r.row(i), init_vel_l.row(i), obj_pose.row(i));
  }

  void final_reset_state(Eigen::Ref<EigenRowMajorMat> &init_state_r, 
                          bool release_hand,
                          bool sim_flag,
                          Eigen::Ref<EigenRowMajorMat>& set_arm,
                          bool no_wait) {
#pragma omp parallel for
      for (int i = 0; i < num_envs_; i++)
          environments_[i]->final_reset_state(init_state_r.row(i), release_hand, sim_flag, set_arm, no_wait);
  }

  bool get_no_wait_finish_flag() {
    return environments_[0]->get_no_wait_finish_flag();
  }

  void move_line(double x,double y,double z) {
    #pragma omp parallel for
      for (int i = 0; i < num_envs_; i++)
      environments_[i]->move_line(x,y,z);
    }

    void set_goals_r(Eigen::Ref<EigenRowMajorMat> &obj_pos_r, 
                     Eigen::Ref<EigenRowMajorMat> &ee_pos_r, 
                     Eigen::Ref<EigenRowMajorMat> &pose_r,
                     Eigen::Ref<EigenRowMajorMat> &qpos_r) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->set_goals_r(obj_pos_r.row(i), 
                                          ee_pos_r.row(i), 
                                          pose_r.row(i), 
                                          qpos_r.row(i)
                                          );
    }

  // Add this method to the VectorizedEnvironment class in VectorizedEnvironment.hpp
  // Place it with the other public methods like set_goals_r, etc.

  /**
   * Set random object properties (mass and friction) for all environments
   * @param mass_lower_limit Lower bound for object mass sampling
   * @param mass_upper_limit Upper bound for object mass sampling
   * @param mu_lower_limit Lower bound for friction coefficient sampling
   * @param mu_upper_limit Upper bound for friction coefficient sampling
   */
  void set_random_obj_prior_info(double random_mass,
                                double random_mu) {
  #pragma omp parallel for
      for (int i = 0; i < num_envs_; i++) {
          environments_[i]->set_random_obj_prior_info(
                  random_mass, random_mu
          );
      }
  }

    void set_goals_r2(Eigen::Ref<EigenRowMajorMat> &obj_pos_r,
                      Eigen::Ref<EigenRowMajorMat> &ee_pos_r,
                      Eigen::Ref<EigenRowMajorMat> &pose_r,
                      Eigen::Ref<EigenRowMajorMat> &qpos_r,
                      Eigen::Ref<EigenRowMajorMat> &contact_r) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->set_goals_r2(obj_pos_r.row(i), 
                                           ee_pos_r.row(i), 
                                           pose_r.row(i), 
                                           qpos_r.row(i),
                                           contact_r.row(i)
                                           );
    }

    void set_imitation_goals(Eigen::Ref<EigenRowMajorMat> &pose_l,
                      Eigen::Ref<EigenRowMajorMat> &pose_r,
                      Eigen::Ref<EigenRowMajorMat> &obj_pose) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->set_imitation_goals(pose_l.row(i),
                                           pose_r.row(i),
                                           obj_pose.row(i)
                                           );
    }

    void set_ext(Eigen::Ref<EigenRowMajorMat> &ext_force, 
                 Eigen::Ref<EigenRowMajorMat> &ext_torque) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->set_ext(ext_force.row(i), 
                                      ext_torque.row(i)    
                                      );
    }

    void set_pregrasp(Eigen::Ref<EigenRowMajorMat> &obj_pos, 
                      Eigen::Ref<EigenRowMajorMat> &ee_pos, 
                      Eigen::Ref<EigenRowMajorMat> &pose) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->set_pregrasp(obj_pos.row(i), 
                                           ee_pos.row(i), 
                                           pose.row(i)
                                           );
    }


    void update_target(Eigen::Ref<EigenRowMajorMat> &target_center) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->update_target(target_center.row(i));
    }

    void set_goals(Eigen::Ref<EigenRowMajorMat> &obj_angle,
                   Eigen::Ref<EigenRowMajorMat> &obj_pos,
                   Eigen::Ref<EigenRowMajorMat> &ee_pos_r,
                   Eigen::Ref<EigenRowMajorMat> &ee_pos_l,
                   Eigen::Ref<EigenRowMajorMat> &pose_r,
                   Eigen::Ref<EigenRowMajorMat> &pose_l,
                   Eigen::Ref<EigenRowMajorMat> &qpos_r,
                   Eigen::Ref<EigenRowMajorMat> &qpos_l,
                   Eigen::Ref<EigenRowMajorMat> &contact_r,
                   Eigen::Ref<EigenRowMajorMat> &contact_l) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->set_goals(obj_angle.row(i),
                                        obj_pos.row(i),
                                        ee_pos_r.row(i),
                                        ee_pos_l.row(i),
                                        pose_r.row(i),
                                        pose_l.row(i),
                                        qpos_r.row(i),
                                        qpos_l.row(i),
                                        contact_r.row(i),
                                        contact_l.row(i)
                                        );
    }

    void set_obj_goal(Eigen::Ref<EigenRowMajorMat> &obj_angle,
                      Eigen::Ref<EigenRowMajorMat> &obj_pos) {
#pragma omp parallel for
        for (int i = 0; i < num_envs_; i++)
            environments_[i]->set_obj_goal(obj_angle.row(i),
                                        obj_pos.row(i)
                                        );
    }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob_r,
               Eigen::Ref<EigenRowMajorMat> &ob_l) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob_r.row(i), ob_l.row(i));
  }

    void get_global_state(Eigen::Ref<EigenRowMajorMat> &gs) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++) 
      environments_[i]->get_global_state(gs.row(i));
  }

    void get_contact_info(Eigen::Ref<Eigen::MatrixXd> points,
                        Eigen::Ref<Eigen::MatrixXd> normals,
                        Eigen::Ref<Eigen::MatrixXd> forces,
                        Eigen::Ref<Eigen::MatrixXd> normal_forces,
                        Eigen::Ref<Eigen::MatrixXi> contact_ids,
                        Eigen::Ref<Eigen::MatrixXd> jacobians_flat,  // 扁平化雅可比
                        Eigen::Ref<Eigen::VectorXi> contact_counts
                        ) {  
        
        int dof = 22;
        int num_envs = num_envs_;
        int max_contacts = 13;
        int jacobian_size_per_contact = 3 * dof;  // 每个接触点的雅可比大小
        
        contact_counts.resize(num_envs);
        
        #pragma omp parallel for
        for (int i = 0; i < num_envs; i++) {
            std::vector<Eigen::Vector3d> env_points, env_normals, env_forces, env_normal_forces;
            std::vector<int> env_ids;
            std::vector<Eigen::MatrixXd> env_jacobians_dense;
            
            environments_[i]->get_contact_info(env_points, env_normals, env_forces, 
                                              env_normal_forces, env_ids, env_jacobians_dense);
            // std::cerr << "Env ["<< i << "] Number of contact Jacobians: " << env_jacobians_dense.size() << std::endl;

            int num_contacts = std::min(static_cast<int>(env_points.size()), max_contacts);
            contact_counts(i) = num_contacts;
            
            // 将雅可比矩阵扁平化存储
            for (int j = 0; j < num_contacts; j++) {
                int flat_start_index = j * jacobian_size_per_contact;
                const Eigen::MatrixXd& jac = env_jacobians_dense[j];
                
                // 将 3 x dof 矩阵展平存储
                for (int row = 0; row < 3; row++) {
                    for (int col = 0; col < dof; col++) {
                        jacobians_flat(i, flat_start_index + row * dof + col) = jac(row, col);
                    }
                }
            }
            
            // 填充剩余部分为0
            for (int j = num_contacts; j < max_contacts; j++) {
                int flat_start_index = j * jacobian_size_per_contact;
                for (int k = 0; k < jacobian_size_per_contact; k++) {
                    jacobians_flat(i, flat_start_index + k) = 0.0;
                }
            }
            
            // 原有的接触信息填充保持不变
            for (int j = 0; j < num_contacts; j++) {
                int col_index = j * 3;
                points(i, col_index)     = env_points[j].x();
                points(i, col_index + 1) = env_points[j].y();
                points(i, col_index + 2) = env_points[j].z();
                
                normals(i, col_index)     = env_normals[j].x();
                normals(i, col_index + 1) = env_normals[j].y();
                normals(i, col_index + 2) = env_normals[j].z();
                
                forces(i, col_index)     = env_forces[j].x();
                forces(i, col_index + 1) = env_forces[j].y();
                forces(i, col_index + 2) = env_forces[j].z();

                normal_forces(i, col_index) = env_normal_forces[j].x();
                normal_forces(i, col_index + 1) = env_normal_forces[j].y();
                normal_forces(i, col_index + 2) = env_normal_forces[j].z();

                contact_ids(i, j) = env_ids[j];
            }
            
            for (int j = num_contacts; j < 13; j++) {
                int col_index = j * 3;
                points(i, col_index) = points(i, col_index + 1) = points(i, col_index + 2) = 0.0;
                normals(i, col_index) = normals(i, col_index + 1) = normals(i, col_index + 2) = 0.0;
                forces(i, col_index) = forces(i, col_index + 1) = forces(i, col_index + 2) = 0.0;
                normal_forces(i, col_index) = normal_forces(i, col_index + 1) = normal_forces(i, col_index + 2) = 0.0;
                contact_ids(i, j) = -1;
            }
        }
    }
void get_obj_weight(Eigen::Ref<EigenVec> weights) {
    #pragma omp parallel for
    for (int i = 0; i < num_envs_; i++) {
        weights[i] = environments_[i]->get_obj_weight(); // 假设get_obj_weight()返回单个环境的权重
    }
}

void get_obj_mu(Eigen::Ref<EigenVec> mus) {
    #pragma omp parallel for
    for (int i = 0; i < num_envs_; i++) {
        mus[i] = environments_[i]->get_obj_mu(); // 假设get_obj_weight()返回单个环境的权重
    }
  }

    void get_global_state_l(Eigen::Ref<EigenRowMajorMat> &gs) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++) 
      environments_[i]->get_global_state_l(gs.row(i));
  }

  void step(Eigen::Ref<EigenRowMajorMat> &action_r,
            Eigen::Ref<EigenRowMajorMat> &action_l,
            Eigen::Ref<EigenVec> &reward_r,
            Eigen::Ref<EigenVec> &reward_l,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action_r, action_l, reward_r, reward_l, done);
  }

  void step2(Eigen::Ref<EigenRowMajorMat> &action_r,
             Eigen::Ref<EigenRowMajorMat> &action_l,
             Eigen::Ref<EigenVec> &reward_r,
             Eigen::Ref<EigenVec> &reward_l,
             Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      perAgentStep2(i, action_r, action_l, reward_r, reward_l, done);
  }

  void reset_right_hand(Eigen::Ref<EigenRowMajorMat>& obj_pose_step_r,
                        Eigen::Ref<EigenRowMajorMat>& hand_ee_step_r,
                        Eigen::Ref<EigenRowMajorMat>& hand_pose_step_r){
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->reset_right_hand(obj_pose_step_r.row(i), hand_ee_step_r.row(i), hand_pose_step_r.row(i));
  }

  void step_imitate(Eigen::Ref<EigenRowMajorMat> &action_r,
                    Eigen::Ref<EigenRowMajorMat> &action_l,
                    Eigen::Ref<EigenRowMajorMat> &obj_pose_r, 
                    Eigen::Ref<EigenRowMajorMat> &hand_ee_r,
                    Eigen::Ref<EigenRowMajorMat> &hand_pose_r,
                    Eigen::Ref<EigenRowMajorMat> &obj_pose_l, 
                    Eigen::Ref<EigenRowMajorMat> &hand_ee_l,
                    Eigen::Ref<EigenRowMajorMat> &hand_pose_l,
                    bool imitate_right,
                    bool imitate_left,
                    Eigen::Ref<EigenVec> &reward_r,
                    Eigen::Ref<EigenVec> &reward_l,
                    Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for
    for (int i = 0; i < num_envs_; i++)
      perAgentImitate(i, action_r, action_l, obj_pose_r, hand_ee_r, hand_pose_r, obj_pose_l, hand_ee_l, hand_pose_l, imitate_right, imitate_left, reward_r, reward_l, done);
  }

  void debugShowObs(Eigen::Ref<EigenRowMajorMat>& ob_r){
        #pragma omp parallel for
              for (int i = 0; i < num_envs_; i++){
                  environments_[i]->debugShowObs(ob_r.row(i));
              }
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }

//  void getPtarget(Eigen::Ref<EigenVec>& ptarget) {ptarget = p;}

  void setSeed(int seed) {
    int seed_inc = seed;
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void set_rootguidance() {
        #pragma omp parallel for
              for (int i = 0; i < num_envs_; i++)
                  environments_[i]->set_rootguidance();
  }

  void switch_root_guidance(bool is_on) {
        #pragma omp parallel for
              for (int i = 0; i < num_envs_; i++)
                  environments_[i]->switch_root_guidance(is_on);
  }

  void switch_obj_pos(Eigen::Ref<EigenRowMajorMat> obj_pos_bias) {
        #pragma omp parallel for
              for (int i = 0; i < num_envs_; i++)
                  environments_[i]->switch_obj_pos(obj_pos_bias.row(i));
  }

  void control_switch(int left, int right) {
        #pragma omp parallel for
              for (int i = 0; i < num_envs_; i++)
                  environments_[i]->control_switch(left, right);
  }

  void control_switch_all(Eigen::Ref<EigenVec>& left, Eigen::Ref<EigenVec>& right) {
        #pragma omp parallel for
              for (int i = 0; i < num_envs_; i++){
                environments_[i]->control_switch(left[i], right[i]);
              }

  }

  void set_joint_sensor_visual(Eigen::Ref<EigenRowMajorMat> &joint_vector){
        #pragma omp parallel for
              for (int i = 0; i < num_envs_; i++){
                environments_[i]->set_joint_sensor_visual(joint_vector.row(i));
              }
  }

  void set_sample_point_visual(Eigen::Ref<EigenRowMajorMat> &joint_vector, Eigen::Ref<EigenRowMajorMat> &obj_pose){
        #pragma omp parallel for
              for (int i = 0; i < num_envs_; i++){
                environments_[i]->set_sample_point_visual(joint_vector.row(i), obj_pose.row(i));
              }
  }

  bool check_collision(Eigen::Ref<EigenRowMajorMat> &joint_vector){
    return environments_[0]->check_collision(joint_vector.row(0));
  }

  void set_joint_sensor_visual_l(Eigen::Ref<EigenRowMajorMat> &joint_vector){
        #pragma omp parallel for
              for (int i = 0; i < num_envs_; i++){
                environments_[i]->set_joint_sensor_visual_l(joint_vector.row(i));
              }
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getRightObDim() { return obDim_r_; }
  int getLeftObDim() {return obDim_l_;}
  int getActionDim() { return actionDim_; }
  int getGSDim() { return gsDim_; }
  int getNumOfEnvs() { return num_envs_; }

  // Return material pair properties for a given environment index and material names.
  // Returns a vector of doubles: {c_f, c_r, r_th, c_static_f, v_static_speed}
  std::vector<double> getMaterialPairProperties(int envIndex, const std::string &mat1, const std::string &mat2) {
    if (envIndex < 0 || envIndex >= num_envs_)
      throw std::runtime_error("envIndex out of range");
    const raisim::MaterialPairProperties &p = environments_[envIndex]->getWorld()->getMaterialPairProperties(mat1, mat2);
    return std::vector<double>{p.c_f, p.c_r, p.r_th, p.c_static_f, p.v_static_speed};
  }

  // Return the total mass of an object in a given env (by object name). If the object is articulated, returns getTotalMass(), otherwise returns getMass(0).
  double getObjectTotalMass(int envIndex, const std::string &objName="object") {
    if (envIndex < 0 || envIndex >= num_envs_)
      throw std::runtime_error("envIndex out of range");
    raisim::Object* obj = environments_[envIndex]->getWorld()->getObject(objName);
    if (!obj) throw std::runtime_error("object not found: " + objName);
    if (auto art = dynamic_cast<raisim::ArticulatedSystem*>(obj))
      return art->getTotalMass();
    else
      return obj->getMass(0);
  }





  ////// optional methods //////
  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

  const std::vector<std::map<std::string, float>>& getRewardInfoRight() { return rewardInformation_r_; }
  const std::vector<std::map<std::string, float>>& getRewardInfoLeft() { return rewardInformation_l_; }

 private:

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action_r,
                           Eigen::Ref<EigenRowMajorMat> &action_l,
                           Eigen::Ref<EigenVec> &reward_r,
                           Eigen::Ref<EigenVec> &reward_l,
                           Eigen::Ref<EigenBoolVec> &done) {
//    std::cout << "action i" << agentId << " " << action.row(agentId)[0] << " " <<  action.row(agentId)[1] << " " << action.row(agentId)[2] << std::endl;
    float* reward_sum;

    reward_sum = environments_[agentId]->step(action_r.row(agentId), action_l.row(agentId));
    reward_r[agentId] = reward_sum[0];
    reward_l[agentId] = reward_sum[1];

    rewardInformation_r_[agentId] = environments_[agentId]->getRewardsRight().getStdMap();
    rewardInformation_l_[agentId] = environments_[agentId]->getRewardsLeft().getStdMap();

    //std::cout << "vsize:" << rewardInformation_l_.size() << std::endl; 
    //std::cout << "rize:" << reward_l.size() << std::endl;
    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward_r[agentId] += terminalReward;
      reward_l[agentId] += terminalReward;
    }
  }

  inline void perAgentStep2(int agentId,
                            Eigen::Ref<EigenRowMajorMat> &action_r,
                            Eigen::Ref<EigenRowMajorMat> &action_l,
                            Eigen::Ref<EigenVec> &reward_r,
                            Eigen::Ref<EigenVec> &reward_l,
                            Eigen::Ref<EigenBoolVec> &done) {
//    std::cout << "action i" << agentId << " " << action.row(agentId)[0] << " " <<  action.row(agentId)[1] << " " << action.row(agentId)[2] << std::endl;
    float* reward_sum;

    reward_sum = environments_[agentId]->step2(action_r.row(agentId), action_l.row(agentId));
    reward_r[agentId] = reward_sum[0];
    reward_l[agentId] = reward_sum[1];

    rewardInformation_r_[agentId] = environments_[agentId]->getRewardsRight().getStdMap();
    rewardInformation_l_[agentId] = environments_[agentId]->getRewardsLeft().getStdMap();

    //std::cout << "vsize:" << rewardInformation_l_.size() << std::endl; 
    //std::cout << "rize:" << reward_l.size() << std::endl;
    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward_r[agentId] += terminalReward;
      reward_l[agentId] += terminalReward;
    }
  }

  inline void perAgentImitate(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action_r,
                           Eigen::Ref<EigenRowMajorMat> &action_l,
                           Eigen::Ref<EigenRowMajorMat> &obj_pose_r,
                           Eigen::Ref<EigenRowMajorMat> &hand_ee_r,
                           Eigen::Ref<EigenRowMajorMat> &hand_pose_r,
                           Eigen::Ref<EigenRowMajorMat> &obj_pose_l,
                           Eigen::Ref<EigenRowMajorMat> &hand_ee_l,
                           Eigen::Ref<EigenRowMajorMat> &hand_pose_l,
                           bool imitate_right,
                           bool imitate_left,
                           Eigen::Ref<EigenVec> &reward_r,
                           Eigen::Ref<EigenVec> &reward_l,
                           Eigen::Ref<EigenBoolVec> &done) {
//    std::cout << "action i" << agentId << " " << action.row(agentId)[0] << " " <<  action.row(agentId)[1] << " " << action.row(agentId)[2] << std::endl;
    float* reward_sum;

    reward_sum = environments_[agentId]->step_imitate(action_r.row(agentId), action_l.row(agentId), obj_pose_r.row(agentId), hand_ee_r.row(agentId), hand_pose_r.row(agentId), obj_pose_l.row(agentId), hand_ee_l.row(agentId), hand_pose_l.row(agentId), imitate_right, imitate_left);
    reward_r[agentId] = reward_sum[0];
    reward_l[agentId] = reward_sum[1];

    rewardInformation_r_[agentId] = environments_[agentId]->getRewardsRight().getStdMap();
    rewardInformation_l_[agentId] = environments_[agentId]->getRewardsLeft().getStdMap();

    //std::cout << "vsize:" << rewardInformation_l_.size() << std::endl; 
    //std::cout << "rize:" << reward_l.size() << std::endl;
    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward_r[agentId] += terminalReward;
      reward_l[agentId] += terminalReward;
    }
  }

  std::vector<ChildEnvironment *> environments_;
  std::vector<std::map<std::string, float>> rewardInformation_r_;
  std::vector<std::map<std::string, float>> rewardInformation_l_;
  Eigen::VectorXd forceClosureCosts_;
  Eigen::VectorXd frictionConeCosts_;
  int num_envs_ = 1;
  int obDim_r_ = 0, obDim_l_ = 0, actionDim_ = 0, gsDim_ = 0;
  bool recordVideo_=false, render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;
  std::string cfgString_;
  Eigen::VectorXd p;
};

class NormalDistribution {
 public:
  NormalDistribution() : normDist_(0.f, 1.f) {}

  float sample() { return normDist_(gen_); }
  void seed(int i) { gen_.seed(i); }

 private:
  std::normal_distribution<float> normDist_;
  static thread_local std::mt19937 gen_;
};
thread_local std::mt19937 raisim::NormalDistribution::gen_;


class NormalSampler {
 public:
  NormalSampler(int dim) {
    dim_ = dim;
    normal_.resize(THREAD_COUNT);
    seed(0);
  }

  void seed(int seed) {
    // this ensures that every thread gets a different seed
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < THREAD_COUNT; i++)
      normal_[0].seed(i + seed);
  }

  inline void sample(Eigen::Ref<EigenRowMajorMat> &mean,
                     Eigen::Ref<EigenVec> &std,
                     Eigen::Ref<EigenRowMajorMat> &samples,
                     Eigen::Ref<EigenVec> &log_prob) {
    int agentNumber = log_prob.rows();

#pragma omp parallel for schedule(auto)
    for (int agentId = 0; agentId < agentNumber; agentId++) {
      log_prob(agentId) = 0;
      for (int i = 0; i < dim_; i++) {
        const float noise = normal_[omp_get_thread_num()].sample();
        samples(agentId, i) = mean(agentId, i) + noise * std(i);
        log_prob(agentId) -= noise * noise * 0.5 + std::log(std(i));
      }
      log_prob(agentId) -= float(dim_) * 0.9189385332f;
    }
  }
  int dim_;
  std::vector<NormalDistribution> normal_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
