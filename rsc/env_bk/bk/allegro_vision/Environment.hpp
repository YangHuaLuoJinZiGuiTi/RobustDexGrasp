//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include "raisim/World.hpp"
#include <vector>
#include "raisim/math.hpp"
#include <math.h>
#include <time.h>

#define USE_RIGHT_HAND(TYPE) ((TYPE) == 0 || (TYPE) == 2)
#define USE_LEFT_HAND(TYPE) ((TYPE) == 1 || (TYPE) == 2)

/**
 * naming rule:  all of them are in world frame.
 * 
 * <OBJECT>_<STATE>_<TYPE>_<FRAME>(_<TRANS>_)
 * 
 * <OBJECT>   t=top, b=bottom, base=base_origin_point, arti=articulation_axis
 * <STATE>    init=resetinit, tar=target, cur=current(default)
 * <TYPE>     pos=3D_position, rot=9D_rotation_matrix
 * <FRAME>    o=object, w=world, wrist=wrist
 * <TRANS>    trans means transpose of the matrix
 * <GLOBAL>   `_` in the end for global value and non for local value
 * 
 * e.g. `wrist_init_pos_o_` means the init position of wrist in object frame and it is a global value
 * 
 * 
**/


namespace raisim 
{
    class global_value {
    public:
        void init(std::unique_ptr<raisim::World>& world, const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) {
            resourceDir_ = resourceDir;

            visualizable_ = cfg["visualize"].As<bool>();
            hand_type_ = cfg["hand_type"].As<int>();

            history_len_ = cfg["history_len"].As<int>();
            encode_dim_ = cfg["step_encode_dim"].As<int>();
            step_obs_dim_ = cfg["step_obs_dim"].As<int>();

            // init world_ 
            world->addGround();
            world->setERP(0.0);
            world->setMaterialPairProp("object", "object", 0.8, 0.0, 0.0, 0.8, 0.1);
            world->setMaterialPairProp("object", "finger", 0.8, 0.0, 0.0, 0.8, 0.1);
            world->setMaterialPairProp("finger", "finger", 0.8, 0.0, 0.0, 0.8, 0.1);
            world->setDefaultMaterial(0.8, 0, 0, 0.8, 0.1);

        }
        
        void quat_rot_quat(raisim::Vec<4> &cur, raisim::Mat<3,3> &rot_r, Eigen::Vector4d &tar_e) {
            raisim::Vec<4> tar;
            raisim::Mat<3,3> cur_r, tar_r;
            raisim::quatToRotMat(cur, cur_r);
            raisim::matmul(rot_r, cur_r, tar_r);
            raisim::rotMatToQuat(tar_r, tar);
            for (int i = 0; i < 4; i++) {
                tar_e[i] = tar[i];
            }
        }

        void set_coordinate_vis(raisim::Visuals **vis, Eigen::Vector3d pos, raisim::Mat<3,3> rot_r) {
            raisim::Vec<4> init_pos[3] = {{0.70710678118, 0, 0.70710678118, 0}, {0.70710678118, -0.70710678118, 0, 0}, {1, 0, 0, 0}};
            for (int i = 0; i < 3; i++) {
                Eigen::Vector4d new_rot;
                quat_rot_quat(init_pos[i], rot_r, new_rot);
                vis[i]->setPosition(pos);
                vis[i]->setOrientation(new_rot);
            }
        }

        void set_cylinder_vis(raisim::Visuals *vis, raisim::Vec<3> &start, raisim::Vec<3> &end, float r=0.001) {
            raisim::Vec<3> mid = (start + end) / 2.0;
            raisim::Vec<3> diff = end - start;
            double dis = sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]);

            Eigen::Vector3d A = diff.e();
            A.normalize();
            Eigen::Vector3d Z(0, 0, 1);
            Eigen::Vector3d axis = Z.cross(A);
            if (axis.norm() < 1e-6) {
                if (A.dot(Z) > 0) {
                } else {
                    axis = Eigen::Vector3d(1, 0, 0);
                }
            }
            Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(Z, A);
            vis->setPosition(mid.e());
            vis->setOrientation(q.w(), q.x(), q.y(), q.z());
            vis->setCylinderSize(r, dis);
        }

    public:
        bool visualizable_ = false, first_reset_ = true;
        std::string resourceDir_;
        bool lift_ = false;
        int hand_type_ = 0; // 0 is right, 1 is left, 2 is two hands
        float rewards_sum_[2];

        int encode_dim_, history_len_, step_obs_dim_;
    };

    class arti_object {
    public:
        void init(global_value *g, const Yaml::Node& cfg) {
            g_ = g;
            load_set_ = cfg["load_set"].As<std::string>();
            
            // init object variables
            obj_init_gc_.setZero(8); obj_init_gc_[3] = 1; obj_tar_gc_.setZero(8); obj_tar_gc_[3] = 1; obj_cur_gc_.setZero(8); obj_cur_gc_[3] = 1; // for quat w = 1
        }

        void load(std::unique_ptr<raisim::World>& world, std::unique_ptr<raisim::RaisimServer>& server, const std::string& obj_model) {
            /// init table collision group is 1
            box_ = static_cast<raisim::Box*>(world->addBox(2, 1, 0.5, 100, "table", raisim::COLLISION(1)));
            box_->setPosition(1.25, 0, 0.25);
            box_->setAppearance("0.0 0.0 0.0 0.0");

            // collision group is 2, will check collision with 0(hand), 1(table), 2(object), 63(all)
            arctic_ = static_cast<raisim::ArticulatedSystem*>(world->addArticulatedSystem(g_->resourceDir_+"/"+load_set_+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
            arctic_->setName("arctic");
            obj_gc_dim_ = arctic_->getGeneralizedCoordinateDim();
            obj_gv_dim_ = arctic_->getDOF(); // same as ->getGeneralizedVelocityDim()
            obj_weight_ = arctic_->getTotalMass();

            arctic_->setState(obj_init_gc_, Eigen::VectorXd::Zero(obj_gv_dim_));
            arctic_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            arctic_->setPdGains(Eigen::VectorXd::Zero(obj_gv_dim_), Eigen::VectorXd::Zero(obj_gv_dim_)); 
            arctic_->setGeneralizedForce(Eigen::VectorXd::Zero(obj_gv_dim_));

            if (g_->visualizable_) { // add the fake target object mesh
                arcticVisual_ = server->addVisualArticulatedSystem("arctic_visual", g_->resourceDir_+"/"+load_set_+"/"+obj_model, 0, 1, 0, 1);

                /// Create table
                table_top = server->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                table_top->setPosition(1.25, 0, 0.475);
                leg1 = server->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg2 = server->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg3 = server->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg4 = server->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                leg1->setPosition(0.2625,0.4675,0.2375);
                leg2->setPosition(2.2275,0.4875,0.2375);
                leg3->setPosition(0.2625,-0.4675,0.2375);
                leg4->setPosition(2.2275,-0.4875,0.2375);

            }

            double non_aff_mass = arctic_->getMass(arctic_->getBodyIdx("bottom"));
            if (non_aff_mass > 0.001) {
                has_non_aff_ = true;
            } else {
                has_non_aff_ = false;
            }
        }

        void reset_init_state() {
            arctic_->setState(obj_init_gc_, Eigen::VectorXd::Zero(obj_gv_dim_));
            arctic_->setGeneralizedForce(Eigen::VectorXd::Zero(obj_gv_dim_));
            arctic_->setPdGains(Eigen::VectorXd::Zero(obj_gv_dim_), Eigen::VectorXd::Zero(obj_gv_dim_)); 

            box_->clearExternalForcesAndTorques();
            box_->setPosition(1.25, 0, 0.25);
            box_->setOrientation(1,0,0,0);
            box_->setVelocity(0,0,0,0,0,0);
        }

        void reset_user_state(const Eigen::Ref<EigenVec>& obj_pose) {
            /// set initial object state
            obj_init_gc_ = obj_pose.cast<double>(); // 8 dof
            arctic_->setState(obj_init_gc_, Eigen::VectorXd::Zero(obj_gv_dim_));
            arctic_->setGeneralizedForce(Eigen::VectorXd::Zero(obj_gv_dim_));
            arctic_->setPdGains(Eigen::VectorXd::Zero(obj_gv_dim_), Eigen::VectorXd::Zero(obj_gv_dim_)); 

            /// reset table position (only required in case for inference)
            box_->setPosition(1.25, 0, 0.25);
            box_->setOrientation(1,0,0,0);
            box_->setVelocity(0,0,0,0,0,0);
        }

        void set_goals(double obj_angle, const Eigen::Ref<EigenVec>& obj_goal_pos) {
            #if 0
            obj_tar_gc_ = obj_goal_pos.cast<double>(); // 7D
            obj_tar_gc_[7] = obj_angle;
            //set_gc_for_arctic(obj_b_tar_7Dstate_, obj_t_tar_7Dstate_, obj_goal_pos_true);
            if (g_->visualizable_) {
                arcticVisual_->setGeneralizedCoordinate(obj_tar_gc_);
            }
            #endif
        }

        void update_object_state() {
            auto affordance_id = arctic_->getBodyIdx("top");
            auto non_affordance_id = arctic_->getBodyIdx("bottom");

            arctic_->getPosition(affordance_id, obj_t_cur_pos_);
            arctic_->getOrientation(affordance_id, obj_t_cur_rot_);
            arctic_->getAngularVelocity(affordance_id, obj_t_cur_anglevel_);
            arctic_->getVelocity(affordance_id, obj_t_cur_linevel_);

            arctic_->getPosition(non_affordance_id, obj_b_cur_pos_);
            arctic_->getOrientation(non_affordance_id, obj_b_cur_rot_);
            arctic_->getAngularVelocity(non_affordance_id, obj_b_cur_anglevel_);
            arctic_->getVelocity(non_affordance_id, obj_b_cur_linevel_);
            
            obj_cur_gc_ = arctic_->getGeneralizedCoordinate().e();
            obj_cur_gv_ = arctic_->getGeneralizedVelocity().e();
        }

        void update_observation() {
            update_object_state();
            
            Eigen::Vector3d rotation_axis_obj(0., 0., 1.);
            obj_arti_axis_cur_rot_ = obj_b_cur_rot_.e() * rotation_axis_obj;
        }
        
        // get the 7D state of top and bottom from the target object state(after move the bottom and articulation angle)
        void set_gc_for_arctic(Eigen::VectorXd& gc_for_b, Eigen::VectorXd& gc_for_t, const Eigen::VectorXd& gc_arctic) {
            
            arctic_->setBasePos(gc_arctic.head(3));

            raisim::Mat<3,3> base_rot;
            raisim::quatToRotMat(gc_arctic.segment(3,4), base_rot);
            arctic_->setBaseOrientation(base_rot);

            Eigen::VectorXd obj_goal_angle = Eigen::VectorXd::Zero(1);
            obj_goal_angle[0] = gc_arctic[7];
            arctic_->setGeneralizedCoordinate(obj_goal_angle);

            update_object_state();

            raisim::Vec<4> obj_b_goal_ori, obj_t_goal_ori;
            raisim::rotMatToQuat(obj_b_cur_rot_, obj_b_goal_ori);
            raisim::rotMatToQuat(obj_t_cur_rot_, obj_t_goal_ori);
            gc_for_b.head(3) = obj_b_cur_pos_.e();
            gc_for_t.head(3) = obj_t_cur_pos_.e();
            gc_for_b.segment(3,4) = obj_b_goal_ori.e();
            gc_for_t.segment(3,4) = obj_t_goal_ori.e();
        }

    public:
        global_value* g_;
        std::string load_set_;
        raisim::Box *box_;
        raisim::ArticulatedSystem *arctic_;
        raisim::ArticulatedSystemVisual *arcticVisual_;
        raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
        bool has_non_aff_ = false;
        double obj_weight_ = 0.0;
        int obj_gc_dim_, obj_gv_dim_;

        // init state value:
        Eigen::VectorXd obj_init_gc_; // 8dof

        //raisim::Mat<3,3> obj_base_init_rot_;
        //raisim::Vec<3> obj_base_init_pos_;
        //double obj_arti_init_angle_=0.0;
        // current state value:
        Eigen::VectorXd obj_cur_gc_, obj_cur_gv_; // 8dof, 7dof
        raisim::Mat<3,3> obj_t_cur_rot_, obj_b_cur_rot_;
        raisim::Vec<3> obj_t_cur_pos_, obj_b_cur_pos_, obj_t_cur_anglevel_, obj_t_cur_linevel_, obj_b_cur_anglevel_, obj_b_cur_linevel_;
        double obj_arti_cur_angle_=0.0, obj_arti_cur_anglevel_=0.0;
        Eigen::Vector3d obj_arti_axis_cur_rot_;
        // target state value:
        Eigen::VectorXd obj_tar_gc_; // 8dof
        raisim::Mat<3,3> obj_t_tar_rot_, obj_b_tar_rot_;
        double obj_arti_tar_angle_=0.0;
    };

    class hand {
    public:
        void init(std::unique_ptr<raisim::World>& world, raisim::Reward *rewards, const Yaml::Node& cfg, global_value *g, arti_object *obj, std::string name) {
            std::string hand_model_name;
            g_ = g;
            obj_ = obj;

            if (name == "left") {
                hand_model_name = "hand_model_l";
            } else {
                hand_model_name = "hand_model_r";
            }

            // collision group is 0, will check collision with 2(object), 63(all)
            mano_ = world->addArticulatedSystem(g_->resourceDir_+"/allegro/"+cfg[hand_model_name].As<std::string>(),"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63));
            mano_->setName(name);

            /// set PD control mode
            mano_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

            /// get actuation dimensions
            gcDim_ = mano_->getGeneralizedCoordinateDim();
            gvDim_ = mano_->getDOF();
            nJoints_ = gcDim_-3;
            
            /// initialize hand variables
            num_contacts_ = cfg["hand_contact_num"].As<int>();
            num_bodies_ = cfg["hand_body_num"].As<int>();
            global_state_dim_ = cfg["gsdim"].As<int>();
            num_finger_ = (num_contacts_ - 1) / 3; // allegro is 4, mano is 5

            vis_joint_mesh_dis_.resize(num_bodies_);
            vis_mesh_sphere_.resize(num_bodies_);
            vis_joint_sphere_.resize(num_bodies_);

            // TBD: set these names in yaml file
            std::string tmp_body_part[17] =  {"z_rotation_joint",
                                            "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
                                            "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
                                            "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
                                            "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"};
            std::string tmp_contact_part[13] =  {"base_link",
                                                "link_1.0", "link_2.0", "link_3.0", 
                                                "link_5.0", "link_6.0", "link_7.0",
                                                "link_9.0", "link_10.0", "link_11.0",
                                                "link_13.0", "link_14.0", "link_15.0"};
            for (int i = 0; i < 17; i++) {
                body_parts_.push_back(tmp_body_part[i]);
            }
            for (int i = 0; i < 13; i++) {
                contact_bodies_.push_back(tmp_contact_part[i]);
            }

            gc_.setZero(gcDim_); gv_.setZero(gvDim_); gc_set_.setZero(gcDim_); gv_set_.setZero(gvDim_);
            pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);
            actionMean_.setZero(gcDim_); actionStd_.setOnes(gcDim_);
            joint_limit_high_.setZero(gcDim_); joint_limit_low_.setZero(gcDim_);
            hand_torque_.setZero(gcDim_); wrist_torque_.setZero(6);
            mano_base_init_rot_w_trans_.setZero();  mano_base_init_rot_w_.setZero(); mano_base_init_pos_w_.setZero();
            wrist_target_rot_o_.setZero(); wrist_vel_in_wrist_.setZero(); wrist_qvel_in_wrist_.setZero();
            handcenter_pos_wrist_.setZero(); handcenter_tar_pos_o_.setZero(); 
            frame_y_in_obj_.setZero(num_bodies_*3);
            joint_pos_in_obj_.setZero(num_bodies_*3);
            joint_pos_in_world_.setZero(num_bodies_*3);
            contacts_af_.setZero(num_contacts_); contacts_non_af_.setZero(num_contacts_); impulses_af_.setZero(num_contacts_); impulses_non_af_.setZero(num_contacts_), contacts_table_.setZero(num_contacts_), impulses_table_.setZero(num_contacts_);
            pTarget_clipped_.setZero(gcDim_);
            joint_height_w_.setZero(num_bodies_);

            /// initialize 3D positions weights for fingertips higher than for other fingerparts , global variables
            finger_weights_contact_.setOnes(num_contacts_);
            for(int i = 1; i < num_finger_ + 1; i++){ // all link part without the first link
                finger_weights_contact_(3*i) *= 3;
            }
            finger_weights_contact_.segment(num_contacts_ - 3, 3) *= 2; // the last link part
            finger_weights_contact_ /= finger_weights_contact_.sum();
            finger_weights_contact_ *= num_contacts_;

            impulse_high_.setZero(num_contacts_); impulse_low_.setZero(num_contacts_);
            for(int i = 0; i < (num_contacts_ - 3); i++){
                impulse_high_[i] = 0.1;
                impulse_low_[i] = -0.0;
            }
            for(int i = (num_contacts_ - 3); i < num_contacts_; i++){ // last part
                impulse_high_[i] = 0.2;
                impulse_low_[i] = -0.0;
            }

            /// set PD gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.head(3).setConstant(wrist_Pgain_);
            jointDgain.head(3).setConstant(wrist_Dgain_);
            jointPgain.tail(gcDim_-3).setConstant(rot_Pgain_);
            jointDgain.tail(gcDim_-3).setConstant(rot_Dgain_);

            mano_->setPdGains(jointPgain, jointDgain);
            mano_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
            mano_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

            /// retrieve joint limits from model
            std::vector<raisim::Vec<2>> joint_limits;
            joint_limits = mano_->getJointLimits();

            for(int i=0; i < int(gcDim_); i++){
                actionMean_[i] = (joint_limits[i][1]+joint_limits[i][0])/2.0;
                joint_limit_low_[i] = joint_limits[i][0];
                joint_limit_high_[i] = joint_limits[i][1];
            }

            /// set actuation parameters
            actionStd_.setConstant(cfg["finger_action_std"].As<float>());
            actionStd_.head(3).setConstant(0.005);
            actionStd_.segment(3,3).setConstant(0.005);

            for(int i = 0; i < num_contacts_ ;i++){
                contactMapping_.insert(std::pair<int,int>(int(mano_->getBodyIdx(contact_bodies_[i])),i));
            }
            
            rewards->initializeFromConfigurationFile(cfg["reward"]);
        }
    
        void init_vis(std::unique_ptr<raisim::RaisimServer>& server) {
            /// initialize Cylinders for sensor
            for(int i = 0; i < num_bodies_; i++){
                vis_joint_mesh_dis_[i] = server->addVisualCylinder(body_parts_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                vis_mesh_sphere_[i] = server->addVisualSphere(body_parts_[i]+"_mesh_sphere", 0.005, 0, 1, 0, 1);
                vis_joint_sphere_[i] = server->addVisualSphere(body_parts_[i]+"_joints_sphere", 0.005, 0, 0, 1, 1);
            }
            vis_random_dir_dis_ = server->addVisualCylinder(body_parts_[0]+"_dir_cylinder", 0.01, 0.15, 1, 1, 1);

            vis_target_center_sphere_ = server->addVisualSphere(body_parts_[0]+"_target_center_sphere", 0.015, 0, 0, 0, 1);
            vis_hand_center_sphere_ = server->addVisualSphere(body_parts_[0]+"_hand_center_sphere", 0.015, 0, 1, 1, 1);

            vis_wrist_center_array_[0] = server->addVisualArrow(body_parts_[0]+"_center_x", 0.03, 0.12, 1, 0, 0, 1);
            vis_wrist_center_array_[1] = server->addVisualArrow(body_parts_[0]+"_center_y", 0.03, 0.12, 0, 1, 0, 1);
            vis_wrist_center_array_[2] = server->addVisualArrow(body_parts_[0]+"_center_z", 0.03, 0.12, 0, 0, 1, 1);

            vis_init_root_array_[0] = server->addVisualArrow(body_parts_[0]+"_init_root_x", 0.035, 0.13, 1, 0, 0, 1);
            vis_init_root_array_[1] = server->addVisualArrow(body_parts_[0]+"_init_root_y", 0.035, 0.13, 0, 1, 0, 1);
            vis_init_root_array_[2] = server->addVisualArrow(body_parts_[0]+"_init_root_z", 0.035, 0.13, 0, 0, 1, 1);
        }

        void joint_distance_vis(const Eigen::Ref<EigenVec>& joint_sensor_visual) {
            raisim::Vec<3> joint_pos_w, mesh_pos_o, joint_pos_o, mesh_pos_w;

            obj_->update_object_state();

            /// initialize Cylinders for sensor
            for(int i = 0; i < num_bodies_; i++) {
                #if 1
                // get the joint position in object frame
                joint_pos_o = joint_pos_in_obj_.segment(i*3,3).cast<double>();
                // rotate the rotation of object + transfer position of object = joint position in world frame
                raisim::matvecmul(obj_->obj_t_cur_rot_, joint_pos_o, joint_pos_w);
                raisim::vecadd(obj_->obj_t_cur_pos_, joint_pos_w);

                // get the mesh point and roate it to 
                mesh_pos_o = joint_sensor_visual.segment(i*3,3).cast<double>();
                raisim::Vec<3> tmpvec;
                raisim::Mat<3,3> obj_t_rot_wrist, wrist_rot_w, wrist_rot_w_trans, obj_pose_wrist_mat_trans;
                mano_->getFrameOrientation(body_parts_[0], wrist_rot_w);
                raisim::transpose(wrist_rot_w, wrist_rot_w_trans);
                raisim::matmul(wrist_rot_w_trans, obj_->obj_t_cur_rot_, obj_t_rot_wrist);
                raisim::transpose(obj_t_rot_wrist, obj_pose_wrist_mat_trans);
                raisim::matvecmul(obj_pose_wrist_mat_trans, mesh_pos_o, tmpvec);
                raisim::vecadd(joint_pos_o, tmpvec);
                raisim::matvecmul(obj_->obj_t_cur_rot_, tmpvec, mesh_pos_w);
                raisim::vecadd(obj_->obj_t_cur_pos_, mesh_pos_w);
                #else // debug the value from raisim ground truth
                raisim::Mat<3,3> Obj_orientation_temp;
                mano_->getFramePosition(body_parts_[i], joint_pos_w);
                obj_->arctic_->getOrientation(obj_->arctic_->getBodyIdx("top"), Obj_orientation_temp);
                mesh_pos_o = joint_sensor_visual.segment(i*3,3).cast<double>();
                raisim::matvecmul(Obj_orientation_temp, mesh_pos_o, mesh_pos_w);
                vecadd(obj_->obj_t_cur_pos_, mesh_pos_w);
                #endif

                // show in raisim UI
                vis_joint_sphere_[i]->setPosition(joint_pos_w.e());
                vis_mesh_sphere_[i]->setPosition(mesh_pos_w.e());
                g_->set_cylinder_vis(vis_joint_mesh_dis_[i], joint_pos_w, mesh_pos_w);
            }
        }

        void reset_init_state() {
            actionMean_.setZero();
            mano_->setBasePos(mano_base_init_pos_w_);
            mano_->setBaseOrientation(mano_base_init_rot_w_);
            mano_->setState(gc_set_, gv_set_);
            mano_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));

            gc_=gc_set_;
            gv_.setZero(gvDim_);
            gv_set_.setZero(gvDim_);
            actionMean_.setZero();
            actionMean_.tail(gcDim_-6) = gc_set_.tail(gcDim_-6);
        }

        void reset_user_state(const Eigen::Ref<EigenVec>& init_state, const Eigen::Ref<EigenVec>& init_vel) {
            /// reset gains (only required in case for inference)
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.head(3).setConstant(wrist_Pgain_);
            jointDgain.head(3).setConstant(wrist_Dgain_);
            jointPgain.tail(gcDim_-3).setConstant(rot_Pgain_);
            jointDgain.tail(gcDim_-3).setConstant(rot_Dgain_);
            mano_->setPdGains(jointPgain, jointDgain);
            mano_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));

            gc_set_.head(6).setZero();
            gc_set_.tail(gcDim_-6) = init_state.tail(gcDim_-6).cast<double>();
            gv_set_ = init_vel.cast<double>();
            mano_->setState(gc_set_, gv_set_);

            /// set initial root position and orientation in global frame as origin in new coordinate frame
            mano_base_init_pos_w_  = init_state.head(3);
            raisim::Vec<4> quat;
            raisim::eulerToQuat(init_state.segment(3,3),quat); // initial base ori, in quat
            raisim::quatToRotMat(quat, mano_base_init_rot_w_); // ..., in matrix
            raisim::transpose(mano_base_init_rot_w_, mano_base_init_rot_w_trans_); // ..., inverse

            mano_->setBasePos(mano_base_init_pos_w_);
            mano_->setBaseOrientation(mano_base_init_rot_w_);
            mano_->setState(gc_set_, gv_set_);

            /// Set action mean to initial pose (first 6DoF since start at 0)
            actionMean_.setZero();
            actionMean_.tail(gcDim_-6) = gc_set_.tail(gcDim_-6);

            obs_history_.clear();
        }

        void set_goals(const Eigen::Ref<EigenVec>& ee_goal_pos) {
            // average position without thumb tip in world frame
            raisim::Vec<3> sphere_pos;
            Eigen::Vector3d handcenter_cur_pos_w = Eigen::Vector3d::Zero();
            for(int i = 0; i < num_finger_ - 1; i++){
                mano_->getFramePosition(body_parts_[i*4+2], sphere_pos); // mano is joint2_x, allegro is 
                handcenter_cur_pos_w += sphere_pos.e();
            }
            handcenter_cur_pos_w /= (float)(num_finger_ - 1);
            Eigen::Vector3d fingercenter_cur_pos_w = handcenter_cur_pos_w;

            // average position with thumb tip in world frame
            mano_->getFramePosition(body_parts_[num_bodies_ - 1], sphere_pos);
            handcenter_cur_pos_w = (handcenter_cur_pos_w + sphere_pos.e()) / 2.0;

            // direction from finger center(no thumb tip) to hand center(with thumb tip)
            Eigen::Vector3d across_axis_w = fingercenter_cur_pos_w - handcenter_cur_pos_w; 
            across_axis_w = across_axis_w / across_axis_w.norm();

            // update object and wrist rotation and position
            raisim::Mat<3,3> wrist_rot_w;
            mano_->getFrameOrientation(body_parts_[0], wrist_rot_w);
            raisim::Vec<3> wrist_pos_w;
            mano_->getFramePosition(body_parts_[0], wrist_pos_w);
            obj_->update_object_state();

            // get wrist init rotation in object frame
            // because we have set that the init orientation is the same as target orientation
            raisim::Mat<3,3> obj_t_rot_w_trans;
            raisim::transpose(obj_->obj_t_cur_rot_, obj_t_rot_w_trans);
            raisim::matmul(obj_t_rot_w_trans, wrist_rot_w, wrist_target_rot_o_);

            // 
            handcenter_tar_pos_o_ = ee_goal_pos.cast<double>();

            if (g_->visualizable_){
                vis_hand_center_sphere_->setPosition(handcenter_cur_pos_w);
                g_->set_coordinate_vis(vis_wrist_center_array_, wrist_pos_w.e(), wrist_rot_w);
                
                raisim::Vec<3> handcenter_tar_pos_w = get_target_hand_center_w();
                vis_target_center_sphere_->setPosition(handcenter_tar_pos_w.e());

                g_->set_coordinate_vis(vis_init_root_array_, handcenter_tar_pos_w.e(), mano_base_init_rot_w_);

                // normalize the direction vector
                //raisim::Vec<3> end = (handcenter_tar_pos_o_ + 0.2*dir_r);
                //set_cylinder_vis(vis_random_dir_dis_, handcenter_tar_pos_o_, end, 0.01);
            }

            Eigen::Vector3d direction_wrist_to_handcenter = handcenter_cur_pos_w - wrist_pos_w.e();
            handcenter_pos_wrist_ = wrist_rot_w.e().transpose() * direction_wrist_to_handcenter;

            Eigen::Vector3d hand_direction_w = direction_wrist_to_handcenter / direction_wrist_to_handcenter.norm();
            grasp_axis_o_ = obj_->obj_t_cur_rot_.e().transpose() * hand_direction_w;
            across_axis_o_ = obj_->obj_t_cur_rot_.e().transpose() * across_axis_w;
            across_axis_wrist_ = wrist_rot_w.e().transpose() * across_axis_w;
        }

        void step(const Eigen::Ref<EigenVec>& action) {
            
            raisim::Vec<3> wrist_pos_w;
            raisim::Mat<3,3> wrist_rot_w;
            mano_->getFrameOrientation(body_parts_[0], wrist_rot_w);
            mano_->getFramePosition(body_parts_[0], wrist_pos_w);
            
            // calculate bias of current and target handcenter position in wrist frame and set to action mean
            raisim::Vec<3> handcenter_tar_pos_w = get_target_hand_center_w();
            Eigen::Vector3d handcenter_cur_pos_w = wrist_rot_w.e() * handcenter_pos_wrist_ + wrist_pos_w.e();
            Eigen::Vector3d handcenter_pos_bias_w;
            handcenter_pos_bias_w.setZero();
            #if 0 // do not calculate bias !!! for small object !!!!
            handcenter_pos_bias_w = handcenter_tar_pos_w.e() - handcenter_cur_pos_w;
            handcenter_pos_bias_w[2] += mano_->getTotalMass() * 9.81 * 0.01;    // add the 0.01G in bias
            #endif
            if (g_->lift_){
                handcenter_pos_bias_w[2] = 0.35;
            }
            Eigen::Vector3d wrist_bias = mano_base_init_rot_w_trans_.e() * handcenter_pos_bias_w;
            actionMean_.head(3) += wrist_bias;

            if (g_->visualizable_){
                g_->set_coordinate_vis(vis_wrist_center_array_, wrist_pos_w.e(), wrist_rot_w);
                vis_target_center_sphere_->setPosition(handcenter_tar_pos_w.e());
                vis_hand_center_sphere_->setPosition(handcenter_cur_pos_w);
            }

            #if 0 // do not calculate bias !!! for small object !!!!
            // calculate bias of current and target handcenter orientation in current wrist frame and set to action mean
            raisim::Mat<3,3> wrist_rot_w_trans, wrist_target_rot_w, wrist_target_rot_wrist;
            raisim::Vec<3> wrist_target_euler;
            raisim::matmul(obj_->obj_t_cur_rot_, wrist_target_rot_o_, wrist_target_rot_w);
            raisim::transpose(wrist_rot_w, wrist_rot_w_trans);
            raisim::matmul(wrist_rot_w_trans, wrist_target_rot_w, wrist_target_rot_wrist);
            raisim::RotmatToEuler(wrist_target_rot_wrist, wrist_target_euler);
            actionMean_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
            #endif

            /// Compute position target for actuators
            pTarget_ = action.cast<double>();
            pTarget_ = pTarget_.cwiseProduct(actionStd_); //residual action * scaling
            pTarget_ += actionMean_; //add wrist bias (first 3DOF) and last pose (23DoF)

            /// Clip targets to limits
            pTarget_clipped_ = pTarget_.cwiseMax(joint_limit_low_).cwiseMin(joint_limit_high_);
            mano_->setPdTarget(pTarget_clipped_, vTarget_);
        }

        float step_calculate_reward(raisim::Reward *rewards) {
            double affordance_contact_reward, not_affordance_contact_reward, affordance_impulse_reward, not_affordance_impulse_reward, wrist_vel_reward, wrist_qvel_reward, obj_vel_reward, obj_qvel_reward, target_grasp_center_reward, target_grasp_direction_reward, table_contact_reward, table_impulse_reward;

            actionMean_ = gc_;

            obj_vel_reward = obj_->obj_t_cur_linevel_.e().squaredNorm();
            obj_qvel_reward = obj_->obj_t_cur_anglevel_.e().squaredNorm();

            affordance_contact_reward = contacts_af_.cwiseProduct(finger_weights_contact_).sum() / num_contacts_;
            if (obj_->has_non_aff_) {
                not_affordance_contact_reward = contacts_non_af_.cwiseProduct(finger_weights_contact_).sum() / num_contacts_;
            } else {
                not_affordance_contact_reward = 0.;
            }
            table_contact_reward = contacts_table_.cwiseProduct(finger_weights_contact_).sum() / num_contacts_;

            Eigen::VectorXd impulses_af_clipped = Eigen::VectorXd::Zero(num_contacts_);
            Eigen::VectorXd impulses_non_af_clipped = Eigen::VectorXd::Zero(num_contacts_);
            Eigen::VectorXd impulses_table_clipped = Eigen::VectorXd::Zero(num_contacts_);
            impulses_af_clipped = impulses_af_.cwiseMax(impulse_low_).cwiseMin(impulse_high_);
            impulses_non_af_clipped = impulses_non_af_.cwiseMax(impulse_low_).cwiseMin(impulse_high_);
            impulses_table_clipped = impulses_table_.cwiseMax(impulse_low_).cwiseMin(impulse_high_);
            affordance_impulse_reward = impulses_af_clipped.cwiseProduct(finger_weights_contact_).sum();
            if (obj_->has_non_aff_){
                not_affordance_impulse_reward = impulses_non_af_clipped.cwiseProduct(finger_weights_contact_).sum();
            } else {
                not_affordance_impulse_reward = 0.;
            }
            table_impulse_reward = impulses_table_clipped.cwiseProduct(finger_weights_contact_).sum();

            wrist_vel_reward = wrist_vel_in_wrist_.squaredNorm();
            wrist_qvel_reward = wrist_qvel_in_wrist_.squaredNorm();
            target_grasp_center_reward = handcenter_pos_bias_wrist_.squaredNorm();
            target_grasp_direction_reward = handcenter_rot_bias_wrist_.squaredNorm();

            rewards->record("affordance_contact_reward", std::max(0.0, affordance_contact_reward));
            rewards->record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward));
            rewards->record("table_contact_reward", std::max(0.0, table_contact_reward));
            rewards->record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward));
            rewards->record("affordance_impulse_reward", std::min(obj_->obj_weight_ * 5, affordance_impulse_reward));
            rewards->record("table_impulse_reward", std::max(0.0, table_impulse_reward));
            rewards->record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward));
            rewards->record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward));
            rewards->record("obj_vel_reward_", std::max(0.0, obj_vel_reward));
            rewards->record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward));
            rewards->record("torque", std::max(0.0, (hand_torque_.squaredNorm() + 4 * wrist_torque_.squaredNorm())));

            return rewards->sum();
        }

        void update_observation() {
            raisim::Mat<3,3> wrist_rot_w, obj_t_rot_w_trans;
            raisim::Vec<3> wrist_vel_w, wrist_qvel_w;

            impulses_af_.setZero(); contacts_af_.setZero(); impulses_non_af_.setZero(); contacts_non_af_.setZero(); contacts_table_.setZero(); impulses_table_.setZero();

            mano_->getState(gc_, gv_);
            mano_->getFrameOrientation(body_parts_[0], wrist_rot_w);
            mano_->getFrameVelocity(body_parts_[0], wrist_vel_w);
            mano_->getFrameAngularVelocity(body_parts_[0], wrist_qvel_w);

            raisim::transpose(obj_->obj_t_cur_rot_, obj_t_rot_w_trans);

            obj_rotation_axis_in_wrist_ = wrist_rot_w.e().transpose() * obj_->obj_arti_axis_cur_rot_;

            // wrist / object angular velocity / velocity in wrist frame
            wrist_vel_in_wrist_ = wrist_rot_w.e().transpose() * wrist_vel_w.e();
            wrist_qvel_in_wrist_ = wrist_rot_w.e().transpose() * wrist_qvel_w.e();
            Eigen::Vector3d obj_vel_wrist = wrist_rot_w.e().transpose() * obj_->obj_t_cur_linevel_.e() - wrist_vel_in_wrist_; 
            Eigen::Vector3d obj_qvel_wrist = wrist_rot_w.e().transpose() * obj_->obj_t_cur_anglevel_.e() - wrist_qvel_in_wrist_;

            /// compute current contacts of hand parts and the contact force
            auto& contact_list_obj = obj_->arctic_->getContacts();
            for(auto& contact_af: mano_->getContacts()) {
                if (contact_af.skip() || contact_af.getPairObjectIndex() != obj_->arctic_->getIndexInWorld()) continue;
                if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != obj_->arctic_->getBodyIdx("top")) continue;
                contacts_af_[contactMapping_[contact_af.getlocalBodyIndex()]] = 1;
                impulses_af_[contactMapping_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
            }

            for(auto& contact_non_af: mano_->getContacts()) {
                if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != obj_->arctic_->getIndexInWorld()) continue;
                if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != obj_->arctic_->getBodyIdx("bottom")) continue;
                contacts_non_af_[contactMapping_[contact_non_af.getlocalBodyIndex()]] = 1;
                impulses_non_af_[contactMapping_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
            }

            for(auto& contact_table: mano_->getContacts()) {
                if (contact_table.skip() || contact_table.getPairObjectIndex() != obj_->box_->getIndexInWorld()) continue;
                contacts_table_[contactMapping_[contact_table.getlocalBodyIndex()]] = 1;
                impulses_table_[contactMapping_[contact_table.getlocalBodyIndex()]] = contact_table.getImpulse().norm();
            }

            for(int i=0; i<num_contacts_; i++) {
                if (contacts_non_af_[i] == 1) {
                    contacts_non_af_[i] = contacts_non_af_[i] - contacts_af_[i];
                    impulses_non_af_[i] = impulses_non_af_[i] - impulses_af_[i];
                }
            }

            if (obj_->has_non_aff_) {
            } else {
                contacts_non_af_.setZero();
                impulses_non_af_.setZero();
            }

            hand_torque_ = (pTarget_clipped_ - gc_);

            // get the bias of wrist current and target position in wrist frame 
            raisim::Vec<3> afford_center_w, wrist_pos_w;
            raisim::Vec<3> handcenter_tar_pos_w = get_target_hand_center_w();
            mano_->getFramePosition(body_parts_[0], wrist_pos_w);
            Eigen::Vector3d handcenter_pos_bias_w = handcenter_tar_pos_w.e() - wrist_pos_w.e();
            handcenter_pos_bias_wrist_ = wrist_rot_w.e().transpose() * handcenter_pos_bias_w - handcenter_pos_wrist_;

            // get the object top position in wrist frame
            raisim::Mat<3,3> obj_t_rot_wrist, wrist_rot_w_trans;
            raisim::Vec<3> obj_t_pos_wrist;
            raisim::transpose(wrist_rot_w, wrist_rot_w_trans);
            raisim::matmul(wrist_rot_w_trans, obj_->obj_t_cur_rot_, obj_t_rot_wrist);
            raisim::RotmatToEuler(obj_t_rot_wrist, obj_t_pos_wrist);

            raisim::Vec<3> frame_y_frame, joint_pos_w, frame_y_w, frame_y_o, joint_pos_o, joint_pos_o_tmp;
            frame_y_frame.setZero();
            frame_y_frame[1] = -1;
            for(int i = 0; i < num_bodies_ ; i++){
                mano_->getFramePosition(body_parts_[i], joint_pos_w);
                raisim::matvecmul(wrist_rot_w, frame_y_frame, frame_y_w);
                raisim::matvecmul(obj_t_rot_w_trans, frame_y_w, frame_y_o);
                joint_pos_in_world_.segment(i*3, 3) = joint_pos_w.e();
                joint_pos_o_tmp = joint_pos_w - obj_->obj_t_cur_pos_;
                raisim::matvecmul(obj_t_rot_w_trans, joint_pos_o_tmp, joint_pos_o);
                frame_y_in_obj_.segment(i*3, 3) = frame_y_o.e();
                joint_pos_in_obj_.segment(i*3, 3) = joint_pos_o.e();

                joint_height_w_[i] = joint_pos_w[2] - 0.5;
            }

            obDouble_ << handcenter_pos_bias_wrist_,           // 3, hand center diff
                        gc_.tail(gcDim_ - 3),      // (mirror) 45, generalized coordinate
                            hand_torque_,
//                            obj_pos_wrist,
                            wrist_vel_in_wrist_,
                            wrist_qvel_in_wrist_,
                            contacts_af_,
                            impulses_af_,
                            joint_height_w_;
            
            obs_history_.push_back(obDouble_);

            Eigen::Vector3d current_across_axis_w = wrist_rot_w.e() * across_axis_wrist_;
            Eigen::Vector3d current_grasp_axis_w = wrist_rot_w.e() * handcenter_pos_wrist_;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();
            Eigen::Vector3d current_grasp_axis_o = obj_->obj_t_cur_rot_.e().transpose() * current_grasp_axis_w;
            Eigen::Vector3d current_across_axis_o = obj_->obj_t_cur_rot_.e().transpose() * current_across_axis_w;

            float across_mul = current_across_axis_o.dot(across_axis_o_);

            raisim::Mat<3,3> wrist_target_rot_w, wrist_target_rot_wrist;
            raisim::Vec<3> wrist_target_euler;
            raisim::matmul(obj_->obj_t_cur_rot_, wrist_target_rot_o_, wrist_target_rot_w);
            raisim::matmul(mano_base_init_rot_w_trans_, wrist_target_rot_w, wrist_target_rot_wrist);
            raisim::RotmatToEuler(wrist_target_rot_wrist, wrist_target_euler);

            global_state_ << obj_t_pos_wrist.e(),
                            frame_y_in_obj_,
                            joint_pos_in_obj_,
                            obj_->obj_t_cur_pos_.e(),
                            current_grasp_axis_o - grasp_axis_o_,
                            across_mul,
                            gc_.segment(3, 3) - wrist_target_euler.e();
        }

    private:
        raisim::Vec<3> get_target_hand_center_w() {
            raisim::Vec<3> tmp;
            raisim::matvecmul(obj_->obj_t_cur_rot_, handcenter_tar_pos_o_, tmp);
            return tmp + obj_->obj_t_cur_pos_;
        }

    public:
        global_value* g_;
        arti_object* obj_;

        int num_contacts_, num_bodies_, global_state_dim_, num_finger_;
        int gcDim_, gvDim_, nJoints_;

        Eigen::VectorXd obDouble_, global_state_;
        raisim::ArticulatedSystem *mano_;
        std::map<int,int> contactMapping_;
        std::deque<Eigen::VectorXd> obs_history_;
        Eigen::VectorXd ob_delay_, ob_concat_;

        raisim::Mat<3,3> mano_base_init_rot_w_, mano_base_init_rot_w_trans_, wrist_target_rot_o_;
        raisim::Vec<3> mano_base_init_pos_w_;

        raisim::Vec<3> random_direction_vector_;
        double  wrist_Pgain_ = 100.0, wrist_Dgain_ = 0.1, rot_Pgain_ = 100.0, rot_Dgain_ = 0.2;
        Eigen::VectorXd impulse_high_, impulse_low_, finger_weights_contact_, finger_weights_aff_;

        Eigen::VectorXd joint_pos_in_world_, gc_, gv_, pTarget_, vTarget_, gc_set_, gv_set_;
        Eigen::VectorXd joint_limit_high_, joint_limit_low_;
        Eigen::VectorXd actionMean_, actionStd_;
        Eigen::VectorXd contacts_af_, impulses_af_, contacts_non_af_, impulses_non_af_, hand_torque_, wrist_torque_, impulses_table_, contacts_table_;
        Eigen::VectorXd frame_y_in_obj_, joint_pos_in_obj_;
        Eigen::VectorXd pTarget_clipped_;
        Eigen::VectorXd joint_height_w_;

        Eigen::Vector3d obj_rotation_axis_in_wrist_;
        Eigen::Vector3d wrist_vel_in_wrist_, wrist_qvel_in_wrist_, across_axis_wrist_;
        Eigen::Vector3d handcenter_pos_wrist_, handcenter_tar_pos_o_;
        Eigen::Vector3d grasp_axis_o_, across_axis_o_;
        Eigen::Vector3d handcenter_pos_bias_wrist_, handcenter_rot_bias_wrist_;

        std::vector<std::string> body_parts_;
        std::vector<std::string> contact_bodies_;
        std::vector<raisim::Visuals *> vis_joint_mesh_dis_;
        std::vector<raisim::Visuals *> vis_mesh_sphere_;
        std::vector<raisim::Visuals *> vis_joint_sphere_;
        raisim::Visuals *vis_target_center_sphere_;
        raisim::Visuals *vis_hand_center_sphere_;
        raisim::Visuals *vis_wrist_center_array_[3];
        raisim::Visuals *vis_init_root_array_[3];
        raisim::Visuals *vis_random_dir_dis_;
    };

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg) {

            /// create world_
            world_ = std::make_unique<raisim::World>();

            g_state_.init(world_, resourceDir, cfg, visualizable);
            obj_state_.init(&g_state_, cfg);
            right_.init(world_, &rewards_r_, cfg, &g_state_, &obj_state_, "right");

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_r_ = 1; obDim_l_ = 1;
            actionDim_ = cfg["actiondim"].As<int>();
            gsDim_ = cfg["gsdim"].As<int>();
            int step_obs_dim = cfg["step_obs_dim"].As<int>();
            if USE_RIGHT_HAND(g_state_.hand_type_) {
                obDim_r_ = g_state_.encode_dim_ * g_state_.history_len_ + step_obs_dim;
                right_.obDouble_.setZero(step_obs_dim);
                right_.global_state_.setZero(gsDim_);
                right_.ob_delay_.setZero(step_obs_dim);
                right_.ob_concat_.setZero(obDim_r_);
            }
            if USE_LEFT_HAND(g_state_.hand_type_) {
            }

            /// start visualization server
            if (visualizable) {
                if(server_) server_->lockVisualizationServerMutex();
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                right_.init_vis(server_);
                if(server_) server_->unlockVisualizationServerMutex();
            }
        }

        void init() final { }
        void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}

        /// This function loads the object into the environment
        void load_articulated(const std::string& obj_model){
            obj_state_.load(world_, server_, obj_model);
        }

        void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
            right_.joint_distance_vis(joint_sensor_visual);
        }

        void set_joint_sensor_visual_l(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
        }

        void debugShowObs(const Eigen::Ref<EigenVec>& ob_r) final {
            // rot_r, pos_r, bias_r, dir_r, rot_l, pos_l, bias_l, random_direction_vector_
            //dir_r = ob_r.segment(0,3).cast<double>();
            //random_direction_vector_ = ob_r.segment(3,3).cast<double>();
        }

        /// Resets the object and hand to its initial pose
        void reset() final {
            g_state_.lift_ = false;
        
            if (g_state_.first_reset_) {
                g_state_.first_reset_ = false;
                return;
            }
            if USE_RIGHT_HAND(g_state_.hand_type_) {
                right_.reset_init_state();
            }
            if USE_LEFT_HAND(g_state_.hand_type_) {
            }

            obj_state_.reset_init_state();
            updateObservation();
        }

        /// Resets the state to a user defined input
        // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
        // init_state_l in right-hand coord
        void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                         const Eigen::Ref<EigenVec>& init_state_l,
                         const Eigen::Ref<EigenVec>& init_vel_r,
                         const Eigen::Ref<EigenVec>& init_vel_l,
                         const Eigen::Ref<EigenVec>& obj_pose) final {
            
            g_state_.lift_ = false;
            obj_state_.reset_user_state(obj_pose);
            
            /// set initial hand pose (20 DoF) and velocity (20 DoF)
            if USE_RIGHT_HAND(g_state_.hand_type_) {
                right_.reset_user_state(init_state_r, init_vel_r);
            }
            if USE_LEFT_HAND(g_state_.hand_type_) {
            }

            updateObservation();
        }

        void set_goals(const Eigen::Ref<EigenVec>& target_center,
                       const Eigen::Ref<EigenVec>& obj_center,
                       const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                       const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                       const Eigen::Ref<EigenVec>& goal_pose_r,
                       const Eigen::Ref<EigenVec>& goal_pose_l,
                       const Eigen::Ref<EigenVec>& goal_qpos_r,
                       const Eigen::Ref<EigenVec>& goal_qpos_l,
                       const Eigen::Ref<EigenVec>& goal_contacts_r,
                       const Eigen::Ref<EigenVec>& goal_contacts_l) final {
            if USE_RIGHT_HAND(g_state_.hand_type_) {
                right_.set_goals(target_center);
            }
            if USE_LEFT_HAND(g_state_.hand_type_) {     // only left hand or two hands
            }
        }

        /// This function takes an environment step given an action (26DoF) input
        // action_l in left-hand coord
        float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

            obj_state_.update_object_state();
            if USE_RIGHT_HAND(g_state_.hand_type_) {  
                right_.step(action_r);
            }
            if USE_LEFT_HAND(g_state_.hand_type_) {
            }

            /// Apply N control steps
            for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }
            
            updateObservation();

            if USE_RIGHT_HAND(g_state_.hand_type_) {
               g_state_.rewards_sum_[0] = right_.step_calculate_reward(&rewards_r_);
            }
            if USE_LEFT_HAND(g_state_.hand_type_) {
            }

            return g_state_.rewards_sum_;
        }

        /// This function computes and updates the observation/state space
        void updateObservation() {
            obj_state_.update_observation();

            if USE_RIGHT_HAND(g_state_.hand_type_) {
                right_.update_observation();
            }
            if USE_LEFT_HAND(g_state_.hand_type_) {
            }
        }

        /// Set observation in wrapper to current observation
        void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
            int lag = 1;
            int vec_size = right_.obs_history_.size();
            right_.ob_delay_ << right_.obs_history_[vec_size - lag];
            // ly 初始化：当第一次observe，即step=0时只能获得一个观测，则先填充与第一个观测一样的（history_len-1）个obs
            if (vec_size == 1)  //
            {
                for (int i = 1; i < g_state_.history_len_; i++)
                {
                    right_.obs_history_.push_back(right_.obDouble_);
                }
            }
            vec_size = right_.obs_history_.size();
            for (int i = 0; i < g_state_.history_len_; i++)
            {
                right_.ob_concat_.segment(g_state_.encode_dim_ * i, g_state_.encode_dim_) << right_.obs_history_[vec_size - g_state_.history_len_ + i].head(g_state_.encode_dim_);
            }
            right_.ob_concat_.tail(g_state_.step_obs_dim_) << right_.ob_delay_;
            ob_r = right_.ob_concat_.cast<float>();
            Eigen::VectorXd obsl; obsl.setZero(1);
            ob_l = obsl.cast<float>();
        }

        void get_global_state(Eigen::Ref<EigenVec> gs) {
            gs = right_.global_state_.cast<float>();
        }

        void get_global_state_l(Eigen::Ref<EigenVec> gs) {
        }

        void set_rootguidance() final {}
        void switch_root_guidance(bool is_on) {
            g_state_.lift_ = true;
        }
        /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
        bool isTerminalState(float& terminalReward) final {
            obj_state_.update_object_state();
            if(g_state_.lift_){}
            else{
                double displacement = (obj_state_.obj_t_cur_pos_.e() - obj_state_.obj_init_gc_.head(3)).squaredNorm();
                if (displacement > 0.1){
                    terminalReward = -10;
                    return true;
                }
            }

            for(int i = 0; i < right_.num_bodies_ ; i++){
                if (right_.joint_height_w_[i] < 0.){
                    return true;
                }
            }

            if(right_.obDouble_.hasNaN() || right_.global_state_.hasNaN())
            {
                std::cout<<"NaN detected"<< right_.obDouble_.transpose()<<std::endl;
                std::cout<<"NaN detected"<< right_.global_state_.transpose()<<std::endl;
                return true;
            }

            return false;
        }

    private:
        global_value g_state_;
        arti_object obj_state_;
        hand right_;
    };
}
