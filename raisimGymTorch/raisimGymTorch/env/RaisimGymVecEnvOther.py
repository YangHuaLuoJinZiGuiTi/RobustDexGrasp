# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//
import faulthandler;

# from fontTools.merge.util import current_time

faulthandler.enable()

import numpy as np
import platform
import os
from scipy.spatial.transform import Rotation as R
import torch
import trimesh
from raisimGymTorch.helper.qp_solver import *

from raisimGymTorch.helper import rotations
class RaisimGymVecEnvTest:

    def __init__(self, obj_list, impl, cfg, normalize_ob=False, seed=0, normalize_rew=True, clip_obs=10., cat_name=None, cent_training=False, two_hand=False):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'

        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs_r = self.wrapper.getRightObDim()
        self.num_obs_l = self.wrapper.getLeftObDim()
        self.num_acts = self.wrapper.getActionDim()
        self.num_gs = self.wrapper.getGSDim()
        self._observation_r = np.zeros([self.num_envs, self.num_obs_r], dtype=np.float32)
        self._observation_l = np.zeros([self.num_envs, self.num_obs_l], dtype=np.float32)
        self._global_state = np.zeros([self.num_envs, self.num_gs], dtype=np.float32)
        self._global_state_l = np.zeros([self.num_envs, self.num_gs], dtype=np.float32)
        self.obs_rms_r = RunningMeanStd(shape=[self.num_envs, self.num_obs_r])
        self.obs_rms_l = RunningMeanStd(shape=[self.num_envs, self.num_obs_l])
        self.gs_rms = RunningMeanStd(shape=[self.num_envs, self.num_gs])
        self._reward_r = np.zeros(self.num_envs, dtype=np.float32)
        self._reward_l = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.jacobians_list = []
        self.dof = 22

        # contact info
        self.max_contacts = 13
        self.points = np.zeros((self.num_envs, self.max_contacts * 3), dtype=np.float64, order='F')
        self.normals = np.zeros((self.num_envs, self.max_contacts * 3), dtype=np.float64, order='F')
        self.forces = np.zeros((self.num_envs, self.max_contacts * 3), dtype=np.float64, order='F')
        self.normal_forces = np.zeros((self.num_envs, self.max_contacts * 3), dtype=np.float64, order='F')
        self.contact_ids = np.full((self.num_envs, self.max_contacts), -1, dtype=np.int32, order='F') 
        self.contact_counts = np.zeros(self.num_envs, dtype=np.int32, order='F')
        self.contact_info_list = [{} for _ in range(self.num_envs)]     # num_envs * contact_info directory

        
        # for QP computation
        self.solver = QP_Solver()

        # object info
        self.object_info_list = [{} for _ in range(self.num_envs)]     # num_envs * object_info directory
        
        # cost
        self._force_closure_cost = np.zeros(self.num_envs, dtype=np.float32)
        self._friction_cone_cost = np.zeros(self.num_envs, dtype=np.float32)
        self.num_constraints = int(cfg['p3o']['num_of_costs'])
                
        self.affordance_pcd = np.zeros([len(obj_list),200,3])
        self.affordance_normals = np.zeros([len(obj_list),200,3])
        self.aff_mesh = [None] * len(obj_list)
        self.non_affordance_pcd = np.zeros([len(obj_list),200,3])
        self.non_affordance_normals = np.zeros([len(obj_list),200,3])
        self.non_aff_mesh = [None] * len(obj_list)
        self.affordance_center = np.zeros((len(obj_list),3), 'float32')
        self.non_affordance_center = np.zeros((len(obj_list),3), 'float32')

        for obj_name in np.unique(obj_list):
            if two_hand == False:
                aff_mesh_name = f'../rsc/{cat_name}/{obj_name}/top_watertight_tiny.obj'
                non_aff_mesh_name = f'../rsc/{cat_name}/{obj_name}/bottom_watertight_tiny.obj'
            else:
                aff_mesh_name = f'../rsc/{cat_name}/{obj_name}/bottom_watertight_tiny.obj'
                non_aff_mesh_name = f'../rsc/{cat_name}/{obj_name}/top_watertight_tiny.obj'
            aff_mesh = trimesh.load_mesh(aff_mesh_name)
            aff_points, aff_face_id = trimesh.sample.sample_surface(aff_mesh, 200)
            aff_normals = aff_mesh.face_normals[aff_face_id]

            aff_center = aff_mesh.centroid

            non_aff_mesh = trimesh.load_mesh(non_aff_mesh_name)
            non_aff_points, non_aff_face_id = trimesh.sample.sample_surface(non_aff_mesh, 200)
            non_aff_normals = non_aff_mesh.face_normals[non_aff_face_id]

            non_aff_center = non_aff_mesh.centroid
            if non_aff_mesh.vertices.shape[0] < 25:
                not_two_parts = True
            else:
                not_two_parts = False

            for i in range(len(obj_list)):
                if obj_list[i] == obj_name:
                    self.affordance_pcd[i] = aff_points
                    self.affordance_normals[i] = aff_normals
                    self.aff_mesh[i] = aff_mesh
                    self.non_affordance_pcd[i] = non_aff_points
                    self.non_affordance_normals[i] = non_aff_normals
                    self.non_aff_mesh[i] = non_aff_mesh
                    self.affordance_center[i] = aff_center
                    if not not_two_parts:
                        self.non_affordance_center[i] = non_aff_center
                    else:
                        self.non_affordance_center[i, :] = 100

        self.affordance_pcd = torch.tensor(self.affordance_pcd).float().to('cuda')
        self.affordance_normals = torch.tensor(self.affordance_normals).float().to('cuda')
        self.non_affordance_pcd = torch.tensor(self.non_affordance_pcd).float().to('cuda')
        self.non_affordance_normals = torch.tensor(self.non_affordance_normals).float().to('cuda')


    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def set_pd_wrist(self):
        self.wrapper.set_pd_wrist()

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)
        
    

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action_r, action_l):
        self.wrapper.step(action_r, action_l, self._reward_r, self._reward_l, self._done)
        return self._reward_r.copy(), self._reward_l.copy(), self._done.copy()

    def step2(self, action_r, action_l):
        self.wrapper.step2(action_r, action_l, self._reward_r, self._reward_l, self._done)
        return self._reward_r.copy(), self._reward_l.copy(), self._done.copy()

    def reset_right_hand(self, obj_pose_step_r, hand_ee_step_r, hand_pose_step_r):
        self.wrapper.reset_right_hand(obj_pose_step_r, hand_ee_step_r, hand_pose_step_r)

    def step_imitate(self, action_r, action_l, obj_pose_r, hand_ee_r, hand_pose_r, obj_pose_l, hand_ee_l, hand_pose_l, imitate_right, imitate_left):
        self.wrapper.step_imitate(action_r, action_l, obj_pose_r, hand_ee_r, hand_pose_r, obj_pose_l, hand_ee_l, hand_pose_l, imitate_right, imitate_left, self._reward_r, self._reward_l, self._done)
        return self._reward_r.copy(), self._reward_l.copy(), self._done.copy()
    
    def get_force_closure_costs(self):
        """获取力闭合成本"""
        self.wrapper.getForceClosureCosts(self._force_closure_cost)
        return self._force_closure_cost.copy()

    def get_friction_cone_costs(self):
        """获取摩擦锥成本"""
        self.wrapper.getFrictionConeCosts(self._friction_cone_cost)
        return self._friction_cone_cost.copy()
    
    def get_cost_info_r(self):
        # print(">>> force_closure_cost: ", self.get_force_closure_costs())
        # print(">>> friction_cone_cost: ", self.get_friction_cone_costs())
        
        force_closure_costs = self.get_force_closure_costs()
        friction_cone_costs = self.get_friction_cone_costs()
        
        # 组合成costs数组
        costs = np.stack([force_closure_costs, friction_cone_costs], axis=1)[:, :self.num_constraints]
        assert costs.shape == (self.num_envs, self.num_constraints)
        return costs
    
    def set_random_obj_prior_info(self, random_mass, random_mu):
        """Set random object properties for all environments"""
        # self.wrapper.set_random_obj_prior_info(mass_lower_limit, mass_upper_limit, mu_lower_limit, mu_upper_limit)
        self.wrapper.set_random_obj_prior_info(random_mass, random_mu)

    def load_scaling(self, dir_name, iteration, count=1e5, cent_training=False):
        mean_file_name_r = dir_name + "/mean_r" + str(iteration) + ".csv"
        var_file_name_r = dir_name + "/var_r" + str(iteration) + ".csv"
        mean_file_name_l = dir_name + "/mean_l" + str(iteration) + ".csv"
        var_file_name_l = dir_name + "/var_l" + str(iteration) + ".csv"
        if cent_training:
            mean_file_name_g = dir_name + "/mean_g" + str(iteration) + ".csv"
            var_file_name_g = dir_name + "/var_g" + str(iteration) + ".csv"
        self.obs_rms_r.count = count
        self.obs_rms_r.mean = np.loadtxt(mean_file_name_r, dtype=np.float32)
        self.obs_rms_r.var = np.loadtxt(var_file_name_r, dtype=np.float32)
        if os.path.exists(mean_file_name_l) and os.path.exists(var_file_name_l):
            self.obs_rms_l.count = count
            self.obs_rms_l.mean = np.loadtxt(mean_file_name_l, dtype=np.float32)
            self.obs_rms_l.var = np.loadtxt(var_file_name_l, dtype=np.float32)
        if cent_training:
            self.gs_rms.count = count
            self.gs_rms.mean = np.loadtxt(mean_file_name_g, dtype=np.float32)
            self.gs_rms.var = np.loadtxt(var_file_name_g, dtype=np.float32)

    def save_scaling(self, dir_name, iteration):
        mean_file_name_r = dir_name + "/mean_r" + iteration + ".csv"
        var_file_name_r = dir_name + "/var_r" + iteration + ".csv"
        mean_file_name_l = dir_name + "/mean_l" + iteration + ".csv"
        var_file_name_l = dir_name + "/var_l" + iteration + ".csv"
        np.savetxt(mean_file_name_r, self.obs_rms_r.mean)
        np.savetxt(var_file_name_r, self.obs_rms_r.var)
        np.savetxt(mean_file_name_l, self.obs_rms_l.mean)
        np.savetxt(var_file_name_l, self.obs_rms_l.var)   

    def euler_to_rotation_matrix(self, euler_angles):
        """Convert Euler angles to rotation matrices."""
        batch_size = euler_angles.shape[0]
        c1 = torch.cos(euler_angles[:, 0])
        s1 = torch.sin(euler_angles[:, 0])
        c2 = torch.cos(euler_angles[:, 1])
        s2 = torch.sin(euler_angles[:, 1])
        c3 = torch.cos(euler_angles[:, 2])
        s3 = torch.sin(euler_angles[:, 2])

        rotation_matrices = torch.zeros((batch_size, 3, 3), device=euler_angles.device)
        rotation_matrices[:, 0, 0] = c2 * c3
        rotation_matrices[:, 0, 1] = -c2 * s3
        rotation_matrices[:, 0, 2] = s2
        rotation_matrices[:, 1, 0] = c1 * s3 + c3 * s1 * s2
        rotation_matrices[:, 1, 1] = c1 * c3 - s1 * s2 * s3
        rotation_matrices[:, 1, 2] = -c2 * s1
        rotation_matrices[:, 2, 0] = s1 * s3 - c1 * c3 * s2
        rotation_matrices[:, 2, 1] = c3 * s1 + c1 * s2 * s3
        rotation_matrices[:, 2, 2] = c1 * c2

        return rotation_matrices    

    def observe_vision_new_sum(self, obs_dim):
        if obs_dim == 153: # Original RobustDexGrasp
            return self.observe_vision_new()
        
        elif obs_dim == 155: # With object mass and friction
            return self.observe_vision_obj_new()
        
        elif obs_dim == 158: # With object mass, friction, and Center of Mass
            return self.observe_vision_obj_com_new()
        
        elif obs_dim == 159: # Grasp Acc extraction.
            return self.observe_vision_GraspAcc_new()
        
        elif obs_dim == 153 + 13*3 + 1: # QP as reference
            return self.observe_vision_QP_new()
        
        elif obs_dim == 153 + 16 + 1: # Wrench Error + QP + IK error as reference
            return self.observe_vision_QP_IK_new()
        else:
            raise ValueError("Invalid observation dimension")
    
    
    def update_contact_info(self):
       
        jacobian_size_per_contact = 3 * self.dof
        total_jacobian_cols = self.max_contacts * jacobian_size_per_contact
        self.jacobians_flat = np.asfortranarray(np.zeros((self.num_envs, total_jacobian_cols), dtype=np.float64))
        
    
        self.wrapper.get_contact_info(
            self.points, self.normals, self.forces, self.normal_forces,
            self.contact_ids, self.jacobians_flat, self.contact_counts
        )
        
        
    def get_object_info(self):
        self.object_info_list = [{} for _ in range(self.num_envs)]
        obj_weight, obj_mu = self.get_obj_weight().reshape(self.num_envs, -1), self.get_obj_mu().reshape(self.num_envs, -1)
        self.wrapper.get_global_state(self._global_state)
        global_state = self._global_state.copy()
        obj_com = global_state[:, 129:132].reshape(self.num_envs, -1)
        for i in range(self.num_envs):
            # print(type(obj_mu[i]), obj_mu[i].shape)
            self.object_info_list[i] = {
                "object_weight": obj_weight[i],
                "miu_coef": [obj_mu[i][0], 0.02],
                "object_gravity_center" : obj_com[i],
                "env_id": i,
            }
        return self.object_info_list
        
    def get_contact_info(self):
        """
        Update self.contact_info_list: a list of dictionaries, each dictionary contains the following keys:
            contact_info_dict = {
                'env_index': env_index,
                'points': points, # contact points in world frame
                'normals': normals, # contact normals point from object to hand
                'forces': forces, # contact forces from hand to object
                "contact_ids": contact_ids, # contact ids, relates to allegro's affordance id
                "num_contact": count,
            }
        """
        
        self.update_contact_info()
        self.contact_info_list = []
        jacobian_size_per_contact = 3 * self.dof
    
        for env_index in range(self.num_envs):
            count = self.contact_counts[env_index]
            contact_info_dict = {}
            points, normals, forces, contact_ids, normal_forces = [], [], [], [], []
            jacobians = []
            

            for j in range(count):
                point_idx = j * 3
                point = self.points[env_index, point_idx:point_idx+3]
                normal = self.normals[env_index, point_idx:point_idx+3]
                force = self.forces[env_index, point_idx:point_idx+3]
                normal_force = self.normal_forces[env_index, point_idx:point_idx+3]
                contact_id = self.contact_ids[env_index, j]
                points.append(point.copy().tolist())
                normals.append(normal.copy().tolist())
                forces.append(force.copy().tolist())
                normal_forces.append(normal_force.copy().tolist())
                contact_ids.append(contact_id)
                
                jac_start_idx = j * jacobian_size_per_contact
                jac_flat = self.jacobians_flat[env_index, jac_start_idx:jac_start_idx + jacobian_size_per_contact]
                jac_matrix = jac_flat.reshape(3, self.dof)
                jacobians.append(jac_matrix)
            
                
            contact_info_dict = {
                'env_index': env_index,
                'points': points, # contact points in world frame
                'normals': np.array(normals), # contact normals point from object to hand
                'forces': np.array(forces), # contact forces from hand to object
                'normal_forces': normal_forces,
                "contact_ids": contact_ids, # contact ids, relates to allegro's affordance id
                "num_contact": count,
                "env_jacobians": np.array(jacobians),
            }
            # ==============================================================================================================
            # print("PYTHON:\n normal_forces:\n",np.array(contact_info_dict['normal_forces']))
            # print("forces:\n",np.array(contact_info_dict['forces']))
            # print("normals:\n",np.array(contact_info_dict['normals']))
            # print(repr(contact_info_dict))
            # print(repr(self.get_object_info()))
            # print(f"Env id in {contact_info_dict['env_index']} Jacobians in Python {np.array(contact_info_dict['env_jacobians']).shape}:\n")
            # print(f"Jacobians in Python {np.array(contact_info_dict['env_jacobians']).shape}:\n")
            # print(np.array(contact_info_dict['forces']).shape)
            
            # ============================================================================================================
            # NOTE: Check Jacobian via contact forces
            # if contact_info_dict['num_contact'] > 0:
            #     tau = np.zeros(22)
            #     for j in range(contact_info_dict['num_contact']):
            #         tau += np.dot(contact_info_dict['env_jacobians'][j].T, contact_info_dict['forces'][j])
            #         # print("Caculate tau: J^T * F :\n", tau)
                    # print("Jacobian:\n", contact_info_dict['env_jacobians'][j])
                # print("Force:\n", contact_info_dict['forces'])
                # print("contact_ids:\n", contact_info_dict['contact_ids'])
                # print("[Computed] tau:\n", tau)
                # print("--------")
            # ============================================================================================================
            
            
            self.contact_info_list.append(contact_info_dict)
            
        return self.contact_info_list
    
    
    def observe_vision_GraspAcc_new(self):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy() 

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 54:105].reshape(num_envs, -1, 3)).to('cuda')

        af_dists = torch.cdist(joints, self.affordance_pcd)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(self.affordance_pcd, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        obj_euler_wrist = torch.from_numpy(global_state[:, :3]).to('cuda')
        obj_euler_world = torch.from_numpy(global_state[:, 118:121]).to('cuda')

        r_obj = self.euler_to_rotation_matrix(obj_euler_world).unsqueeze(1).repeat(1, joints.shape[1], 1, 1).to('cuda')
        
        af_vec_rotated = torch.matmul(r_obj, af_vec.reshape(num_envs, -1, 3).to('cuda').unsqueeze(-1)).squeeze(-1)
        af_vec = af_vec_rotated.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        # obj_weight, obj_mu = self.get_obj_weight().reshape(num_envs, -1), self.get_obj_mu().reshape(num_envs, -1)
        # obj_com = global_state[:, 129:132].reshape(num_envs, -1)
        grasp_acc = global_state[:, 132:132+6].reshape(num_envs, -1)
        # print('>>> Grasp Acc: ', grasp_acc)
        
        obs_r = np.concatenate([obs_r, af_vec, grasp_acc], axis=-1)

        show_af_point = af_points.reshape(-1, 3).cpu().numpy().reshape(num_envs, -1).astype('float32')
        dis_info = np.concatenate([min_dis_af.cpu().numpy(), show_af_point], axis=-1)
        return obs_r, dis_info
    
    def observe_vision_obj_com_new(self):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy()

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 54:105].reshape(num_envs, -1, 3)).to('cuda')

        af_dists = torch.cdist(joints, self.affordance_pcd)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(self.affordance_pcd, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        obj_euler_wrist = torch.from_numpy(global_state[:, :3]).to('cuda')
        obj_euler_world = torch.from_numpy(global_state[:, 118:121]).to('cuda')

        r_obj = self.euler_to_rotation_matrix(obj_euler_world).unsqueeze(1).repeat(1, joints.shape[1], 1, 1).to('cuda')
        
        af_vec_rotated = torch.matmul(r_obj, af_vec.reshape(num_envs, -1, 3).to('cuda').unsqueeze(-1)).squeeze(-1)
        af_vec = af_vec_rotated.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        obj_weight, obj_mu = self.get_obj_weight().reshape(num_envs, -1), self.get_obj_mu().reshape(num_envs, -1)
        obj_com = global_state[:, 129:132].reshape(num_envs, -1)

        obs_r = np.concatenate([obs_r, af_vec, obj_weight, obj_mu, obj_com], axis=-1)

        show_af_point = af_points.reshape(-1, 3).cpu().numpy().reshape(num_envs, -1).astype('float32')
        dis_info = np.concatenate([min_dis_af.cpu().numpy(), show_af_point], axis=-1)

        # raise NotImplementedError("Not implemented yet")
        return obs_r, dis_info
        
    

    def observe_vision_obj_new(self):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy()

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 54:105].reshape(num_envs, -1, 3)).to('cuda')

        af_dists = torch.cdist(joints, self.affordance_pcd)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(self.affordance_pcd, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        obj_euler_wrist = torch.from_numpy(global_state[:, :3]).to('cuda')
        obj_euler_world = torch.from_numpy(global_state[:, 118:121]).to('cuda')

        r_obj = self.euler_to_rotation_matrix(obj_euler_world).unsqueeze(1).repeat(1, joints.shape[1], 1, 1).to('cuda')
        af_vec_rotated = torch.matmul(r_obj, af_vec.reshape(num_envs, -1, 3).to('cuda').unsqueeze(-1)).squeeze(-1)
        af_vec = af_vec_rotated.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        obj_weight, obj_mu = self.get_obj_weight().reshape(num_envs, -1), self.get_obj_mu().reshape(num_envs, -1)

        obs_r = np.concatenate([obs_r, af_vec, obj_weight, obj_mu], axis=-1)

        show_af_point = af_points.reshape(-1, 3).cpu().numpy().reshape(num_envs, -1).astype('float32')
        dis_info = np.concatenate([min_dis_af.cpu().numpy(), show_af_point], axis=-1)


        return obs_r, dis_info
    
    def observe_vision_QP_IK_new(self):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy()

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 54:105].reshape(num_envs, -1, 3)).to('cuda')


        af_dists = torch.cdist(joints, self.affordance_pcd)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(self.affordance_pcd, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        obj_euler_wrist = torch.from_numpy(global_state[:, :3]).to('cuda')
        obj_euler_world = torch.from_numpy(global_state[:, 118:121]).to('cuda')

        r_obj = self.euler_to_rotation_matrix(obj_euler_world).unsqueeze(1).repeat(1, joints.shape[1], 1, 1).to('cuda')
        af_vec_rotated = torch.matmul(r_obj, af_vec.reshape(num_envs, -1, 3).to('cuda').unsqueeze(-1)).squeeze(-1)
        af_vec = af_vec_rotated.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        # compute qp result
        # forces_error, wrench_error_list = self.solver.qp_force_as_ref(self.get_contact_info(), self.get_object_info())
        tau_error_list, wrench_error_list = self.solver.qp_ik_torque_as_ref(self.get_contact_info(), self.get_object_info())
        wrench_error_list = np.array(wrench_error_list).reshape(num_envs, -1)

        obs_r = np.concatenate([obs_r, af_vec, tau_error_list, wrench_error_list], axis=-1)
        
        show_af_point = af_points.reshape(-1, 3).cpu().numpy().reshape(num_envs, -1).astype('float32')
        dis_info = np.concatenate([min_dis_af.cpu().numpy(), show_af_point], axis=-1)


        return obs_r, dis_info
    def observe_vision_QP_new(self):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy()

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 54:105].reshape(num_envs, -1, 3)).to('cuda')


        af_dists = torch.cdist(joints, self.affordance_pcd)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(self.affordance_pcd, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        obj_euler_wrist = torch.from_numpy(global_state[:, :3]).to('cuda')
        obj_euler_world = torch.from_numpy(global_state[:, 118:121]).to('cuda')

        r_obj = self.euler_to_rotation_matrix(obj_euler_world).unsqueeze(1).repeat(1, joints.shape[1], 1, 1).to('cuda')
        af_vec_rotated = torch.matmul(r_obj, af_vec.reshape(num_envs, -1, 3).to('cuda').unsqueeze(-1)).squeeze(-1)
        af_vec = af_vec_rotated.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        # compute qp result
        forces_error, wrench_error_list = self.solver.qp_force_as_ref(self.get_contact_info(), self.get_object_info())
        # print("wrench_error_list:\n", wrench_error_list)
        # print("forces_error:\n", forces_error)
        
        obs_r = np.concatenate([obs_r, af_vec, forces_error, wrench_error_list], axis=-1)


        show_af_point = af_points.reshape(-1, 3).cpu().numpy().reshape(num_envs, -1).astype('float32')
        dis_info = np.concatenate([min_dis_af.cpu().numpy(), show_af_point], axis=-1)


        return obs_r, dis_info
    
            
    
    def observe_vision_new(self):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy()

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 54:105].reshape(num_envs, -1, 3)).to('cuda')


        af_dists = torch.cdist(joints, self.affordance_pcd)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(self.affordance_pcd, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        obj_euler_wrist = torch.from_numpy(global_state[:, :3]).to('cuda')
        obj_euler_world = torch.from_numpy(global_state[:, 118:121]).to('cuda')

        r_obj = self.euler_to_rotation_matrix(obj_euler_world).unsqueeze(1).repeat(1, joints.shape[1], 1, 1).to('cuda')
        af_vec_rotated = torch.matmul(r_obj, af_vec.reshape(num_envs, -1, 3).to('cuda').unsqueeze(-1)).squeeze(-1)
        af_vec = af_vec_rotated.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        obs_r = np.concatenate([obs_r, af_vec], axis=-1)
        obj_com = global_state[:, 129:132]
        # print(len(global_state[0]))
        # print("obj_com: ", obj_com)
        
        show_af_point = af_points.reshape(-1, 3).cpu().numpy().reshape(num_envs, -1).astype('float32')
        dis_info = np.concatenate([min_dis_af.cpu().numpy(), show_af_point], axis=-1)


        return obs_r, dis_info

    def get_obj_mu(self):
        mu = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.get_obj_mu(mu)
        return mu
    
    
    def observe_student_aff(self, visible_points):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy()

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 128:179].reshape(num_envs, -1, 3)).to('cuda')

        af_dists = torch.cdist(joints, visible_points)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(visible_points, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        af_vec = af_vec.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        show_af_point = af_points.reshape(-1, 3).cpu().numpy().reshape(num_envs, -1).astype('float32')

        return af_vec, show_af_point


    def observe_student_deploy(self, visible_points):
        self.wrapper.observe(self._observation_r, self._observation_l)
        self.wrapper.get_global_state(self._global_state)

        global_state = self._global_state.copy()
        obs_r = self._observation_r.copy()

        num_envs = global_state.shape[0]

        joints = torch.from_numpy(global_state[:, 128:179].reshape(num_envs, -1, 3)).to('cuda')

        af_dists = torch.cdist(joints, visible_points)
        min_dis_af, min_idx_af = torch.min(af_dists, dim=2)

        af_points = torch.gather(visible_points, 1, min_idx_af.unsqueeze(2).expand(-1, -1, 3))
        af_vec = af_points - joints

        af_vec = af_vec.reshape(num_envs, -1).float().cpu().numpy().astype('float32')

        obs_r = np.concatenate([obs_r, af_vec], axis=-1)

        return obs_r, af_vec

    def get_obj_weight(self):
        weights = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.get_obj_weight(weights)
        return weights
    
    def get_global_state(self, update_mean=True):
        self.wrapper.get_global_state(self._global_state)

        if self.normalize_ob:
            if update_mean:
                self.gs_rms.update(self._global_state)

            return self._normalize_global_state(self._global_state)
        else:
            return self._global_state.copy()

    def get_global_state_l(self, update_mean=True):
        self.wrapper.get_global_state_l(self._global_state_l)

        if self.normalize_ob:
            if update_mean:
                self.gs_rms.update(self._global_state_l)

            return self._normalize_global_state(self._global_state_l)
        else:
            return self._global_state_l.copy()


    def set_rootguidance(self):
        self.wrapper.set_rootguidance()

    def switch_root_guidance(self, obj_pos_bias):
        self.wrapper.switch_root_guidance(obj_pos_bias)

    def switch_obj_pos(self, is_on):
        self.wrapper.switch_obj_pos(is_on)

    def control_switch(self, left, right):
        self.wrapper.control_switch(left, right)

    def control_switch_all(self, left, right):
        self.wrapper.control_switch_all(left, right)

    def reset(self):
        self._reward_r = np.zeros(self.num_envs, dtype=np.float32)
        self._reward_l = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def add_stage(self, stage_dim, stage_pos):
        self.wrapper.add_stage(stage_dim, stage_pos)

    def switch_arctic(self, idx):
        self.wrapper.switch_arctic(idx)

    def load_object(self, obj_idx, obj_weight, obj_dim, obj_type):
        self.wrapper.load_object(obj_idx, obj_weight, obj_dim, obj_type)

    def load_articulated(self, obj_model):
        self.wrapper.load_articulated(obj_model)

    def load_multi_articulated(self, obj_models):
        self.wrapper.load_multi_articulated(obj_models)

    def get_obj_weight(self):
        weights = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.get_obj_weight(weights)  # 调用 C++ 接口
        return weights
    def reset_state(self, init_state_r, init_state_l, init_vel_r, init_vel_l, obj_pose):
        self.wrapper.reset_state(init_state_r, init_state_l, init_vel_r, init_vel_l, obj_pose)

    def set_goals_r(self, obj_pos_r, ee_pos_r, pose_r, qpos_r):
        self.wrapper.set_goals_r(obj_pos_r, ee_pos_r, pose_r, qpos_r)

    def set_imitation_goals(self, pose_l, pose_r, obj_pose):
        self.wrapper.set_imitation_goals(pose_l, pose_r, obj_pose)

    def set_goals_r2(self, obj_pos_r, ee_pos_r, pose_r, qpos_r, contact_r):
        self.wrapper.set_goals_r2(obj_pos_r, ee_pos_r, pose_r, qpos_r, contact_r)

    def set_ext(self, ext_force, ext_torque):
        self.wrapper.set_ext(ext_force, ext_torque)

    def set_pregrasp(self, obj_pos, ee_pos, pose):
        self.wrapper.set_pregrasp(obj_pos, ee_pos, pose)

    def set_goals(self, obj_angle, obj_pos, ee_pos_r, ee_pos_l, pose_r, pose_l, qpos_r, qpos_l, contact_r, contact_l):
        self.wrapper.set_goals(obj_angle, obj_pos, ee_pos_r, ee_pos_l, pose_r, pose_l, qpos_r, qpos_l, contact_r, contact_l)

    def update_target(self, target_center):
        self.wrapper.update_target(target_center)

    def set_joint_sensor_visual(self, joint_sensor_visual):
        self.wrapper.set_joint_sensor_visual(joint_sensor_visual)

    def set_sample_point_visual(self, joint_sensor_visual):
        self.wrapper.set_sample_point_visual(joint_sensor_visual)

    def getObjectTotalMass(self, env_id, object_name):
        return self.wrapper.getObjectTotalMass(env_id, object_name)

    def check_collision(self, joint_state):
        return self.wrapper.check_collision(joint_state)

    def getMaterialPairProperties(self, envIndex, mat1, mat2):
        return self.wrapper.getMaterialPairProperties(envIndex, mat1, mat2)

    def set_joint_sensor_visual_l(self, joint_sensor_visual):
        self.wrapper.set_joint_sensor_visual_l(joint_sensor_visual)

    def set_obj_goal(self, obj_angle, obj_pos):
        self.wrapper.set_obj_goal(obj_angle, obj_pos)

    def _normalize_observation(self, obs, is_rhand):
        if self.normalize_ob:
            if is_rhand:
                return np.clip((obs - self.obs_rms_r.mean) / np.sqrt(self.obs_rms_r.var + 1e-8), -self.clip_obs,
                               self.clip_obs)
            else:
                return np.clip((obs - self.obs_rms_l.mean) / np.sqrt(self.obs_rms_l.var + 1e-8), -self.clip_obs,
                               self.clip_obs)
        else:
            return obs

    def _normalize_global_state(self, gs):
        if self.normalize_ob:
            return np.clip((gs - self.gs_rms.mean) / np.sqrt(self.gs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return gs

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def get_reward_info_l(self):
        reward_info = self.wrapper.rewardInfoLeft()
        return reward_info

    def get_reward_info_r(self):
        reward_info = self.wrapper.rewardInfoRight()
        return reward_info


    def get_pca_rewards(self, obs, is_right):
        num_envs = obs.shape[0]
        eulers = obs.reshape(num_envs,-1, 3).copy()

        eulers = eulers.reshape(-1,3)
        rotvec = R.from_euler('XYZ', eulers, degrees=False)
        rotvec = rotvec.as_rotvec().reshape(num_envs,-1)

        if is_right:
            joint_pca = np.matmul(rotvec, np.linalg.inv(self.th_comps_r))
            pca_target = self.mean_pca_r.repeat(num_envs, 0)

        else:
            joint_pca = np.matmul(rotvec, np.linalg.inv(self.th_comps_l))
            pca_target = self.mean_pca_l.repeat(num_envs, 0)

        joint_pca_norm = joint_pca / (np.linalg.norm(joint_pca, axis=-1, keepdims=True)+1e-5)
        pca_target_norm = pca_target / (np.linalg.norm(pca_target, axis=-1, keepdims=True)+1e-5)
        pca_dist = joint_pca_norm * pca_target_norm
        cos_sim = 1 - pca_dist.sum(-1)
        cos_sim[cos_sim<0.4] *= 0.1
        return cos_sim

    def debugShowObs(self, obs):
        self.wrapper.debugShowObs(obs)

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

