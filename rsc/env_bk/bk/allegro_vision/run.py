from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_vision as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
from raisimGymTorch.env.bin.allegro_vision import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_allegro_new, get_initial_pose_faive_random


import sys
import os
import time
import torch.nn as nn
import numpy as np
import torch
from raisimGymTorch.helper import rotations
import wandb
import torch
from random import choice

class cfg_file():
    def __init__(self, task_path, cfg_name):
        self.cfg = YAML().load(open(task_path + '/cfgs/' + cfg_name, 'r'))
        self.use_wandb = self.cfg['base']['use_wandb']
        self.project_name = self.cfg['base']['project_name']
        self.task_name = self.cfg['base']['task_name']
        self.eval_mode = self.cfg['base']['eval_mode']
        self.fixed_mode = self.cfg['base']['fixed_mode']
        self.iteration = self.cfg["train"]['iteration']
        self.datasets_name = self.cfg['train']['datasets_name']
        objs = self.cfg['train']['datasets_specific_obj_name']
        if objs != '':
            self.datasets_objs = objs.split(" ")
        else:
            self.datasets_objs = None
        self.grasp_step = self.cfg['train']['grasp_step']
        self.arti_step = self.cfg['train']['arti_step']
        self.trail_step = self.cfg['train']['trail_step']
        self.ppo_obs_dim = self.cfg['environment']['ppo_obs_dim']
        self.action_dim = self.cfg['environment']['actiondim']
        self.hand_type = self.cfg['environment']['hand_type']
        self.finger_num = int ((self.cfg['environment']['hand_contact_num'] - 1) / 3)
        self.body_num = self.cfg['environment']['hand_body_num']

        # dagger observation dim
        self.dagger_mode = self.cfg["train"]['dagger_mode']
        self.tobeEncode_dim = self.cfg['environment']['step_encode_dim']
        self.t_steps = self.cfg['environment']['history_len']
        self.prop_latent_dim = self.cfg['environment']['prop_latent_dim']
        self.total_obs_dim = self.tobeEncode_dim*self.t_steps + self.ppo_obs_dim

        # update
        self.cfg['environment']['load_set'] = self.datasets_name
        if self.eval_mode == True:
            self.cfg['environment']['render'] = True
            self.cfg['environment']['visualize'] = True
        else:
            self.cfg['environment']['render'] = False
            self.cfg['environment']['visualize'] = False
            
class global_value():
    def __init__(self, cfg_name):
        # set directories
        self.task_path_ = os.path.dirname(os.path.realpath(__file__))
        self.cfg_name_ = cfg_name
        self.cfg_ = cfg_file(self.task_path_, cfg_name)
        self.home_path_ = self.task_path_ + "/../../../../.." # raisim/raisimGymTprch
        self.directory_path_ = self.home_path_ + f"/rsc/{self.cfg_.datasets_name}/"
        self.checkpoint_path_ = self.home_path_ + "/raisimGymTorch/data_all/"
        
        # set object
        self.obj_path_list_ = []
        self.obj_list_ = []
        self.obj_pose_reset_ = None
        self.num_envs_ = 1
        self.load_object()

        # init other values
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_steps_ = self.cfg_.grasp_step + self.cfg_.trail_step
        self.total_steps_ = self.n_steps_ * self.num_envs_
        self.finger_weights_ = self.set_finger_wight()

        # init environment
        self.env_ = VecEnv(self.obj_list_, mano.RaisimGymEnv(self.home_path_ + "/rsc", dump(self.cfg_.cfg['environment'], Dumper=RoundTripDumper)), self.cfg_.cfg['environment'], cat_name=self.cfg_.datasets_name)
        self.env_.load_multi_articulated(self.obj_path_list_)
        self.saver_ = ConfigurationSaver(log_dir=self.checkpoint_path_ + self.cfg_.project_name, save_items=[self.task_path_ + "/cfgs/" + cfg_name], test_dir=self.cfg_.eval_mode, task_name=self.cfg_.task_name)
        if self.cfg_.use_wandb == True:
            wandb.init(project=self.cfg_.project_name, config=self.cfg_, name=self.cfg_.task_name)

    def load_object(self):
        items = os.listdir(self.directory_path_)

        # # Filter out only the folders (directories) from the list of items
        if self.cfg_.datasets_objs is None:
            folder_names = [item for item in items if os.path.isdir(os.path.join(self.directory_path_, item))]
        else:
            folder_names = self.cfg_.datasets_objs

        obj_it_num = 3
        if self.cfg_.eval_mode == True:
            folder_names = [choice(folder_names)]
            obj_it_num = 1

        for item in folder_names:
            for i in range(obj_it_num):
                self.obj_list_.append(item)
                if self.cfg_.fixed_mode == True:
                    self.obj_path_list_.append(os.path.join(f"{item}/{item}_fixed_base.urdf"))
                else:
                    self.obj_path_list_.append(os.path.join(f"{item}/{item}.urdf"))
            
        self.num_envs_ = len(self.obj_list_)
        self.cfg_.cfg['environment']['num_envs'] = self.num_envs_

        self.obj_pose_reset_ = np.zeros((self.num_envs_, 8), dtype='float32')
        for i in range(self.num_envs_):
            txt_file_path = os.path.join(self.directory_path_, self.obj_list_[i]) + "/lowest_point_new.txt"
            with open(txt_file_path, 'r') as txt_file:
                lowest_point = float(txt_file.read())
            self.obj_pose_reset_[i, :] = [1., -0., 0.502, 1., -0., -0., 0., 0.]
            self.obj_pose_reset_[i, 2] -= lowest_point

    def set_finger_wight(self):
        finger_weights = np.ones((self.num_envs_, self.cfg_.body_num)).astype('float32')
        for i in range(self.cfg_.finger_num):
            finger_weights[:, 4 * i+4] *= 4.0
        finger_weights /= finger_weights.sum(axis=1).reshape(-1, 1)
        finger_weights *= float(self.cfg_.body_num)

        return finger_weights
    
    # 'left', 'right', 'only_left', 'only_right'
    def use_hand(self, type:str):
        if type == 'right':
            return (self.cfg_.hand_type == 0 or self.cfg_.hand_type == 2)
        elif type == 'left':
            return (self.cfg_.hand_type == 1 or self.cfg_.hand_type == 2)
        elif type == 'only_left':
            return (self.cfg_.hand_type == 1)
        elif type == 'only_right':
            return (self.cfg_.hand_type == 0)

class mano_hand():
    def __init__(self, g:global_value, type:str="right"):
        # for RL network arch
        if g.cfg_.dagger_mode == False:
            self.ppo_init_arch(g, type)
        else:
            self.dagger_init_arch(g, type)

        self.pt_pth_ = ''
        self.obs_ = None
        self.obs_info_ = None
        self.reward_ = None
        self.rewards_sum_ = None
        self.action_ = None
        self.new_allegro_ = True

        # set init pose for each episode
        self.qpos_reset_ = np.zeros((g.num_envs_, g.cfg_.action_dim), dtype='float32')
        for i in range(g.num_envs_):
            if self.new_allegro_:
                self.qpos_reset_[i, 6:] = 0.2
                self.qpos_reset_[i, -4] = 1.0
                self.qpos_reset_[i, 7] = 0.8
                self.qpos_reset_[i, 11] = 0.8
                self.qpos_reset_[i, 15] = 0.8
                self.qpos_reset_[i, 19] = 0.8
            else:
                self.qpos_reset_[i, -4] = 1.7
    
    def ppo_init_arch(self, g:global_value, type:str="right"):
        import raisimGymTorch.algo.ppo.module as ppo_module
        import raisimGymTorch.algo.ppo.ppo as PPO

        self.actor_ = ppo_module.Actor(
                ppo_module.MLP(g.cfg_.cfg['architecture']['policy_net'], nn.LeakyReLU, g.cfg_.ppo_obs_dim, g.cfg_.action_dim),
                ppo_module.MultivariateGaussianDiagonalCovariance(g.cfg_.action_dim, g.num_envs_, 1.0, NormalSampler(g.cfg_.action_dim)), 
                g.device_)
        self.critic_ = ppo_module.Critic(
            ppo_module.MLP(g.cfg_.cfg['architecture']['value_net'], nn.LeakyReLU, g.cfg_.ppo_obs_dim, 1), 
            g.device_)
        self.arch_ = PPO.PPO(actor=self.actor_,
                    critic=self.critic_,
                    num_envs=g.num_envs_,
                    num_transitions_per_env=g.n_steps_,
                    num_learning_epochs=4,
                    gamma=0.996,
                    lam=0.95,
                    num_mini_batches=4,
                    device=g.device_,
                    log_dir=g.saver_.data_dir,
                    shuffle_batch=False
                    # learning_rate=1e-4
                    )
        
    def dagger_init_arch(self, g:global_value, type:str="right"):
        import raisimGymTorch.algo.ppo_dagger_recon.module as ppo_module
        from raisimGymTorch.algo.ppo_dagger_recon.dagger_new import Dagger

        self.actor_expert_ = ppo_module.Actor(
            ppo_module.MLP(g.cfg_.cfg['architecture']['policy_net'], nn.LeakyReLU, g.cfg_.ppo_obs_dim, g.cfg_.action_dim),
            ppo_module.MultivariateGaussianDiagonalCovariance(g.cfg_.action_dim, g.num_envs_, 1.0, NormalSampler(g.cfg_.action_dim)), g.device_)
        self.actor_ = ppo_module.Actor(
            ppo_module.MLP(g.cfg_.cfg['architecture']['policy_net'], nn.LeakyReLU, g.cfg_.ppo_obs_dim, g.cfg_.action_dim),
            ppo_module.MultivariateGaussianDiagonalCovariance(g.cfg_.action_dim, g.num_envs_, 1.0, NormalSampler(g.cfg_.action_dim)), g.device_)
        self.critic_ = ppo_module.Critic(ppo_module.MLP(g.cfg_.cfg['architecture']['value_net'], nn.LeakyReLU, g.cfg_.ppo_obs_dim, 1), g.device_)

        checkpoint = torch.load(g.checkpoint_path_ + g.cfg_.cfg['base']['teacher_pth'] + 'r.pt', map_location=torch.device('cpu'))
        self.actor_expert_.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
        self.actor_expert_.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
        self.actor_.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
        self.actor_.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
        self.critic_.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
        self.prop_latent_encoder_ = ppo_module.LSTM_StateHistoryEncoder(g.cfg_.tobeEncode_dim, g.cfg_.prop_latent_dim, g.cfg_.t_steps, g.device_).to(g.device_)

        self.arch_ = Dagger(expert_policy=self.actor_expert_.architecture,
                        actor_student=self.actor_,
                        critic_student=self.critic_,
                        prop_latent_encoder=self.prop_latent_encoder_,
                        tobeEncode_dim=g.cfg_.tobeEncode_dim,
                        prop_latent_dim=g.cfg_.prop_latent_dim,
                        total_obs_dim=g.cfg_.total_obs_dim,
                        mlp_obs_dim=g.cfg_.ppo_obs_dim,
                        t_steps=g.cfg_.t_steps,
                        num_envs=g.num_envs_,
                        num_transitions_per_env=g.n_steps_,
                        num_learning_epochs=4,
                        gamma=0.996,
                        lam=0.95,
                        num_mini_batches=4,
                        device=g.device_,
                        log_dir=g.saver_.data_dir,
                        shuffle_batch=False,
                        update_mlp=g.cfg_.cfg['train']['update_mlp'],
                        ppo_ratio=g.cfg_.cfg['train']['ppo_ratio']
                        )
        
    def load_student_policy(self, weight_path): 
        checkpoint_student = torch.load(weight_path, map_location=torch.device('cpu'))
        self.actor_.architecture.load_state_dict(checkpoint_student['actor_architecture_state_dict'])
        self.actor_.distribution.load_state_dict(checkpoint_student['actor_distribution_state_dict'])
        self.critic_.architecture.load_state_dict(checkpoint_student['critic_architecture_state_dict'])
        self.prop_latent_encoder_.load_state_dict(checkpoint_student['prop_latent_encoder_state_dict'])

    def save_state(self, dagger_mode):
        ret = {'actor_architecture_state_dict': self.actor_.architecture.state_dict(),
            'actor_distribution_state_dict': self.actor_.distribution.state_dict(),
            'critic_architecture_state_dict': self.critic_.architecture.state_dict(),
            'optimizer_state_dict': self.arch_.optimizer.state_dict(),}
        if dagger_mode == True:
            ret['prop_latent_encoder_state_dict'] = self.prop_latent_encoder_.state_dict()
        return ret
    
    def reset_state(self, i, obj_pos, rot, pos, bias):
        obj_mat = rotations.quat2mat(obj_pos[3:7])

        wrist_pose_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
        wrist_mat = rotations.euler2mat(wrist_pose_obj)
        wrist_in_world = np.matmul(obj_mat, wrist_mat)
        wrist_pose = rotations.mat2euler(wrist_in_world)
        self.qpos_reset_[i, :3] = obj_pos[:3] + np.matmul(obj_mat, pos[0, :])
        self.qpos_reset_[i, 3:6] = wrist_pose[0, :]

    def obs(self, g:global_value, type:str):
        if type == "right":
            self.obs_, self.obs_info_ = g.env_.observe_vision(0, allegro=True)
        if g.cfg_.eval_mode == True:
            show_points = self.obs_[:, -51:].astype('float32').copy()
            if type == "right":
                g.env_.set_joint_sensor_visual(show_points)
        else:
            self.obs_ = self.obs_[:].astype('float32')

    def reset_reward_value(self, g:global_value, type:str):
        if type == "right":
            self.rewards_sum_ = g.env_.get_reward_info_r()

        for i in range(len(self.rewards_sum_)):
            self.rewards_sum_[i]['affordance_reward'] = 0
            self.rewards_sum_[i]['not_affordance_reward'] = 0
            self.rewards_sum_[i]['direction_reward'] = 0
            self.rewards_sum_[i]['center_reward'] = 0
            self.rewards_sum_[i]['table_reward'] = 0
            for k in self.rewards_sum_[i].keys():
                self.rewards_sum_[i][k] = 0

    def dagger_student_get_action(self, g:global_value, device):
        encode_obs = torch.from_numpy(self.obs_[:, :g.cfg_.tobeEncode_dim*g.cfg_.t_steps]).to(device)
        student_latent = self.prop_latent_encoder_(encode_obs)
        student_mlp_obs = torch.cat((torch.from_numpy(self.obs_[:, -g.cfg_.ppo_obs_dim:-g.cfg_.ppo_obs_dim+g.cfg_.tobeEncode_dim]), student_latent.cpu(), torch.from_numpy(self.obs_[:, -g.cfg_.ppo_obs_dim+g.cfg_.tobeEncode_dim+g.cfg_.prop_latent_dim:])), dim=1).to(device)
        self.action_ = self.actor_.architecture.architecture(student_mlp_obs.to(device))
        self.action_ = self.action_.cpu().detach().numpy()

    def actor_get_action(self, device):
        self.action_ = self.actor_.architecture.architecture(torch.from_numpy(self.obs_.astype('float32')).to(device))
        self.action_ = self.action_.cpu().detach().numpy()

    def ppo_get_action(self, student_driven_ratio):
        self.action_ = self.arch_.act(self.obs_, student_driven_ratio)

    def calculate_reward(self, g:global_value, type:str):
        if type == "right":
            obs, dis_info = g.env_.observe_vision(0, allegro=True)
            global_state = g.env_.get_global_state()
            
        direction_loss = np.sum(np.square(global_state[:, 112:115]), axis=1)
        abs_dis = np.linalg.norm(obs[:, :3], axis=1)
        # if distance > 0.1 then give center reward otherwise set to 0
        center_loss = (abs_dis > 0.07) * (np.square(abs_dis - 0.07))

        if type == "right":
            rewards = g.env_.get_reward_info_r() 
        else:
            rewards = g.env_.get_reward_info_l()

        affordance_reward = - np.sum((dis_info[:, :g.cfg_.body_num]) * g.finger_weights_, axis=1)
        table_reward = - np.sum(np.log(np.maximum(np.abs(obs[:, 76:93]), 0.01*np.ones_like(np.abs(obs[:, 76:93])))) * g.finger_weights_ * (np.abs(obs[:, 76:93]) < 0.03), axis=1)

        for i in range(g.num_envs_):
            rewards[i]['affordance_reward'] = affordance_reward[i] * g.cfg_.cfg['environment']['reward']['affordance_reward']['coeff']
            rewards[i]['not_affordance_reward'] = 0
            
            rewards[i]['direction_reward'] = direction_loss[i] * g.cfg_.cfg['environment']['reward']['direction_reward']['coeff']
            rewards[i]['center_reward'] = center_loss[i] * g.cfg_.cfg['environment']['reward']['center_reward']['coeff']
            rewards[i]['table_reward'] = table_reward[i] * g.cfg_.cfg['environment']['reward']['table_reward']['coeff']
            obj_vel_pul = rewards[i]['obj_vel_reward_']
            if obj_vel_pul < -0.75:
                obj_vel_pul = (obj_vel_pul + 0.75) / 4.0 - 0.75
            if obj_vel_pul < -1.0:
                obj_vel_pul = -1.0
            obj_qvel_pul = rewards[i]['obj_qvel_reward_']
            if obj_qvel_pul < -0.75:
                obj_qvel_pul = (obj_qvel_pul + 0.75) / 4.0 - 0.75
            if obj_qvel_pul < -1.0:
                obj_qvel_pul = -1.0

            wrist_vel_pul = rewards[i]['wrist_vel_reward_']
            if wrist_vel_pul < -0.75:
                wrist_vel_pul = (wrist_vel_pul + 0.75) / 4.0 - 0.75
            if wrist_vel_pul < -1.0:
                wrist_vel_pul = -1.0
            wrist_qvel_pul = rewards[i]['wrist_qvel_reward_']
            if wrist_qvel_pul < -0.75:
                wrist_qvel_pul = (wrist_qvel_pul + 0.75) / 4.0 - 0.75
            if wrist_qvel_pul < -1.0:
                wrist_qvel_pul = -1.0

            rewards[i]['reward_sum'] = (rewards[i]['reward_sum'] + rewards[i]['affordance_reward'] + rewards[i]['direction_reward'] + rewards[i]['center_reward'] + rewards[i]['table_reward'] - rewards[i]['obj_vel_reward_'] - rewards[i]['obj_qvel_reward_'] + obj_vel_pul + obj_qvel_pul + rewards[i]['not_affordance_reward'] - rewards[i]['wrist_vel_reward_'] - rewards[i]['wrist_qvel_reward_'] + wrist_vel_pul + wrist_qvel_pul)
            rewards[i]['obj_vel_reward_'] = obj_vel_pul
            rewards[i]['obj_qvel_reward_'] = obj_qvel_pul
            rewards[i]['wrist_vel_reward_'] = wrist_vel_pul
            rewards[i]['wrist_qvel_reward_'] = wrist_qvel_pul
            self.reward_[i] = rewards[i]['reward_sum']
        self.reward_.clip(min=-2.0)

        for i in range(len(self.rewards_sum_)):
            for k in self.rewards_sum_[i].keys():
                self.rewards_sum_[i][k] = self.rewards_sum_[i][k] + rewards[i][k]

    def ppo_step(self, dones):
        self.arch_.step(self.obs_, self.reward_, dones)

    def dagger_update(self, ob_dim):
        value_obs = self.obs_[:, -ob_dim:]
        self.prop_mse_loss, self.action_mse_loss = self.arch_.update(value_obs)

    def ppo_update(self, update: int, g:global_value):
        self.arch_.update(actor_obs=self.obs_, value_obs=self.obs_, log_this_iteration=update % g.cfg_.cfg["train"]["log_interval_ppo_eval"] == 0, update=update)
        self.actor_.distribution.enforce_minimum_std((torch.ones(g.cfg_.action_dim) * 0.2).to(g.device_))
        return self.arch_.check_exploding_gradient()
    
    def set_log(self, dagger_mode, ave_reward, suffix:str, n_steps):
        for k in self.rewards_sum_[0].keys():
            ave_reward[k + suffix] = 0
            for i in range(len(self.rewards_sum_)):
                ave_reward[k + suffix] = ave_reward[k + suffix] + self.rewards_sum_[i][k]
            ave_reward[k + suffix] = ave_reward[k + suffix] / (len(self.rewards_sum_) * n_steps)
        
        if dagger_mode == True:
            ave_reward['recon_loss'] = self.prop_mse_loss
            ave_reward['action_loss'] = self.action_mse_loss
    
    def limit_reward(self, r):
        out = r
        if r > 0.75:
            out = (r - 0.75) / 4.0 + 0.75
            if out > 1.0:
                out = 1.0
        return out

class test_train():
    def __init__(self, cfg_name):
        self.g_ = global_value(cfg_name)
        self.right_ = mano_hand(self.g_, "right")
        self.dones_ = None
        self.save_dict_cnt_ = 0
        self.update_cnt_ = 0
        self.exploding_cnt_ = 0

    def load_param(self, exploding_gradient_mode:bool):
        if exploding_gradient_mode == True:
            load_pth = self.g_.saver_.data_dir + "/full_" + str(self.save_dict_cnt_)
        else:
            load_pth = self.g_.checkpoint_path_ + self.g_.cfg_.cfg['base']['retrain_pth']
        
        if self.g_.cfg_.cfg['base']['retrain_pth'] != '':
            if self.g_.use_hand("right"):
                if self.g_.cfg_.dagger_mode == True:
                    self.right_.load_student_policy(load_pth + 'r.pt')
                else:
                    load_param(load_pth + 'r.pt', self.g_.env_, self.right_.actor_, self.right_.critic_, self.right_.arch_.optimizer, self.g_.saver_.data_dir, self.g_.cfg_name_)

    def save_state(self):
        if self.g_.cfg_.eval_mode == True:
            return

        if self.update_cnt_ % self.g_.cfg_.cfg['train']['log_interval_pt'] == 0:
            if self.g_.use_hand("right"):
                torch.save(self.right_.save_state(self.g_.cfg_.dagger_mode), self.g_.saver_.data_dir + "/full_" + str(self.update_cnt_) + '_r.pt')
            self.g_.env_.save_scaling(self.g_.saver_.data_dir, str(self.update_cnt_))
            self.save_dict_cnt_ = self.update_cnt_

    def set_episode_target(self):
        target_center = np.zeros_like(self.g_.env_.affordance_center)
        object_center = np.zeros_like(self.g_.env_.affordance_center)

        for i in range(self.g_.num_envs_):
            if np.linalg.norm(self.g_.env_.non_aff_mesh[i].centroid - [0.346408, 0.346408, 0.346408]) < 0.01:
                non_aff_mesh = None
            else:
                non_aff_mesh = self.g_.env_.non_aff_mesh[i]

            # rot, pos, bias = get_initial_pose_faive(env.aff_mesh[i], non_aff_mesh, "allegro")
            if self.right_.new_allegro_:
                rot, pos, bias = get_initial_pose_allegro_new(self.g_.env_.aff_mesh[i], non_aff_mesh, "allegro", top=True, easy=False)
            else:
                rot, pos, bias = get_initial_pose_faive_random(self.g_.env_.aff_mesh[i], non_aff_mesh, "allegro", top=True, easy=False)
            self.right_.reset_state(i, self.g_.obj_pose_reset_[i], rot, pos, bias)

            target_center[i, :] = bias[:]
            object_center[i, :] = self.g_.env_.affordance_center[i]

        self.g_.env_.reset_state(self.right_.qpos_reset_,
                        np.zeros((self.g_.num_envs_, self.g_.cfg_.action_dim), 'float32'),
                        np.zeros((self.g_.num_envs_, self.g_.cfg_.action_dim), 'float32'),
                        np.zeros((self.g_.num_envs_, self.g_.cfg_.action_dim), 'float32'),
                        self.g_.obj_pose_reset_,
                        )

        self.g_.env_.set_goals(target_center, object_center,
                    np.zeros((self.g_.num_envs_, 1), 'float32'),
                    np.zeros((self.g_.num_envs_, 1), 'float32'),
                    np.zeros((self.g_.num_envs_, 1), 'float32'),
                    np.zeros((self.g_.num_envs_, 1), 'float32'),
                    np.zeros((self.g_.num_envs_, 1), 'float32'),
                    np.zeros((self.g_.num_envs_, 1), 'float32'),
                    np.zeros((self.g_.num_envs_, 1), 'float32'),
                    np.zeros((self.g_.num_envs_, 1), 'float32'),
                    )
    
    def obs(self):
        if self.g_.use_hand("right"):
            self.right_.obs(self.g_, "right")

    def reset_reward_value(self):
        if self.g_.cfg_.eval_mode == False:
            if self.g_.use_hand("right"):
                self.right_.reset_reward_value(self.g_, "right")

    def act_get_action(self):
        if self.g_.cfg_.eval_mode == True:
            if self.g_.use_hand("right"):
                if self.g_.cfg_.dagger_mode == True:
                    self.right_.dagger_student_get_action(self.g_, self.g_.device_)
                else:
                    self.right_.actor_get_action(self.g_.device_)
        else:
            if self.g_.use_hand("right"):
                self.right_.ppo_get_action(self.g_.cfg_.cfg["train"]["student_driven_ratio"])
    def step(self):
        self.right_.reward_, _, self.dones_ = self.g_.env_.step(self.right_.action_.astype('float32'), self.right_.action_.astype('float32'))

    def calculate_reward(self):
        if self.g_.use_hand("right"):
            self.right_.calculate_reward(self.g_, "right")

    def ppo_step(self):
        if self.g_.use_hand("right"):
            self.right_.ppo_step(self.dones_)

    def ppo_update(self):
        reload_flag = [False, False]
        if self.g_.use_hand("right"):
            if self.g_.cfg_.dagger_mode == True:
                self.right_.dagger_update(self.g_.cfg_.ppo_obs_dim)
                return False
            else:
                reload_flag[0] = self.right_.ppo_update(self.update_cnt_, self.g_)

        if True == reload_flag[0] or True == reload_flag[1]:
            print("------------------- exploding gradient !!! will reload param --------------------")
            self.load_param(True)
            self.right_.arch_.is_exploding_gradient = False
            self.exploding_cnt_ += 1
            return True
        else:
            return False

    def log_show(self):
        ave_reward = {}
        if self.g_.use_hand("right"):
            if (self.g_.cfg_.use_wandb == True) and (self.update_cnt_ % self.g_.cfg_.cfg["train"]["log_interval_wandb"] == 0):
                self.right_.set_log(self.g_.cfg_.dagger_mode, ave_reward, "", self.g_.n_steps_)
                wandb.log(ave_reward)

        if (self.update_cnt_ % self.g_.cfg_.cfg["train"]["log_interval_terminal"] == 0):
            print(f'{self.update_cnt_}th: explod_cnt={self.exploding_cnt_}')

def main():
    if len(sys.argv) > 1 and sys.argv[1] is not None:
        CFG_NAME = sys.argv[1]
    else:
        CFG_NAME = 'cfg_dagger_eval.yaml'
    
    demo = test_train(CFG_NAME)
    demo.load_param(False)

    while demo.update_cnt_ < demo.g_.cfg_.iteration:
        
        demo.save_state()
        demo.set_episode_target()
        demo.reset_reward_value()
        
        for step in range(demo.g_.n_steps_):
            demo.obs()
            demo.act_get_action()
            
            frame_start = time.time()
            demo.step()
            if demo.g_.cfg_.eval_mode == True:
                frame_end = time.time()
                wait_time = demo.g_.cfg_.cfg['environment']['control_dt'] - (frame_end - frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)
                continue

            demo.calculate_reward()
            demo.ppo_step()
            
        if demo.g_.cfg_.eval_mode == True:
            demo.update_cnt_ += 1
            continue
        
        # if demo.g_.cfg_.dagger_mode == False:
        demo.obs()
            
        if True == demo.ppo_update():
            demo.update_cnt_ = demo.save_dict_cnt_
            continue
        
        demo.log_show()
        demo.update_cnt_ += 1

if __name__ == '__main__':
    main()
