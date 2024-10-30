#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import arm_rand_student as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.arm_rand_student import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_faive, get_initial_pose_faive_random, get_initial_pose_allegro_new, get_initial_pose_allegro_arm_rand, get_initial_pose_allegro_arm_rand_test
from scipy.spatial.transform import Rotation as R
from random import choice

# import pdb

import os
import math
import time
import raisimGymTorch.algo.ppo_dagger_recon.module as ppo_module
# import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
from datetime import datetime
import argparse
from raisimGymTorch.helper import rotations
from raisimGymTorch.helper.inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter
import joblib
import random
import wandb
import torch


exp_name = "arm_rand_student"

# weight_saved = '2024-10-22-16-46-35/full_24500_r.pt'
# weight_saved = '2024-10-23-08-13-36/full_20000_r.pt'
# weight_saved = '2024-10-23-08-21-47/full_20000_r.pt'
# weight_saved = '2024-10-23-08-34-20/full_20000_r.pt'
# weight_saved = '2024-10-23-09-52-39/full_17500_r.pt'
# weight_saved = '2024-10-23-12-15-12/full_15500_r.pt'
# weight_saved = '2024-10-23-14-33-50/full_14500_r.pt'
# weight_saved = '2024-10-23-15-05-08/full_12500_r.pt'
# weight_saved = '2024-10-23-17-15-16/full_21000_r.pt'
# weight_saved = '2024-10-23-17-22-34/full_21000_r.pt'
# weight_saved = '2024-10-25-17-09-30/full_23000_r.pt'
# weight_saved = '2024-10-25-17-16-54/full_23000_r.pt'
# weight_saved = '2024-10-26-15-58-30/full_40000_r.pt'
# weight_saved = '2024-10-26-16-03-00/full_40500_r.pt'
weight_saved = './../arm_rand/2024-10-26-17-02-58/full_17000_r.pt'

weight_path_student = '2024-10-30-09-05-20/full_0_r.pt'


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_saved)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-seed', '--seed', type=int, default=1)
parser.add_argument('-itr', '--num_iterations', type=int, default=50001)
parser.add_argument('-nr', '--num_repeats', type=int, default=1)
parser.add_argument('-re', '--load_trained_policy', action="store_true")
parser.add_argument('-renew', '--renew', help='update labels every iteration', action="store_true")
parser.add_argument('-ln', '--log_name', type=str, default='single_obj')
parser.add_argument('-mean', '--mean_pose', action="store_true")

new_allegro = True

args = parser.parse_args()
weight_path = args.weight
cfg_grasp = args.cfg

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")

# task specification
task_name = args.exp_name
# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

# config
cfg = YAML().load(open(task_path + '/cfgs/' + args.cfg, 'r'))

if args.seed != 1:
    cfg['seed'] = args.seed

num_envs = args.num_repeats
activations = nn.LeakyReLU

cfg['environment']['visualize'] = True
cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)


# cat_name = 'mixed_test'
# cat_name = 'mixed_unseen_test'
# cat_name = 'mixed_unseen_category_test'
# cat_name = 'mixed_train'
cat_name = 'ycb_urdf_all'
# cat_name = 'affordance_level'
cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)

items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

obj_path_list = []
obj_ori_list = folder_names

# obj_item = choice(obj_ori_list)
# obj_item = '002_master_chef_can'
# obj_item = '003_cracker_box'
# obj_item = '004_sugar_box'
# obj_item = '005_tomato_soup_can'
# obj_item = '006_mustard_bottle'
obj_item = '007_tuna_fish_can'
# obj_item = '008_pudding_box'
# obj_item = '009_gelatin_box'
# obj_item = '010_potted_meat_can'
# obj_item = '011_banana'
# obj_item = '019_pitcher_base'
# obj_item = '021_bleach_cleanser'
# obj_item = '024_bowl'
# obj_item = '025_mug'
# obj_item = '035_power_drill'
# obj_item = '036_wood_block'
# obj_item = '037_scissors'
# obj_item = '040_large_marker'
# obj_item = '051_large_clamp'
# obj_item = '052_extra_large_clamp'
# obj_item = '061_foam_brick'

# Environment definition
env = VecEnv([obj_item], mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished")

obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)


ob_dim_r = 153
# act_dim = env.num_acts
act_dim = 22
print('ob dim', ob_dim_r)
print('act dim', act_dim)

tobeEncode_dim = 44
t_steps = 10
prop_latent_dim=26
total_obs_dim = tobeEncode_dim*t_steps + ob_dim_r

# Training
trail_steps = 80
reward_clip = -2.0
grasp_steps = 100
n_steps_r = grasp_steps + trail_steps
total_steps_r = n_steps_r * env.num_envs

# RL network

actor_student_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)
prop_latent_encoder = ppo_module.LSTM_StateHistoryEncoder(tobeEncode_dim, prop_latent_dim, t_steps, device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)


test_dir = True

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[], test_dir=test_dir)

checkpoint_student = torch.load(saver.data_dir.split('eval')[0] + weight_path_student, map_location=torch.device('cpu'))
actor_student_r.architecture.load_state_dict(checkpoint_student['actor_architecture_state_dict'])
actor_student_r.distribution.load_state_dict(checkpoint_student['actor_distribution_state_dict'])
prop_latent_encoder.load_state_dict(checkpoint_student['prop_latent_encoder_state_dict'])

# ppo_r = PPO.PPO(actor=actor_r,
#                 critic=critic_r,
#                 num_envs=num_envs,
#                 num_transitions_per_env=n_steps_r,
#                 num_learning_epochs=4,
#                 gamma=0.996,
#                 lam=0.95,
#                 num_mini_batches=4,
#                 device=device,
#                 log_dir=saver.data_dir,
#                 shuffle_batch=False
#                 )
# load_param(saver.data_dir.split('eval')[0]+weight_path, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)

lowest_points = np.zeros((num_envs, 1), dtype='float32')
for i in range(num_envs):
    txt_file_path = os.path.join(directory_path, obj_item) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())

for update in range(args.num_iterations):
    start = time.time()

    qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

    target_center = np.zeros_like(env.affordance_center)

    qpos_reset_r[:, 6:] = 0.2
    qpos_reset_r[:, -4] = 1.57
    qpos_reset_r[:, 7] = 0.8
    qpos_reset_r[:, 11] = 0.8
    qpos_reset_r[:, 15] = 0.8
    qpos_reset_r[:, 19] = 0.
    qpos_reset_r[:, 20] = -0.5

    # obj_pose_reset[:, 0] = np.random.uniform(0.5, 1.1, num_envs)
    # obj_pose_reset[:, 1] = np.random.uniform(0.0, 0.4, num_envs)
    # obj_pose_reset[:, 2] = 0.773 - lowest_points
    # obj_pose_reset[:, 3:] = [1., -0., -0., 0., 0.]
    #
    # axis_angles = np.zeros((num_envs, 3))
    # axis_angles[:, 2] = np.random.uniform(-np.pi, np.pi, num_envs)
    # quats = rotations.axisangle2quat(axis_angles)
    # obj_pose_reset[:, 3:7] = quats
    #
    # hand_center_w = np.zeros((num_envs, 3))
    # hand_center_w[:, 0] = 0.711334
    # hand_center_w[:, 1] = 0.243815
    # hand_center_w[:, 2] = 1.34026
    #
    # obj_aff_center_in_obj = env.affordance_center.copy()
    # obj_mat = rotations.quat2mat(quats)

    for i in range(num_envs):
        get_meaningful_ik = False
        while not get_meaningful_ik:
            obj_pose_reset[i, 0] = np.random.uniform(0.5, 1.1)
            obj_pose_reset[i, 1] = np.random.uniform(0.0, 0.4)
            obj_pose_reset[i, 2] = 0.773 - lowest_points[i]
            obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]

            axis_angles = np.zeros((1, 3))
            axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
            quats = rotations.axisangle2quat(axis_angles)
            # quats[:] = 0
            # quats[:, 0] = 1
            obj_pose_reset[i, 3:7] = quats

            hand_center_w = np.zeros((1, 3))
            # hand_center_w[0, 0] = 0.711334
            # hand_center_w[0, 1] = 0.243815
            # hand_center_w[0, 2] = 1.34026
            hand_center_w[0, 0] = 0.669872
            hand_center_w[0, 1] = 0.141735
            hand_center_w[0, 2] = 1.5   #1.11052

            obj_aff_center_in_obj = env.affordance_center[i].copy()
            obj_mat_single = rotations.quat2mat(quats).reshape(3, 3)

            obj_aff_center_in_world = np.matmul(obj_mat_single, obj_aff_center_in_obj.T).T
            obj_aff_center_in_world = obj_aff_center_in_world + obj_pose_reset[i, :3]

            hand_dir_x = hand_center_w - obj_aff_center_in_world
            hand_dir_x = hand_dir_x / np.linalg.norm(hand_dir_x, axis=1, keepdims=True)
            hand_dir_x_in_obj = np.matmul(obj_mat_single.T, hand_dir_x.T).T

            rot, pos, target = get_initial_pose_allegro_arm_rand_test(env.aff_mesh[i], hand_dir_x_in_obj, env.affordance_center[i], top=False)
            if rot is None:
                hand_dir_x_in_obj[0, :] = 0
                hand_dir_x_in_obj[0, 2] = 1
                rot, pos, target = get_initial_pose_allegro_arm_rand_test(env.aff_mesh[i], hand_dir_x_in_obj, env.affordance_center[i], top=True)

            wrist_mat = rot
            # wrist_pose_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
            # wrist_mat = rotations.euler2mat(wrist_pose_obj)
            wrist_in_world = np.matmul(obj_mat_single, wrist_mat)
            wrist_pose = rotations.mat2euler(wrist_in_world)
            qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat_single, pos[0, :])
            # qpos_reset_r[i, 3:6] = wrist_pose[0, :]

            target_center[i, :] = target[:]

            wrist_bias = np.zeros((1, 3))
            # wrist_bias[0, 0] = -0.0091
            # wrist_bias[0, 2] = -0.095
            wrist_bias[0, 0] = -0.0091
            wrist_bias[0, 2] = -0.085
            wrist_bias_in_world = np.matmul(wrist_in_world, wrist_bias.T).T

            ur5_to_world = np.eye(3)
            ur5_to_world[0, 0] = -1
            ur5_to_world[1, 1] = -1

            pos_in_ur5 = np.zeros((3, 1))
            pos_in_ur5[0, 0] = qpos_reset_r[i, 0] - 0.55 + wrist_bias_in_world[0, 0]
            pos_in_ur5[1, 0] = qpos_reset_r[i, 1] - 0.75152 + wrist_bias_in_world[0, 1]
            pos_in_ur5[2, 0] = qpos_reset_r[i, 2] - 0.771 + wrist_bias_in_world[0, 2]
            pos_in_ur5_new = np.matmul(ur5_to_world.T, pos_in_ur5)

            wrist_mat_in_ur5 = np.matmul(ur5_to_world.T, wrist_in_world)

            gd = np.eye(4)
            gd[:3, :3] = wrist_mat_in_ur5
            gd[0, 3] = pos_in_ur5_new[0, 0]
            gd[1, 3] = pos_in_ur5_new[1, 0]
            gd[2, 3] = pos_in_ur5_new[2, 0]

            theta0 = [-1.57, -1.57, 1.57, 0., 1.57, -1.57]
            joint_weights = [1, 1, 1, 1, 1, 1]
            ik = InverseKinematicsUR5()
            ik.setJointWeights(joint_weights)
            ik.setJointLimits(-3.14, 3.14)
            if ik.findClosestIK(gd, theta0) is None:
                continue
            else:
                qpos_reset_r[i, :6] = ik.findClosestIK(gd, theta0)

            if math.isnan(qpos_reset_r[i, 0]):
                continue
            else:
                get_meaningful_ik = True

    # qpos_reset_r[:, :6] = [-1.57, -1.57, 1.57, 1.57, 3.14, -1.57]
    # env.reset_state(qpos_reset_r,
    #                 qpos_reset_l,
    #                 np.zeros((num_envs, 22), 'float32'),
    #                 np.zeros((num_envs, 22), 'float32'),
    #                 obj_pose_reset,
    #                 )
    # temp_action_r = np.zeros((num_envs, act_dim), dtype='float32')
    # temp_action_l = np.zeros((num_envs, act_dim), dtype='float32')
    # _, _, _ = env.step(temp_action_r, temp_action_l)
    # global_state = env.get_global_state()
    # one_check = global_state[:, 124:128]
    # contains_one = np.any(one_check == 1, axis=1)
    # true_indices = np.where(contains_one)[0]
    # for true_idx in true_indices:
    #     current_obj_idx = true_idx // 3
    #     current_obj_env_indices = [current_obj_idx * 3, current_obj_idx * 3 + 1, current_obj_idx * 3 + 2]
    #     false_indices = [idx for idx in current_obj_env_indices if not contains_one[idx]]
    #     if len(false_indices)>0:
    #         chosen_index = np.random.choice(false_indices)
    #         qpos_reset_r[true_idx, :] = qpos_reset_r[chosen_index, :]
    #         obj_pose_reset[true_idx, :] = obj_pose_reset[chosen_index, :]
    #     else:
    #         qpos_reset_r[true_idx, :6] = [-1.57, -1.57, 1.57, 0., 1.57, -1.57]

    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )

    obs_new_r, dis_info = env.observe_vision_new()
    show_point = dis_info[:, 17:68].astype('float32').copy()
    env.set_joint_sensor_visual(show_point)
    env.update_target(target_center)
    for step in range(n_steps_r):
        obs_r = obs_new_r
        obs_r = obs_r[:, :].astype('float32')

        # if step > 0:
        #     time.sleep(10)

        encode_obs = torch.from_numpy(obs_r[:, :tobeEncode_dim * t_steps]).to(device)

        student_latent = prop_latent_encoder(encode_obs)
        student_mlp_obs = torch.cat((torch.from_numpy(obs_r[:, -ob_dim_r:-ob_dim_r + tobeEncode_dim]),
                                     student_latent.cpu(),
                                     torch.from_numpy(obs_r[:, -ob_dim_r + tobeEncode_dim + prop_latent_dim:])),
                                    dim=1).to(device)

        # print(student_latent)
        # print(obs_r[:, -ob_dim_r + tobeEncode_dim:-ob_dim_r + tobeEncode_dim+prop_latent_dim])

        action_r = actor_student_r.architecture.architecture(student_mlp_obs.to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)
        # action_r[:, :6] = 0

        frame_start = time.time()

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe_vision_new()
        show_point = dis_info[:, 17:68].astype('float32').copy()
        env.set_joint_sensor_visual(show_point)

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    print("end")



