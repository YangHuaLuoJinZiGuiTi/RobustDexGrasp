#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_teacher_biased as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.allegro_teacher_biased import NormalSampler
from raisimGymTorch.helper.initial_pose_final import sample_rot_mats
from scipy.spatial.transform import Rotation as R
from random import choice

# import pdb

import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
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


exp_name = "arm_rand"

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
# weight_saved = '2024-10-26-17-02-58/full_17000_r.pt'
# weight_saved = '2024-10-28-10-29-30/full_22500_r.pt'
# weight_saved = '2024-10-28-16-13-40/full_5500_r.pt'
# weight_saved = '2024-10-28-16-53-10/full_6000_r.pt'
# weight_saved = '2024-10-29-18-45-29/full_11000_r.pt'
# weight_saved = '2024-10-29-18-49-44/full_9500_r.pt'
# weight_saved = '2024-10-31-14-38-19/full_50000_r.pt'
# weight_saved = '2024-10-31-16-40-52/full_50000_r.pt'
# weight_saved = '2024-10-31-16-43-20/full_50000_r.pt'
# weight_saved = '2024-11-04-16-42-02/full_11500_r.pt'
# weight_saved = '2024-11-04-16-42-02/full_50000_r.pt'
# weight_saved = '2024-11-19-18-53-10/full_44000_r.pt'
# weight_saved = '2024-11-19-18-55-45/full_15000_r.pt'
# weight_saved = '2024-11-26-18-01-13/full_12000_r.pt'
# weight_saved = '2024-11-26-18-04-50/full_10000_r.pt'
# weight_saved = '2024-11-26-18-08-39/full_10000_r.pt'
# weight_saved = '2024-11-27-14-04-03/full_14000_r.pt'
# weight_saved = '2024-11-28-15-40-26/full_7500_r.pt'
# weight_saved = '2024-11-28-15-42-46/full_12500_r.pt'
# weight_saved = '2024-11-28-15-47-05/full_11500_r.pt'
# weight_saved = '2024-11-28-18-22-16/full_9000_r.pt'
# weight_saved = '2024-11-29-11-20-34/full_9000_r.pt'
# weight_saved = '2024-11-29-11-23-28/full_9000_r.pt'
# weight_saved = '2024-11-29-11-24-39/full_9000_r.pt'
# weight_saved = '2024-11-29-12-32-22/full_20000_r.pt'
# weight_saved = '2024-11-29-12-56-22/full_19500_r.pt'
# weight_saved = '2024-12-01-08-29-14/full_22500_r.pt'
# weight_saved = '2024-12-02-13-52-26/full_18000_r.pt'
# weight_saved = '2024-12-02-15-42-28/full_16000_r.pt'
# weight_saved = '2024-12-02-15-44-36/full_9500_r.pt'
# weight_saved = '2024-12-04-10-15-15/full_14500_r.pt'
# weight_saved = '2024-12-04-17-35-23/full_15500_r.pt'
# weight_saved = '2024-12-06-10-18-17/full_12500_r.pt'
# weight_saved = '2024-12-06-17-04-47/full_31000_r.pt'
# weight_saved = '2024-12-11-15-33-22/full_14000_r.pt'
# weight_saved = '2024-12-12-09-55-30/full_27000_r.pt'
# weight_saved = '2024-12-13-10-04-10/full_24000_r.pt'
# weight_saved = '2024-12-13-15-47-58/full_25000_r.pt'
# weight_saved = '2024-12-15-09-58-40/full_8500_r.pt'
# weight_saved = '2024-12-15-11-00-37/full_8500_r.pt'
# weight_saved = '2024-12-17-10-54-17/full_24000_r.pt'
# weight_saved = '2024-12-17-10-56-52/full_18000_r.pt'
# weight_saved = '2024-12-17-13-02-44/full_19000_r.pt'
# weight_saved = '2024-12-19-18-02-00/full_30000_r.pt'
# weight_saved = '2024-12-20-13-06-20/full_17500_r.pt'
# weight_saved = '2024-12-20-13-10-39/full_14000_r.pt'
# weight_saved = '2024-12-20-13-16-37/full_17500_r.pt'
# weight_saved = '2024-12-25-18-27-31/full_8500_r.pt'
# weight_saved = '2024-12-25-18-51-39/full_8500_r.pt'
# weight_saved = '2024-12-25-18-57-50/full_8000_r.pt'
# weight_saved = '2024-12-27-18-08-06/full_12500_r.pt'
# weight_saved = '2024-12-27-18-09-12/full_12500_r.pt'
# weight_saved = '2024-12-31-18-57-49/full_16000_r.pt'
# weight_saved = '2024-12-31-19-13-20/full_12000_r.pt'
# weight_saved = '2025-01-01-10-01-41/full_16000_r.pt'
# weight_saved = '2025-01-02-11-39-41/full_4500_r.pt'
# weight_saved = '2025-01-03-18-29-42/full_20000_r.pt'
# weight_saved = '2025-01-03-18-34-11/full_3500_r.pt'
# weight_saved = '2025-01-04-07-55-08/full_13000_r.pt'
# weight_saved = '2025-01-05-11-08-48/full_5500_r.pt'
# weight_saved = '2025-01-05-11-24-07/full_8500_r.pt'
# weight_saved = '2025-01-06-14-32-58/full_8000_r.pt'
# weight_saved = '2025-01-06-14-37-46/full_9000_r.pt'
# weight_saved = '2025-01-06-19-53-12/full_3500_r.pt'
# weight_saved = '2025-01-07-21-04-51/full_10000_r.pt'
# weight_saved = '2025-01-08-10-20-03/full_4000_r.pt'
# weight_saved = '2025-01-08-10-22-54/full_5500_r.pt'
# weight_saved = '2025-01-12-18-19-14/full_16000_r.pt'
# weight_saved = '2025-01-15-16-15-07/full_1000_r.pt'
# weight_saved = '2025-01-16-10-36-16/full_16000_r.pt'
# weight_saved = '2025-01-16-17-45-39/full_4500_r.pt'
# weight_saved = '2025-01-17-08-07-58/full_18000_r.pt'

# weight_saved = '2024-12-27-18-11-12/full_15000_r.pt'
# weight_saved = '2025-01-07-20-39-18/full_15000_r.pt'

# weight_saved = '2025-02-24-16-01-33/full_5000_r.pt'
# weight_saved = '2025-02-24-16-04-36/full_5000_r.pt'

# weight_saved = '2025-03-08-20-27-59/full_9500_r.pt'
# weight_saved = '2025-03-10-10-34-54/full_9000_r.pt'
# weight_saved = '2025-03-10-10-38-43/full_10000_r.pt'
# weight_saved = '2025-03-10-15-39-47/full_12000_r.pt'
# weight_saved = '2025-03-10-18-26-04/full_14000_r.pt'

# weight_saved = '2025-03-12-18-35-33/full_5500_r.pt'
# weight_saved = '2025-03-12-18-36-32/full_5500_r.pt'
# weight_saved = '2025-03-13-08-57-11/full_3500_r.pt'
# weight_saved = '2025-03-13-12-35-31/full_2500_r.pt'
# weight_saved = '2025-03-13-16-25-41/full_4000_r.pt'

# weight_saved = '2025-03-10-10-38-43/full_10000_r.pt'
# weight_saved = '2025-03-13-12-33-15/full_9000_r.pt'

# weight_saved = '2025-03-14-10-18-43/full_7000_r.pt'
# weight_saved = '2025-03-14-10-20-46/full_6000_r.pt'
# weight_saved = '2025-03-14-10-22-19/full_6500_r.pt'
# weight_saved = '2025-03-14-10-57-52/full_7500_r.pt'
# weight_saved = '2025-03-14-18-41-15/full_6500_r.pt'
# weight_saved = '2025-03-16-09-50-07/full_3000_r.pt'
# weight_saved = '2025-03-16-17-31-08/full_2000_r.pt'

weight_saved = '2025-03-14-10-22-19/full_9500_r.pt'
# weight_saved = '2025-03-14-10-57-52/full_11500_r.pt'

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
# cat_name = 'ycb_urdf_sim'
# cat_name = 'ycb_urdf_all'
# cat_name = 'ycb_urdf_light'
# cat_name = 'ycb_urdf_sim_light'
# cat_name = 'real_obj'
# cat_name = 'affordance_level'
# cat_name = 'large_scale_light_stable'
cat_name = 'new_training_set_eval'

cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)
cfg['environment']['num_threads'] = 1

items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

obj_path_list = []
obj_ori_list = folder_names

obj_item = choice(obj_ori_list)
# obj_item = '002_master_chef_can'
# obj_item = '003_cracker_box'
# obj_item = '004_sugar_box'
# obj_item = '005_tomato_soup_can'
# obj_item = '006_mustard_bottle'
# obj_item = '007_tuna_fish_can'
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
# obj_item = 'blue_pitcher'
# obj_item = 'brush_functional'
# obj_item = 'car_down'
# obj_item = 'fan_small_head'
# obj_item = 'gun_functional'
# obj_item = 'hammer'
# obj_item = 'loopy_head_side'
# obj_item = 'mouse'
# obj_item = 'off_water_body'
# obj_item = 'solder_iron_head'
# obj_item = 'wrench'
# obj_item = 'big_tape'
# obj_item = 'small_tape'
# obj_item = 'small_block'
obj_item = 'wood_block_oriented'
# obj_item = 'suger_box_oriented'
# obj_item = 'cracker_box_oriented'
# obj_item = 'power_drill_oriented'

print(obj_item)

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

# Training
reward_clip = -2.0
grasp_steps = cfg['environment']['grasp_steps']
lift_steps = 30
n_steps_r = grasp_steps + lift_steps
total_steps_r = n_steps_r * env.num_envs

# RL network

actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)


test_dir = True

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/runner_eval.py"], test_dir=test_dir)


ppo_r = PPO.PPO(actor=actor_r,
                critic=critic_r,
                num_envs=num_envs,
                num_transitions_per_env=n_steps_r,
                num_learning_epochs=4,
                gamma=0.996,
                lam=0.95,
                num_mini_batches=4,
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False
                )
load_param(saver.data_dir.split('eval')[0]+weight_path, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)

lowest_points = np.zeros((num_envs, 1), dtype='float32')
for i in range(num_envs):
    txt_file_path = os.path.join(directory_path, obj_item) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())

for update in range(args.num_iterations):
    # np.random.seed(int(time.time()))
    start = time.time()

    qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

    target_center = np.zeros_like(env.affordance_center)

    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']


    hand_center_sample_w = np.zeros((1, 3))
    hand_center_sample_w[0, 0] = cfg['environment']['camera_position'][0]
    hand_center_sample_w[0, 1] = cfg['environment']['camera_position'][1]
    hand_center_sample_w[0, 2] = cfg['environment']['camera_position'][2]


    wrist_bias = np.zeros((1, 3))
    wrist_bias[0, 0] = -0.0091
    wrist_bias[0, 2] = -0.095

    ur5_to_world = np.eye(3)
    ur5_to_world[0, 0] = 0
    ur5_to_world[0, 1] = -1
    ur5_to_world[1, 0] = 1
    ur5_to_world[1, 1] = 0

    theta0 = [0.0, -1.57, 1.57, 0., 1.57, -1.57]
    joint_weights = [1, 1, 1, 1, 1, 1]

    ik = InverseKinematicsUR5()
    ik.setJointWeights(joint_weights)
    ik.setJointLimits(-3.14, 3.14)

    sample_num = cfg['environment']['sample_num']


    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

    for i in range(num_envs):
        # sample object states
        # sample_x = 0.15
        # sample_y = 0.2 - 0.75152



        while True:
            angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
            distance = np.random.uniform(0.45, 0.75)
            sample_x = distance * np.cos(angle)
            sample_y = distance * np.sin(angle)
            if sample_x < 0.25 and sample_x > -0.25:
                break

        
            # 采样物体位置
        # if np.random.random() < 0.5:
        #     print('uniform')
        #     while True:
        #         # 50%概率使用均匀采样
        #         angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
        #         distance = np.random.uniform(0.45, 0.75)
        #         sample_x = distance * np.cos(angle)
        #         sample_y = distance * np.sin(angle)
        #         if sample_x < 0.25 and sample_x > -0.25:
        #             break
        # else:
        #     print('biased')
        #     while True:
        #         # 50%概率使用偏向边缘的采样
        #         # 对于角度，使用Beta分布使边缘概率更大
        #         # Beta(0.5, 0.5)是U形分布，在0和1附近概率更高
        #         beta_param = 0.5
        #         angle_normalized = np.random.beta(beta_param, beta_param)  # 在[0,1]范围内，两端概率高
        #         # 映射到[-0.7π, -0.3π]范围
        #         angle = -0.7 * np.pi + angle_normalized * (0.4 * np.pi)
        #         # 对于距离，同样使用Beta(0.5, 0.5)分布使两端概率更高
        #         distance_normalized = np.random.beta(beta_param, beta_param)  # 在[0,1]范围内，两端概率高
        #         distance = 0.45 + distance_normalized * 0.3  # 映射到[0.45, 0.75]
        #         sample_x = distance * np.cos(angle)
        #         sample_y = distance * np.sin(angle)
        #         if sample_x < 0.25 and sample_x > -0.25:
        #             break
        
        # print(angle/np.pi, distance)

        # print(sample_x, sample_y, angle, distance)
        # print(angle)
        # print(angle + np.pi/2)

        # 均匀采样
        # while True:
        #     # if is_evaluation or not non_uniform_sampling:
        #     #     # 均匀采样
        #     #     sample_x = np.random.uniform(-0.25, 0.25)
        #     #     sample_y = np.random.uniform(-0.75, -0.40)
        #     # else:
        #     if np.random.random() < 0.5:
        #         # 均匀采样
        #         sample_x = np.random.uniform(-0.25, 0.25)
        #         sample_y = np.random.uniform(-0.75, -0.40)
        #     else:
        #         # 使用Beta分布进行采样，使边缘概率更大但中心概率不为0
        #         # 对于x，使用Beta分布使两端概率更大
        #         x_beta = np.random.beta(0.7, 0.7)  # 在0和1附近概率更高
        #         sample_x = -0.25 + x_beta * (0.25 - (-0.25))  # 映射到[-0.35, 0.35]
                
        #         # 对于y，使用Beta分布使两端概率更大
        #         y_beta = np.random.beta(0.7, 0.7)  # 在0和1附近概率更高
        #         sample_y = -0.75 + y_beta * (-0.40 - (-0.75))  # 映射到[-0.8, -0.35]

        #     distance = np.sqrt(sample_x**2 + sample_y**2)
        #     if distance < 0.75 and distance > 0.45:
        #         break
        # print(sample_x, sample_y, distance)
        
        # 计算对应的角度
        # angle2 = np.arctan2(sample_y, sample_x)
        # print(angle)
        # print(angle2)
        # print(sample_x, sample_y, angle)

        obj_pose_reset[i, 0] = sample_x
        obj_pose_reset[i, 1] = sample_y
        obj_pose_reset[i, 2] = 0.773 - lowest_points[i]
        obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]

        axis_angles = np.zeros((1, 3))
        axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
        quats = rotations.axisangle2quat(axis_angles)
        obj_pose_reset[i, 3:7] = quats

        # get the partial point cloud
        obj_mat_single = rotations.quat2mat(quats).reshape(3, 3)

        view_point_obj_diff = view_point_world - obj_pose_reset[i, :3]
        view_point_obj = np.matmul(obj_mat_single.T, view_point_obj_diff.T).T
        obj_pcd = env.affordance_pcd[i].reshape(200, 3).cpu().numpy()
        directions = obj_pcd - view_point_obj
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        locations, index_ray, index_tri = env.aff_mesh[i].ray.intersects_location(ray_origins=view_point_obj,
                                                                                  ray_directions=directions,
                                                                                  multiple_hits=False)
        if locations.shape != (200, 3):
            expanded_locations = np.zeros((200, 3))
            expanded_locations[:, :] = locations[0, :]
            expanded_locations[:locations.shape[0], :] = locations
            locations = expanded_locations
        visible_points_obj[i, :] = locations
        visible_points_w[i, :] = np.matmul(obj_mat_single, locations.T).T + obj_pose_reset[i, :3]

        obj_aff_center_in_w = np.mean(visible_points_w[i].reshape(200, 3), axis=0)

        top_grasp = cfg['environment']['top']

        if top_grasp:
            hand_dir_x_w = np.zeros((1, 3))
            hand_dir_x_w[0, 2] = 1
        else:
            hand_dir_x_w = hand_center_sample_w - obj_aff_center_in_w
            hand_dir_x_w = hand_dir_x_w / np.linalg.norm(hand_dir_x_w, axis=1, keepdims=True)

        # get position and orientation of the wrist
        pos = obj_aff_center_in_w + 0.25 * hand_dir_x_w

        rot_mats, projection_lengths = sample_rot_mats(hand_dir_x_w, sample_num, visible_points_w[i])

        feasible_ik_flag = np.zeros((sample_num), dtype='bool')
        ik_results = np.zeros((sample_num, 6), dtype='float32')
        for j in range(sample_num):
            rot_mat = rot_mats[j]
            wrist_in_world = rot_mat
            wrist_bias_in_world = np.matmul(wrist_in_world, wrist_bias.T).T
            pos_in_ur5 = np.zeros((3, 1))
            pos_in_ur5[0, 0] = pos[0, 0] - 0. + wrist_bias_in_world[0, 0]
            pos_in_ur5[1, 0] = pos[0, 1] - 0. + wrist_bias_in_world[0, 1]
            pos_in_ur5[2, 0] = pos[0, 2] - 0.771 + wrist_bias_in_world[0, 2]
            pos_in_ur5_new = np.matmul(ur5_to_world.T, pos_in_ur5)

            wrist_mat_in_ur5 = np.matmul(ur5_to_world.T, wrist_in_world)

            gd = np.eye(4)
            gd[:3, :3] = wrist_mat_in_ur5
            gd[0, 3] = pos_in_ur5_new[0, 0]
            gd[1, 3] = pos_in_ur5_new[1, 0]
            gd[2, 3] = pos_in_ur5_new[2, 0]

            ik_result = ik.findClosestIK(gd, theta0)
            if ik_result is None or np.isnan(ik_result).any():
                feasible_ik_flag[j] = False
                continue
            else:
                qpos_reset_r[i, :6] = ik_result

                # check self collision
                env.reset_state(qpos_reset_r,
                                qpos_reset_l,
                                np.zeros((num_envs, 22), 'float32'),
                                np.zeros((num_envs, 22), 'float32'),
                                obj_pose_reset,
                                )
                temp_action_r = np.zeros((num_envs, act_dim), dtype='float32')
                temp_action_l = np.zeros((num_envs, act_dim), dtype='float32')
                _, _, _ = env.step(temp_action_r, temp_action_l)
                global_state = env.get_global_state()
                one_check = global_state[:, 124:128]
                contains_one = np.any(one_check == 1, axis=1)
                true_indices = np.where(contains_one)[0]
                if len(true_indices) > 0:
                    feasible_ik_flag[j] = False
                    continue
                else:
                    feasible_ik_flag[j] = True
                    ik_results[j, :] = ik_result
        feasible_indices = np.where(feasible_ik_flag)[0]
        scores = np.ones(sample_num, dtype='float32')
        scores = scores * 10000.
        if min(projection_lengths) < 0.18:
            for j in feasible_indices:
                if projection_lengths[j] < 0.18:
                    score1 = projection_lengths[j] * cfg['environment']['length_score_coeff']
                    score2 = abs(ik_results[j, 4] - 1.57) * cfg['environment']['angle_score_coeff']
                    score3 = ((projection_lengths[j] / min(projection_lengths)) ** 2) * cfg['environment']['length_ratio_coeff']
                    score4 = (abs(ik_results[j, 4]) - 3.2) * cfg['environment']['angle_score_coeff'] * 0.5
                    scores[j] = score1 + score2 + score3 + score4
                else:
                    scores[j] = 10000.
            best_index = np.argmin(scores)
            qpos_reset_r[i, :6] = ik_results[best_index]
        else:
            best_index = np.argmin(projection_lengths)
            qpos_reset_r[i, :6] = ik_results[best_index]

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

    final_actions = np.zeros((num_envs, act_dim), dtype='float32')

    biased = cfg['environment']['biased']
    if biased:
        obj_biased = np.zeros((num_envs, 1), dtype='float32')
        obj_pos_bias = np.random.uniform(-0.05, 0.05, (num_envs, 3)).astype('float32')
    else:
        obj_pos_bias = np.zeros((num_envs, 3), dtype='float32')


    for step in range(n_steps_r):
        frame_start = time.time()
        obs_r = obs_new_r
        obs_r = obs_r[:, :].astype('float32')

        # if step > 0:
        #     time.sleep(3)

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)
        # action_r[:, 6:] *= 0.1

        # print(action_r[:, :6])
        # print(action_r[:, 6:])

        if step < grasp_steps:
            final_actions = action_r
        else:
            action_r = final_actions
            action_r[:, :6] = theta0
            if step == grasp_steps:
                print("lift")
                env.switch_root_guidance(True)

        # # clip the first 6 dim of action to (-2, 2)
        # action_r[:, :6] = np.clip(action_r[:, :6], -2., 2.)

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe_vision_new()
        show_point = dis_info[:, 17:68].astype('float32').copy()
        env.set_joint_sensor_visual(show_point)

        # gs = env.get_global_state()
        # print(gs[0, 107])

        if biased:
            obj_pos_bias_current = np.zeros((num_envs, 3), dtype='float32')
            for i in range(num_envs):
                if np.min(dis_info[i, 0:17]) < 0.07 and obj_biased[i] == 0:
                    obj_biased[i] = 1
                    obj_pos_bias_current[i] = obj_pos_bias[i]
            env.switch_obj_pos(obj_pos_bias_current)

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    # global_state = env.get_global_state()
    # lifted = (global_state[:, 107] - obj_pose_reset[:, 2] > 0.1) * (np.linalg.norm(global_state[:, 112:115] - global_state[:, 105:108], axis=1) < 0.2)
    # print(lifted)
    print("end")



