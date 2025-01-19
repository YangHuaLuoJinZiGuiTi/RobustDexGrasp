#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_teacher_biased as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.allegro_teacher_biased import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_faive, get_initial_pose_faive_random, get_initial_pose_allegro_new, get_initial_pose_allegro_arm_rand, get_initial_pose_allegro_arm_rand_test, get_initial_pose_allegro_arm_partial_safe
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
# weight_saved = '2024-12-15-09-58-40/full_19000_r.pt'
# weight_saved = '2024-12-15-11-00-37/full_19000_r.pt'
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
# weight_saved = '2024-12-27-18-08-06/full_16500_r.pt'
# weight_saved = '2024-12-27-18-09-12/full_12500_r.pt'
# weight_saved = '2024-12-27-18-11-12/full_15000_r.pt'
# weight_saved = '2024-12-31-18-57-49/full_16000_r.pt'
# weight_saved = '2024-12-31-19-13-20/full_12000_r.pt'
# weight_saved = '2024-12-31-20-31-15/full_14500_r.pt'
# weight_saved = '2025-01-01-10-01-41/full_16000_r.pt'
# weight_saved = '2025-01-02-11-39-41/full_4500_r.pt'
# weight_saved = '2025-01-03-18-29-42/full_20000_r.pt'
# weight_saved = '2025-01-05-11-24-07/full_8500_r.pt'
# weight_saved = '2025-01-06-14-32-58/full_8000_r.pt'
# weight_saved = '2025-01-06-14-37-46/full_9000_r.pt'
# weight_saved = '2025-01-06-19-53-12/full_3500_r.pt'
# weight_saved = '2025-01-07-21-04-51/full_10000_r.pt'
# weight_saved = '2025-01-08-10-20-03/full_4000_r.pt'
# weight_saved = '2025-01-08-10-22-54/full_5500_r.pt'
# weight_saved = '2025-01-12-18-16-55/full_12000_r.pt'
# weight_saved = '2025-01-12-18-18-27/full_14000_r.pt'
# weight_saved = '2025-01-12-18-19-14/full_16000_r.pt'
weight_saved = '2025-01-15-11-35-22/full_3000_r.pt'

# weight_saved = '2024-12-27-18-11-12/full_15000_r.pt'
# weight_saved = '2025-01-07-20-39-18/full_15000_r.pt'

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

# num_envs = args.num_repeats
# activations = nn.LeakyReLU

cfg['environment']['visualize'] = False


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
cat_name = 'large_scale_light_stable'
cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)

items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

obj_list = []
obj_path_list = []
obj_ori_list = folder_names

if cat_name == 'large_scale_light_stable':
    obj_ori_list = obj_ori_list[:50]

# obj_item = choice(obj_ori_list)
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

if cat_name != 'large_scale_light_stable':
    num_envs = len(obj_ori_list) * 3
else:
    num_envs = len(obj_ori_list)
activations = nn.LeakyReLU
cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)

if cat_name != 'large_scale_light_stable':
    for i in range(3):
        for item in obj_ori_list:
            obj_list.append(item)
else:
    for item in obj_ori_list:
        obj_list.append(item)

# Environment definition
env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished")

for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)


ob_dim_r = 153
# act_dim = env.num_acts
act_dim = 22
print('ob dim', ob_dim_r)
print('act dim', act_dim)

# Training
reward_clip = -2.0
grasp_steps = cfg['environment']['grasp_steps'] + 30
lift_steps = 100
n_steps_r = grasp_steps + lift_steps
total_steps_r = n_steps_r * env.num_envs

# RL network

actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)


test_dir = True

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/quantitative_eval.py"], test_dir=test_dir)


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
stable_states = np.zeros((num_envs, 7), dtype='float32')
for i in range(num_envs):
    txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())
    if cat_name == 'large_scale_light_stable':
        stable_state_path = home_path + f"/rsc/stable_states/{cat_name}/{obj_list[i]}.npy"
        stable_states[i] = np.load(stable_state_path)

success_rate = 0.0

for update in range(5):
    np.random.seed(int(time.time()))
    start = time.time()

    qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

    target_center = np.zeros_like(env.affordance_center)

    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']

    hand_center_sample_w = np.zeros((1, 3))
    hand_center_sample_w[0, 0] = 0.669872 - 0.55
    hand_center_sample_w[0, 1] = 0.141735 - 0.75152
    hand_center_sample_w[0, 2] = 1.5  # 1.11052

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

    # partial_obs = cfg['environment']['partial_pt_init_pose']
    #
    # if partial_obs:

    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

    for i in range(num_envs):
        # sample object states
        sample_x = 0.15
        sample_y = 0.2 - 0.75152
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            # angle = np.random.uniform(3*np.pi/2, 2 * np.pi)
            distance = np.random.uniform(0.45, 0.75)
            sample_x = distance * np.cos(angle)
            sample_y = distance * np.sin(angle)
            if sample_y < 0.3 - 0.75152:
                # print(sample_x, sample_y, distance)
                break
        obj_pose_reset[i, 0] = sample_x
        obj_pose_reset[i, 1] = sample_y
        if cat_name == 'large_scale_light_stable':
            obj_pose_reset[i, 2:7] = stable_states[i, 2:7]
            obj_pose_reset[i, 2] += 0.005
            quats = stable_states[i, 3:7]
        else:
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

        no_feasible_ik = False
        top_grasp = cfg['environment']['top']
        # inverse_grasp = False
        while True:
            if not no_feasible_ik:
                # get the x_dir of the grasping frame
                if top_grasp:
                    hand_dir_x_w = np.zeros((1, 3))
                    hand_dir_x_w[0, 2] = 1
                else:
                    hand_dir_x_w = hand_center_sample_w - obj_aff_center_in_w
                    hand_dir_x_w = hand_dir_x_w / np.linalg.norm(hand_dir_x_w, axis=1, keepdims=True)

                # get position and orientation of the wrist
                pos = obj_aff_center_in_w + 0.25 * hand_dir_x_w
                rot = get_initial_pose_allegro_arm_partial_safe(visible_points_w[i], hand_dir_x_w, np.eye(3), top=False)
                if rot is None:
                    if top_grasp:
                        # if inverse_grasp:
                        #     no_feasible_ik = True
                        # else:
                        #     inverse_grasp = True
                        no_feasible_ik = True
                    else:
                        top_grasp = True
                    continue
                wrist_in_world = rot
                qpos_reset_r[i, :3] = pos[0, :]

                # from grasping frame pos to wrist pos
                wrist_bias_in_world = np.matmul(wrist_in_world, wrist_bias.T).T
                pos_in_ur5 = np.zeros((3, 1))
                pos_in_ur5[0, 0] = qpos_reset_r[i, 0] - 0. + wrist_bias_in_world[0, 0]
                pos_in_ur5[1, 0] = qpos_reset_r[i, 1] - 0. + wrist_bias_in_world[0, 1]
                pos_in_ur5[2, 0] = qpos_reset_r[i, 2] - 0.771 + wrist_bias_in_world[0, 2]
                pos_in_ur5_new = np.matmul(ur5_to_world.T, pos_in_ur5)

                wrist_mat_in_ur5 = np.matmul(ur5_to_world.T, wrist_in_world)

                gd = np.eye(4)
                gd[:3, :3] = wrist_mat_in_ur5
                gd[0, 3] = pos_in_ur5_new[0, 0]
                gd[1, 3] = pos_in_ur5_new[1, 0]
                gd[2, 3] = pos_in_ur5_new[2, 0]

                if ik.findClosestIK(gd, theta0) is None:
                    if top_grasp:
                        # if inverse_grasp:
                        #     no_feasible_ik = True
                        # else:
                        #     inverse_grasp = True
                        no_feasible_ik = True
                    else:
                        top_grasp = True
                    continue
                else:
                    qpos_reset_r[i, :6] = ik.findClosestIK(gd, theta0)

                if math.isnan(qpos_reset_r[i, 0]):
                    if top_grasp:
                        # if inverse_grasp:
                        #     no_feasible_ik = True
                        # else:
                        #     inverse_grasp = True
                        no_feasible_ik = True
                    else:
                        top_grasp = True
                    continue
                else:
                    # # check self collision
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
                    # if len(true_indices) > 0:
                    #     no_feasible_ik = True
                    #     continue
                    # else:
                    # if qpos_reset_r[i, 4] < -1.57 or qpos_reset_r[i, 4] > 2:
                    #     if top_grasp:
                    #         # if inverse_grasp:
                    #         #     no_feasible_ik = True
                    #         # else:
                    #         #     inverse_grasp = True
                    #         no_feasible_ik = True
                    #     else:
                    #         top_grasp = True
                    #     continue
                    # else:
                    break
            else:
                # get the x_dir of the grasping frame
                hand_dir_x_w = np.zeros((1, 3))
                hand_dir_x_w[0, 2] = 1

                z_dir_in_world = -obj_aff_center_in_w.copy().reshape(1, 3)
                z_dir_in_world[:, 2] = 0.
                z_dir_in_world = z_dir_in_world / np.linalg.norm(z_dir_in_world, axis=1, keepdims=True)

                # get position and orientation of the wrist
                pos = obj_aff_center_in_w + 0.25 * hand_dir_x_w
                rot = get_initial_pose_allegro_arm_partial_safe(visible_points_w[i], hand_dir_x_w, np.eye(3), top=True, z_dir_cmd=z_dir_in_world)
                if rot is None:
                    qpos_reset_r[i, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                    break
                wrist_in_world = rot
                qpos_reset_r[i, :3] = pos[0, :]

                # from grasping frame pos to wrist pos
                wrist_bias_in_world = np.matmul(wrist_in_world, wrist_bias.T).T
                pos_in_ur5 = np.zeros((3, 1))
                pos_in_ur5[0, 0] = qpos_reset_r[i, 0] - 0. + wrist_bias_in_world[0, 0]
                pos_in_ur5[1, 0] = qpos_reset_r[i, 1] - 0. + wrist_bias_in_world[0, 1]
                pos_in_ur5[2, 0] = qpos_reset_r[i, 2] - 0.771 + wrist_bias_in_world[0, 2]
                pos_in_ur5_new = np.matmul(ur5_to_world.T, pos_in_ur5)

                wrist_mat_in_ur5 = np.matmul(ur5_to_world.T, wrist_in_world)

                gd = np.eye(4)
                gd[:3, :3] = wrist_mat_in_ur5
                gd[0, 3] = pos_in_ur5_new[0, 0]
                gd[1, 3] = pos_in_ur5_new[1, 0]
                gd[2, 3] = pos_in_ur5_new[2, 0]

                if ik.findClosestIK(gd, theta0) is None:
                    qpos_reset_r[i, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                    break
                else:
                    qpos_reset_r[i, :6] = ik.findClosestIK(gd, theta0)

                if math.isnan(qpos_reset_r[i, 0]):
                    qpos_reset_r[i, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                    break
                else:
                    # # check self collision
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
                    # if len(true_indices) > 0:
                    #     qpos_reset_r[i, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                    #     break
                    # else:
                    # if qpos_reset_r[i, 4] < -1.57 or qpos_reset_r[i, 4] > 2:
                    #     qpos_reset_r[i, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                    break

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
    for true_idx in true_indices:
        if cat_name == 'ycb_urdf_light':
            current_obj_idx = true_idx // 3
            current_obj_env_indices = [current_obj_idx * 3, current_obj_idx * 3 + 1, current_obj_idx * 3 + 2]
            false_indices = [idx for idx in current_obj_env_indices if not contains_one[idx]]
            if len(false_indices) > 0:
                chosen_index = np.random.choice(false_indices)
                qpos_reset_r[true_idx, :] = qpos_reset_r[chosen_index, :]
                obj_pose_reset[true_idx, :] = obj_pose_reset[chosen_index, :]
            else:
                qpos_reset_r[true_idx, :6] = [angle + np.pi / 2, -1.57, 1.57, 0., 1.57, -1.57]
                obj_pose_reset[true_idx, 0] = 0.15
                obj_pose_reset[true_idx, 1] = 0.2 - 0.75152
        else:
            qpos_reset_r[true_idx, :6] = [angle + np.pi / 2, -1.57, 1.57, 0., 1.57, -1.57]
            obj_pose_reset[true_idx, 0] = 0.15
            obj_pose_reset[true_idx, 1] = 0.2 - 0.75152

    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )

    obs_new_r, dis_info = env.observe_vision_new()

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

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)

        if step < grasp_steps:
            final_actions = action_r
        else:
            action_r = final_actions
            action_r[:, :6] = theta0
            if step == grasp_steps:
                env.switch_root_guidance(True)

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe_vision_new()

        if biased:
            obj_pos_bias_current = np.zeros((num_envs, 3), dtype='float32')
            for i in range(num_envs):
                if np.min(dis_info[i, 0:17]) < 0.07 and obj_biased[i] == 0:
                    obj_biased[i] = 1
                    obj_pos_bias_current[i] = obj_pos_bias[i]
            env.switch_obj_pos(obj_pos_bias_current)

        # frame_end = time.time()
        # wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        # if wait_time > 0.:
        #     time.sleep(wait_time)

    global_state = env.get_global_state()
    # lifted = (global_state[:, 107] - obj_pose_reset[:, 2] > 0.1) * (np.linalg.norm(global_state[:, 112:115] - global_state[:, 105:108], axis=1) < 0.2)
    lifted = global_state[:, 107] - obj_pose_reset[:, 2] > 0.1
    print("current success rate", np.sum(lifted) / num_envs)

    success_rate = (update * success_rate + np.sum(lifted) / num_envs) / (update + 1)
    print("average success rate", success_rate)


