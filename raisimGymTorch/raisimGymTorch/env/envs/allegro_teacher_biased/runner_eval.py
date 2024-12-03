#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_teacher_biased as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.allegro_teacher_biased import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_faive, get_initial_pose_faive_random, get_initial_pose_allegro_new, get_initial_pose_allegro_arm_rand, get_initial_pose_allegro_arm_rand_test, get_initial_pose_allegro_arm_partial
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
weight_saved = '2024-11-04-16-42-02/full_50000_r.pt'

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
grasp_steps = 120
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


    partial_obs = True

    if partial_obs:

        visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
        visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

        view_point_world = np.zeros((200, 3))
        view_point_world[:, 0] = 0.8 - 0.55
        view_point_world[:, 1] = 0.2 - 0.75152
        view_point_world[:, 2] = 1.5

        for i in range(num_envs):
            get_meaningful_ik = False
            while not get_meaningful_ik:
                # sample object states
                sample_x = 0.15
                sample_y = 0.2 - 0.75152
                while True:
                    angle = np.random.uniform(0, 2 * np.pi)
                    distance = np.random.uniform(0.45, 0.75)
                    sample_x = distance * np.cos(angle)
                    sample_y = distance * np.sin(angle)
                    if sample_y < 0.3 - 0.75152:
                        # print(sample_x, sample_y, distance)
                        break
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

                # # get the x_dir of the grasping frame
                # obj_aff_center_in_obj = np.mean(visible_points_obj[i].reshape(200, 3), axis=0)
                # hand_center_obj = np.matmul(obj_mat_single.T, (hand_center_sample_w - obj_pose_reset[i, :3]).T).T
                # hand_dir_x_in_obj = hand_center_obj - obj_aff_center_in_obj
                # hand_dir_x_in_obj = hand_dir_x_in_obj / np.linalg.norm(hand_dir_x_in_obj, axis=1, keepdims=True)
                #
                # target_center[i, :] = obj_aff_center_in_obj - 0.01 * hand_dir_x_in_obj
                # pos = obj_aff_center_in_obj + 0.25 * hand_dir_x_in_obj
                # rot = get_initial_pose_allegro_arm_partial(visible_points_obj[i], hand_dir_x_in_obj, obj_mat_single,
                #                                            top=False)
                # if rot is None:
                #     hand_dir_x_in_obj[0, :] = 0
                #     hand_dir_x_in_obj[0, 2] = 1
                #     rot = get_initial_pose_allegro_arm_partial(visible_points_obj[i], hand_dir_x_in_obj, obj_mat_single,
                #                                                top=True)
                #
                # wrist_in_world = np.matmul(obj_mat_single, rot)
                # wrist_pose = rotations.mat2euler(wrist_in_world)
                # qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat_single, pos[0, :])

                # get the x_dir of the grasping frame
                obj_aff_center_in_w = np.mean(visible_points_w[i].reshape(200, 3), axis=0)
                hand_dir_x_w = hand_center_sample_w - obj_aff_center_in_w
                hand_dir_x_w = hand_dir_x_w / np.linalg.norm(hand_dir_x_w, axis=1, keepdims=True)

                # get position and orientation of the wrist
                pos = obj_aff_center_in_w + 0.25 * hand_dir_x_w
                rot = get_initial_pose_allegro_arm_partial(visible_points_w[i], hand_dir_x_w, np.eye(3), top=False)
                if rot is None:
                    hand_dir_x_w[0, :] = 0
                    hand_dir_x_w[0, 2] = 1
                    rot = get_initial_pose_allegro_arm_partial(visible_points_w[i], hand_dir_x_w, np.eye(3), top=True)
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
                        continue
                    else:
                        get_meaningful_ik = True

    else:
        for i in range(num_envs):
            get_meaningful_ik = False
            while not get_meaningful_ik:
                sample_x = 0.15
                sample_y = 0.2 - 0.75152
                while True:
                    angle = np.random.uniform(0, 2 * np.pi)
                    distance = np.random.uniform(0.45, 0.75)
                    sample_x = distance * np.cos(angle)
                    sample_y = distance * np.sin(angle)
                    if sample_y < (0.3-0.75152):
                        break
                obj_pose_reset[i, 0] = sample_x
                obj_pose_reset[i, 1] = sample_y
                obj_pose_reset[i, 2] = 0.773 - lowest_points[i]
                obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]

                axis_angles = np.zeros((1, 3))
                axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
                quats = rotations.axisangle2quat(axis_angles)
                obj_pose_reset[i, 3:7] = quats

                obj_aff_center_in_obj = env.affordance_center[i].copy()
                obj_mat_single = rotations.quat2mat(quats).reshape(3, 3)

                obj_aff_center_in_world = np.matmul(obj_mat_single, obj_aff_center_in_obj.T).T
                obj_aff_center_in_world = obj_aff_center_in_world + obj_pose_reset[i, :3]

                hand_dir_x = hand_center_sample_w - obj_aff_center_in_world
                hand_dir_x = hand_dir_x / np.linalg.norm(hand_dir_x, axis=1, keepdims=True)
                hand_dir_x_in_obj = np.matmul(obj_mat_single.T, hand_dir_x.T).T

                rot, pos, target = get_initial_pose_allegro_arm_rand_test(env.aff_mesh[i], env.affordance_pcd[i],
                                                                          hand_dir_x_in_obj, env.affordance_center[i], obj_mat_single,
                                                                          top=False)
                if rot is None:
                    hand_dir_x_in_obj[0, :] = 0
                    hand_dir_x_in_obj[0, 2] = 1
                    rot, pos, target = get_initial_pose_allegro_arm_rand_test(env.aff_mesh[i], env.affordance_pcd[i],
                                                                              hand_dir_x_in_obj,
                                                                              env.affordance_center[i], obj_mat_single, top=True)

                wrist_mat = rot
                wrist_in_world = np.matmul(obj_mat_single, wrist_mat)
                wrist_pose = rotations.mat2euler(wrist_in_world)
                qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat_single, pos[0, :])

                target_center[i, :] = target[:]

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
                        continue
                    else:
                        get_meaningful_ik = True




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

    biased = False
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
        #     time.sleep(2)

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)
        # action_r[:, :6] = 0

        if step < grasp_steps:
            final_actions = action_r
        else:
            action_r = final_actions
            action_r[:, :6] = theta0
            if step == grasp_steps:
                print("lift")
                env.switch_root_guidance(True)

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe_vision_new()
        show_point = dis_info[:, 17:68].astype('float32').copy()
        env.set_joint_sensor_visual(show_point)

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

    print("end")



