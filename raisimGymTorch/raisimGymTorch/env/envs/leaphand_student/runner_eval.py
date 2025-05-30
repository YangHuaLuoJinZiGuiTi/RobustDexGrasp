#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import leaphand_student as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.leaphand_student import NormalSampler
from raisimGymTorch.helper.initial_pose_final import sample_rot_mats
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


exp_name = "leaphand_student"

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
weight_saved = './../leaphand_teacher/2024-11-21-16-02-19/full_35000_r.pt'

weight_path_student = 'left/full_1000_r.pt'




# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg_left_sw_noinertia.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_path_student)
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

cfg['environment']['num_threads'] = 1


if args.seed != 1:
    cfg['seed'] = args.seed

num_envs = args.num_repeats
activations = nn.LeakyReLU

cfg['environment']['visualize'] = True
cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)


cat_name = 'new_training_set_eval'

if cat_name == 'shapenet-30obj':
    stable = True
else:
    stable = False

cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)

items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

print(len(folder_names))

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
# obj_item = 'box'

if not cfg['environment']['randomization_eval']:
    print("no randomization")
    cfg['environment']['hardware']['randomize_friction'] = "0.8"
    cfg['environment']['hardware']['randomize_gains_hand_p'] = 0.
    cfg['environment']['hardware']['randomize_gains_hand_d'] = 0.
    cfg['environment']['hardware']['randomize_gains_arm_p'] = 0.
    cfg['environment']['hardware']['randomize_gains_arm_d'] = 0.
    cfg['environment']['hardware']['randomize_gc_hand'] = 0.
    cfg['environment']['hardware']['randomize_gc_arm'] = 0.
    cfg['environment']['hardware']['randomize_frame_position'] = 0.
    cfg['environment']['hardware']['randomize_frame_orientation'] = 0.
else:
    print("randomization")

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
aff_vec_dim = 51
total_obs_dim = tobeEncode_dim*t_steps + ob_dim_r

# Training
reward_clip = -2.0
grasp_steps = cfg['environment']['grasp_steps']
lift_steps = 20
n_steps_r = grasp_steps + lift_steps
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

lowest_points = np.zeros((num_envs, 1), dtype='float32')
if stable:
    stable_states = np.zeros((num_envs, 7), dtype='float32')
for i in range(num_envs):
    txt_file_path = os.path.join(directory_path, obj_item) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())

    if stable:
        stable_state_path = home_path + f"/rsc/{cat_name}/{obj_item}.npy"
        stable_states[i] = np.load(stable_state_path)[-1, :7]
        if obj_item == '2024-09-16_5_Plate_gold_69':
            stable_states[i, 2:] = [0.788, -0.00839127, 0.690102, 0.0161601, 0.723483]

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

    # Set wrist bias (offset for proper hand positioning, from hand center to wrist for IK calculation)
    # allegro hand
    # Ttarget2eef = np.array([[ 1., 0., 0., -0.01],
    #                         [ 0., 1., 0., 0.],
    #                         [ 0., 0., 1., -0.095],
    #                         [ 0., 0., 0., 1.]])
    # leap hand
    Ttarget2eef = np.array([[ 0., 0., 1., -0.11],
                            [ -1., 0., 0., 0.],
                            [ 0, -1., 0., -0.05],
                            [ 0., 0., 0., 1.]])
    # Transformation matrix from UR5 robot frame to world frame
    Teefraisim2ik = np.array([[ 0., 1., 0., 0.],
                                [ -1., 0.,  0., 0.],
                                [ 0., 0.,  1., -0.771],
                                [ 0., 0.,  0., 1.]])

    # Initial joint angles for UR5 robot arm
    theta0 = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    # Joint weights for inverse kinematics
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
        # # get_meaningful_ik = False
        # # while not get_meaningful_ik:
        # # sample object states (not relavent for hardware deployment)
        # sample_x = 0.15
        # sample_y = 0.2 - 0.75152
        # while True:
        #     angle = np.random.uniform(0, 2 * np.pi)
        #     distance = np.random.uniform(0.45, 0.75)
        #     sample_x = distance * np.cos(angle)
        #     sample_y = distance * np.sin(angle)
        #     if sample_y < 0.3 - 0.75152:
        #         # print(sample_x, sample_y, distance)
        #         break

        # 均匀采样
        # sample_x = np.random.uniform(-0.35, 0.35)
        # sample_y = np.random.uniform(-0.8, -0.35)
        
        # # 计算对应的角度
        # angle = np.arctan2(sample_y, sample_x)
        while True:
            angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
            distance = np.random.uniform(0.45, 0.75)
            sample_x = distance * np.cos(angle)
            sample_y = distance * np.sin(angle)
            if sample_x < 0.25 and sample_x > -0.25:
                break
        print(sample_x, sample_y, angle)

        obj_pose_reset[i, 0] = sample_x
        obj_pose_reset[i, 1] = sample_y

        if stable:
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

        # get the partial point cloud (not relavent for hardware deployment)
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

        # get the x_dir of the grasping frame
        obj_aff_center_in_w = np.mean(visible_points_w[i].reshape(200,3), axis=0)

        top_grasp = cfg['environment']['top']
        # get the x_dir of the grasping frame
        if top_grasp:
            hand_dir_x_w = np.zeros((1, 3))
            hand_dir_x_w[0, 2] = 1
        else:
            hand_dir_x_w = hand_center_sample_w - obj_aff_center_in_w
            hand_dir_x_w = hand_dir_x_w / np.linalg.norm(hand_dir_x_w, axis=1, keepdims=True)
        pos = obj_aff_center_in_w + 0.15 * hand_dir_x_w


        rot_mats, projection_lengths = sample_rot_mats(hand_dir_x_w, sample_num, visible_points_w[i])

        feasible_ik_flag = np.zeros((sample_num), dtype='bool')
        ik_results = np.zeros((sample_num, 6), dtype='float32')
        for j in range(sample_num):
            Ttarget = np.eye(4)
            Ttarget[:3, :3] = rot_mats[j, :]
            Ttarget[:3, 3] = pos[0, :3]
            Teef = Ttarget @ Ttarget2eef
            gd = Teefraisim2ik @ Teef

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
                    ik_results[j] = ik_result
        feasible_indices = np.where(feasible_ik_flag)[0]
        scores = np.ones(sample_num, dtype='float32')
        scores = scores * 10000.
        if min(projection_lengths) < 0.14:
            for j in feasible_indices:
                if projection_lengths[j] < 0.14:
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
    # show_point = dis_info[:, 17:68].astype('float32').copy()
    aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
    #env.set_joint_sensor_visual(show_point)
    # env.update_target(target_center)

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
        #     time.sleep(2)

        encode_obs = torch.from_numpy(obs_r[:, :tobeEncode_dim * t_steps]).to(device)

        student_latent = prop_latent_encoder(encode_obs)
        student_mlp_obs = torch.cat((torch.from_numpy(obs_r[:, -ob_dim_r:-ob_dim_r + tobeEncode_dim]),
                                     student_latent.cpu(),
                                     torch.from_numpy(obs_r[:, -ob_dim_r + tobeEncode_dim + prop_latent_dim:-aff_vec_dim]),
                                     torch.from_numpy(aff_vec)), dim=1).to(device)

        action_r = actor_student_r.architecture.architecture(student_mlp_obs.to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)

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
        aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
        # show_point = dis_info[:, 17:68].astype('float32').copy()
        #env.set_joint_sensor_visual(show_point)
        
        time.sleep(0.2)

        if biased:
            obj_pos_bias_current = np.zeros((num_envs, 3), dtype='float32')
            for i in range(num_envs):
                if np.min(dis_info[i, 0:17]) < 0.07 and obj_biased[i] == 0:
                # if random.uniform (0, 1) < 0.3 and obj_biased[i] == 0 and np.min(dis_info[i, 0:17]) > 0.07:
                    obj_biased[i] = 1
                    obj_pos_bias_current[i] = obj_pos_bias[i]
            env.switch_obj_pos(obj_pos_bias_current)

        # frame_end = time.time()
        # wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        # if wait_time > 0.:
        #     time.sleep(wait_time)

    print("end")

