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
import sys
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+


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
weight_saved = './../arm_rand/2024-11-04-16-42-02/full_50000_r.pt'

# weight_path_student = '2024-12-11-09-43-04/full_3000_r.pt'
# weight_path_student = '2024-12-16-22-33-03/full_8000_r.pt'
# weight_path_student = '2024-12-23-12-41-34/full_7500_r.pt'
# weight_path_student = '2024-12-29-10-01-25/full_6500_r.pt'
# weight_path_student = '2024-12-30-14-33-27/full_4500_r.pt'
# weight_path_student = '2024-12-30-14-47-53/full_4500_r.pt'
# weight_path_student = '2024-12-30-14-50-50/full_4500_r.pt'
# weight_path_student = '2024-12-31-10-43-32/full_4500_r.pt'
# weight_path_student = '2024-12-31-10-45-09/full_4000_r.pt'
# weight_path_student = '2024-12-31-14-08-48/full_4000_r.pt'
# weight_path_student = '2025-01-01-11-39-58/full_4000_r.pt'
# weight_path_student = '2025-01-02-10-43-23/full_5000_r.pt'
# weight_path_student = '2025-01-03-16-21-27/full_4000_r.pt'
# weight_path_student = '2025-01-03-16-22-25/full_3500_r.pt'

# weight_path_student = '2024-12-29-10-02-39/full_4500_r.pt'
# weight_path_student = '2025-01-02-00-16-48/full_5000_r.pt'
# weight_path_student = '2025-01-09-14-18-00/full_4500_r.pt'
# weight_path_student = '2025-01-09-14-19-41/full_6500_r.pt'
# weight_path_student = '2025-01-11-13-50-06/full_4500_r.pt'
# weight_path_student = '2025-01-11-13-51-35/full_6000_r.pt'
# weight_path_student = '2025-01-12-18-12-37/full_5000_r.pt'
# weight_path_student = '2025-01-15-10-48-01/full_3500_r.pt'
# weight_path_student = '2025-01-15-10-50-43/full_4000_r.pt'
# weight_path_student = '2025-01-15-10-51-51/full_4000_r.pt'
# weight_path_student = '2025-01-18-15-13-50/full_4500_r.pt'

# weight_path_student = '2025-01-12-17-56-20/full_3000_r.pt'


# weight_path_student = '2025-03-14-19-22-14/full_5000_r.pt'
weight_path_student = 'baseline/full_4500_r.pt'

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

print(f"Configuration file: \"{args.cfg}\"", file=sys.stdout)
print(f"Experiment name: \"{args.exp_name}\"", file=sys.stdout)

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
# cat_name = 'real_obj'
# cat_name = 'affordance_level'
# cat_name = 'large_scale_light_stable'
# cat_name = 'shapenet-30obj'
cat_name = 'new_training_set_eval'

if cat_name == 'shapenet-30obj':
    stable = True
else:
    stable = False

repeat_per_obj = 2

cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path, file=sys.stdout)

items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

obj_list = []
obj_path_list = []
obj_ori_list = folder_names

# if cat_name == 'large_scale_light_stable':
#     obj_ori_list = obj_ori_list[:50]

# Environment definition

num_envs = len(obj_ori_list) * repeat_per_obj
for i in range(repeat_per_obj):
    for item in obj_ori_list:
        obj_list.append(item)
        
activations = nn.LeakyReLU
cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs, file=sys.stdout)

if not cfg['environment']['randomization_eval']:
    print("no randomization", file=sys.stdout)
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
    print("randomization", file=sys.stdout)

env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished", file=sys.stdout)

for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)


ob_dim_r = 153
# act_dim = env.num_acts
act_dim = 22
print('ob dim', ob_dim_r, file=sys.stdout)
print('act dim', act_dim, file=sys.stdout)

tobeEncode_dim = 44
t_steps = 10
prop_latent_dim=26
aff_vec_dim = 51
total_obs_dim = tobeEncode_dim*t_steps + ob_dim_r

# Training
reward_clip = -2.0
grasp_steps = cfg['environment']['grasp_steps'] + 30
lift_steps = 100
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

print(f"load weight from {saver.data_dir.split('eval')[0] + weight_path_student}", file=sys.stdout)

checkpoint_student = torch.load(saver.data_dir.split('eval')[0] + weight_path_student, map_location=torch.device('cpu'))
actor_student_r.architecture.load_state_dict(checkpoint_student['actor_architecture_state_dict'])
actor_student_r.distribution.load_state_dict(checkpoint_student['actor_distribution_state_dict'])
prop_latent_encoder.load_state_dict(checkpoint_student['prop_latent_encoder_state_dict'])

lowest_points = np.zeros((num_envs, 1), dtype='float32')
if stable:
    stable_states = np.zeros((num_envs, 7), dtype='float32')
for i in range(num_envs):
    txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())
    if stable:
        stable_state_path = home_path + f"/rsc/{cat_name}/{obj_list[i]}/{obj_list[i]}.npy"
        stable_states[i] = np.load(stable_state_path)[-1, :7]
        if obj_list[i] == '2024-09-16_5_Plate_gold_69':
            stable_states[i, 2:] = [0.788, -0.00839127, 0.690102, 0.0161601, 0.723483]

success_rate = 0.0

# 创建一个字典来跟踪每个物体的失败次数和总尝试次数
object_failure_stats = {}
for obj_name in obj_list:
    if obj_name not in object_failure_stats:
        object_failure_stats[obj_name] = {"failures": 0, "attempts": 0}

for update in range(5):
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

    Ttarget2eef = np.array([[ 0., 0., 1., -0.095],
                            [ -1., 0., 0., 0.],
                            [ 0, -1., 0., 0.],
                            [ 0., 0., 0., 1.]])
    # Transformation matrix from UR5 robot frame to world frame
    Teefraisim2ik = np.array([[ 0., 1., 0., 0.],
                                [ -1., 0.,  0., 0.],
                                [ 0., 0.,  1., -0.771],
                                [ 0., 0.,  0., 1.]])

    theta0 = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    joint_weights = [1, 1, 1, 1, 1, 1]

    ik = InverseKinematicsUR5()
    ik.setJointWeights(joint_weights)
    ik.setJointLimits(-3.14, 3.14)


    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

    sample_num = cfg['environment']['sample_num']

    for i in range(num_envs):
        # print(i, obj_list[i])
        # get_meaningful_ik = False
        # while not get_meaningful_ik:
        # sample object states (not relavent for hardware deployment)
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
        # obj_pose_reset[i, 0] = sample_x
        # obj_pose_reset[i, 1] = sample_y
        # if cat_name == 'large_scale_light_stable':
        #     obj_pose_reset[i, 2:7] = stable_states[i, 2:7]
        #     obj_pose_reset[i, 2] += 0.005
        #     quats = stable_states[i, 3:7]
        # else:
        #     obj_pose_reset[i, 2] = 0.773 - lowest_points[i]
        #     obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]

        #     axis_angles = np.zeros((1, 3))
        #     axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
        #     quats = rotations.axisangle2quat(axis_angles)
        #     obj_pose_reset[i, 3:7] = quats
        # 均匀采样
        # sample_x = np.random.uniform(-0.35, 0.35)
        # sample_y = np.random.uniform(-0.8, -0.35)
        # angle = np.arctan2(sample_y, sample_x)
        while True:
            angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
            distance = np.random.uniform(0.45, 0.75)
            sample_x = distance * np.cos(angle)
            sample_y = distance * np.sin(angle)
            if sample_x < 0.25 and sample_x > -0.25:
                break

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
        obj_aff_center_in_w = np.mean(visible_points_w[i].reshape(200, 3), axis=0)

        top_grasp = cfg['environment']['top']
        # get the x_dir of the grasping frame
        if top_grasp:
            hand_dir_x_w = np.zeros((1, 3))
            hand_dir_x_w[0, 2] = 1
        else:
            hand_dir_x_w = hand_center_sample_w - obj_aff_center_in_w
            hand_dir_x_w = hand_dir_x_w / np.linalg.norm(hand_dir_x_w, axis=1, keepdims=True)

        # get position and orientation of the wrist
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
                ik_results[j, :] = ik_result
                feasible_ik_flag[j] = True

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
        current_obj_idx = true_idx // repeat_per_obj
        current_obj_env_indices = []
        for i in range(repeat_per_obj):
            current_obj_env_indices.append(current_obj_idx * repeat_per_obj + i)
        false_indices = [idx for idx in current_obj_env_indices if not contains_one[idx]]
        if len(false_indices)>0:
            chosen_index = np.random.choice(false_indices)
            qpos_reset_r[true_idx, :] = qpos_reset_r[chosen_index, :]
            obj_pose_reset[true_idx, :] = obj_pose_reset[chosen_index, :]
        else:
            qpos_reset_r[true_idx, :6] = [angle+np.pi/2-0.3, -1.57, 1.57, 0., 1.57, -1.57]
            obj_pose_reset[true_idx, 0] = 0.1
            obj_pose_reset[true_idx, 1] = -0.5

    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )

    obs_new_r, dis_info = env.observe_vision_new()
    # show_point = dis_info[:, 17:68].astype('float32').copy()
    aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
    # env.set_joint_sensor_visual(show_point)
    env.update_target(target_center)

    final_actions = np.zeros((num_envs, act_dim), dtype='float32')

    biased = cfg['environment']['biased']
    if biased:
        obj_biased = np.zeros((num_envs, 1), dtype='float32')
        obj_pos_bias = np.random.uniform(-0.05, 0.05, (num_envs, 3)).astype('float32')
    else:
        obj_pos_bias = np.zeros((num_envs, 3), dtype='float32')

    for step in range(n_steps_r):
        # print("step", step)
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
                # print("lift")
                env.switch_root_guidance(True)

        # # clip the first 6 dim of action to (-2, 2)
        # action_r[:, :6] = np.clip(action_r[:, :6], -2., 2.)

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe_vision_new()
        aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
        # show_point = dis_info[:, 17:68].astype('float32').copy()
        # env.set_joint_sensor_visual(show_point)

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

    global_state = env.get_global_state()
    # lifted = (global_state[:, 107] - obj_pose_reset[:, 2] > 0.1) * (
    #             np.linalg.norm(global_state[:, 112:115] - global_state[:, 105:108], axis=1) < 0.2)
    lifted = global_state[:, 107] - obj_pose_reset[:, 2] > 0.1
    print("current success rate", np.sum(lifted) / num_envs, file=sys.stdout)

    success_rate = (update * success_rate + np.sum(lifted) / num_envs) / (update + 1)
    print("average success rate", success_rate, file=sys.stdout)

    # 更新每个物体的失败统计
    for i in range(num_envs):
        obj_name = obj_list[i]  # 直接使用物体名称
        object_failure_stats[obj_name]["attempts"] += 1
        if not lifted[i]:
            object_failure_stats[obj_name]["failures"] += 1

    # Print the names of failed objects
    failed_indices = np.where(lifted == 0)[0]
    if len(failed_indices) > 0:
        print("Failed objects:", file=sys.stdout)
        for idx in failed_indices:
            print(f"  - {obj_list[idx]}", file=sys.stdout)
    else:
        print("All objects were successfully grasped!", file=sys.stdout)


# 在所有评估结束后，打印每个物体的失败统计
print("\n===== Object Failure Statistics =====", file=sys.stdout)
print(f"{'Object Name':<30} {'Failures':<10} {'Attempts':<10} {'Failure Rate (%)':<20}", file=sys.stdout)
print("-" * 70, file=sys.stdout)

# 按失败率从高到低排序
sorted_stats = sorted(object_failure_stats.items(), 
                     key=lambda x: x[1]["failures"] / x[1]["attempts"] if x[1]["attempts"] > 0 else 0, 
                     reverse=True)

for obj_name, stats in sorted_stats:
    failure_rate = (stats["failures"] / stats["attempts"] * 100) if stats["attempts"] > 0 else 0
    print(f"{obj_name:<30} {stats['failures']:<10} {stats['attempts']:<10} {failure_rate:.2f}%", file=sys.stdout)

# 计算总尝试次数和总失败次数
total_attempts = sum(stats["attempts"] for stats in object_failure_stats.values())
total_failures = sum(stats["failures"] for stats in object_failure_stats.values())
total_success_rate = ((total_attempts - total_failures) / total_attempts * 100) if total_attempts > 0 else 0

print("\nTotal success rate: {:.2f}%".format(total_success_rate), file=sys.stdout)





