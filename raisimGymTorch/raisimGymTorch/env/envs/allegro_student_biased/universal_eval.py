#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_student_biased as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.allegro_student_biased import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_faive, get_initial_pose_faive_random, get_initial_pose_allegro_new, get_initial_pose_allegro_arm_rand, get_initial_pose_allegro_arm_rand_test, get_initial_pose_allegro_arm_partial_safe
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

# weight_saved = './../arm_rand/2024-11-04-16-42-02/full_50000_r.pt'

# weight_path_student = '2024-12-29-10-02-39/full_4500_r.pt'
# weight_path_student = '2025-01-02-00-16-48/full_5000_r.pt'
# weight_path_student = '2025-01-09-14-18-00/full_4500_r.pt'
# weight_path_student = '2025-01-09-14-19-41/full_6500_r.pt'
# weight_path_student = '2025-01-11-13-51-35/full_6000_r.pt'

# weight_path_student = '2025-01-11-13-50-06/full_4500_r.pt'
weight_path_student = '2025-01-12-17-56-20/full_3000_r.pt'

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_path_student)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-itr', '--num_iterations', type=int, default=1)
parser.add_argument('-group', '--group_name', type=int, default=0)
parser.add_argument('-mode', '--mode', type=int, default=-1)

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

# if args.seed != 1:
#     cfg['seed'] = args.seed

mode = args.mode
if mode == -1:
    # cat_name = "large_scale_light_stable"
    cat_name = "temp"
elif mode == 0:
    cat_name = f"surdf_group{args.group_name}_s"
elif mode == 1:
    cat_name = f"surdf_group{args.group_name}_m"
elif mode == 2:
    cat_name = f"surdf_group{args.group_name}_l"

cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)

# # Filter out only the folders (directories) from the list of items
if mode == -1:
    # load_ids = np.load(home_path + f'/rsc/{cat_name}.npy').tolist()
    items = os.listdir(directory_path)
    load_ids = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
else:
    load_ids = np.load(home_path + f'/rsc/stable_ids/{cat_name}.npy').tolist()
# elif mode == 0:
#     load_ids = np.load(f'/mnt/ssd/data3/hui/selected_ids_new/surdf_group{args.group_name}_s.npy').tolist()
#     # load_ids = np.load(f'/home/huizhang/work/data_related/data/processed_objaverse_obj/selected_ids_new/surdf_group{args.group_name}_s.npy').tolist()
# elif mode == 1:
#     load_ids = np.load(f'/mnt/ssd/data3/hui/selected_ids_new/surdf_group{args.group_name}_m.npy').tolist()
#     # load_ids = np.load(f'/home/huizhang/work/data_related/data/processed_objaverse_obj/selected_ids_new/surdf_group{args.group_name}_m.npy').tolist()
# elif mode == 2:
#     load_ids = np.load(f'/mnt/ssd/data3/hui/selected_ids_new/surdf_group{args.group_name}_l.npy').tolist()
print("number of objects", len(load_ids))



obj_path_list = []
obj_ori_list = load_ids

num_env_per_iter = 500
iter_num = args.num_iterations
obj_list = []
if iter_num < len(obj_ori_list) // num_env_per_iter:
    num_envs = num_env_per_iter
else:
    num_envs = len(obj_ori_list) % num_env_per_iter
for i in range(num_envs):
    obj_list.append(obj_ori_list[i + iter_num * num_env_per_iter])
print("iter_num", iter_num)

# cfg['environment']['visualize'] = True
# obj_list = [choice(obj_ori_list) for _ in range(1)]

num_envs = len(obj_list)

activations = nn.LeakyReLU
cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)

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

print(f"load weight from {saver.data_dir.split('eval')[0] + weight_path_student}")

checkpoint_student = torch.load(saver.data_dir.split('eval')[0] + weight_path_student, map_location=torch.device('cpu'))
actor_student_r.architecture.load_state_dict(checkpoint_student['actor_architecture_state_dict'])
actor_student_r.distribution.load_state_dict(checkpoint_student['actor_distribution_state_dict'])
prop_latent_encoder.load_state_dict(checkpoint_student['prop_latent_encoder_state_dict'])

lowest_points = np.zeros((num_envs, 1), dtype='float32')
stable_states = np.zeros((num_envs, 7), dtype='float32')
for i in range(num_envs):
    txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())
    stable_state_path = home_path + f"/rsc/stable_states/{cat_name}/{obj_list[i]}.npy"
    stable_states[i] = np.load(stable_state_path)

success_rate = 0.0

for update in range(1):
    np.random.seed(int(time.time()))
    start = time.time()

    qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

    target_center = np.zeros_like(env.affordance_center)

    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']


    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

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

    for i in range(num_envs):
        # sample object states (not relavent for hardware deployment)
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
        obj_pose_reset[i, 2:7] = stable_states[i, 2:7]
        obj_pose_reset[i, 2] += 0.005

        # obj_pose_reset[i, 2] = 0.773 - lowest_points[i]
        # obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]
        # axis_angles = np.zeros((1, 3))
        # axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
        # quats = rotations.axisangle2quat(axis_angles)
        quats = stable_states[i, 3:7]


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
                    qpos_reset_r[i, :6] = [angle + np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
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
                    qpos_reset_r[i, :6] = [angle + np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                    break
                else:
                    qpos_reset_r[i, :6] = ik.findClosestIK(gd, theta0)

                if math.isnan(qpos_reset_r[i, 0]):
                    qpos_reset_r[i, :6] = [angle + np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                    break
                else:
                    # if qpos_reset_r[i, 4] < -1.57 or qpos_reset_r[i, 4] > 2:
                    #     qpos_reset_r[i, :6] = [angle + np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
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
        qpos_reset_r[true_idx, :6] = [angle + np.pi / 2, -1.57, 1.57, 0., 1.57, -1.57]
        obj_pose_reset[true_idx, 0] = 0.15
        obj_pose_reset[true_idx, 1] = 0.2 - 0.75152
        # current_obj_idx = true_idx // 3
        # current_obj_env_indices = [current_obj_idx * 3, current_obj_idx * 3 + 1, current_obj_idx * 3 + 2]
        # false_indices = [idx for idx in current_obj_env_indices if not contains_one[idx]]
        # if len(false_indices) > 0:
        #     chosen_index = np.random.choice(false_indices)
        #     qpos_reset_r[true_idx, :] = qpos_reset_r[chosen_index, :]
        #     obj_pose_reset[true_idx, :] = obj_pose_reset[chosen_index, :]
        # else:
        #     qpos_reset_r[true_idx, :6] = [angle + np.pi / 2, -1.57, 1.57, 0., 1.57, -1.57]
        #     obj_pose_reset[true_idx, 0] = 0.15
        #     obj_pose_reset[true_idx, 1] = 0.2 - 0.75152

    print("complete initial pose generation")

    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )

    obs_new_r, dis_info = env.observe_vision_new()
    aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))

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
                env.switch_root_guidance(True)


        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe_vision_new()
        aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))

        if biased:
            obj_pos_bias_current = np.zeros((num_envs, 3), dtype='float32')
            for i in range(num_envs):
                if np.min(dis_info[i, 0:17]) < 0.07 and obj_biased[i] == 0:
                # if random.uniform (0, 1) < 0.3 and obj_biased[i] == 0 and np.min(dis_info[i, 0:17]) > 0.07:
                    obj_biased[i] = 1
                    obj_pos_bias_current[i] = obj_pos_bias[i]
            env.switch_obj_pos(obj_pos_bias_current)

    global_state = env.get_global_state()
    # lifted = (global_state[:, 107] - obj_pose_reset[:, 2] > 0.1) * (
    #             np.linalg.norm(global_state[:, 112:115] - global_state[:, 105:108], axis=1) < 0.2)
    lifted = global_state[:, 107] - obj_pose_reset[:, 2] > 0.1
    print("current success rate", np.sum(lifted) / num_envs)

    success_rate = (update * success_rate + np.sum(lifted) / num_envs) / (update + 1)
    print("average success rate", success_rate)

# direct_save = "/mnt/ssd/data3/hui/anydex_results/"
direct_save = home_path + f'/rsc/anydex_results/'

try:
    if mode == -1:
        pre_result = np.load(home_path + f"/rsc/shapenet_{iter_num - 1}.npy", allow_pickle=True).item()
    elif mode == 0:
        pre_result = np.load(direct_save + f"group{args.group_name}_s_results_{iter_num - 1}.npy", allow_pickle=True).item()
    elif mode == 1:
        pre_result = np.load(direct_save + f"group{args.group_name}_m_results_{iter_num - 1}.npy", allow_pickle=True).item()
    elif mode == 2:
        pre_result = np.load(direct_save + f"group{args.group_name}_l_results_{iter_num - 1}.npy", allow_pickle=True).item()
    print("load previous result")
except:
    pre_result = {}
    pre_result["success_rate"] = 0
    pre_result["obj_num"] = 0
    print("no previous result")
pre_success_rate = pre_result["success_rate"]
pre_obj_num = pre_result["obj_num"]

result = {}
result["current_success_rate"] = success_rate
result["current_obj_num"] = num_envs
result["success_rate"] = (success_rate * num_envs + pre_success_rate * pre_obj_num) / (num_envs + pre_obj_num)
result["obj_num"] = num_envs + pre_obj_num
if mode == -1:
    # np.save(home_path + f"/rsc/shapenet_{iter_num}.npy", result)
    np.save(home_path + f"/rsc/temp_{iter_num}.npy", result)
elif mode == 0:
    np.save(direct_save + f"group{args.group_name}_s_results_{iter_num}.npy", result)
elif mode == 1:
    np.save(direct_save + f"group{args.group_name}_m_results_{iter_num}.npy", result)
elif mode == 2:
    np.save(direct_save + f"group{args.group_name}_l_results_{iter_num}.npy", result)


print("iter num", iter_num)
if mode != -1:
    print("group", args.group_name)
if mode == -1:
    # print("mode", "shapenet")
    print("mode", "temp")
elif mode == 0:
    print("mode", "small")
elif mode == 1:
    print("mode", "medium")
elif mode == 2:
    print("mode", "large")
print("current averate success rate", result["success_rate"])
print("current obj num", result["obj_num"])
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(" ")



