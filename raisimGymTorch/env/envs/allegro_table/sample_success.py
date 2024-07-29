#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_table as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.allegro_table import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_faive, get_initial_pose_faive_random, get_initial_pose_allegro_new
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
import joblib
import random
import wandb
import torch


exp_name = "pt_allegro_table"


# weight_saved = '2024-01-15-08-14-11/full_5700_r.pt'
# weight_saved = '/../faive_fixed/2024-01-19-16-53-55/full_10200_r.pt'
# weight_saved = '/../faive_fixed/2024-04-22-15-17-24/full_12000_r.pt'
# weight_saved = '/../faive_floating/2024-04-24-09-34-42/full_6800_r.pt'
# weight_saved = '2024-06-18-20-54-14/full_12200_r.pt'
# weight_saved = '2024-06-19-14-15-49/full_15000_r.pt'
# weight_saved = '2024-06-20-13-17-42/full_19200_r.pt'
weight_saved = '2024-06-27-11-55-08/full_32000_r.pt'

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

# cfg['environment']['visualize'] = True
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

num_envs = len(obj_ori_list)

activations = nn.LeakyReLU

cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)

# Environment definition
env = VecEnv(obj_ori_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished")

for obj_item in obj_ori_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)


ob_dim_r = 144
# act_dim = env.num_acts
act_dim = 22
print('ob dim', ob_dim_r)
print('act dim', act_dim)

# Training
trail_steps = 100
reward_clip = -2.0
grasp_steps = 130
n_steps_r = grasp_steps + trail_steps
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


for update in range(args.num_iterations):
    start = time.time()

    qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

    target_center = np.zeros_like(env.affordance_center)
    object_center = np.zeros_like(env.affordance_center)
    fake_non_aff_center = [0.346408, 0.346408, 0.346408]
    contain_non_aff = np.zeros((num_envs, 1), dtype='float32')

    rot_all = []
    pos_all = []
    bias_all = []

    for i in range(num_envs):
        got_proper_initial_pose = False
        lowest_point = 0.
        txt_file_path = os.path.join(directory_path, obj_item) + "/lowest_point_new.txt"
        with open(txt_file_path, 'r') as txt_file:
            lowest_point = float(txt_file.read())

        obj_pose_reset[i, :] = [1., -0., 0.502, 1., -0., -0., 0., 0.]
        obj_pose_reset[i, 2] -= lowest_point

        if new_allegro:
            qpos_reset_r[i, 6:] = 0.2
            qpos_reset_r[i, -4] = 1.0
            qpos_reset_r[i, 7] = 0.8
            qpos_reset_r[i, 11] = 0.8
            qpos_reset_r[i, 15] = 0.8
            qpos_reset_r[i, 19] = 0.8
        else:
            qpos_reset_r[i, -4] = 1.7

        if np.linalg.norm(env.non_aff_mesh[i].centroid - fake_non_aff_center) < 0.01:
            non_aff_mesh = None
            # print("no non-aff mesh")
        else:
            non_aff_mesh = env.non_aff_mesh[i]
            contain_non_aff[i, 0] = 1.

        # rot, pos, bias = get_initial_pose_faive(env.aff_mesh[i], non_aff_mesh, "allegro")
        if new_allegro:
            rot, pos, bias = get_initial_pose_allegro_new(env.aff_mesh[i], non_aff_mesh, "allegro", top=True,
                                                          easy=False)
        else:
            rot, pos, bias = get_initial_pose_faive_random(env.aff_mesh[i], non_aff_mesh, "allegro", top=True,
                                                           easy=False)

        rot_all.append(rot)
        pos_all.append(pos)
        bias_all.append(bias)

        obj_mat = rotations.quat2mat(obj_pose_reset[i, 3:7])
        wrist_pose_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
        wrist_mat = rotations.euler2mat(wrist_pose_obj)
        wrist_in_world = np.matmul(obj_mat, wrist_mat)
        wrist_pose = rotations.mat2euler(wrist_in_world)
        qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat, pos[0, :])
        qpos_reset_r[i, 3:6] = wrist_pose[0, :]

        target_center[i, :] = bias[:]
        object_center[i, :] = env.affordance_center[i]

    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )

    env.set_goals(target_center,
                  object_center,
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  )

    obs_new_r, dis_info = env.observe(contain_non_aff, allegro=True)
    # show_point = dis_info[:, 17:].astype('float32').copy()
    # env.set_joint_sensor_visual(show_point)

    # print("reset finished")
    for step in range(n_steps_r):
        obs_r = obs_new_r
        obs_r = obs_r[:, :].astype('float32')
        if step == grasp_steps:
            env.switch_root_guidance(True)

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)

        frame_start = time.time()

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe(contain_non_aff, allegro=True)
        # show_point = dis_info[:, 17:].astype('float32').copy()
        # env.set_joint_sensor_visual(show_point)

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    # print(f"Update {update} finished in {time.time() - start} seconds")

    global_state = env.get_global_state()

    lifted = (global_state[:, 107] - obj_pose_reset[:, 2] > 0.1) * (np.linalg.norm(obs_new_r[:, :3], axis=1) < 0.2)
    print(lifted.sum())
    for i in range(num_envs):
        if lifted[i]:
            obj_item = obj_ori_list[i]
            data = {}
            data['rot'] = np.float32(rot_all[i])
            data['pos'] = np.float32(pos_all[i])
            data['bias'] = np.float32(bias_all[i])

            data_folder = os.path.join(directory_path, f"./../{cat_name}_successful_labels/", obj_item)
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            seq_itr = 0
            while os.path.exists(os.path.join(data_folder, f"{seq_itr}.npy")):
                seq_itr += 1

            np.save(os.path.join(data_folder, f"{seq_itr}.npy"), data)

            print(f"Saved {obj_item} to {data_folder}/{seq_itr}.npy")



