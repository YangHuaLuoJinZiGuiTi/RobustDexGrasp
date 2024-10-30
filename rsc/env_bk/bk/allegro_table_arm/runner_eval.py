#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_table_arm as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.allegro_table_arm import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_faive, get_initial_pose_faive_random, get_initial_pose_allegro_new, get_initial_pose_allegro_arm
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


exp_name = "pt_allegro_arm"

# weight_saved = './../pt_allegro_table/2024-06-27-11-55-08/full_32000_r.pt'
# weight_saved = '2024-10-09-17-02-07/full_1000_r.pt'
# weight_saved = '2024-10-10-08-41-07/full_8500_r.pt'
# weight_saved = '2024-10-11-15-11-44/full_33000_r.pt'
# weight_saved = '2024-10-11-15-59-16/full_33000_r.pt'
# weight_saved = '2024-10-15-12-20-24/full_3500_r.pt'
# weight_saved = '2024-10-15-12-42-54/full_5000_r.pt'
# weight_saved = '2024-10-15-13-26-23/full_3500_r.pt'
# weight_saved = '2024-10-15-13-57-56/full_21500_r.pt'
weight_saved = '2024-10-15-15-08-23/full_22000_r.pt'

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
# obj_item = '037_scissors'

# Environment definition
env = VecEnv([obj_item], mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished")

obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)


ob_dim_r = 150
# act_dim = env.num_acts
act_dim = 22
print('ob dim', ob_dim_r)
print('act dim', act_dim)

# Training
trail_steps = 30
reward_clip = -2.0
grasp_steps = 100
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

    for i in range(num_envs):
        got_proper_initial_pose = False
        lowest_point = 0.
        txt_file_path = os.path.join(directory_path, obj_item) + "/lowest_point_new.txt"
        with open(txt_file_path, 'r') as txt_file:
            lowest_point = float(txt_file.read())

        obj_pose_reset[i, :] = [0.8, 0.2, 0.773, 1., -0., -0., 0., 0.]
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
            print("no non-aff mesh")
        else:
            non_aff_mesh = env.non_aff_mesh[i]
            contain_non_aff[i, 0] = 1.

        # rot, pos, bias = get_initial_pose_faive(env.aff_mesh[i], non_aff_mesh, "allegro")
        # rot, pos, bias = get_initial_pose_faive_random(env.aff_mesh[i], non_aff_mesh, "allegro", top=True, easy=False)


        get_meaningful_ik = False
        while not get_meaningful_ik:
            if new_allegro:
                rot, pos, bias = get_initial_pose_allegro_arm(env.aff_mesh[i], non_aff_mesh,  top=True, easy=False)
                # rot, pos, bias = get_initial_pose_allegro_new(env.aff_mesh[i], non_aff_mesh, "allegro", top=True, easy=False)
            else:
                rot, pos, bias = get_initial_pose_faive_random(env.aff_mesh[i], non_aff_mesh, "allegro", top=True,
                                                               easy=False)

            obj_mat = rotations.quat2mat(obj_pose_reset[i, 3:7])
            wrist_pose_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
            wrist_mat = rotations.euler2mat(wrist_pose_obj)
            wrist_in_world = np.matmul(obj_mat, wrist_mat)
            wrist_pose = rotations.mat2euler(wrist_in_world)
            qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat, pos[i, :])
            qpos_reset_r[i, 3:6] = wrist_pose[i, :]

            target_center[i, :] = bias[:]
            object_center[i, :] = env.affordance_center[i]



            wrist_bias = np.zeros((1,3))
            wrist_bias[0, 0] = -0.0091
            wrist_bias[0, 2] = -0.095
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

            theta0 = [-1.57, -1.57, 1.57, 0, 1.57, -1]
            joint_weights = [1, 1, 1, 1, 1, 1]
            ik = InverseKinematicsUR5()
            ik.setJointWeights(joint_weights)
            ik.setJointLimits(-3.14, 3.14)
            if ik.findClosestIK(gd, theta0) is None:
                continue
            else:
                qpos_reset_r[i, :6] = ik.findClosestIK(gd, theta0)
                # print("found a solution")

            if math.isnan(qpos_reset_r[i, 0]):
                continue
                # print("no meaningful ik")
                # print(qpos_reset_r[i, :6])
                # print(rot)
            else:
                get_meaningful_ik = True
                # print(qpos_reset_r[i, :6])





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

    obs_new_r, dis_info = env.observe_vision(contain_non_aff, allegro=True)
    show_point = dis_info[:, 17:].astype('float32').copy()
    env.set_joint_sensor_visual(show_point)
    for step in range(n_steps_r):
        obs_r = obs_new_r
        obs_r = obs_r[:, :].astype('float32')

        # time.sleep(10)

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)
        # action_r[:, :6] = 0

        frame_start = time.time()

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe_vision(contain_non_aff, allegro=True)
        show_point = dis_info[:, 17:].astype('float32').copy()
        env.set_joint_sensor_visual(show_point)

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    print("end")



