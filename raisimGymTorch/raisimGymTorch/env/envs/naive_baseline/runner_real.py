#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import naive_baseline as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.initial_pose_final import  get_initial_pose_allegro_arm_partial_safe
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
import argparse
from raisimGymTorch.helper import rotations
from raisimGymTorch.helper.inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter
import torch

from raisimGymTorch.env.hardware.realsense.PointCloud import Realsense
from raisimGymTorch.env.hardware.FoundationPose.interactive import FoundationData
from raisimGymTorch.env.hardware.FoundationPose.RGBDPointCloud import GetPointCloud
from raisimGymTorch.env.hardware.log_data import d435_record

import csv
exp_name = "arm_rand_student"

weight_saved = './../arm_rand/2024-11-17-12-27-38/full_7000_r.pt'
weight_path_student = '2025-03-14-19-24(curriculum)/full_5000_r.pt'

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

cfg['environment']['visualize'] = True
cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)

sample_pc_mode = cfg['environment']['hardware']['pointcloud_real']['sample_pointcloud_mode']
obj_item = cfg['environment']['hardware']['pointcloud_real']['obj_mesh']

# cat_name = 'mixed_test'
# cat_name = 'mixed_unseen_test'
# cat_name = 'mixed_unseen_category_test'
# cat_name = 'mixed_train'
cat_name = 'test'
# cat_name = 'affordance_level'
cat_name = cfg['environment']['load_set']
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)

items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

obj_path_list = []
obj_ori_list = folder_names

if obj_item == 'random' or obj_item == 'dummy':
    obj_item = choice(obj_ori_list)

# Environment definition
env = VecEnv([obj_item], mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished")

obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)


rs = Realsense(cfg['environment']['hardware']['pointcloud_real']['camera_K_path'], 200)


while True:
    start = time.time()

    final_grasp_pose = np.zeros((num_envs, 6), dtype='float32')
    qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')
    target_center = np.zeros_like(env.affordance_center)
    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']
    graspqpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
    graspqpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']

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

    # wrist 到 handcenter 那个球的bias
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

    obj_init_xyz_qwxyz = None
    obj_pointcloud = None
    obj_pos_mean, obj_pointcloud = rs.GetPointCloud()
    obj_h_half = ((obj_pos_mean[0][2] - 0.771) / 2.0)
    if obj_h_half > 0.1:
        obj_h_half = 0.1
    obj_init_xyz_qwxyz = np.array([obj_pos_mean[0][0], obj_pos_mean[0][1], obj_pos_mean[0][2] - 0.0, 0.707, 0, 0.707, 0])

    for i in range(num_envs):
        obj_pose_reset[i, :7] = obj_init_xyz_qwxyz # mean of pointcloud
        visible_points_w[i, :] = obj_pointcloud # sample randomly from RGBD in mask
        angle = math.atan2(obj_init_xyz_qwxyz[1], obj_init_xyz_qwxyz[0])

        # get the x_dir of the grasping frame
        obj_aff_center_in_w = np.mean(visible_points_w[i].reshape(200,3), axis=0)

        # get the x_dir of the grasping frame
        hand_dir_x_w = np.zeros((1, 3))
        hand_dir_x_w[0, 2] = 1
        # get position and orientation of the wrist
        pos = obj_aff_center_in_w + 0.25 * hand_dir_x_w
        grasppos = obj_aff_center_in_w + 0.08 * hand_dir_x_w
        
        max_h_pc = 0
        for j in range(200):
            if obj_pointcloud[j][2] > max_h_pc:
                 max_h_pc = obj_pointcloud[j][2]
        
        grasppos[i, 2] = max_h_pc + 0.04
        
        if (grasppos[i, 2] < 0.92):
            grasppos[i, 2] = 0.92
        print(f"grasp pose = {grasppos}")
        
        rot = get_initial_pose_allegro_arm_partial_safe(visible_points_w[i], hand_dir_x_w, np.eye(3), top=False)
        if rot is None:
            print("obj len more then 18cm !!!")
            exit(0)
        wrist_in_world = rot
        qpos_reset_r[i, :3] = pos[0, :]

        # from grasping frame pos to wrist pos
        wrist_bias_in_world = np.matmul(wrist_in_world, wrist_bias.T).T
        pos_in_ur5 = np.zeros((3, 1))
        pos_in_ur5[0, 0] = qpos_reset_r[i, 0] - 0. + wrist_bias_in_world[0, 0]
        pos_in_ur5[1, 0] = qpos_reset_r[i, 1] - 0. + wrist_bias_in_world[0, 1]
        pos_in_ur5[2, 0] = qpos_reset_r[i, 2] - 0.771 + wrist_bias_in_world[0, 2]
        pos_in_ur5_new = np.matmul(ur5_to_world.T, pos_in_ur5)

        grasppos_in_ur5 = np.zeros((3, 1))
        grasppos_in_ur5[0, 0] = grasppos[i, 0] - 0. + wrist_bias_in_world[0, 0]
        grasppos_in_ur5[1, 0] = grasppos[i, 1] - 0. + wrist_bias_in_world[0, 1]
        grasppos_in_ur5[2, 0] = grasppos[i, 2] - 0.771 + wrist_bias_in_world[0, 2]
        grasppos_in_ur5_new = np.matmul(ur5_to_world.T, grasppos_in_ur5)
        
        wrist_mat_in_ur5 = np.matmul(ur5_to_world.T, wrist_in_world)

        gd = np.eye(4)
        gd[:3, :3] = wrist_mat_in_ur5
        gd[0, 3] = pos_in_ur5_new[0, 0]
        gd[1, 3] = pos_in_ur5_new[1, 0]
        gd[2, 3] = pos_in_ur5_new[2, 0]

        graspgd = np.eye(4)
        graspgd[:3, :3] = wrist_mat_in_ur5
        graspgd[0, 3] = grasppos_in_ur5_new[0, 0]
        graspgd[1, 3] = grasppos_in_ur5_new[1, 0]
        graspgd[2, 3] = grasppos_in_ur5_new[2, 0]
        
        if ik.findClosestIK(gd, theta0) is None:
            print("no IK solve")
            exit(0)
        else:
            qpos_reset_r[i, :6] = ik.findClosestIK(gd, theta0)
            graspqpos_reset_r[i, :6] = ik.findClosestIK(graspgd, theta0)

        if math.isnan(qpos_reset_r[i, 0]):
            print("no feasible ik ")
            exit(0)

    if qpos_reset_r[i, 4] < -2.0:
        print(f"------------------{qpos_reset_r} no safety, break or not !!!")
        exit(0)
        

    if qpos_reset_r[0, 3] > np.pi - 1.0:
        print("=============== danger pose !!!!!!!!!!!!!!!!!!")
        exit(0)

    collision_check = env.check_collision(qpos_reset_r)
    if not collision_check:
        print("======== self collision ??? " + str(qpos_reset_r))
        exit(0)

    if qpos_reset_r[0, 0] > np.pi:
        print("===================== will reset arm0 ======================== ")
        qpos_reset_r[0, 0] -= 2*np.pi

    vis_point = visible_points_w.reshape(200*3, -1).astype('float32')
    env.set_sample_point_visual(vis_point, obj_pose_reset)

    # safety check
    if obj_pose_reset[0, 0] < -0.4 or obj_pose_reset[0, 0] > 0.4 or obj_pose_reset[0, 1] < -0.83 or obj_pose_reset[0, 1] > -0.4 or obj_pose_reset[0, 2] > 0.98:
        print("--------------object pose check error !!! ")
        #continue

    print(f" ================== obj  pose = {obj_pose_reset} ===============")
    print(f" ================== hand pose = {qpos_reset_r} ================")
    
    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset, 
                    False
                    )

    delta_grasp = [[0.0, 0.3, 0.0, 0.0,   0.0, 0.3, 0.0, 0.0,   0.0, 0.3, 0.0, 0.0,   0.6, 0.0, -0.1, 0.2],
                   [0.0, 0.4, 0.2, 0.2,   0.0, 0.4, 0.1, 0.2,   0.0, 0.4, 0.1, 0.2,   1.1, 0.4, -0.1, 0.2],
                   [0.0, 0.5, 0.3, 0.3,   0.0, 0.5, 0.2, 0.3,   0.0, 0.5, 0.2, 0.3,   1.35, 0.6, -0.1, 0.2],
                   [0.0, 0.6, 0.4, 0.4,   0.0, 0.6, 0.3, 0.4,   0.0, 0.6, 0.3, 0.4,   1.35, 0.85, 0.0, 0.2],
                   [0.0, 0.7, 0.5, 0.5,   0.0, 0.7, 0.4, 0.5,   0.0, 0.7, 0.4, 0.5,   1.35, 0.85, 0.2, 0.2],
                       
                   [0.0, 0.8, 0.6, 0.5,   0.0, 0.8, 0.6, 0.5,   0.0, 0.8, 0.6, 0.5,   1.35, 0.85, 0.3, 0.3],
                   [0.0, 0.9, 0.7, 0.5,   0.0, 0.9, 0.7, 0.5,   0.0, 0.9, 0.7, 0.5,   1.35, 0.85, 0.3, 0.5],
                   [0.0, 1.0, 0.8, 0.5,   0.0, 1.0, 0.8, 0.8,   0.0, 1.0, 0.8, 0.5,   1.35, 0.85, 0.3, 0.7],
                   [0.0, 1.1, 0.8, 0.5,   0.0, 1.1, 0.8, 0.8,   0.0, 1.1, 0.8, 0.5,   1.35, 0.85, 0.3, 0.9],
                   [0.0, 1.2, 0.8, 0.5,   0.0, 1.2, 0.8, 0.8,   0.0, 1.2, 0.8, 0.5,   1.35, 0.85, 0.3, 1.1],
                   [0.0, 1.3, 0.8, 0.5,   0.0, 1.3, 0.8, 0.8,   0.0, 1.3, 0.8, 0.5,   1.35, 0.85, 0.3, 1.3]]
    for i in range(11):
        graspqpos_reset_r[0, 6:] = delta_grasp[i]
        env.reset_state(graspqpos_reset_r,
                        qpos_reset_l,
                        np.zeros((num_envs, 22), 'float32'),
                        np.zeros((num_envs, 22), 'float32'),
                        obj_pose_reset, 
                        False
                        )
        time.sleep(0.05)

    action_r = qpos_reset_r
    ## for test
    # print("will move ..... ")
    env.final_reset_state(action_r, False, False, True)
    print("will release ..... ")
    env.final_reset_state(action_r, True, False, True)
    print("will move left ..... ")
    env.final_reset_state(action_r, True, False, False)
    print("finsh all")
