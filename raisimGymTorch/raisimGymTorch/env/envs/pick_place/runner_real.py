#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import pick_place as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.pick_place import NormalSampler
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
import argparse
from raisimGymTorch.helper import rotations
from raisimGymTorch.helper.inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter
import torch
import cv2

from raisimGymTorch.env.hardware.planning.vlm_planner import vlm_planner
from raisimGymTorch.env.hardware.sam.sam_predict import sam_predict
from raisimGymTorch.env.hardware.realsense.PointCloud import Realsense
from raisimGymTorch.env.hardware.FoundationPose.interactive import FoundationData
from raisimGymTorch.env.hardware.FoundationPose.RGBDPointCloud import GetPointCloudClass
from raisimGymTorch.env.hardware.log_data import d435_record

import csv
exp_name = "arm_rand_student"

weight_saved = './../arm_rand/2024-11-17-12-27-38/full_7000_r.pt'
weight_path_student = 'last/full_1500_r.pt'
#weight_path_student = 'hui_reset_pose/2025-03-20-18-47-17/full_5500_r.pt'

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
activations = nn.LeakyReLU

cfg['environment']['visualize'] = True
cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)

sample_pc_mode = cfg['environment']['hardware']['pointcloud_real']['sample_pointcloud_mode']
obj_item = cfg['environment']['hardware']['pointcloud_real']['obj_mesh']

# cat_name = 'mixed_test'
# cat_name = 'mixed_unseen_test'
# cat_name = 'mixed_unseen_category_test'
# cat_name = 'mixed_train'
# cat_name = 'test'
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
        
lift_top = np.zeros((num_envs, 6), dtype='float32')
lift_top[0, :] = [0.0, -1.57, 1.57, 0., 1.57, -1.57]
lift_topright = np.zeros((num_envs, 6), dtype='float32')
lift_topright[0, :] = [1.0, -1.57, 1.57, 0., 1.57, -1.57]
lift_topleft = np.zeros((num_envs, 6), dtype='float32')
lift_topleft[0, :] = [-0.733, -1.57, 1.57, 0., 1.57, -1.57]
        
# env.final_reset_state(np.zeros((num_envs, 22), dtype='float32'), True, False, lift_topleft)
# exit(0)

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
lift_steps = 0
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

getpc = GetPointCloudClass(cfg['environment']['hardware']['pointcloud_real']['camera_K_path'])
while True:
    start = time.time()

    qpos_place_r = np.zeros((num_envs, 6), dtype='float32')
    qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')
    target_center = np.zeros_like(env.affordance_center)
    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']

    hand_center_sample_w = np.zeros((1, 3))
    hand_center_sample_w[0, 0] = cfg['environment']['camera_position'][0]
    hand_center_sample_w[0, 1] = cfg['environment']['camera_position'][1]
    hand_center_sample_w[0, 2] = cfg['environment']['camera_position'][2]

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

    sample_num = cfg['environment']['sample_num']


    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

    obj_init_xyz_qwxyz = None
    obj_pointcloud = None
    if sample_pc_mode == 'sam' or sample_pc_mode == 'manual':
        obj_pointcloud = getpc.GetPointCloud()
        obj_pointcloud[:, 0] += 0.01 # 0.04 for飞行棋
        obj_pointcloud[:, 2] -= 0.01 #0.01 for 飞行棋
        obj_pos_mean = np.mean(obj_pointcloud.reshape(200,3), axis=0)
        obj_init_xyz_qwxyz = np.array([obj_pos_mean[0], obj_pos_mean[1], obj_pos_mean[2], 0.707, 0, 0.707, 0])

        
        place_pc = getpc.GetPointCloud()
        place_pc[:, 2] += 0.15 #0.15, 0.28 0.35
        plaec_pos_mean = np.mean(place_pc.reshape(200,3), axis=0)
        
        print(f"--------------obj pose={obj_pos_mean}")
        print(f"--------------place pose={plaec_pos_mean}")

    for i in range(num_envs):
        # get_meaningful_ik = False
        # while not get_meaningful_ik:
        # sample object states (not relavent for hardware deployment)
        if sample_pc_mode == 'manual' or sample_pc_mode == 'auto' or sample_pc_mode == 'sam':
            obj_pose_reset[i, :7] = obj_init_xyz_qwxyz # mean of pointcloud
            visible_points_w[i, :] = obj_pointcloud # sample randomly from RGBD in mask

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
                collision_check = env.check_collision(qpos_reset_r)
                if not collision_check:
                    feasible_ik_flag[j] = False
                    continue
                else:
                    #print(f"feasible ik in {j}")
                    feasible_ik_flag[j] = True
                    ik_results[j] = ik_result
        feasible_indices = np.where(feasible_ik_flag)[0]
        scores = np.ones(sample_num, dtype='float32')
        scores = scores * 10000.
        if min(projection_lengths) < 0.15:
            for j in feasible_indices:
                if projection_lengths[j] < 0.15:
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

        best_place_mat = rotations.euler2mat([1.5707963, 0, -1.5707963])
        wrist_mat_place_in_ur5 = np.matmul(ur5_to_world.T, best_place_mat)
        wrist_bias_place = np.zeros((1, 3))
        wrist_bias_place[0,1] = -0.08
        wrist_bias_in_world = np.matmul(wrist_mat_place_in_ur5, wrist_bias_place.T).T
        pos_in_ur5 = np.zeros((3, 1))
        pos_in_ur5[0, 0] = plaec_pos_mean[0] - 0. + wrist_bias_in_world[0, 0]
        pos_in_ur5[1, 0] = plaec_pos_mean[1] - 0. + wrist_bias_in_world[0, 1]
        pos_in_ur5[2, 0] = plaec_pos_mean[2] - 0.771 + wrist_bias_in_world[0, 2]
        pos_in_ur5_new = np.matmul(ur5_to_world.T, pos_in_ur5)

        gd = np.eye(4)
        gd[:3, :3] = wrist_mat_place_in_ur5
        gd[:3, 3] = pos_in_ur5_new[:3, 0]

        ik_result = ik.findClosestIK(gd, theta0)
        if ik_result is None or np.isnan(ik_result).any():
            print('error ik for place point')
            exit(0)
        else:
            qpos_place_r[i, :6] = ik_result

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

    #qpos_reset_r[0, :6] = [-0.11907104, -1.1725091, 1.3224226, -0.14991362, -0.11905771, -1.5707964] # 009_gelatin_box 边缘侧放
    #qpos_reset_r[0, :6]  = [-0.12826417, -1.2702502, 1.4753474, -0.20509712,   1.5790964, -1.5707964] # 009_gelatin_box 中间正放
    #qpos_reset_r[0, :6]  = [-0.12826417, -1.2702502, 1.4753474, -0.20509712,   1.5790964, -1.5707964] # 009_gelatin_box 中间侧放

    if qpos_reset_r[0, 4] < -2.5:
        print("===================== change joint 4 position ======================== ")
        qpos_reset_r[0, 4] = 3.0
        
    if qpos_reset_r[0, 0] < -1.0 or qpos_reset_r[0, 0] > 1.0:
        print(f"===================== danger arm joint 0: {qpos_reset_r}  ======================== ")
        exit(0)
        
    if qpos_reset_r[0, 4] < -0.5 or qpos_reset_r[0, 4] > 3.14159:
        print(f"===================== danger arm joint 4: {qpos_reset_r}  ======================== ")
        exit(0)
    if qpos_reset_r[0, 4] < 0.4 or qpos_reset_r[0, 4] > 2.75:
        grasp_steps = 40
        n_steps_r = grasp_steps + lift_steps
        print("===================== will add grasp step ======================== ")
    else:
        grasp_steps = cfg['environment']['grasp_steps']
        n_steps_r = grasp_steps + lift_steps
    
    print(f" ================== obj  pose = {obj_pose_reset} ===============")
    print(f" ================== hand pose = {qpos_reset_r} ================")

    # safety check
    # check_dis = obj_pose_reset[0, 0]*obj_pose_reset[0, 0] + obj_pose_reset[0, 1]*obj_pose_reset[0, 1]
    # if obj_pose_reset[0, 0] < -0.25 or obj_pose_reset[0, 0] > 0.25 or check_dis < 0.45*0.45 or check_dis > 0.75*0.75 or obj_pose_reset[0, 2] > 1.0:
    #     print(f"--------------object pose check error !!! {obj_pose_reset}")
    #     exit(0)
    
    for sim_flag in [False]: # True, False

        print(f"--------------------------- test in {sim_flag} flag ---------------------- ")
        env.reset_state(qpos_reset_r,
                        qpos_reset_l,
                        np.zeros((num_envs, 22), 'float32'),
                        np.zeros((num_envs, 22), 'float32'),
                        obj_pose_reset, 
                        sim_flag
                        )

        obs_new_r, aff_vec = env.observe_student_deploy(torch.from_numpy(visible_points_w).to(device))
        #obs_new_r, dis_info = env.observe_vision_new()
        aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
        env.set_joint_sensor_visual(show_point)

        final_actions = np.zeros((num_envs, act_dim), dtype='float32')

        step = 0
        
        #csvfile = open(f"/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/log_data/csv/recon{sim_flag}.csv","w")
        #writer = csv.writer(csvfile)
        while step < n_steps_r:

            frame_start = time.time()

            # cost 0.3~1.3ms 
            obs_r = obs_new_r
            obs_r = obs_r[:, :].astype('float32')
            encode_obs = torch.from_numpy(obs_r[:, :tobeEncode_dim * t_steps]).to(device)
            student_latent = prop_latent_encoder(encode_obs)
            student_mlp_obs = torch.cat((torch.from_numpy(obs_r[:, -ob_dim_r:-ob_dim_r + tobeEncode_dim]), # -153, -153+44 总共44个
                                        student_latent.cpu(),   # 26个
                                        torch.from_numpy(obs_r[:, -ob_dim_r + tobeEncode_dim + prop_latent_dim:-aff_vec_dim]), # -153 + 77 + 26, -51
                                        torch.from_numpy(aff_vec)), dim=1).to(device) # 51(17*3)

            #newobs = student_mlp_obs.cpu().detach().numpy()
            #writer.writerows(newobs)
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
            frame_start2 = time.time()

            # cost 0.3~1ms in simulation
            reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'), sim_flag)

            #time.sleep(0.5)
            frame_start3 = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_start3 - frame_start2)
            if wait_time > 0.:
                time.sleep(wait_time)
            frame_start4 = time.time()

            # cost 1-5ms
            obs_new_r, aff_vec = env.observe_student_deploy(torch.from_numpy(visible_points_w).to(device))
            # obs_new_r, dis_info = env.observe_vision_new()
            #aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
            #env.set_joint_sensor_visual(show_point)

            end = time.time()
            print(f"{step} --- policy:{frame_start2 - frame_start},  step:{frame_start3 - frame_start2},  obscalculate:{frame_start4 - frame_start3},  all:{end - frame_start}")
            step = step + 1
        print("end")
        #csvfile.close()  

        # demo
        print("will move up ..... ")
        env.move_line(0, 0, 0.2)
        print("will move place ..... ")
        env.final_reset_state(action_r, False, sim_flag, qpos_place_r)
        print("will release ..... ")
        idx = [7,8,11,12,15,16,20,21]
        action_r[0, idx] -= 20
        env.final_reset_state(action_r, False, sim_flag, qpos_place_r)
        action_r[0, idx] -= 20
        env.final_reset_state(action_r, False, sim_flag, qpos_place_r)
        
        env.final_reset_state(action_r, True, sim_flag, qpos_place_r)
        print("will move up ..... ")
        env.move_line(0, 0, 0.2)
        print("finsh all")
        env.final_reset_state(action_r, True, sim_flag, lift_topleft)
        exit(0)