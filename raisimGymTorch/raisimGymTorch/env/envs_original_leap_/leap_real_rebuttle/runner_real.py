#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import leap_real_rebuttle as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.leap_real_rebuttle import NormalSampler
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

from raisimGymTorch.env.hardware.realsense.PointCloud import Realsense

import csv
exp_name = "leaphand_student"

weight_saved = './../arm_rand/2024-11-17-12-27-38/full_7000_r.pt'
#weight_path_student = 'last/full_1500_r.pt'
weight_path_student = 'left/full_6500_r.pt'

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg_left.yaml')
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
lift_top[0, :] = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
lift_topright = np.zeros((num_envs, 6), dtype='float32')
lift_topright[0, :] = [1.0, -1.57, 1.57, -1.57, -1.57, 0.0]

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
lift_steps = 6
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

if sample_pc_mode == 'foundationpose' or sample_pc_mode == 'foundationpose_fullpc':
    data_producer = FoundationData(os.path.join(f"{directory_path}/{obj_item}/top_watertight_tiny.obj"), cfg['environment']['hardware']['pointcloud_real']['camera_K_path'])
elif sample_pc_mode == 'sam' or sample_pc_mode == 'manual':
    rgbd = Realsense(cfg['environment']['hardware']['pointcloud_real']['camera_K_path'], 200)
    vlm = vlm_planner()
    sam = sam_predict()
elif sample_pc_mode == 'auto':
    rs = Realsense(cfg['environment']['hardware']['pointcloud_real']['camera_K_path'], 200)
elif sample_pc_mode == 'mesh':
    lowest_points = np.zeros((num_envs, 1), dtype='float32')
    for i in range(num_envs):
        txt_file_path = os.path.join(directory_path, obj_item) + "/lowest_point_new.txt"
        with open(txt_file_path, 'r') as txt_file:
            lowest_points[i] = float(txt_file.read())
else:
    print(f"unknow sample pc mode input {sample_pc_mode}")
    exit(0)

while True:
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

    obj_init_xyz_qwxyz = None
    obj_pointcloud = None
    if sample_pc_mode == 'foundationpose' or sample_pc_mode == 'foundationpose_fullpc':
        try:
            while True:
                time.sleep(1)
                obj_init_xyz_qwxyz = data_producer.get_data()
                if obj_init_xyz_qwxyz is not None:
                    #print(f"init success!!! pose is \n {obj_init_xyz_qwxyz}" )
                    time.sleep(0.5)
                    obj_init_xyz_qwxyz = data_producer.get_data()
                    obj_pointcloud = data_producer.get_pcd()
                    print(f"after filter .... pose is \n {obj_init_xyz_qwxyz}" )
                    break
        except KeyboardInterrupt:
            data_producer.end_thread()
            print("end")
            exit(0)
    elif sample_pc_mode == 'sam' or sample_pc_mode == 'manual':
        rgb_frame, depth_frame = rgbd.get_rgbd_frame()
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        input_point = np.array(rgbd.get_point_from_image(rgb_frame))
        input_label = np.array([1])  # 为分割对象的性质（背景|前景）
        bbox = rgbd.get_point_from_image(rgb_frame)
        input_box = np.array([bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]])
        mask = sam.calculate_mask(rgb_frame, input_point, input_label, input_box)
        obj_pos_mean, obj_pointcloud = rgbd.GetPointCloud(mask)
        obj_pointcloud[:, 0] += 0.035 # 0.04 for飞行棋
        obj_pos_mean[0][0] += 0.035
        obj_pointcloud[:, 2] -= 0.01 # 0.04 for飞行棋
        obj_pos_mean[0][2] -= 0.01

        obj_pos_mean = np.mean(obj_pointcloud.reshape(200,3), axis=0)
        obj_init_xyz_qwxyz = np.array([obj_pos_mean[0], obj_pos_mean[1], obj_pos_mean[2], 0.707, 0, 0.707, 0])
        print(f" ================== mean of point cloud (obj pose center) = {obj_pos_mean}")
    elif sample_pc_mode == 'auto':
        obj_pos_mean, obj_pointcloud = rs.GetPointCloud()
        obj_pointcloud[:, 0] += 0.025 # 0.04 for飞行棋
        obj_pos_mean[0][0] += 0.025

        # 按z轴（第三列）对点云进行排序
        sorted_point_cloud = obj_pointcloud[obj_pointcloud[:, 2].argsort()]
        z_weighted_mean = np.average(sorted_point_cloud[:,2], weights=sorted_point_cloud[:,2]*10)
        max_z = np.max(obj_pointcloud[:,2])
        print(f"========== max z = {max_z}, mean z = {z_weighted_mean}")
        while max_z - z_weighted_mean > 0.05:
            # 确定z轴的15%分位数作为阈值
            z_threshold = np.percentile(sorted_point_cloud[:, 2], 15)

            # 保留z大于或等于阈值的点
            sorted_point_cloud = sorted_point_cloud[sorted_point_cloud[:, 2] >= z_threshold]
            
            # 计算剩余点云的xyz均值
            z_weighted_mean = np.average(sorted_point_cloud[:,2], weights=sorted_point_cloud[:,2]*10)

            print(f"==========mean = {obj_pos_mean[0][2]}, update new weight mean = {z_weighted_mean}")
        obj_init_xyz_qwxyz = np.array([obj_pos_mean[0][0], obj_pos_mean[0][1], obj_pos_mean[0][1], 0.707, 0, 0.707, 0])
    elif sample_pc_mode == 'mesh':
        pass

    for i in range(num_envs):
        # get_meaningful_ik = False
        # while not get_meaningful_ik:
        # sample object states (not relavent for hardware deployment)
        if sample_pc_mode == 'manual' or sample_pc_mode == 'auto' or sample_pc_mode == 'sam':
            obj_pose_reset[i, :7] = obj_init_xyz_qwxyz # mean of pointcloud
            visible_points_w[i, :] = obj_pointcloud # sample randomly from RGBD in mask
            angle = math.atan2(obj_init_xyz_qwxyz[1], obj_init_xyz_qwxyz[0])
        elif sample_pc_mode == 'mesh' or sample_pc_mode == 'foundationpose' or sample_pc_mode == 'foundationpose_fullpc':
            if sample_pc_mode == 'foundationpose' or sample_pc_mode == 'foundationpose_fullpc':
                obj_pose_reset[i, :7] = obj_init_xyz_qwxyz
                angle = math.atan2(obj_init_xyz_qwxyz[1], obj_init_xyz_qwxyz[0])
                print("------obj reset pose = ", obj_pose_reset)
                print("------obj reset angle = ", angle)
            else:
                while True:
                    angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
                    distance = np.random.uniform(0.45, 0.75)
                    sample_x = distance * np.cos(angle)
                    sample_y = distance * np.sin(angle)
                    if sample_x < 0.25 and sample_x > -0.25:
                        break
                obj_pose_reset[i, 0] = sample_x
                obj_pose_reset[i, 1] = sample_y
                obj_pose_reset[i, 2] = 0.773 - lowest_points[i]
                obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]

                axis_angles = np.zeros((1, 3))
                axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
                obj_pose_reset[i, 3:7] = rotations.axisangle2quat(axis_angles)

            print(f" ================================= sample obj pose: {obj_pose_reset}")

            ################## set default xyz and quats ############### for debug          
            # 一下精选   037_scissors 
            #obj_pose_reset[i, :8] = [-0.24842523, -0.6717001, 0.780741, 0.10874034, 0., 0., -0.9940702, 0.] # 临界值
            # -0.57723176 -0.9771907   1.1137106  -0.9462291  -0.67278194 -0.99736685 # 正常pose
            # -0.5753135  -0.9651766   1.0831178  -0.88515085 -0.71819276 -1.0162001  # 异常pose
            
            
            #obj_pose_reset[i, :8] = [0.22135608, -0.5619688, 0.8170985, 0.707, 0.,     0.707,  0., 0.] # 009_gelatin_box 边缘侧放
            #obj_pose_reset[i, :8] = [0.01878602, -0.6946416, 0.8113495, 0.707, -0.707, 0.,     0., 0.] # 009_gelatin_box 中间正放
            #obj_pose_reset[i, :8]  = [0.01878602, -0.6946416, 0.8273495, 0.707, 0.,     0.707,  0., 0.] # 009_gelatin_box 中间侧放

            # get the partial point cloud
            obj_mat_single = rotations.quat2mat(obj_pose_reset[i, 3:7]).reshape(3, 3)

            view_point_obj_diff = view_point_world - obj_pose_reset[i, :3]
            view_point_obj = np.matmul(obj_mat_single.T, view_point_obj_diff.T).T
            obj_pcd = env.affordance_pcd[i].reshape(200, 3).cpu().numpy()
            
            if sample_pc_mode == 'foundationpose_fullpc':
                visible_points_w[i, :] = np.matmul(obj_mat_single, obj_pcd.T).T + obj_pose_reset[i, :3]
            else:
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

    # if qpos_reset_r[0, 4] < -2.5:
    #     print("===================== change joint 4 position ======================== ")
    #     qpos_reset_r[0, 4] = 3.0
        
    if qpos_reset_r[0, 0] < -1.0 or qpos_reset_r[0, 0] > 1.0:
        print(f"===================== danger arm joint 0: {qpos_reset_r}  ======================== ")
        exit(0)
        
    # if qpos_reset_r[0, 4] < -0.5 or qpos_reset_r[0, 4] > 3.14159:
    #     print(f"===================== danger arm joint 4: {qpos_reset_r}  ======================== ")
    #     exit(0)
    # if qpos_reset_r[0, 4] < 0.4 or qpos_reset_r[0, 4] > 2.75:
    #     grasp_steps = 40
    #     n_steps_r = grasp_steps + lift_steps
    #     print("===================== will add grasp step ======================== ")
    # else:
    grasp_steps = cfg['environment']['grasp_steps']
    n_steps_r = grasp_steps + lift_steps
    
    print(f" ================== obj  pose = {obj_pose_reset} ===============")
    print(f" ================== hand pose = {qpos_reset_r} ================")

    # safety check
    check_dis = obj_pose_reset[0, 0]*obj_pose_reset[0, 0] + obj_pose_reset[0, 1]*obj_pose_reset[0, 1]
    if obj_pose_reset[0, 0] < -0.25 or obj_pose_reset[0, 0] > 0.25 or check_dis < 0.45*0.45 or check_dis > 0.75*0.75 or obj_pose_reset[0, 2] > 1.0:
        print(f"--------------object pose check error !!! {obj_pose_reset}")
        exit(0)
    
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
        
        csvfile = open(f"/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/log_data/csv/recon_leap_real.csv","w")
        writer = csv.writer(csvfile)
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

            newobs = student_mlp_obs.cpu().detach().numpy()
            writer.writerows(newobs)
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
        csvfile.close()

        # demo
        print("will move left ..... ")
        env.final_reset_state(action_r, False, sim_flag, lift_topright)
        print("will release ..... ")
        env.final_reset_state(action_r, True, sim_flag, lift_topright)
        print("finsh all")
