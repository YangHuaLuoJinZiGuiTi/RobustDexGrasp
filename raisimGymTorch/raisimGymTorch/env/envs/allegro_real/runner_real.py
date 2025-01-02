#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_real as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.allegro_real import NormalSampler
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
import argparse
from raisimGymTorch.helper import rotations
from raisimGymTorch.helper.inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter
import torch

from raisimGymTorch.env.hardware.realsense.PointCloud import Realsense
from raisimGymTorch.env.hardware.FoundationPose.interactive import FoundationData
from raisimGymTorch.env.hardware.FoundationPose.RGBDPointCloud import GetPointCloud
from raisimGymTorch.env.hardware.log_data import d435_record

exp_name = "arm_rand_student"

weight_saved = './../arm_rand/2024-11-17-12-27-38/full_7000_r.pt'
weight_path_student = 'hui/full_2000_r.pt'

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
cat_name = 'ycb_urdf_all'
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

if 'real' in obj_item:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item[5:]}.urdf"))
else:
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

checkpoint_student = torch.load(saver.data_dir.split('eval')[0] + weight_path_student, map_location=torch.device('cpu'))
actor_student_r.architecture.load_state_dict(checkpoint_student['actor_architecture_state_dict'])
actor_student_r.distribution.load_state_dict(checkpoint_student['actor_distribution_state_dict'])
prop_latent_encoder.load_state_dict(checkpoint_student['prop_latent_encoder_state_dict'])

while True:

    obj_init_xyz_qwxyz = None
    obj_pointcloud = None
    if sample_pc_mode == 'foundationpose':
        data_producer = FoundationData(os.path.join(f"{directory_path}/{obj_item}/top_watertight_tiny.obj"), cfg['environment']['hardware']['pointcloud_real']['camera_K_path'])
        try:
            while True:
                time.sleep(1)
                obj_init_xyz_qwxyz = data_producer.get_data()
                if obj_init_xyz_qwxyz is not None:
                    print(f"init success!!! pose is \n {obj_init_xyz_qwxyz}" )
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
        obj_pointcloud = GetPointCloud(cfg['environment']['hardware']['pointcloud_real']['camera_K_path'], sample_pc_mode)
        obj_pos_mean = np.mean(obj_pointcloud.reshape(200,3), axis=0)
        obj_init_xyz_qwxyz = np.array([obj_pos_mean[0], obj_pos_mean[1], obj_pos_mean[2], 0.707, 0, 0.707, 0])
        print(f" ================== mean of point cloud (obj pose center) = {obj_pos_mean}")
    elif sample_pc_mode == 'auto':
        rs = Realsense(cfg['environment']['hardware']['pointcloud_real']['camera_K_path'], 200)
        obj_pos_mean, obj_pointcloud = rs.GetPointCloud()
        obj_init_xyz_qwxyz = np.array([obj_pos_mean[0][0], obj_pos_mean[0][1], obj_pos_mean[0][2], 0.707, 0, 0.707, 0])
    elif sample_pc_mode == 'mesh':
        lowest_points = np.zeros((num_envs, 1), dtype='float32')
        for i in range(num_envs):
            txt_file_path = os.path.join(directory_path, obj_item) + "/lowest_point_new.txt"
            with open(txt_file_path, 'r') as txt_file:
                lowest_points[i] = float(txt_file.read())
    else:
        print(f"unknow sample pc mode input {sample_pc_mode}")
        exit(0)
        
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
        # get_meaningful_ik = False
        # while not get_meaningful_ik:
        # sample object states (not relavent for hardware deployment)
        if sample_pc_mode == 'manual' or sample_pc_mode == 'auto' or sample_pc_mode == 'foundationpose' or sample_pc_mode == 'sam':
            obj_pose_reset[i, :7] = obj_init_xyz_qwxyz # mean of pointcloud
            visible_points_w[i, :] = obj_pointcloud # sample randomly from RGBD in mask
            angle = math.atan2(obj_init_xyz_qwxyz[1], obj_init_xyz_qwxyz[0])
        elif sample_pc_mode == 'mesh':
            sample_x = 0.15
            sample_y = 0.2 - 0.75152
            while True:
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0.45, 0.75)
                sample_x = distance * np.cos(angle)
                sample_y = distance * np.sin(angle)
                if sample_y < 0.3 - 0.75152 and sample_x < 0.3 and sample_x > -0.3:
                    # print(sample_x, sample_y, distance)
                    break
            obj_pose_reset[i, 0] = sample_x
            obj_pose_reset[i, 1] = sample_y
            obj_pose_reset[i, 2] = 0.773 - lowest_points[i]
            obj_pose_reset[i, 3:] = [1., -0., -0., 0., 0.]

            axis_angles = np.zeros((1, 3))
            axis_angles[0, 2] = np.random.uniform(-np.pi, np.pi)
            obj_pose_reset[i, 3:7] = rotations.axisangle2quat(axis_angles)

            # get the partial point cloud
            obj_mat_single = rotations.quat2mat(obj_pose_reset[i, 3:7]).reshape(3, 3)

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
        inverse_grasp = False
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
                        if inverse_grasp:
                            no_feasible_ik = True
                        else:
                            inverse_grasp = True
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
                        if inverse_grasp:
                            no_feasible_ik = True
                        else:
                            inverse_grasp = True
                    else:
                        top_grasp = True
                    continue
                else:
                    qpos_reset_r[i, :6] = ik.findClosestIK(gd, theta0)

                if math.isnan(qpos_reset_r[i, 0]):
                    if top_grasp:
                        if inverse_grasp:
                            no_feasible_ik = True
                        else:
                            inverse_grasp = True
                    else:
                        top_grasp = True
                    continue
                else:
                    # check self collision
                    get_meaningful_ik = env.check_collision(qpos_reset_r)
                    if not get_meaningful_ik:
                        if top_grasp:
                            if inverse_grasp:
                                no_feasible_ik = True
                            else:
                                inverse_grasp = True
                        else:
                            top_grasp = True
                        continue
                    else:
                        if qpos_reset_r[i, 4] < -1.57 or qpos_reset_r[i, 4] > 2:
                            if top_grasp:
                                if inverse_grasp:
                                    no_feasible_ik = True
                                else:
                                    inverse_grasp = True
                            else:
                                top_grasp = True
                            continue
                        else:
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
                rot = get_initial_pose_allegro_arm_partial_safe(visible_points_w[i], hand_dir_x_w, np.eye(3), top=True,
                                                            z_dir_cmd=z_dir_in_world)
                if rot is None:
                    qpos_reset_r[i, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                    print("======== cannot get rot, use defalut pose = " + str(qpos_reset_r))
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
                    print("======== cannot found ik, use defalut pose = " + str(qpos_reset_r))
                    break
                else:
                    qpos_reset_r[i, :6] = ik.findClosestIK(gd, theta0)
                    if qpos_reset_r[i, 4] < -65.0/180.0*np.pi:
                        qpos_reset_r[i, 4] += 2*np.pi
                        
                if math.isnan(qpos_reset_r[i, 0]):
                    qpos_reset_r[i, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                    print("======== reset pose is nan, use defalut pose = " + str(qpos_reset_r))
                    break
                else:
                    # check self collision
                    get_meaningful_ik = env.check_collision(qpos_reset_r)
                    if not get_meaningful_ik:
                        qpos_reset_r[i, :6] = [angle + np.pi / 2, -1.57, 1.57, 0., 1.57, -1.57]
                        print("======== self collision, use defalut pose = " + str(qpos_reset_r))
                        break
                    else:
                        if qpos_reset_r[i, 4] < -1.57 or qpos_reset_r[i, 4] > 2:
                            qpos_reset_r[i, :6] = [angle + np.pi / 2, -1.57, 1.57, 0., 1.57, -1.57]
                            print("======== bad reset pose, use defalut pose = " + str(qpos_reset_r))
                        break

    if qpos_reset_r[0, 0] > np.pi:
        qpos_reset_r[0, 0] -= 2*np.pi
    print(f" ================== samble obj reset pose = {obj_pose_reset}")

    vis_point = visible_points_w.reshape(200*3, -1).astype('float32')
    env.set_sample_point_visual(vis_point, obj_pose_reset)

    for sim_flag in [True, False]: # True, False
        print(f"--------------------------- test in {sim_flag} flag ---------------------- ")
        env.reset_state(qpos_reset_r,
                        qpos_reset_l,
                        np.zeros((num_envs, 22), 'float32'),
                        np.zeros((num_envs, 22), 'float32'),
                        obj_pose_reset, 
                        sim_flag
                        )

        #obs_new_r, aff_vec = env.observe_student_deploy(torch.from_numpy(visible_points_w).to(device))
        obs_new_r, dis_info = env.observe_vision_new()
        aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
        env.set_joint_sensor_visual(show_point)

        final_actions = np.zeros((num_envs, act_dim), dtype='float32')

        step = 0
        while step < n_steps_r:

            frame_start = time.time()

            # cost 0.3~1.3ms 
            obs_r = obs_new_r
            obs_r = obs_r[:, :].astype('float32')
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

            frame_start2 = time.time()

            # cost 0.3~1ms in simulation
            reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'), sim_flag)

            #time.sleep(0.5)
            frame_start3 = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_start3 - frame_start2)
            #if wait_time > 0.:
            #    time.sleep(wait_time)
            frame_start4 = time.time()

            # cost 1-5ms
            #obs_new_r, aff_vec = env.observe_student_deploy(torch.from_numpy(visible_points_w).to(device))
            obs_new_r, dis_info = env.observe_vision_new()
            aff_vec, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
            env.set_joint_sensor_visual(show_point)

            end = time.time()
            #print(f"{step} --- policy:{frame_start2 - frame_start},  step:{frame_start3 - frame_start2},  obscalculate:{frame_start4 - frame_start3},  all:{end - frame_start}")
            step = step + int(reward_r)
        print("end")
