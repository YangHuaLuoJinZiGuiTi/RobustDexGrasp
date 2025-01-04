import numpy
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_teacher_biased as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.allegro_teacher_biased import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_allegro_arm_partial, get_initial_pose_allegro_arm_partial_safe

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

from random import choices
from raisimGymTorch.helper.inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter

exp_name = "arm_rand"

# weight_saved = '/../../faive_fixed/2024-01-29-22-01-24/full_2700_r.pt'
# weight_saved = '/../../pt_allegro_fixed/2024-05-30-18-50-43/full_9000_r.pt'
# weight_saved = '/../../pt_allegro_fixed/2024-05-30-18-50-43/full_9000_r.pt'
# weight_saved = '/../../pt_allegro_fixed/2024-06-11-17-56-47/full_48000_r.pt'
# weight_saved = '/../2024-10-28-16-53-10/full_6000_r.pt'
weight_saved = '/../2024-12-12-09-55-30/full_27000_r.pt'


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_saved)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-seed', '--seed', type=int, default=1)
parser.add_argument('-itr', '--num_iterations', type=int, default=50001)
# parser.add_argument('-nr', '--num_repeats', type=int, default=95)
parser.add_argument('-re', '--load_trained_policy', action="store_true")
parser.add_argument('-renew', '--renew', help='update labels every iteration', action="store_true")
parser.add_argument('-ln', '--log_name', type=str, default=None)
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

if args.log_name is not None:
    wandb.init(project=task_name, config=cfg, name=args.log_name)

if args.seed != 1:
    cfg['seed'] = args.seed

obj_path_list = []
obj_list = []

# directory_path = home_path + "/rsc/mixed_train/"
cat_name = 'ycb_urdf_sim'
# cat_name = 'ycb_urdf_all'
# cat_name = 'ycb_urdf_light'
# cat_name = 'ycb_urdf_sim_light'
cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)
items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

obj_path_list = []
obj_ori_list = folder_names
# obj_ori_list.append('Mug_6a9b31e1298ca1109c515ccf0f61e75f_handle')
# obj_ori_list.append('Mug_40f9a6cc6b2c3b3a78060a3a3a55e18f_handle')
# obj_ori_list.append('Mug_46ed9dad0440c043d33646b0990bb4a_handle')
# obj_ori_list.append('Mug_8556_handle')

# label = {}

num_envs = len(obj_ori_list) * 3
activations = nn.LeakyReLU

for i in range(3):
    for item in obj_ori_list:
        obj_list.append(item)

if args.log_name is None:
    num_envs = 3
    obj_list = choices(obj_list, k=1)
    obj_list.append(obj_list[0])
    obj_list.append(obj_list[0])
    cfg['environment']['visualize'] = True


cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)
# Environment definition
env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)


ob_dim_r = 153
act_dim = 22
print('ob dim', ob_dim_r)
print('act dim', act_dim)

# Training
reward_clip = -2.0
n_steps_r = cfg['environment']['grasp_steps']
total_steps_r = n_steps_r * env.num_envs

# print(n_steps_r)

# RL network
actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)

test_dir = False

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp",
                                       task_path + "/runner.py", task_path + "/runner_eval.py", task_path + "/../../RaisimGymVecEnvOther.py"], test_dir=test_dir)


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
                # learning_rate=1e-4
                )

if args.load_trained_policy:
    load_param(saver.data_dir.split('eval')[0] + weight_path, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir,
               cfg_grasp)


finger_weights = np.ones((num_envs, 17)).astype('float32')
for i in range(4):
    finger_weights[:, 4 * i+4] *= 4.0
finger_weights /= finger_weights.sum(axis=1).reshape(-1, 1)
finger_weights *= 17.0
affordance_reward_r = np.zeros((num_envs, 1))
center_reward_r = np.zeros((num_envs, 1))
table_reward_r = np.zeros((num_envs, 1))
arm_height_reward_r = np.zeros((num_envs, 1))
arm_action_reward_r = np.zeros((num_envs, 1))
hand_action_reward_r = np.zeros((num_envs, 1))


qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

saved_update_idx = 0

lowest_points = np.zeros((num_envs, 1), dtype='float32')
for i in range(num_envs):
    txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())

for update in range(args.num_iterations):
    start = time.time()

    ### Evaluate trained model visually (note always the first environment gets visualized)

    if update % cfg['environment']['eval_every_n'] == 0 and args.log_name is not None:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor_r.architecture.state_dict(),
            'actor_distribution_state_dict': actor_r.distribution.state_dict(),
            'critic_architecture_state_dict': critic_r.architecture.state_dict(),
            'optimizer_state_dict': ppo_r.optimizer.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '_r.pt')

        env.save_scaling(saver.data_dir, str(update))
        saved_update_idx = update

    target_center = np.zeros_like(env.affordance_center)

    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']

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

    # partial_obs = cfg['environment']['partial_pt_init_pose']
    #
    # if partial_obs:
    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

    for i in range(num_envs):
        if cfg['environment']['train_safe']:
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

            # get the x_dir of the grasping frame
            obj_aff_center_in_w = np.mean(visible_points_w[i].reshape(200, 3), axis=0)

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
                        if qpos_reset_r[i, 4] < -1.57 or qpos_reset_r[i, 4] > 2:
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
                        qpos_reset_r[i, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
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
                        break
                    else:
                        qpos_reset_r[i, :6] = ik.findClosestIK(gd, theta0)

                    if math.isnan(qpos_reset_r[i, 0]):
                        qpos_reset_r[i, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                        break
                    else:
                        if qpos_reset_r[i, 4] < -1.57 or qpos_reset_r[i, 4] > 2:
                            qpos_reset_r[i, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
                            break
                        else:
                            break
        else:
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
                    get_meaningful_ik = True

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
        current_obj_idx = true_idx // 3
        current_obj_env_indices = [current_obj_idx * 3, current_obj_idx * 3 + 1, current_obj_idx * 3 + 2]
        false_indices = [idx for idx in current_obj_env_indices if not contains_one[idx]]
        if len(false_indices)>0:
            chosen_index = np.random.choice(false_indices)
            qpos_reset_r[true_idx, :] = qpos_reset_r[chosen_index, :]
            obj_pose_reset[true_idx, :] = obj_pose_reset[chosen_index, :]
        else:
            qpos_reset_r[true_idx, :6] = [angle+np.pi/2, -1.57, 1.57, 0., 1.57, -1.57]
            obj_pose_reset[true_idx, 0] = 0.15
            obj_pose_reset[true_idx, 1] = 0.2 - 0.75152
            # obj_pose_reset[true_idx, 3:] = [1., -0., -0., 0., 0.]

    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )

    obs_new_r, dis_info = env.observe_vision_new()
    env.update_target(target_center)
    rewards_r_sum = env.get_reward_info_r()
    for i in range(len(rewards_r_sum)):
        rewards_r_sum[i]['affordance_reward'] = 0
        # rewards_r_sum[i]['not_affordance_reward'] = 0
        rewards_r_sum[i]['center_reward'] = 0
        rewards_r_sum[i]['table_reward'] = 0
        rewards_r_sum[i]['arm_height_reward'] = 0
        rewards_r_sum[i]['arm_action_reward'] = 0
        rewards_r_sum[i]['hand_action_reward'] = 0

        for k in rewards_r_sum[i].keys():
            rewards_r_sum[i][k] = 0


    biased = cfg['environment']['biased']
    if biased:
        obj_biased = np.zeros((num_envs, 1), dtype='float32')
        obj_pos_bias = np.random.uniform(-0.05, 0.05, (num_envs, 3)).astype('float32')
    else:
        obj_pos_bias = np.zeros((num_envs, 3), dtype='float32')


    for step in range(n_steps_r):
        obs_r = obs_new_r
        obs_r = obs_r[:].astype('float32')

        action_r = ppo_r.act(obs_r)
        action_l = np.zeros_like(action_r)

        # # clip the first 6 dim of action to (-2, 2)
        # action_r[:, :6] = np.clip(action_r[:, :6], -2., 2.)

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe_vision_new()
        obs_new_r = obs_new_r[:].astype('float32')
        # env.update_target(target_center)


        if biased:
            obj_pos_bias_current = np.zeros((num_envs, 3), dtype='float32')
            for i in range(num_envs):
                if np.min(dis_info[i, 0:17]) < 0.07 and obj_biased[i] == 0:
                    obj_biased[i] = 1
                    obj_pos_bias_current[i] = obj_pos_bias[i]
            env.switch_obj_pos(obj_pos_bias_current)


        global_state = env.get_global_state()

        abs_dis = np.linalg.norm(global_state[:, 115:118], axis=1)
        # if distance > 0.1 then give center reward otherwise set to 0
        center_loss = (abs_dis > 0.07) * (np.square(abs_dis - 0.07))

        rewards_r = env.get_reward_info_r()
        affordance_reward_r = - np.sum((dis_info[:, :17]) * finger_weights, axis=1)
        # table_reward_r = - np.sum(np.log(np.maximum(np.abs(obs_new_r[:, 70:87]), 0.01*np.ones_like(np.abs(obs_new_r[:, 70:87])))) * finger_weights * (np.abs(obs_new_r[:, 70:87]) < 0.03), axis=1)
        # arm_height_reward_r = - np.sum(np.log(20 * np.clip(obs_new_r[:, 89:93], a_min=0.001 , a_max=0.05)), axis=1)
        table_reward_r = -np.sum(np.log(50*np.clip(obs_new_r[:, 70:87], a_min=0.002, a_max=0.02)) * finger_weights, axis=1)
        arm_height_reward_r = -np.sum(np.log(50*np.clip(obs_new_r[:, 89:93], a_min=0.002, a_max=0.02)), axis=1)
        # if the abs of the first 6 dim of action_r are larger than 6, then give a negative reward arm_action_reward_r
        arm_action_reward_r = np.sum((np.abs(action_r[:, :6])-4) * (np.abs(action_r[:, :6]) > 4), axis=1)
        hand_action_reward_r = np.sum((np.abs(action_r[:, 6:])-2) * (np.abs(action_r[:, 6:]) > 2), axis=1)

        for i in range(num_envs):
            rewards_r[i]['affordance_reward'] = affordance_reward_r[i] * cfg['environment']['reward']['affordance_reward']['coeff']
            rewards_r[i]['center_reward'] = center_loss[i] * cfg['environment']['reward']['center_reward']['coeff']
            rewards_r[i]['table_reward'] = table_reward_r[i] * cfg['environment']['reward']['table_reward']['coeff']
            rewards_r[i]['arm_height_reward'] = arm_height_reward_r[i] * cfg['environment']['reward']['arm_height_reward']['coeff']
            rewards_r[i]['arm_action_reward'] = arm_action_reward_r[i] * min(update/1000, 1.0) * cfg['environment']['reward']['arm_action_reward']['coeff']
            rewards_r[i]['hand_action_reward'] = hand_action_reward_r[i] * min(update/1000, 1.0) * cfg['environment']['reward']['hand_action_reward']['coeff']

            # rewards_r[i]['reward_sum'] = (
            #             rewards_r[i]['reward_sum'] + rewards_r[i]['affordance_reward'] + rewards_r[i]['center_reward'] +
            #             rewards_r[i]['table_reward'] + rewards_r[i]['arm_height_reward'])
            rewards_r[i]['reward_sum'] = (
                        rewards_r[i]['reward_sum'] + rewards_r[i]['affordance_reward'] + rewards_r[i]['center_reward'] +
                        rewards_r[i]['table_reward'] + rewards_r[i]['arm_height_reward'] + rewards_r[i]['arm_action_reward'] + rewards_r[i]['hand_action_reward'])

            reward_r[i] = rewards_r[i]['reward_sum']
        reward_r.clip(min=reward_clip)

        for i in range(len(rewards_r_sum)):
            for k in rewards_r_sum[i].keys():
                rewards_r_sum[i][k] = rewards_r_sum[i][k] + rewards_r[i][k]

        ppo_r.step(value_obs=obs_r, rews=reward_r, dones=dones)

    obs_r, _ = env.observe_vision_new()

    obs_r = obs_r[:, :].astype('float32')

    if np.isnan(obs_r).any():
        print('nan in obs')
        print(obs_r)

    # update policy
    ppo_r.update(actor_obs=obs_r, value_obs=obs_r, log_this_iteration=update % 10 == 0, update=update)

    actor_r.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))

    if ppo_r.check_exploding_gradient():
        print("------------------- exploding gradient !!! will reload param --------------------")
        ppo_r.is_exploding_gradient = False
        load_pth = saver.data_dir + "/full_" + str(saved_update_idx) + '_r.pt'
        load_param(load_pth, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)


    end = time.time()

    ave_reward = {}
    for k in rewards_r_sum[0].keys():
        ave_reward[k] = 0
    for k in rewards_r_sum[0].keys():
        for i in range(len(rewards_r_sum)):
            ave_reward[k] = ave_reward[k] + rewards_r_sum[i][k]
        ave_reward[k] = ave_reward[k] / (len(rewards_r_sum) * n_steps_r)
    if args.log_name is not None:
        wandb.log(ave_reward)

    if args.log_name is None:
        print(ave_reward)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average reward: ", '{:0.10f}'.format(ave_reward['reward_sum'])))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps_r / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps_r / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    # print('std: ')
    # print(np.exp(actor_r.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')