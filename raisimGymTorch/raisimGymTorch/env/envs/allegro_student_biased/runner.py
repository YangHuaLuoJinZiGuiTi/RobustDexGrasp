import numpy
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_student_biased as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.allegro_student_biased import NormalSampler
from raisimGymTorch.helper.initial_pose_final import sample_rot_mats

import sys
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

import os
import math
import time
# import raisimGymTorch.algo.ppo.module as ppo_module
# import raisimGymTorch.algo.ppo.ppo as PPO
import raisimGymTorch.algo.ppo_dagger_recon.module as ppo_module
from raisimGymTorch.algo.ppo_dagger_recon.dagger_partial import Dagger
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

exp_name = "arm_rand_student"

# weight_saved = '/../../arm_rand/2024-12-27-18-08-06/full_12500_r.pt'
# weight_saved = '/../../arm_rand/2024-12-27-18-11-12/full_15000_r.pt'
# weight_saved = '/../../arm_rand/2025-01-07-20-39-18/full_15000_r.pt'
# weight_saved = '/../../arm_rand/2025-01-12-18-16-55/full_12000_r.pt'
# weight_saved = '/../../arm_rand/2025-01-12-18-18-27/full_14000_r.pt'
# weight_saved = '/../../arm_rand/2025-01-12-18-19-14/full_16000_r.pt'
# weight_saved = '/../../arm_rand/2025-01-16-10-36-16/full_16000_r.pt'
# weight_saved = '/../../arm_rand/2025-01-17-08-07-58/full_18000_r.pt'
# weight_saved = '/../../arm_rand/2025-01-17-07-33-45/full_14000_r.pt'
# weight_saved = '/../../arm_rand/2025-01-16-17-45-39/full_8500_r.pt'

# weight_saved = '/../../arm_rand/2025-01-22-14-00-13/full_12500_r.pt'
# weight_saved = '/../../arm_rand/2025-01-22-14-00-13/full_12500_r.pt'

# weight_saved = '/../../arm_rand/2025-03-10-10-38-43/full_10000_r.pt'

weight_saved = '/../../arm_rand/2025-03-14-10-22-19/full_10000_r.pt'
# weight_saved = '/../../arm_rand/2025-03-14-10-57-52/full_11500_r.pt'

weight_path_student = '2024-10-28-14-49-02/full_1000_r.pt'


# weight_saved = '/../../arm_rand/2025-01-12-18-19-14/full_16000_r.pt'
# weight_saved = '/../../arm_rand/2025-01-16-10-36-16/full_16000_r.pt'

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

if args.log_name is not None:
    wandb.init(project=task_name, config=cfg, name=args.log_name)

if args.seed != 1:
    cfg['seed'] = args.seed

obj_path_list = []
obj_list = []

# directory_path = home_path + "/rsc/mixed_train/"
# cat_name = 'large_training'
# cat_name = 'ycb_urdf_sim'
# cat_name = 'ycb_urdf_all'
# cat_name = 'ycb_urdf_light'
# cat_name = 'ycb_urdf_sim_light'
cat_name = 'new_training_set'
# cat_name = 'new_training_set_eval'

if cat_name == 'new_training_set' or cat_name == 'new_training_set_eval':
    repeat_per_obj = 2
else:
    repeat_per_obj = 3

cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path, file=sys.stdout)
items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

obj_path_list = []
obj_ori_list = folder_names
if cat_name == 'new_training_set' or cat_name == 'new_training_set_eval':
    obj_ori_list.append('009_gelatin_box')
    # obj_ori_list.append('011_banana')
    # obj_ori_list.append('011_banana')
    # obj_ori_list.append('019_pitcher_base')
    # obj_ori_list.append('037_scissors')
    obj_ori_list.append('052_extra_large_clamp')
    # obj_ori_list.append('big_tape')
    # obj_ori_list.append('big_tape')
    obj_ori_list.append('big_tape')
    # obj_ori_list.append('big_tape')
    # obj_ori_list.append('hammer')
    obj_ori_list.append('hammer')
    obj_ori_list.append('loopy_head_side')
    # obj_ori_list.append('loopy_head_side')
    # obj_ori_list.append('small_block')
    obj_ori_list.append('small_block')

    obj_ori_list.append('003_cracker_box')
    # obj_ori_list.append('off_water_body')
    # obj_ori_list.append('037_scissors')
    obj_ori_list.append('019_pitcher_base')
    obj_ori_list.append('mouse')
    obj_ori_list.append('011_banana')
    obj_ori_list.append('gun_functional')
    # obj_ori_list.append('wood_block_oriented')
    # obj_ori_list.append('suger_box_oriented')
    # obj_ori_list.append('cracker_box_oriented')
    obj_ori_list.append('power_drill_oriented')
# label = {}

num_envs = len(obj_ori_list) * repeat_per_obj
obj_list = []
for i in range(repeat_per_obj):
    for item in obj_ori_list:
        obj_list.append(item)

activations = nn.LeakyReLU

if args.log_name is None:
    num_envs = repeat_per_obj
    obj_list = choices(obj_list, k=num_envs)
    # obj_list = obj_list[:2]
    cfg['environment']['visualize'] = True


cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs, file=sys.stdout)

# 检查配置中是否有非均匀采样的标志位，如果没有则默认为False
non_uniform_sampling = cfg['environment'].get('non_uniform_sampling', False)
if non_uniform_sampling:
    print("Training with non-uniform sampling (biased towards edges)", file=sys.stdout)
else:
    print("Training with uniform sampling", file=sys.stdout)
print("Evaluation will always use uniform sampling for fair assessment", file=sys.stdout)

# Environment definition
env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)



# Training
reward_clip = -2.0
n_steps_r = cfg['environment']['grasp_steps']
total_steps_r = n_steps_r * env.num_envs

test_dir = False

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp",
                                       task_path + "/runner.py", task_path + "/runner_eval.py", task_path + "/../../RaisimGymVecEnvOther.py"], test_dir=test_dir)


ob_dim_r = 153
act_dim = 22
print('ob dim', ob_dim_r, file=sys.stdout)
print('act dim', act_dim, file=sys.stdout)

tobeEncode_dim = 44
t_steps = 10
prop_latent_dim=26
aff_vec_dim = 51
total_obs_dim = tobeEncode_dim*t_steps + ob_dim_r

update_mlp = True
student_driven_ratio=1.0
if update_mlp:
    ppo_ratio = 0.5
else:
    ppo_ratio = 0

print('update mlp: ', update_mlp, file=sys.stdout)
print('student driven ratio: ', student_driven_ratio, file=sys.stdout)
print('ppo ratio: ', ppo_ratio, file=sys.stdout)

# RL network
actor_expert_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)
actor_student_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)
critic_student_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)


print('loading expert policy from: ', saver.data_dir.split('eval')[0] + weight_path, file=sys.stdout)

checkpoint = torch.load(saver.data_dir.split('eval')[0] + weight_path, map_location=torch.device(device))
actor_expert_r.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
actor_expert_r.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])

expert_policy = actor_expert_r.architecture
prop_latent_encoder = ppo_module.LSTM_StateHistoryEncoder(tobeEncode_dim, prop_latent_dim, t_steps, device).to(device)

dagger = Dagger(expert_policy=expert_policy,
                actor_student=actor_student_r,
                critic_student=critic_student_r,
                prop_latent_encoder=prop_latent_encoder,
                tobeEncode_dim=tobeEncode_dim,
                prop_latent_dim=prop_latent_dim,
                total_obs_dim=total_obs_dim,
                mlp_obs_dim=ob_dim_r,
                t_steps=t_steps,
                num_envs=num_envs,
                num_transitions_per_env=n_steps_r,
                num_learning_epochs=4,
                gamma=0.996,
                lam=0.95,
                num_mini_batches=4,
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False,
                update_mlp=update_mlp,
                ppo_ratio=ppo_ratio
                )

if args.load_trained_policy:
    print('loading trained policy from: ', saver.data_dir.split('eval')[0] + weight_path_student)
    checkpoint_student = torch.load(saver.data_dir.split('eval')[0] + weight_path_student, map_location=torch.device(device))

    actor_student_r.architecture.load_state_dict(checkpoint_student['actor_architecture_state_dict'])
    actor_student_r.distribution.load_state_dict(checkpoint_student['actor_distribution_state_dict'])
    critic_student_r.architecture.load_state_dict(checkpoint_student['critic_architecture_state_dict'])
    prop_latent_encoder.load_state_dict(checkpoint_student['prop_latent_encoder_state_dict'])
    # dagger.optimizer.load_state_dict(checkpoint_student['optimizer_state_dict'])

else:
    actor_student_r.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor_student_r.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    critic_student_r.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])



finger_weights = np.ones((num_envs, 17)).astype('float32')
for i in range(4):
    finger_weights[:, 4 * i+4] *= 4.0
finger_weights[:, 16] *= 2.0
finger_weights /= finger_weights.sum(axis=1).reshape(-1, 1)
finger_weights *= 17.0
affordance_reward_r = np.zeros((num_envs, 1))
center_reward_r = np.zeros((num_envs, 1))
table_reward_r = np.zeros((num_envs, 1))
arm_height_reward_r = np.zeros((num_envs, 1))
arm_action_reward_r = np.zeros((num_envs, 1))
hand_action_reward_r = np.zeros((num_envs, 1))
arm_collision_reward_r = np.zeros((num_envs, 1))

qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

lowest_points = np.zeros((num_envs, 1), dtype='float32')
stable_states = np.zeros((num_envs, 7), dtype='float32')
for i in range(num_envs):
    txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
    with open(txt_file_path, 'r') as txt_file:
        lowest_points[i] = float(txt_file.read())
    # if cat_name == 'large_training':
    #     stable_state_path = home_path + f"/rsc/stable_states/surdf_group27_m/{obj_list[i]}.npy"
    #     stable_states[i] = np.load(stable_state_path)

for update in range(args.num_iterations):
    # if update % 300 == 0 and ppo_ratio < 1.0 and update > 10:
    #     ppo_ratio += 0.2
    #     student_driven_ratio += 0.2
    #     dagger.update_ppo_ratio(ppo_ratio)
    #     print('ppo ratio: ', ppo_ratio)
    #     print('student driven ratio: ', student_driven_ratio)
    # np.random.seed(int(time.time()))

    if cfg['environment']['curriculum']:
        ppo_ratio = min(update*0.0005, 1.0)
        # student_driven_ratio = min(update*0.0005, 1.0)
        dagger.update_ppo_ratio(ppo_ratio)
    if update % 1000 == 0:
        print('ppo ratio: ', ppo_ratio, file=sys.stdout)
        print('student driven ratio: ', student_driven_ratio, file=sys.stdout)

    if cfg['environment']['eval_during_training']:
        # 每100个迭代周期执行一次评估
        is_evaluation = update % 100 == 0 
    else:
        is_evaluation = False
    
    if is_evaluation:
        print(f"Evaluating policy at iteration {update}...", file=sys.stdout)
        print("Using uniform sampling for fair evaluation", file=sys.stdout)
        # 在评估模式下，设置lift_step为100
        lift_steps = 100
        current_steps = n_steps_r + lift_steps
    else:
        # 正常训练模式
        lift_steps = 0
        current_steps = n_steps_r
    
    # 更新当前迭代的总步数
    total_steps = current_steps * env.num_envs

    start = time.time()

    ### Evaluate trained model visually (note always the first environment gets visualized)

    if update % cfg['environment']['eval_every_n'] == 0 and args.log_name is not None:
        print("Visualizing and evaluating the current policy", file=sys.stdout)
        torch.save({
            'actor_architecture_state_dict': actor_student_r.architecture.state_dict(),
            'actor_distribution_state_dict': actor_student_r.distribution.state_dict(),
            'critic_architecture_state_dict': critic_student_r.architecture.state_dict(),
            'optimizer_state_dict': dagger.optimizer.state_dict(),
            'prop_latent_encoder_state_dict': prop_latent_encoder.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '_r.pt')

        env.save_scaling(saver.data_dir, str(update))

    target_center = np.zeros_like(env.affordance_center)

    qpos_reset_r[:, 6:] = cfg['environment']['hardware']['init_finger_pose']


    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = cfg['environment']['camera_position'][0]
    view_point_world[:, 1] = cfg['environment']['camera_position'][1]
    view_point_world[:, 2] = cfg['environment']['camera_position'][2]

    hand_center_sample_w = np.zeros((1, 3))
    hand_center_sample_w[0, 0] = cfg['environment']['camera_position'][0]
    hand_center_sample_w[0, 1] = cfg['environment']['camera_position'][1]
    hand_center_sample_w[0, 2] = cfg['environment']['camera_position'][2]

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

    for i in range(num_envs):
        # sample object states
        # 在评估阶段始终使用均匀采样，在训练阶段根据配置决定
        # if is_evaluation or not non_uniform_sampling:
        #     # 均匀采样
        #     sample_x = np.random.uniform(-0.35, 0.35)
        #     sample_y = np.random.uniform(-0.8, -0.35)
        # else:
        #     if np.random.random() < 0.5:
        #         # 均匀采样
        #         sample_x = np.random.uniform(-0.35, 0.35)
        #         sample_y = np.random.uniform(-0.8, -0.35)
        #     else:
        #         # 使用Beta分布进行采样，使边缘概率更大但中心概率不为0
        #         # 对于x，使用Beta分布使两端概率更大
        #         x_beta = np.random.beta(0.7, 0.7)  # 在0和1附近概率更高
        #         sample_x = -0.35 + x_beta * (0.35 - (-0.35))  # 映射到[-0.35, 0.35]
                
        #         # 对于y，使用Beta分布使两端概率更大
        #         y_beta = np.random.beta(0.7, 0.7)  # 在0和1附近概率更高
        #         sample_y = -0.8 + y_beta * (-0.35 - (-0.8))  # 映射到[-0.8, -0.35]
        
        # # 计算对应的角度
        # angle = np.arctan2(sample_y, sample_x)
        if is_evaluation or not non_uniform_sampling:
            while True:
                angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
                distance = np.random.uniform(0.45, 0.75)
                sample_x = distance * np.cos(angle)
                sample_y = distance * np.sin(angle)
                if sample_x < 0.25 and sample_x > -0.25:
                    break
        else:        
            if np.random.random() < 0.5:
                # 50%概率使用均匀采样
                while True:
                    angle = np.random.uniform(-0.7 * np.pi, -0.3 * np.pi)
                    distance = np.random.uniform(0.45, 0.75)
                    sample_x = distance * np.cos(angle)
                    sample_y = distance * np.sin(angle)
                    if sample_x < 0.25 and sample_x > -0.25:
                        break
            else:
                # 50%概率使用偏向边缘的采样
                while True:
                    # 对于角度，使用Beta分布使边缘概率更大 Beta(0.5, 0.5)是U形分布，在0和1附近概率更高
                    beta_param = 0.5
                    angle_normalized = np.random.beta(beta_param, beta_param)  # 在[0,1]范围内，两端概率高
                    angle = -0.7 * np.pi + angle_normalized * (0.4 * np.pi) # 映射到[-0.7π, -0.3π]范围
                    # 对于距离，同样使用Beta(0.5, 0.5)分布使两端概率更高
                    distance_normalized = np.random.beta(beta_param, beta_param)  # 在[0,1]范围内，两端概率高
                    distance = 0.45 + distance_normalized * 0.3  # 映射到[0.45, 0.75]
                    sample_x = distance * np.cos(angle)
                    sample_y = distance * np.sin(angle)
                    if sample_x < 0.25 and sample_x > -0.25:
                        break

        obj_pose_reset[i, 0] = sample_x
        obj_pose_reset[i, 1] = sample_y
        # if cat_name == 'large_training':
        #     obj_pose_reset[i, 2:7] = stable_states[i, 2:7]
        #     obj_pose_reset[i, 2] += 0.005
        #     quats = stable_states[i, 3:7]
        # else:
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
                ik_results[j, :] = ik_result
                feasible_ik_flag[j] = True

        feasible_indices = np.where(feasible_ik_flag)[0]
        scores = np.ones(sample_num, dtype='float32')
        scores = scores * 10000.
        if min(projection_lengths) < 0.18:
            for j in feasible_indices:
                if projection_lengths[j] < 0.18:
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
        # if cat_name == 'large_training':
        #     qpos_reset_r[true_idx, :6] = [angle + np.pi / 2 - 0.3, -1.57, 1.57, 0., 1.57, -1.57]
        #     obj_pose_reset[true_idx, 0] = 0.15
        #     obj_pose_reset[true_idx, 1] = 0.2 - 0.75152
        # else:
        current_obj_idx = true_idx // repeat_per_obj
        current_obj_env_indices = []
        for i in range(repeat_per_obj):
            current_obj_env_indices.append(current_obj_idx * repeat_per_obj + i)
        false_indices = [idx for idx in current_obj_env_indices if not contains_one[idx]]
        if len(false_indices) > 0:
            chosen_index = np.random.choice(false_indices)
            qpos_reset_r[true_idx, :] = qpos_reset_r[chosen_index, :]
            obj_pose_reset[true_idx, :] = obj_pose_reset[chosen_index, :]
        else:
            qpos_reset_r[true_idx, :6] = [angle + np.pi / 2 - 0.3, -1.57, 1.57, 0., 1.57, -1.57]
            obj_pose_reset[true_idx, 0] = 0.1
            obj_pose_reset[true_idx, 1] = -0.5
            # obj_pose_reset[true_idx, 3:] = [1., -0., -0., 0., 0.]

    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 22), 'float32'),
                    np.zeros((num_envs, 22), 'float32'),
                    obj_pose_reset,
                    )

    obs_new_r, dis_info = env.observe_vision_new()
    env.update_target(target_center)
    aff_vec_new, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
    rewards_r_sum = env.get_reward_info_r()
    for i in range(len(rewards_r_sum)):
        rewards_r_sum[i]['affordance_reward'] = 0
        # rewards_r_sum[i]['not_affordance_reward'] = 0
        rewards_r_sum[i]['center_reward'] = 0
        rewards_r_sum[i]['table_reward'] = 0
        rewards_r_sum[i]['arm_height_reward'] = 0
        rewards_r_sum[i]['arm_action_reward'] = 0
        rewards_r_sum[i]['hand_action_reward'] = 0
        rewards_r_sum[i]['arm_collision_reward'] = 0

        for k in rewards_r_sum[i].keys():
            rewards_r_sum[i][k] = 0

    biased = cfg['environment']['biased']
    if biased:
        obj_biased = np.zeros((num_envs, 1), dtype='float32')
        obj_pos_bias = np.random.uniform(-0.05, 0.05, (num_envs, 3)).astype('float32')
    else:
        obj_pos_bias = np.zeros((num_envs, 3), dtype='float32')

    for step in range(n_steps_r + lift_steps):
        obs_r = obs_new_r
        obs_r = obs_r[:].astype('float32')
        aff_vec = aff_vec_new.astype('float32')

        action_r, student_mlp_obs = dagger.act(obs_r, student_driven_ratio, aff_vec, aff_vec_dim)

        # 如果在评估模式下且已经完成了抓取阶段，则进入提升阶段
        if is_evaluation and step >= n_steps_r:
            # 在提升阶段，使用固定的手臂姿态
            action_r[:, :6] = theta0
            if step == n_steps_r:
                # 开始提升时启用根部引导
                env.switch_root_guidance(True)
                print("Starting lift phase...", file=sys.stdout)

        reward_r, _, dones = env.step(action_r.astype('float32'), np.zeros_like(action_r).astype('float32'))

        obs_new_r, dis_info = env.observe_vision_new()
        obs_new_r = obs_new_r[:].astype('float32')
        aff_vec_new, _ = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))

        if biased:
            obj_pos_bias_current = np.zeros((num_envs, 3), dtype='float32')
            for i in range(num_envs):
                if np.min(dis_info[i, 0:17]) < 0.07 and obj_biased[i] == 0:
                # if random.uniform (0, 1) < 0.3 and obj_biased[i] == 0 and np.min(dis_info[i, 0:17]) > 0.07:
                    obj_biased[i] = 1
                    obj_pos_bias_current[i] = obj_pos_bias[i]
            env.switch_obj_pos(obj_pos_bias_current)

        global_state = env.get_global_state()

        abs_dis = np.linalg.norm(global_state[:, 115:118], axis=1)
        # if distance > 0.1 then give center reward otherwise set to 0
        center_loss = (abs_dis > 0.07) * (np.square(abs_dis - 0.07))

        rewards_r = env.get_reward_info_r()
        affordance_reward_r = - np.sum((dis_info[:, 1:17]) * finger_weights[:, 1:17], axis=1)
        # table_reward_r = - np.sum(np.log(np.maximum(np.abs(obs_new_r[:, -ob_dim_r+70:-ob_dim_r+87]), 0.01*np.ones_like(np.abs(obs_new_r[:, -ob_dim_r+70:-ob_dim_r+87])))) * finger_weights * (np.abs(obs_new_r[:, -ob_dim_r+70:-ob_dim_r+87]) < 0.03), axis=1)
        # arm_height_reward_r = - np.sum(np.log(20 * np.clip(obs_new_r[:, -ob_dim_r+89:-ob_dim_r+93], a_min=0.001 , a_max=0.05)), axis=1)
        table_reward_r = -np.sum(np.log(50 * np.clip(obs_new_r[:, -ob_dim_r+70:-ob_dim_r+87], a_min=0.002, a_max=0.02)) * finger_weights, axis=1)
        arm_height_reward_r = -np.sum(np.log(50 * np.clip(obs_new_r[:, -ob_dim_r+89:-ob_dim_r+93], a_min=0.002, a_max=0.02)), axis=1)
        # if the abs of the first 6 dim of action_r are larger than 6, then give a negative reward arm_action_reward_r
        arm_action_reward_r = np.sum((np.abs(action_r[:, :6])-4) * (np.abs(action_r[:, :6]) > 4), axis=1)
        hand_action_reward_r = np.sum((np.abs(action_r[:, 6:])-4) * (np.abs(action_r[:, 6:]) > 4), axis=1)

        one_check = global_state[:, 124:128]
        arm_collision_reward_r = np.sum(one_check, axis=1)

        for i in range(num_envs):
            rewards_r[i]['affordance_reward'] = affordance_reward_r[i] * cfg['environment']['reward']['affordance_reward']['coeff']
            rewards_r[i]['center_reward'] = center_loss[i] * cfg['environment']['reward']['center_reward']['coeff']
            rewards_r[i]['table_reward'] = table_reward_r[i] * cfg['environment']['reward']['table_reward']['coeff']
            rewards_r[i]['arm_height_reward'] = arm_height_reward_r[i] * cfg['environment']['reward']['arm_height_reward']['coeff']
            rewards_r[i]['arm_action_reward'] = arm_action_reward_r[i] * min(update/1000, 1.0) * cfg['environment']['reward']['arm_action_reward']['coeff']
            rewards_r[i]['hand_action_reward'] = hand_action_reward_r[i] * min(update/1000, 1.0) * cfg['environment']['reward']['hand_action_reward']['coeff']
            rewards_r[i]['arm_collision_reward'] = arm_collision_reward_r[i] * cfg['environment']['reward']['arm_collision_reward']['coeff']
            # rewards_r[i]['reward_sum'] = (
            #             rewards_r[i]['reward_sum'] + rewards_r[i]['affordance_reward'] + rewards_r[i]['center_reward'] +
            #             rewards_r[i]['table_reward'] + rewards_r[i]['arm_height_reward'])
            rewards_r[i]['reward_sum'] = (
                        rewards_r[i]['reward_sum'] + rewards_r[i]['affordance_reward'] + rewards_r[i]['center_reward'] +
                        rewards_r[i]['table_reward'] + rewards_r[i]['arm_height_reward'] + rewards_r[i]['arm_action_reward'] + rewards_r[i]['hand_action_reward'] + rewards_r[i]['arm_collision_reward'])
            reward_r[i] = rewards_r[i]['reward_sum']
        reward_r.clip(min=reward_clip)

        for i in range(len(rewards_r_sum)):
            for k in rewards_r_sum[i].keys():
                rewards_r_sum[i][k] = rewards_r_sum[i][k] + rewards_r[i][k]

        # 只在非评估模式下收集训练数据
        if not is_evaluation:
            obs_r_student = np.concatenate([obs_r[:, :tobeEncode_dim*t_steps], student_mlp_obs], axis=1)
            dagger.step(total_obs=obs_r_student, rews=reward_r, dones=dones, value_obs=obs_r[:, -ob_dim_r:])

    # 如果是评估模式，计算成功率
    if is_evaluation:
        # 获取全局状态以检查物体是否被提升
        global_state = env.get_global_state()
        lifted = global_state[:, 107] - obj_pose_reset[:, 2] > 0.1
        success_rate = np.sum(lifted) / num_envs
        
        # 打印当前成功率
        print(f"Evaluation success rate at iteration {update}: {success_rate:.4f}", file=sys.stdout)
        
        # 计算并打印各个物体的成功率
        print("\n===== Per-Object Success Rate =====", file=sys.stdout)
        # 创建一个字典来存储每个物体的成功和尝试次数
        object_success = {}
        for i in range(num_envs):
            obj_name = obj_list[i]
            if obj_name not in object_success:
                object_success[obj_name] = {"success": 0, "attempts": 0}    
            object_success[obj_name]["attempts"] += 1
            if lifted[i]:
                object_success[obj_name]["success"] += 1
        
        # 打印物体类型统计
        print(f"Total object types: {len(object_success)}", file=sys.stdout)
        print(f"Total environments: {num_envs}", file=sys.stdout)

        # 打印每个物体的成功率，按成功率从低到高排序
        sorted_objects = sorted(
            object_success.items(),
            key=lambda x: (x[1]["success"] / x[1]["attempts"]) if x[1]["attempts"] > 0 else 0
        )
        for obj_name, stats in sorted_objects:
            success_rate_obj = (stats["success"] / stats["attempts"]) if stats["attempts"] > 0 else 0
            print(f"{obj_name:<30} Success: {stats['success']}/{stats['attempts']} ({success_rate_obj:.2%})", file=sys.stdout)
        
        print("=====================================\n", file=sys.stdout)

        # 如果使用wandb，记录评估结果
        if args.log_name is not None:
            wandb.log({"evaluation_success_rate": success_rate}, step=update)
            # 也记录每个物体的成功率
            for obj_name, stats in object_success.items():
                success_rate_obj = (stats["success"] / stats["attempts"]) if stats["attempts"] > 0 else 0
                wandb.log({f"object_success_rate/{obj_name}": success_rate_obj}, step=update)

        # 关闭根部引导
        env.switch_root_guidance(False)

    obs_r, _ = env.observe_vision_new()
    value_obs = obs_r[:, -ob_dim_r:]

    # 只在非评估模式下更新策略
    if not is_evaluation:
        prop_mse_loss, action_mse_loss = dagger.update(value_obs)

    # TODO: not sure whether should keep
    actor_student_r.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))

    end = time.time()

    ave_reward = {}
    for k in rewards_r_sum[0].keys():
        ave_reward[k] = 0
    for k in rewards_r_sum[0].keys():
        for i in range(len(rewards_r_sum)):
            ave_reward[k] = ave_reward[k] + rewards_r_sum[i][k]
        # 在计算平均奖励时，只考虑抓取阶段的步数
        ave_reward[k] = ave_reward[k] / (len(rewards_r_sum) * n_steps_r)
    ave_reward['recon_loss'] = prop_mse_loss
    ave_reward['action_loss'] = action_mse_loss

    # 只在非评估模式下记录训练奖励
    if args.log_name is not None and not is_evaluation:
        wandb.log(ave_reward, step=update)

    if args.log_name is None:
        print(ave_reward, file=sys.stdout)

    print('----------------------------------------------------', file=sys.stdout)
    print('{:>6}th iteration'.format(update), file=sys.stdout)
    print('{:<40} {:>6}'.format("average reward: ", '{:0.10f}'.format(ave_reward['reward_sum'])), file=sys.stdout)
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)), file=sys.stdout)
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps_r / (end - start))), file=sys.stdout)
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps_r / (end - start)
                                                                * cfg['environment']['control_dt'])), file=sys.stdout)
    print('{:<40} {:>6}'.format("prop mse loss: ", '{:0.10f}'.format(prop_mse_loss)), file=sys.stdout)
    print('{:<40} {:>6}'.format("action mse loss: ", '{:0.10f}'.format(action_mse_loss)), file=sys.stdout)
    # print('std: ')
    # print(np.exp(actor_r.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n', file=sys.stdout)