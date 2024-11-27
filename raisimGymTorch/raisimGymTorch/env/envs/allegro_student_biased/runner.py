import numpy
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import arm_rand_student_partial as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.arm_rand_student_partial import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_faive, get_initial_pose_faive_random, get_initial_pose_allegro_arm_rand, get_initial_pose_allegro_arm_rand_test, get_initial_pose_allegro_arm_partial

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

weight_saved = '/../../arm_rand/2024-11-04-16-42-02/full_50000_r.pt'
weight_path_student = '2024-10-28-14-49-02/full_1000_r.pt'


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

if args.log_name is not None:
    wandb.init(project=task_name, config=cfg, name=args.log_name)

if args.seed != 1:
    cfg['seed'] = args.seed

obj_path_list = []
obj_list = []

# directory_path = home_path + "/rsc/mixed_train/"
cat_name = 'ycb_urdf_all'
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
    num_envs = 2
    obj_list = choices(obj_list, k=num_envs)
    # obj_list = obj_list[:2]
    cfg['environment']['visualize'] = True


cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)
# Environment definition
env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)



# Training
reward_clip = -2.0
n_steps_r = 120
total_steps_r = n_steps_r * env.num_envs

# print(env.num_envs)

test_dir = False

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp",
                                       task_path + "/runner.py", task_path + "/runner_eval.py", task_path + "/../../RaisimGymVecEnvOther.py"], test_dir=test_dir)


ob_dim_r = 153
act_dim = 22
print('ob dim', ob_dim_r)
print('act dim', act_dim)

tobeEncode_dim = 44
t_steps = 10
prop_latent_dim=26
aff_vec_dim = 54
total_obs_dim = tobeEncode_dim*t_steps + ob_dim_r

update_mlp = True
student_driven_ratio=0.5
if update_mlp:
    ppo_ratio = 0.5
else:
    ppo_ratio = 0

print('update mlp: ', update_mlp)
print('student driven ratio: ', student_driven_ratio)
print('ppo ratio: ', ppo_ratio)

# RL network
actor_expert_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)
actor_student_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)
critic_student_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)

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
finger_weights /= finger_weights.sum(axis=1).reshape(-1, 1)
finger_weights *= 17.0
affordance_reward_r = np.zeros((num_envs, 1))
center_reward_r = np.zeros((num_envs, 1))
table_reward_r = np.zeros((num_envs, 1))
arm_height_reward_r = np.zeros((num_envs, 1))


qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

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
            'actor_architecture_state_dict': actor_student_r.architecture.state_dict(),
            'actor_distribution_state_dict': actor_student_r.distribution.state_dict(),
            'critic_architecture_state_dict': critic_student_r.architecture.state_dict(),
            'optimizer_state_dict': dagger.optimizer.state_dict(),
            'prop_latent_encoder_state_dict': prop_latent_encoder.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '_r.pt')

        env.save_scaling(saver.data_dir, str(update))

    target_center = np.zeros_like(env.affordance_center)

    qpos_reset_r[:, 6:] = 0.2
    qpos_reset_r[:, -4] = 1.57
    qpos_reset_r[:, 7] = 0.8
    qpos_reset_r[:, 11] = 0.8
    qpos_reset_r[:, 15] = 0.8
    qpos_reset_r[:, 19] = 0
    qpos_reset_r[:, 20] = -0.5

    visible_points_w = np.zeros((num_envs, 200, 3), dtype='float32')
    visible_points_obj = np.zeros((num_envs, 200, 3), dtype='float32')

    view_point_world = np.zeros((200, 3))
    view_point_world[:, 0] = 0.8 - 0.55
    view_point_world[:, 1] = 0.2 - 0.75152
    view_point_world[:, 2] = 1.5

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

    for i in range(num_envs):
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

            # # get the x_dir of the grasping frame
            # obj_aff_center_in_obj = np.mean(visible_points_obj[i].reshape(200, 3), axis=0)
            # hand_center_obj = np.matmul(obj_mat_single.T, (hand_center_sample_w - obj_pose_reset[i, :3]).T).T
            # hand_dir_x_in_obj = hand_center_obj - obj_aff_center_in_obj
            # hand_dir_x_in_obj = hand_dir_x_in_obj / np.linalg.norm(hand_dir_x_in_obj, axis=1, keepdims=True)
            #
            # target_center[i, :] = obj_aff_center_in_obj - 0.01 * hand_dir_x_in_obj
            # pos = obj_aff_center_in_obj + 0.25 * hand_dir_x_in_obj
            # rot = get_initial_pose_allegro_arm_partial(visible_points_obj[i], hand_dir_x_in_obj, obj_mat_single,
            #                                            top=False)
            # if rot is None:
            #     hand_dir_x_in_obj[0, :] = 0
            #     hand_dir_x_in_obj[0, 2] = 1
            #     rot = get_initial_pose_allegro_arm_partial(visible_points_obj[i], hand_dir_x_in_obj, obj_mat_single,
            #                                                top=True)
            #
            # wrist_in_world = np.matmul(obj_mat_single, rot)
            # wrist_pose = rotations.mat2euler(wrist_in_world)
            # qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat_single, pos[0, :])

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
            qpos_reset_r[true_idx, :6] = [0, -1.57, 1.57, 0., 1.57, -1.57]
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
    aff_vec_new, show_point = env.observe_student_aff(torch.from_numpy(visible_points_w).to(device))
    rewards_r_sum = env.get_reward_info_r()
    for i in range(len(rewards_r_sum)):
        rewards_r_sum[i]['affordance_reward'] = 0
        # rewards_r_sum[i]['not_affordance_reward'] = 0
        rewards_r_sum[i]['center_reward'] = 0
        rewards_r_sum[i]['table_reward'] = 0
        rewards_r_sum[i]['arm_height_reward'] = 0

        for k in rewards_r_sum[i].keys():
            rewards_r_sum[i][k] = 0

    biased = False
    if biased:
        obj_biased = np.zeros((num_envs, 1), dtype='float32')
        obj_pos_bias = np.random.uniform(-0.05, 0.05, (num_envs, 3)).astype('float32')
    else:
        obj_pos_bias = np.zeros((num_envs, 3), dtype='float32')

    for step in range(n_steps_r):
        obs_r = obs_new_r
        obs_r = obs_r[:].astype('float32')
        aff_vec = aff_vec_new.astype('float32')

        action_r, student_mlp_obs = dagger.act(obs_r, student_driven_ratio, aff_vec, aff_vec_dim)

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
        affordance_reward_r = - np.sum((dis_info[:, :17]) * finger_weights, axis=1)
        table_reward_r = - np.sum(np.log(np.maximum(np.abs(obs_new_r[:, -ob_dim_r+70:-ob_dim_r+87]), 0.01*np.ones_like(np.abs(obs_new_r[:, -ob_dim_r+70:-ob_dim_r+87])))) * finger_weights * (np.abs(obs_new_r[:, -ob_dim_r+70:-ob_dim_r+87]) < 0.03), axis=1)
        arm_height_reward_r = - np.sum(np.log(20 * np.clip(obs_new_r[:, -ob_dim_r+89:-ob_dim_r+93], a_min=0.001 , a_max=0.05)), axis=1)

        for i in range(num_envs):
            rewards_r[i]['affordance_reward'] = affordance_reward_r[i] * cfg['environment']['reward']['affordance_reward']['coeff']
            rewards_r[i]['center_reward'] = center_loss[i] * cfg['environment']['reward']['center_reward']['coeff']
            rewards_r[i]['table_reward'] = table_reward_r[i] * cfg['environment']['reward']['table_reward']['coeff']
            rewards_r[i]['arm_height_reward'] = arm_height_reward_r[i] * cfg['environment']['reward']['arm_height_reward']['coeff']

            rewards_r[i]['reward_sum'] = (
                        rewards_r[i]['reward_sum'] + rewards_r[i]['affordance_reward'] + rewards_r[i]['center_reward'] +
                        rewards_r[i]['table_reward'] + rewards_r[i]['arm_height_reward'])

            reward_r[i] = rewards_r[i]['reward_sum']
        reward_r.clip(min=reward_clip)

        for i in range(len(rewards_r_sum)):
            for k in rewards_r_sum[i].keys():
                rewards_r_sum[i][k] = rewards_r_sum[i][k] + rewards_r[i][k]

        obs_r_student = np.concatenate([obs_r[:, :tobeEncode_dim*t_steps], student_mlp_obs], axis=1)
        dagger.step(total_obs=obs_r_student, rews=reward_r, dones=dones, value_obs=obs_r[:, -ob_dim_r:])

    obs_r, _ = env.observe_vision_new()
    value_obs = obs_r[:, -ob_dim_r:]
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
        ave_reward[k] = ave_reward[k] / (len(rewards_r_sum) * n_steps_r)
    ave_reward['recon_loss'] = prop_mse_loss
    ave_reward['action_loss'] = action_mse_loss
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
    print('{:<40} {:>6}'.format("prop mse loss: ", '{:0.10f}'.format(prop_mse_loss)))
    print('{:<40} {:>6}'.format("action mse loss: ", '{:0.10f}'.format(action_mse_loss)))
    # print('std: ')
    # print(np.exp(actor_r.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')