import numpy
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import allegro_dagger as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.allegro_dagger import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_faive, get_initial_pose_faive_random, get_initial_pose_allegro_new

import os
import math
import time
# import raisimGymTorch.algo.ppo.module as ppo_module
# import raisimGymTorch.algo.ppo.ppo as PPO
import raisimGymTorch.algo.ppo_dagger_recon.module as ppo_module
from raisimGymTorch.algo.ppo_dagger_recon.dagger_new import Dagger

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

exp_name = "pt_allegro_student"

# weight_saved = '/../../faive_fixed/2024-01-29-22-01-24/full_2700_r.pt'
weight_saved = '/../../pt_allegro_floating/2024-06-20-13-23-53/full_17500_r.pt'
# weight_saved = '/../../pt_allegro_fixed/2024-05-30-18-50-43/full_9000_r.pt'
weight_path_student = '/../2024-06-21-10-52-17/full_800_r.pt'

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

wandb.init(project=task_name, config=cfg, name=args.log_name)

if args.seed != 1:
    cfg['seed'] = args.seed

obj_path_list = []
obj_list = []

# directory_path = home_path + "/rsc/mixed_train/"
cat_name = 'mixed_train'

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

# cfg['environment']['visualize'] = True
# num_envs = 1
# obj_list = choices(obj_list, k=num_envs)


cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)
# Environment definition
env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)



# Training
trail_steps = 30
reward_clip = -2.0
grasp_steps = 100
n_steps_r = grasp_steps + trail_steps
total_steps_r = n_steps_r * env.num_envs

print(env.num_envs)

test_dir = False

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp",
                                       task_path + "/runner.py", task_path + "/../../RaisimGymVecEnvOther.py"], test_dir=test_dir)




ob_dim_r = 144
act_dim = 22
print('ob dim', ob_dim_r)
print('act dim', act_dim)

tobeEncode_dim = 44
t_steps = 10
prop_latent_dim=26
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

checkpoint = torch.load(saver.data_dir.split('eval')[0] + weight_path, map_location=torch.device('cpu'))
actor_expert_r.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
actor_expert_r.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])



# expert_path = saver.data_dir.split('eval')[0] + weight_path
# expert_policy = DaggerExpert(expert_path, total_obs_dim, t_steps,
#                              tobeEncode_dim, prop_latent_dim, num_envs, actor_expert_r, critic_r, device)

expert_policy = actor_expert_r.architecture
prop_latent_encoder = ppo_module.LSTM_StateHistoryEncoder(tobeEncode_dim, prop_latent_dim, t_steps, device).to(device)

# if args.load_trained_policy:
#     trained_prop_loaded_encoder = torch.jit.load(saver.data_dir.split('eval')[0] + encoder_saved)
#     prop_latent_encoder.load_state_dict(trained_prop_loaded_encoder.state_dict())


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

# agent = DaggerAgent(expert_policy,
#                     prop_latent_encoder,
#                     t_steps, tobeEncode_dim, device)
# dagger = DaggerTrainer(
#     agent=agent,
#     num_envs=num_envs,
#     num_transitions_per_env=n_steps_r,
#     obs_shape=total_obs_dim,
#     latent_shape=prop_latent_dim,
#     num_learning_epochs=4,
#     num_mini_batches=4,
#     device=device,
#     learning_rate=1e-3,
#     update_mlp=update_mlp
# )


# ppo_r = PPO.PPO(actor=actor_r,
#                 critic=critic_r,
#                 num_envs=num_envs,
#                 num_transitions_per_env=n_steps_r,
#                 num_learning_epochs=4,
#                 gamma=0.996,
#                 lam=0.95,
#                 num_mini_batches=4,
#                 device=device,
#                 log_dir=saver.data_dir,
#                 shuffle_batch=False
#                 # learning_rate=1e-4
#                 )

if args.load_trained_policy:
    print('loading trained policy from: ', saver.data_dir.split('eval')[0] + weight_path_student)
    checkpoint_student = torch.load(saver.data_dir.split('eval')[0] + weight_path_student, map_location=torch.device('cpu'))

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
not_affordance_reward_r = np.zeros((num_envs, 1))
direction_reward_r = np.zeros((num_envs, 1))
center_reward_r = np.zeros((num_envs, 1))
table_reward_r = np.zeros((num_envs, 1))

qpos_reset_r = np.zeros((num_envs, 22), dtype='float32')
qpos_reset_l = np.zeros((num_envs, 22), dtype='float32')
obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

for update in range(args.num_iterations):
    # reward_sum = 0
    start = time.time()

    ### Evaluate trained model visually (note always the first environment gets visualized)

    if update % cfg['environment']['eval_every_n'] == 0:
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
    object_center = np.zeros_like(env.affordance_center)
    fake_non_aff_center = [0.346408, 0.346408, 0.346408]
    contain_non_aff = np.zeros((num_envs, 1), dtype='float32')

    for i in range(num_envs):
        got_proper_initial_pose = False
        lowest_point = 0.
        txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
        with open(txt_file_path, 'r') as txt_file:
            lowest_point = float(txt_file.read())
        obj_pose_reset[i, :] = [1., -0., 0.502, 1., -0., -0., 0., 0.]
        obj_pose_reset[i, 2] -= lowest_point

        # TODO: 調整大拇指
        if new_allegro:
            qpos_reset_r[i, 6:] = 0.2
            qpos_reset_r[i, -4] = 1.0
            qpos_reset_r[i, 7] = 0.8
            qpos_reset_r[i, 11] = 0.8
            qpos_reset_r[i, 15] = 0.8
            qpos_reset_r[i, 19] = 0.8

            # qpos_reset_r[i, -4] = 1.3
            # qpos_reset_r[i, 7] = 0.8
            # qpos_reset_r[i, 11] = 0.8
            # qpos_reset_r[i, 15] = 0.8
            # qpos_reset_r[i, 19] = 0.8
        else:
            qpos_reset_r[i, -4] = 1.7


        if np.linalg.norm(env.non_aff_mesh[i].centroid - fake_non_aff_center) < 0.01:
            non_aff_mesh = None
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

    obs_new_r, _ = env.observe_vision(contain_non_aff, allegro=True)
    rewards_r_sum = env.get_reward_info_r()
    for i in range(len(rewards_r_sum)):
        rewards_r_sum[i]['affordance_reward'] = 0
        rewards_r_sum[i]['not_affordance_reward'] = 0
        rewards_r_sum[i]['direction_reward'] = 0
        rewards_r_sum[i]['center_reward'] = 0
        rewards_r_sum[i]['table_reward'] = 0


        for k in rewards_r_sum[i].keys():
            rewards_r_sum[i][k] = 0

    for step in range(n_steps_r):
        # print('step', step)
        obs_r = obs_new_r
        obs_r = obs_r[:].astype('float32')

        # if np.isnan(obs_r).any():
        #     print('nan in obs')

        # time.sleep(0.05)

        action_r = dagger.act(obs_r, student_driven_ratio)

        reward_r, _, dones = env.step(action_r.astype('float32'), np.zeros_like(action_r).astype('float32'))

        obs_new_r, dis_info = env.observe_vision(contain_non_aff, allegro=True)
        obs_new_r = obs_new_r[:].astype('float32')

        global_state = env.get_global_state()

        direction_loss = np.sum(np.square(global_state[:, 112:115]), axis=1)
        abs_dis = np.linalg.norm(obs_new_r[:, :3], axis=1)
        # if distance > 0.1 then give center reward otherwise set to 0
        center_loss = (abs_dis > 0.07) * (np.square(abs_dis - 0.07))

        rewards_r = env.get_reward_info_r()
        affordance_reward_r = - np.sum((dis_info[:, :17]) * finger_weights, axis=1)
        # non_affordance_reward_r = - np.sum((dis_info[:, 17:]) * finger_weights, axis=1)
        table_reward_r = - np.sum(np.log(np.maximum(np.abs(obs_new_r[:, 76:93]), 0.01*np.ones_like(np.abs(obs_new_r[:, 76:93])))) * finger_weights * (np.abs(obs_new_r[:, 76:93]) < 0.03), axis=1)

        for i in range(num_envs):
            rewards_r[i]['affordance_reward'] = affordance_reward_r[i] * cfg['environment']['reward']['affordance_reward']['coeff']
            rewards_r[i]['not_affordance_reward'] = 0
            rewards_r[i]['direction_reward'] = direction_loss[i] * cfg['environment']['reward']['direction_reward']['coeff']
            rewards_r[i]['center_reward'] = center_loss[i] * cfg['environment']['reward']['center_reward']['coeff']
            rewards_r[i]['table_reward'] = table_reward_r[i] * cfg['environment']['reward']['table_reward']['coeff']
            obj_vel_pul = rewards_r[i]['obj_vel_reward_']
            if obj_vel_pul < -0.75:
                obj_vel_pul = (obj_vel_pul + 0.75) / 4.0 - 0.75
            if obj_vel_pul < -1.0:
                obj_vel_pul = -1.0
            obj_qvel_pul = rewards_r[i]['obj_qvel_reward_']
            if obj_qvel_pul < -0.75:
                obj_qvel_pul = (obj_qvel_pul + 0.75) / 4.0 - 0.75
            if obj_qvel_pul < -1.0:
                obj_qvel_pul = -1.0

            wrist_vel_pul = rewards_r[i]['wrist_vel_reward_']
            if wrist_vel_pul < -0.75:
                wrist_vel_pul = (wrist_vel_pul + 0.75) / 4.0 - 0.75
            if wrist_vel_pul < -1.0:
                wrist_vel_pul = -1.0
            wrist_qvel_pul = rewards_r[i]['wrist_qvel_reward_']
            if wrist_qvel_pul < -0.75:
                wrist_qvel_pul = (wrist_qvel_pul + 0.75) / 4.0 - 0.75
            if wrist_qvel_pul < -1.0:
                wrist_qvel_pul = -1.0

            rewards_r[i]['reward_sum'] = (rewards_r[i]['reward_sum'] + rewards_r[i]['affordance_reward'] + rewards_r[i]['direction_reward'] + rewards_r[i]['center_reward'] + rewards_r[i]['table_reward']
                                          - rewards_r[i]['obj_vel_reward_'] - rewards_r[i]['obj_qvel_reward_'] + obj_vel_pul + obj_qvel_pul + rewards_r[i]['not_affordance_reward'] - rewards_r[i]['wrist_vel_reward_'] - rewards_r[i]['wrist_qvel_reward_'] + wrist_vel_pul + wrist_qvel_pul)
            rewards_r[i]['obj_vel_reward_'] = obj_vel_pul
            rewards_r[i]['obj_qvel_reward_'] = obj_qvel_pul
            rewards_r[i]['wrist_vel_reward_'] = wrist_vel_pul
            rewards_r[i]['wrist_qvel_reward_'] = wrist_qvel_pul
            # rewards_r[i]['reward_sum'] = rewards_r[i]['reward_sum'] + rewards_r[i]['pca_reward'] + rewards_r[i]['affordance_reward'] + rewards_r[i]['not_affordance_reward'] + rewards_r[i]['anatomy_reward'] + rewards_r[i]['direction_reward'] + rewards_r[i]['finger_tip_reward']
            reward_r[i] = rewards_r[i]['reward_sum']
        reward_r.clip(min=reward_clip)
        # reward_sum = reward_sum + np.sum(reward_r)

        for i in range(len(rewards_r_sum)):
            for k in rewards_r_sum[i].keys():
                rewards_r_sum[i][k] = rewards_r_sum[i][k] + rewards_r[i][k]

        # ppo_r.step(value_obs=obs_r, rews=reward_r, dones=dones)
        dagger.step(total_obs=obs_r, rews=reward_r, dones=dones)
        #
        # for i in range(num_envs):
        #     rewards_r[i]['recon_loss'] = prop_mse_loss
        #     rewards_r[i]['action_loss'] = action_mse_loss


    obs_r, _ = env.observe_vision(contain_non_aff, allegro=True)
    # # if swing_diverse:
    # #     obs_r = np.concatenate([obs_r, swing_random], axis=1)
    # obs_r = obs_r[:, :].astype('float32')
    #
    # if np.isnan(obs_r).any():
    #     print('nan in obs')
    #     print(obs_r)
    #
    # # update policy
    # ppo_r.update(actor_obs=obs_r, value_obs=obs_r, log_this_iteration=update % 10 == 0, update=update)
    #
    # actor_r.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))
    #
    # end = time.time()
    value_obs = obs_r[:, -ob_dim_r:]
    prop_mse_loss, action_mse_loss = dagger.update(value_obs)

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
    wandb.log(ave_reward)


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