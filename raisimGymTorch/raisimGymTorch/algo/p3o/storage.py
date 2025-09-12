import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, device, num_constraints=1):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.num_constraints = num_constraints
        self.step = 0

        # Core
        self.critic_obs = np.zeros([num_transitions_per_env, num_envs, *critic_obs_shape], dtype=np.float32)
        self.actor_obs = np.zeros([num_transitions_per_env, num_envs, *actor_obs_shape], dtype=np.float32)
        self.rewards = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.costs = np.zeros([num_transitions_per_env, num_envs, num_constraints], dtype=np.float32)  # 修改为支持多个约束
        self.actions = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.dones = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.bool_)

        # For PPO/P3O
        self.actions_log_prob = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.reward_values = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.cost_values = np.zeros([num_transitions_per_env, num_envs, num_constraints], dtype=np.float32)  # 修改为支持多个约束
        self.reward_returns = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.cost_returns = np.zeros([num_transitions_per_env, num_envs, num_constraints], dtype=np.float32)  # 修改为支持多个约束
        self.reward_advantages = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.cost_advantages = np.zeros([num_transitions_per_env, num_envs, num_constraints], dtype=np.float32)  # 修改为支持多个约束
        self.mu = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.sigma = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)

        # Initialize torch tensors
        self._init_torch_tensors()

    def _init_torch_tensors(self):
        """Initialize torch tensors from numpy arrays"""
        self.critic_obs_tc = torch.from_numpy(self.critic_obs).to(self.device)
        self.actor_obs_tc = torch.from_numpy(self.actor_obs).to(self.device)
        self.actions_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.reward_values_tc = torch.from_numpy(self.reward_values).to(self.device)
        self.cost_values_tc = torch.from_numpy(self.cost_values).to(self.device)
        self.reward_returns_tc = torch.from_numpy(self.reward_returns).to(self.device)
        self.cost_returns_tc = torch.from_numpy(self.cost_returns).to(self.device)
        self.reward_advantages_tc = torch.from_numpy(self.reward_advantages).to(self.device)
        self.cost_advantages_tc = torch.from_numpy(self.cost_advantages).to(self.device)
        self.mu_tc = torch.from_numpy(self.mu).to(self.device)
        self.sigma_tc = torch.from_numpy(self.sigma).to(self.device)

    def add_transitions(self, actor_obs, critic_obs, actions, mu, sigma, rewards, costs, dones, actions_log_prob):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        
        self.actor_obs[self.step] = actor_obs
        self.critic_obs[self.step] = critic_obs
        self.actions[self.step] = actions
        self.mu[self.step] = mu
        self.sigma[self.step] = sigma
        self.rewards[self.step] = rewards.reshape(-1, 1)
        
        # 处理多个costs
        if len(costs.shape) == 1:
            # 单个cost，reshape到(num_envs, num_constraints)
            self.costs[self.step] = costs.reshape(-1, 1)
        else:
            # 多个costs，已经是正确形状
            self.costs[self.step] = costs
        
        self.dones[self.step] = dones.reshape(-1, 1)
        self.actions_log_prob[self.step] = actions_log_prob.reshape(-1, 1)
        
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_reward_values, last_cost_values, reward_critic, cost_critic, gamma, lam, cost_gamma, cost_lam):
        # Compute values for all states using the critics
        with torch.no_grad():
            # Flatten the observations for batch processing
            critic_obs_flat = self.critic_obs.reshape(-1, *self.critic_obs.shape[2:])
            critic_obs_tensor = torch.from_numpy(critic_obs_flat).to(self.device)
            
            # Get predictions from both critics
            reward_preds = reward_critic.predict(critic_obs_tensor).cpu().numpy()
            cost_preds = cost_critic.predict(critic_obs_tensor).cpu().numpy()
            
            # Reshape back to original shape
            self.reward_values = reward_preds.reshape(self.num_transitions_per_env, self.num_envs, 1)
            
            # 处理多个cost约束的情况
            if cost_preds.shape[1] == 1 and self.num_constraints > 1:
                # 如果critic只输出一个值但有多个约束，复制到所有约束
                self.cost_values = np.tile(
                    cost_preds.reshape(self.num_transitions_per_env, self.num_envs, 1), 
                    (1, 1, self.num_constraints)
                )
            else:
                # 正常情况，critic输出与约束数相同的值
                self.cost_values = cost_preds.reshape(
                    self.num_transitions_per_env, self.num_envs, self.num_constraints
                )

        # Compute reward advantages and returns
        reward_advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_reward_values = last_reward_values.cpu().numpy()
            else:
                next_reward_values = self.reward_values[step + 1]
            
            next_is_not_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_is_not_terminal * next_reward_values - self.reward_values[step]
            reward_advantage = delta + gamma * lam * next_is_not_terminal * reward_advantage
            self.reward_returns[step] = reward_advantage + self.reward_values[step]

        # 为每个cost约束单独计算advantages和returns
        for cost_idx in range(self.num_constraints):
            cost_advantage = 0
            for step in reversed(range(self.num_transitions_per_env)):
                if step == self.num_transitions_per_env - 1:
                    # 处理last_cost_values的形状
                    if last_cost_values.dim() == 1:
                        next_cost_values = last_cost_values[cost_idx].cpu().numpy()
                    else:
                        next_cost_values = last_cost_values[:, cost_idx].cpu().numpy()
                    next_cost_values = next_cost_values.reshape(-1, 1)
                else:
                    next_cost_values = self.cost_values[step + 1, :, cost_idx:cost_idx+1]
                
                next_is_not_terminal = 1.0 - self.dones[step]
                delta = (self.costs[step, :, cost_idx:cost_idx+1] + 
                        cost_gamma * next_is_not_terminal * next_cost_values - 
                        self.cost_values[step, :, cost_idx:cost_idx+1])
                cost_advantage = delta + cost_gamma * cost_lam * next_is_not_terminal * cost_advantage
                
                # 更新当前约束的returns
                if step == self.num_transitions_per_env - 1:
                    # 初始化returns数组
                    self.cost_returns = np.zeros_like(self.cost_values)
                
                self.cost_returns[step, :, cost_idx:cost_idx+1] = (
                    cost_advantage + self.cost_values[step, :, cost_idx:cost_idx+1]
                )
                
                # 更新当前约束的advantages
                if step == self.num_transitions_per_env - 1:
                    # 初始化advantages数组
                    self.cost_advantages = np.zeros_like(self.cost_values)
                
                self.cost_advantages[step, :, cost_idx:cost_idx+1] = cost_advantage

        # 计算reward advantages
        self.reward_advantages = self.reward_returns - self.reward_values

        # Update torch tensors
        self._init_torch_tensors()

    def normalize_advantages(self):
        # Normalize reward advantages
        reward_advantage_mean = np.mean(self.reward_advantages)
        reward_advantage_std = np.std(self.reward_advantages)
        self.reward_advantages = (self.reward_advantages - reward_advantage_mean) / (reward_advantage_std + 1e-8)
        
        # Normalize cost advantages for each constraint
        for cost_idx in range(self.num_constraints):
            cost_advantage_mean = np.mean(self.cost_advantages[:, :, cost_idx])
            cost_advantage_std = np.std(self.cost_advantages[:, :, cost_idx])
            self.cost_advantages[:, :, cost_idx] = (
                (self.cost_advantages[:, :, cost_idx] - cost_advantage_mean) / 
                (cost_advantage_std + 1e-8)
            )
        
        # Update torch tensors
        self.reward_advantages_tc = torch.from_numpy(self.reward_advantages).to(self.device)
        self.cost_advantages_tc = torch.from_numpy(self.cost_advantages).to(self.device)

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            actor_obs_batch = self.actor_obs_tc.view(-1, *self.actor_obs_tc.size()[2:])[indices]
            critic_obs_batch = self.critic_obs_tc.view(-1, *self.critic_obs_tc.size()[2:])[indices]
            actions_batch = self.actions_tc.view(-1, self.actions_tc.size(-1))[indices]
            sigma_batch = self.sigma_tc.view(-1, self.sigma_tc.size(-1))[indices]
            mu_batch = self.mu_tc.view(-1, self.mu_tc.size(-1))[indices]
            reward_values_batch = self.reward_values_tc.view(-1, 1)[indices]
            cost_values_batch = self.cost_values_tc.view(-1, self.num_constraints)[indices]  # 修改为支持多个约束
            reward_advantages_batch = self.reward_advantages_tc.view(-1, 1)[indices]
            cost_advantages_batch = self.cost_advantages_tc.view(-1, self.num_constraints)[indices]  # 修改为支持多个约束
            reward_returns_batch = self.reward_returns_tc.view(-1, 1)[indices]
            cost_returns_batch = self.cost_returns_tc.view(-1, self.num_constraints)[indices]  # 修改为支持多个约束
            old_actions_log_prob_batch = self.actions_log_prob_tc.view(-1, 1)[indices]
            
            yield (actor_obs_batch, critic_obs_batch, actions_batch, sigma_batch, mu_batch,
                   reward_values_batch, cost_values_batch,
                   reward_advantages_batch, cost_advantages_batch,
                   reward_returns_batch, cost_returns_batch,
                   old_actions_log_prob_batch)

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            start_idx = batch_id * mini_batch_size
            end_idx = (batch_id + 1) * mini_batch_size
            
            yield (self.actor_obs_tc.view(-1, *self.actor_obs_tc.size()[2:])[start_idx:end_idx],
                   self.critic_obs_tc.view(-1, *self.critic_obs_tc.size()[2:])[start_idx:end_idx],
                   self.actions_tc.view(-1, self.actions_tc.size(-1))[start_idx:end_idx],
                   self.sigma_tc.view(-1, self.sigma_tc.size(-1))[start_idx:end_idx],
                   self.mu_tc.view(-1, self.mu_tc.size(-1))[start_idx:end_idx],
                   self.reward_values_tc.view(-1, 1)[start_idx:end_idx],
                   self.cost_values_tc.view(-1, self.num_constraints)[start_idx:end_idx],  # 修改为支持多个约束
                   self.reward_advantages_tc.view(-1, 1)[start_idx:end_idx],
                   self.cost_advantages_tc.view(-1, self.num_constraints)[start_idx:end_idx],  # 修改为支持多个约束
                   self.reward_returns_tc.view(-1, 1)[start_idx:end_idx],
                   self.cost_returns_tc.view(-1, self.num_constraints)[start_idx:end_idx],  # 修改为支持多个约束
                   self.actions_log_prob_tc.view(-1, 1)[start_idx:end_idx])