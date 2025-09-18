import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from .storage import RolloutStorage
import numpy as np

class P3O:
    def __init__(self,
                 actor,
                 critic_reward,
                 critic_cost,  # Single critic handling all cost constraints    
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 max_grad_norm=0.5,
                 learning_rate_schedule='adaptive',
                 desired_kl=0.01,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True,
                 kappa=20.0,  # Penalty factor from P3O paper
                 kappa_max = 100,
                 cost_limits=None,  # List of cost limits for each constraint
                 cost_discount_factor=0.99,  # Discount factor for cost returns
                 cost_value_loss_coef=0.5,  # Coefficient for cost value loss
                 rou=1.2,  # Factor to increase kappa
                 cost_lam=0.95,  # GAE parameter for cost advantages
                 advantage_normalization=True):  # Whether to normalize advantages

        # P3O components
        self.actor = actor
        self.critic_reward = critic_reward
        self.critic_cost = critic_cost  # 单个critic处理所有cost约束
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, 
                                     actor.obs_shape, critic_reward.obs_shape, actor.action_shape, device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        # Separate optimizers for actor and critics
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        # self.critic_reward_optimizer = optim.Adam(self.critic_reward.parameters(), lr=learning_rate)
        # self.critic_cost_optimizer = optim.Adam(self.critic_cost.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic_reward.parameters(), *self.critic_cost.parameters()], lr=learning_rate)
        
        self.device = device

        # Environment parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # P3O parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        # P3O-specific parameters
        self.kappa = kappa
        self.kappa_max = kappa_max
        self.rou = rou
        self.cost_limits = cost_limits if cost_limits is not None else []
        self.cost_discount_factor = cost_discount_factor
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_lam = cost_lam
        self.advantage_normalization = advantage_normalization
        self.num_constraints = len(self.cost_limits)
        
        # Logging
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # Learning rate scheduling
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = learning_rate_schedule

        # Temporary variables
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

        # Tracking
        self.is_exploding_gradient = False
        self.episode_costs = []  # Track costs for adaptive penalty

    def act(self, actor_obs):
        self.actor_obs = actor_obs
        with torch.no_grad():
            self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).float().to(self.device))
        return self.actions

    def step(self, value_obs, rews, costs, dones):
        # Store transitions with both rewards and costs
        self.storage.add_transitions(
            self.actor_obs, value_obs, self.actions, 
            self.actor.action_mean, self.actor.distribution.std_np, 
            rews, costs, dones, self.actions_log_prob
        )

    def update(self, actor_obs, value_obs, log_this_iteration, update):
        # Compute returns and advantages for both reward and cost
        last_reward_values = self.critic_reward.predict(torch.from_numpy(value_obs).float().to(self.device))
        last_cost_values = self.critic_cost.predict(torch.from_numpy(value_obs).float().to(self.device))

        # Compute returns and advantages
        self.storage.compute_returns(
            last_reward_values, last_cost_values, 
            self.critic_reward, self.critic_cost,
            self.gamma, self.lam, self.cost_discount_factor, self.cost_lam
        )
        
        # Normalize advantages if enabled
        if self.advantage_normalization:
            self.storage.normalize_advantages()
        
        # Train step
        mean_value_loss, mean_surrogate_loss, mean_cost_loss, infos = self._train_step(log_this_iteration)
        self.storage.clear()

        if infos is None:
            self.is_exploding_gradient = True
            return

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()
        
        self.writer.add_scalar('P3O/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('P3O/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('P3O/cost_loss', variables['mean_cost_loss'], variables['it'])
        self.writer.add_scalar('P3O/mean_noise_std', mean_std.item(), variables['it'])
        self.writer.add_scalar('P3O/learning_rate', self.learning_rate, variables['it'])
        
        # Log constraint satisfaction
        for i, cost_limit in enumerate(self.cost_limits):
            self.writer.add_scalar(f'P3O/cost_{i}_limit', cost_limit, variables['it'])

    def _train_step(self, log_this_iteration):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_cost_loss = 0
        
        for epoch in range(self.num_learning_epochs):
            for batch in self.batch_sampler(self.num_mini_batches):
                # Unpack batch
                (actor_obs_batch, critic_obs_batch, actions_batch, old_sigma_batch, 
                 old_mu_batch, current_reward_values_batch, current_cost_values_batch,
                 reward_advantages_batch, cost_advantages_batch, 
                 reward_returns_batch, cost_returns_batch, old_actions_log_prob_batch) = batch

                # Evaluate current policy
                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                reward_value_batch = self.critic_reward.evaluate(critic_obs_batch)
                cost_value_batch = self.critic_cost.evaluate(critic_obs_batch)

                # Adjust learning rate using KL divergence (adaptive)
                if self.desired_kl is not None and self.schedule == 'adaptive':
                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(self.actor.distribution.std / old_sigma_batch + 1.e-5) + 
                            (torch.square(old_sigma_batch) + torch.square(old_mu_batch - self.actor.action_mean)) / 
                            (2.0 * torch.square(self.actor.distribution.std)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.2)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.2)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Reward surrogate loss (same as PPO)
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                reward_surrogate = -torch.squeeze(reward_advantages_batch) * ratio
                reward_surrogate_clipped = -torch.squeeze(reward_advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                reward_surrogate_loss = torch.max(reward_surrogate, reward_surrogate_clipped).mean()
                
                # Cost surrogate loss for all constraints
                total_cost_surrogate_loss = 0
                for i in range(self.num_constraints):
                    # Get the cost advantage and return for this constraint
                    cost_advantage_i = cost_advantages_batch[:, i] if cost_advantages_batch.shape[1] > 1 else cost_advantages_batch.squeeze()
                    cost_return_i = cost_returns_batch[:, i] if cost_returns_batch.shape[1] > 1 else cost_returns_batch.squeeze()
                    
                    # Calculate surrogate loss for this constraint
                    cost_surrogate_i = cost_advantage_i * ratio
                    cost_surrogate_clipped_i = cost_advantage_i * torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    
                    # Use max operator as in P3O paper (ReLU equivalent)
                    cost_surrogate_loss_i = torch.max(cost_surrogate_i, cost_surrogate_clipped_i)
                    
                    # Add the constraint offset term (1-γ)(J_C(π_k) - d_i)
                    # We approximate J_C(π_k) with the mean of cost returns for this constraint
                    cost_offset_i = (1 - self.cost_discount_factor) * (
                        torch.mean(cost_return_i) - self.cost_limits[i])
                    
                    cost_surrogate_loss_i = torch.mean(cost_surrogate_loss_i) + cost_offset_i
                    
                    # Apply ReLU operator to cost loss (exact penalty method)
                    cost_surrogate_loss_i = torch.relu(cost_surrogate_loss_i)
                    
                    # Add to total cost loss
                    total_cost_surrogate_loss += cost_surrogate_loss_i

                # If no constraints, set to zero
                if self.num_constraints == 0:
                    total_cost_surrogate_loss = torch.tensor(0.0, device=self.device)

                # Value function losses
                if self.use_clipped_value_loss:
                    # Reward value loss
                    reward_value_clipped = current_reward_values_batch + (
                        reward_value_batch - current_reward_values_batch).clamp(-self.clip_param, self.clip_param)
                    reward_value_losses = (reward_value_batch - reward_returns_batch).pow(2)
                    reward_value_losses_clipped = (reward_value_clipped - reward_returns_batch).pow(2)
                    reward_value_loss = torch.max(reward_value_losses, reward_value_losses_clipped).mean()
                    
                    # Cost value loss for each constraint
                    cost_value_loss = 0
                    for i in range(self.num_constraints):
                        cost_value_i = cost_value_batch[:, i] if cost_value_batch.shape[1] > 1 else cost_value_batch
                        cost_return_i = cost_returns_batch[:, i] if cost_returns_batch.shape[1] > 1 else cost_returns_batch
                        current_cost_value_i = current_cost_values_batch[:, i] if current_cost_values_batch.shape[1] > 1 else current_cost_values_batch
                        
                        cost_value_clipped_i = current_cost_value_i + (
                            cost_value_i - current_cost_value_i).clamp(-self.clip_param, self.clip_param)
                        cost_value_losses_i = (cost_value_i - cost_return_i).pow(2)
                        cost_value_losses_clipped_i = (cost_value_clipped_i - cost_return_i).pow(2)
                        cost_value_loss += torch.max(cost_value_losses_i, cost_value_losses_clipped_i).mean()
                    
                    # Average cost value loss across constraints
                    if self.num_constraints > 0:
                        cost_value_loss = cost_value_loss / self.num_constraints
                else:
                    reward_value_loss = (reward_returns_batch - reward_value_batch).pow(2).mean()
                    
                    # Cost value loss for each constraint
                    cost_value_loss = 0
                    for i in range(self.num_constraints):
                        cost_value_i = cost_value_batch[:, i] if cost_value_batch.shape[1] > 1 else cost_value_batch
                        cost_return_i = cost_returns_batch[:, i] if cost_returns_batch.shape[1] > 1 else cost_returns_batch
                        cost_value_loss += (cost_return_i - cost_value_i).pow(2).mean()
                    
                    # Average cost value loss across constraints
                    if self.num_constraints > 0:
                        cost_value_loss = cost_value_loss / self.num_constraints

                # Total actor loss (P3O objective)
                actor_loss = (reward_surrogate_loss + 
                             self.kappa * total_cost_surrogate_loss - 
                             self.entropy_coef * entropy_batch.mean())

                total_loss = actor_loss + self.value_loss_coef * reward_value_loss + self.cost_value_loss_coef * cost_value_loss
                # Update kappa if not normalize the cost 
                if not self.advantage_normalization:
                    self.kappa = np.min(self.kappa * self.rou, self.kappa_max)

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic_reward.parameters(), *self.critic_cost.parameters()], self.max_grad_norm)
                self.optimizer.step()
                
                # Check for exploding gradients
                for name, parameters in self.actor.architecture.state_dict().items():
                    if "weight" in name:
                        w = parameters.detach()
                        if torch.isnan(w).any():
                            print("------------------------ find exploding gradient")
                            return None, None, None, None

                if log_this_iteration:
                    mean_value_loss += (reward_value_loss.item() + cost_value_loss.item()) / 2
                    mean_surrogate_loss += reward_surrogate_loss.item()
                    mean_cost_loss += total_cost_surrogate_loss.item()

        if log_this_iteration:
            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_cost_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, mean_cost_loss, locals()

    def check_exploding_gradient(self):
        return self.is_exploding_gradient