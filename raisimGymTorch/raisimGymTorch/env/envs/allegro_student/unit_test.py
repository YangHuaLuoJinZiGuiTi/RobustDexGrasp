import safety_gymnasium
import numpy as np
import torch
import torch.nn as nn
from raisimGymTorch.algo.p3o.module import Actor, Critic, MLP, MultivariateGaussianDiagonalCovariance
from raisimGymTorch.algo.p3o.storage import RolloutStorage
from raisimGymTorch.algo.p3o.p3o import P3O
import time
from collections import deque
import torch.multiprocessing as mp  # 支持共享内存
from multiprocessing import Process, Queue

save_path = '/home/hang/raisim/RobustDexGrasp/raisimGymTorch/data_all/unit_test'


class FastSampler:
    def __init__(self):
        pass
    
    def sample(self, logits, std, samples, logprob):
        # Simple sampling without using external C++ code
        noise = np.random.randn(*samples.shape)
        samples[:] = logits + noise * std
        logprob[:] = -0.5 * np.sum(noise ** 2 + np.log(2 * np.pi * std ** 2), axis=-1)
    
    def seed(self, seed):
        np.random.seed(seed)

def create_networks(env, device='cpu'):
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Create actor network
    actor_mlp = MLP(
        shape=[64, 64],
        actionvation_fn=nn.Tanh,
        input_size=obs_dim,
        output_size=act_dim
    )
    
    # Create distribution for actor
    fast_sampler = FastSampler()
    distribution = MultivariateGaussianDiagonalCovariance(
        dim=act_dim,
        size=1,  # For single environment
        init_std=torch.tensor(0.5),
        fast_sampler=fast_sampler,
        seed=0
    )
    
    actor = Actor(actor_mlp, distribution, device)
    
    # Create reward critic network
    reward_critic_mlp = MLP(
        shape=[64, 64],
        actionvation_fn=nn.Tanh,
        input_size=obs_dim,
        output_size=1
    )
    
    reward_critic = Critic(reward_critic_mlp, device)
    
    # Create cost critic network
    cost_critic_mlp = MLP(
        shape=[64, 64],
        actionvation_fn=nn.Tanh,
        input_size=obs_dim,
        output_size=1
    )
    
    cost_critic = Critic(cost_critic_mlp, device)
    
    return actor, reward_critic, cost_critic

def train_p3o():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    
    # Create networks
    actor, reward_critic, cost_critic = create_networks(env, device)
    
    # Create P3O agent
    agent = P3O(
        actor=actor,
        critic_reward=reward_critic,
        critic_cost=cost_critic,
        num_envs=1,
        num_transitions_per_env=2048,  # Number of steps per rollout
        num_learning_epochs=10,
        num_mini_batches=32,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        learning_rate=3e-4,
        max_grad_norm=0.5,
        learning_rate_schedule='adaptive',
        desired_kl=0.01,
        use_clipped_value_loss=True,
        log_dir=save_path,
        device=device,
        shuffle_batch=True,
        kappa=20,  # Penalty factor
        cost_limits=[25.0],  # Cost limit for the environment
        cost_discount_factor=0.99,
        cost_lam=0.95,
        advantage_normalization=True
    )
    
    
    # Training parameters
    num_updates = 1000
    save_interval = 50
    
    # Metrics tracking
    episode_rewards = deque(maxlen=100)
    episode_costs = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    # Reset environment
    obs, _ = env.reset()
    actor_obs = obs
    critic_obs = obs
    
    
    total_episode_reward = 0
    total_episode_cost = 0
    episode_length = 0
    

    # Training loop
    for update in range(num_updates):
        # Rollout phase
        for bla in range(agent.num_transitions_per_env):
            # Get action
            action = agent.act(actor_obs)
            
            # Step environment
            # next_obs, reward, terminated, truncated, info = env.step(action[0])
            next_obs, reward, cost, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            
            total_episode_reward += reward
            total_episode_cost += cost
            episode_length += 1
            
            
            # Store transition
            agent.storage.add_transitions(
                actor_obs=actor_obs,
                critic_obs=critic_obs,
                actions=action,
                mu=agent.actor.action_mean,
                sigma=agent.actor.distribution.std_np,
                rewards=np.array([reward]),
                costs=np.array([cost]),
                dones=np.array([done]),
                actions_log_prob=agent.actions_log_prob
            )

            # Update observations
            actor_obs = next_obs
            critic_obs = next_obs
            
            # Reset if done
            if done:
                episode_rewards.append(total_episode_reward)
                episode_costs.append(total_episode_cost)
                episode_lengths.append(episode_length)
                
                total_episode_reward = 0
                total_episode_cost = 0
                episode_length = 0
                
                obs, _ = env.reset()
                actor_obs = obs
                critic_obs = obs
                
        # Update agent
        agent.update(actor_obs, critic_obs, update % 10 == 0, update)
        
        # Log metrics
        if update % 10 == 0:
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0
            mean_cost = np.mean(episode_costs) if episode_costs else 0
            mean_length = np.mean(episode_lengths) if episode_lengths else 0
            
            print(f"Update: {update}, "
                  f"Mean Reward: {mean_reward:.2f}, "
                  f"Mean Cost: {mean_cost:.2f}, "
                  f"Mean Length: {mean_length:.2f}")
            
            # Reset metrics
            episode_rewards.clear()
            episode_costs.clear()
            episode_lengths.clear()
        
        # Save model periodically
        if update % save_interval == 0:
            torch.save({
                'actor': agent.actor.architecture.state_dict(),
                'reward_critic': agent.critic_reward.architecture.state_dict(),
                'cost_critic': agent.critic_cost.architecture.state_dict(),
                'optimizer': agent.optimizer.state_dict()
            }, f'{save_path}/p3o_checkpoint_{update}.pth')

    env.close()

def test_p3o(model_path):
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create environment
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    
    # Create networks
    actor, reward_critic, cost_critic = create_networks(env, device)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    actor.architecture.load_state_dict(checkpoint['actor'])
    
    # Test the policy
    obs, _ = env.reset()
    done = False
    total_reward = 0
    total_cost = 0
    steps = 0
    
    while not done and steps < 1000:
        # Get action from policy
        with torch.no_grad():
            action = actor.noiseless_action(obs).cpu().numpy()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Accumulate metrics
        total_reward += reward
        total_cost += info.get('cost', 0)
        steps += 1
    
    print(f"Test Results - Reward: {total_reward}, Cost: {total_cost}, Steps: {steps}")
    env.close()

if __name__ == "__main__":
    # Train the agent
    train_p3o()
    
    # Test the best model
    test_p3o('./models/p3o_checkpoint_950.pth')

