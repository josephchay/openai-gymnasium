import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import random
from collections import deque
import gymnasium as gym

class ReSine(nn.Module):
    def __init__(self):
        super(ReSine, self).__init__()
    
    def forward(self, x):
        return F.leaky_relu(torch.sin(x), 0.1)

class FourierSeries(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FourierSeries, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = ReSine()
        
    def forward(self, x):
        return self.activation(self.linear(x))

class FeedForwardTransformer(nn.Module):
    """
    Environment model for predicting future states based on state-action sequences
    """
    def __init__(self, state_dim, action_dim, hidden_dim=512, num_layers=6, num_heads=8):
        super(FeedForwardTransformer, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim + 1  # +1 for reward
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(100, hidden_dim))  # Max sequence length 100
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, sequence):
        """
        Args:
            sequence: [batch_size, seq_len, state_dim + action_dim + reward_dim]
        Returns:
            next_state: [batch_size, state_dim]
            next_reward: [batch_size, 1]
            next_done: [batch_size, 1]
        """
        batch_size, seq_len, _ = sequence.shape
        
        # Embed input
        embedded = self.input_embedding(sequence)
        
        # Add positional encoding
        embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transform
        transformed = self.transformer(embedded)
        
        # Use last timestep for prediction
        last_hidden = transformed[:, -1, :]
        
        # Predict next state, reward, done
        next_state = self.state_head(last_hidden)
        next_reward = self.reward_head(last_hidden)
        next_done = torch.sigmoid(self.done_head(last_hidden))
        
        return next_state, next_reward, next_done

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fourier1 = FourierSeries(state_dim, hidden_dim)
        self.fourier2 = FourierSeries(hidden_dim, hidden_dim)
        self.fourier3 = FourierSeries(hidden_dim, hidden_dim)
        
        self.output = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = self.fourier1(state)
        x = self.fourier2(x)
        x = self.fourier3(x)
        action = torch.tanh(self.output(x)) * self.max_action
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        input_dim = state_dim + action_dim
        
        # Three Q networks for variance estimation
        self.q1_fourier1 = FourierSeries(input_dim, hidden_dim)
        self.q1_fourier2 = FourierSeries(hidden_dim, hidden_dim)
        self.q1_output = nn.Linear(hidden_dim, 1)
        
        self.q2_fourier1 = FourierSeries(input_dim, hidden_dim)
        self.q2_fourier2 = FourierSeries(hidden_dim, hidden_dim)
        self.q2_output = nn.Linear(hidden_dim, 1)
        
        self.q3_fourier1 = FourierSeries(input_dim, hidden_dim)
        self.q3_fourier2 = FourierSeries(hidden_dim, hidden_dim)
        self.q3_output = nn.Linear(hidden_dim, 1)
        
        # Variance estimation
        self.var_fourier1 = FourierSeries(input_dim, hidden_dim)
        self.var_fourier2 = FourierSeries(hidden_dim, hidden_dim)
        self.var_output = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        
        # Q1
        q1 = self.q1_fourier1(sa)
        q1 = self.q1_fourier2(q1)
        q1 = self.q1_output(q1)
        
        # Q2
        q2 = self.q2_fourier1(sa)
        q2 = self.q2_fourier2(q2)
        q2 = self.q2_output(q2)
        
        # Q3
        q3 = self.q3_fourier1(sa)
        q3 = self.q3_fourier2(q3)
        q3 = self.q3_output(q3)
        
        # Variance
        var = self.var_fourier1(sa)
        var = self.var_fourier2(var)
        var = F.softplus(self.var_output(var))
        
        return q1, q2, q3, var

class ReHE(nn.Module):
    """Rectified Huber Error Loss"""
    def __init__(self):
        super(ReHE, self).__init__()
    
    def forward(self, error):
        a = torch.abs(error).mean()
        return (a * error * torch.tanh(a * error)).mean()

class ReHaE(nn.Module):
    """Rectified Huber Asymmetric Error Loss"""
    def __init__(self):
        super(ReHaE, self).__init__()
    
    def forward(self, error):
        e = error.mean()
        return (torch.abs(e) * torch.tanh(e)).mean()

class DreamReplayBuffer:
    def __init__(self, capacity=100000, fade_factor=7.0, stall_penalty=0.07):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.fade_factor = fade_factor
        self.stall_penalty = stall_penalty
        self.episode_cache = []  # Store last episode for dreaming
        
    def add(self, state, action, reward, next_state, done):
        # Calculate stall penalty
        if len(self.buffer) > 0:
            last_state = self.buffer[-1][0]
            delta = np.linalg.norm(np.array(next_state) - np.array(last_state)) + 1e-8
            reward -= self.stall_penalty * math.log10(1.0/delta)
        
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.episode_cache.append(experience)
        
        if done:
            # Episode finished, reset cache for next episode
            self.episode_cache = []
    
    def sample(self, batch_size):
        # Implement fading memory sampling
        indices = np.arange(len(self.buffer))
        norm_indices = indices / (len(self.buffer) - 1) if len(self.buffer) > 1 else [0]
        weights = np.tanh(self.fade_factor * norm_indices) ** 2
        weights = weights / weights.sum()
        
        sampled_indices = np.random.choice(indices, batch_size, p=weights, replace=True)
        batch = [self.buffer[i] for i in sampled_indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # OPTIMIZED: Convert lists to numpy arrays first, then to tensors for performance
        return (
            torch.from_numpy(np.array(states, dtype=np.float32)),
            torch.from_numpy(np.array(actions, dtype=np.float32)),
            torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(-1),
            torch.from_numpy(np.array(next_states, dtype=np.float32)),
            torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(-1)
        )
    
    def get_last_episode_steps(self, num_steps=7):
        """Get first num_steps from last episode for dreaming"""
        if len(self.episode_cache) < num_steps:
            return None
        return self.episode_cache[:num_steps]
    
    def __len__(self):
        return len(self.buffer)

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self, scale=1.0):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * scale

# Environment-specific configurations
BASE_CONFIG = {
    'gamma': 0.99,
    'tau': 0.005,
    'policy_freq': 2,
    'hidden_dim': 256,
    'transformer_layers': 6,
    'transformer_heads': 8,
    'transformer_hidden': 512
}

ENV_CONFIGS = {
    'BipedalWalker-v3': {
        'actor_lr': 1e-4,
        'critic_lr': 3e-4,
        'env_model_lr': 5e-5,
        'dream_frequency': 5,
        'dream_horizon': 40,
        'noise_scale': 0.1,
        'buffer_capacity': 100000,
        'dream_rollouts': 1000
    },
    'LunarLanderContinuous-v3': {
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'env_model_lr': 1e-4,
        'dream_frequency': 3,
        'dream_horizon': 30,
        'noise_scale': 0.15,
        'buffer_capacity': 50000,
        'dream_rollouts': 500
    },
    'HalfCheetah-v3': {
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'env_model_lr': 1e-4,
        'dream_frequency': 3,
        'dream_horizon': 30,
        'noise_scale': 0.2,
        'buffer_capacity': 200000,
        'dream_rollouts': 1000
    },
    'Humanoid-v3': {
        'actor_lr': 1e-4,
        'critic_lr': 3e-4,
        'env_model_lr': 3e-5,
        'dream_frequency': 10,
        'dream_horizon': 50,
        'noise_scale': 0.05,
        'buffer_capacity': 500000,
        'dream_rollouts': 1500
    },
    'Ant-v3': {
        'actor_lr': 2e-4,
        'critic_lr': 5e-4,
        'env_model_lr': 8e-5,
        'dream_frequency': 4,
        'dream_horizon': 35,
        'noise_scale': 0.12,
        'buffer_capacity': 150000,
        'dream_rollouts': 800
    }
}

# Exploration configurations
EXPLORATION_CONFIGS = {
    'BipedalWalker-v3': {
        'exploration_episodes': 150,
        'pure_random_ratio': 0.3,
        'exploration_noise_scale': 0.4,
        'env_model_pretrain_steps': 500
    },
    'LunarLanderContinuous-v3': {
        'exploration_episodes': 100,
        'pure_random_ratio': 0.4,
        'exploration_noise_scale': 0.5,
        'env_model_pretrain_steps': 300
    },
    'HalfCheetah-v3': {
        'exploration_episodes': 200,
        'pure_random_ratio': 0.2,
        'exploration_noise_scale': 0.3,
        'env_model_pretrain_steps': 600
    },
    'Humanoid-v3': {
        'exploration_episodes': 300,
        'pure_random_ratio': 0.1,
        'exploration_noise_scale': 0.2,
        'env_model_pretrain_steps': 800
    },
    'Ant-v3': {
        'exploration_episodes': 180,
        'pure_random_ratio': 0.25,
        'exploration_noise_scale': 0.35,
        'env_model_pretrain_steps': 450
    }
}

class Config:
    def __init__(self, env_name):
        self.base_config = BASE_CONFIG.copy()
        self.env_specific = ENV_CONFIGS.get(env_name, ENV_CONFIGS['LunarLanderContinuous-v3'])
        self.exploration_config = EXPLORATION_CONFIGS.get(env_name, EXPLORATION_CONFIGS['LunarLanderContinuous-v3'])
        
        # Merge configurations
        self.config = {**self.base_config, **self.env_specific, **self.exploration_config}
        
    def get_config(self):
        return self.config

class ExplorationStrategy:
    def __init__(self, env_name, agent, env):
        self.config = EXPLORATION_CONFIGS.get(env_name, EXPLORATION_CONFIGS['LunarLanderContinuous-v3'])
        self.agent = agent
        self.env = env
        
    def get_exploration_action(self, state, episode, max_exploration_episodes):
        exploration_progress = episode / max_exploration_episodes
        
        if exploration_progress < self.config['pure_random_ratio']:
            # Pure random exploration
            return self.env.action_space.sample()
        else:
            # Noisy policy exploration
            action = self.agent.select_action(state, add_noise=True)
            # Add extra exploration noise
            noise = self.agent.noise.sample() * self.config['exploration_noise_scale']
            return np.clip(action + noise, -self.agent.max_action, self.agent.max_action)

class Agent:
    def __init__(self, state_dim, action_dim, max_action, env_name='LunarLanderContinuous-v3', device='cuda'):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.env_name = env_name
        
        # Load configuration for environment
        self.config = Config(env_name).get_config()
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action, 
                                   self.config['hidden_dim']).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, 
                                          self.config['hidden_dim']).to(device)
        self.critic = Critic(state_dim, action_dim, 
                                     self.config['hidden_dim']).to(device)
        self.critic_target = Critic(state_dim, action_dim, 
                                            self.config['hidden_dim']).to(device)
        
        # Environment model (FeedForward Transformer)
        self.env_model = FeedForwardTransformer(
            state_dim, action_dim, 
            self.config['transformer_hidden'],
            self.config['transformer_layers'],
            self.config['transformer_heads']
        ).to(device)
        
        # Copy target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config['critic_lr'])
        self.env_model_optimizer = optim.Adam(self.env_model.parameters(), lr=self.config['env_model_lr'])
        
        # Loss functions
        self.rehe_loss = ReHE()
        self.rehae_loss = ReHaE()
        
        # Replay buffer
        self.replay_buffer = DreamReplayBuffer(capacity=self.config['buffer_capacity'])
        
        # Noise
        self.noise = OUNoise(action_dim)
        
        # Training parameters
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']
        self.policy_freq = self.config['policy_freq']
        self.total_it = 0
        
        # Dreaming parameters
        self.dream_rollouts = self.config['dream_rollouts']
        self.dream_horizon = self.config['dream_horizon']
        self.context_length = 7
        
        # Exploration tracking
        self.exploration_phase = True
        self.current_noise_scale = self.config['noise_scale']
        
    def select_action(self, state, add_noise=True):
        state = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = self.noise.sample() * self.current_noise_scale
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def update_noise_scale(self, episode, total_episodes, exploration_episodes):
        """Gradually reduce noise as training progresses"""
        if episode >= exploration_episodes:
            # After exploration phase, gradually reduce noise
            training_progress = (episode - exploration_episodes) / (total_episodes - exploration_episodes)
            self.current_noise_scale = self.config['noise_scale'] * max(0.1, 1.0 - training_progress * 0.8)
        else:
            # During exploration, keep higher noise
            self.current_noise_scale = self.config['noise_scale'] * 2.0
    
    def train_env_model(self, batch_size=256):
        """Train the environment model on replay buffer data"""
        if len(self.replay_buffer) < batch_size * 10:
            return 0
        
        # Sample sequences of length context_length + 1
        sequences = []
        targets = []
        
        for _ in range(batch_size):
            # Sample a random starting point with enough sequence length
            max_start = len(self.replay_buffer) - self.context_length - 1
            if max_start <= 0:
                continue
                
            start_idx = random.randint(0, max_start)
            
            sequence = []
            for i in range(self.context_length):
                state, action, reward, next_state, done = self.replay_buffer.buffer[start_idx + i]
                sequence.append(np.concatenate([state, action, [reward]]))
            
            # Target is the next transition
            target_state, target_action, target_reward, target_next_state, target_done = \
                self.replay_buffer.buffer[start_idx + self.context_length]
            
            sequences.append(sequence)
            targets.append((target_next_state, target_reward, float(target_done)))
        
        if len(sequences) == 0:
            return 0
        
        # OPTIMIZED: Convert to numpy arrays first for performance
        sequences = torch.from_numpy(np.array(sequences, dtype=np.float32)).to(self.device)
        target_states = torch.from_numpy(np.array([t[0] for t in targets], dtype=np.float32)).to(self.device)
        target_rewards = torch.from_numpy(np.array([t[1] for t in targets], dtype=np.float32)).unsqueeze(-1).to(self.device)
        target_dones = torch.from_numpy(np.array([t[2] for t in targets], dtype=np.float32)).unsqueeze(-1).to(self.device)
        
        # Forward pass
        pred_states, pred_rewards, pred_dones = self.env_model(sequences)
        
        # Compute losses
        state_loss = F.mse_loss(pred_states, target_states)
        reward_loss = F.mse_loss(pred_rewards, target_rewards)
        done_loss = F.binary_cross_entropy(pred_dones, target_dones)
        
        total_loss = state_loss + reward_loss + done_loss
        
        # Backward pass
        self.env_model_optimizer.zero_grad()
        total_loss.backward()
        self.env_model_optimizer.step()
        
        return total_loss.item()
    
    def generate_dreams(self):
        """Generate synthetic rollouts using the environment model"""
        context_steps = self.replay_buffer.get_last_episode_steps(self.context_length)
        if context_steps is None:
            return []
        
        dreams = []
        
        for _ in range(self.dream_rollouts):
            dream_episode = []
            
            # Initialize with context steps
            current_sequence = []
            for step in context_steps:
                state, action, reward, next_state, done = step
                current_sequence.append(np.concatenate([state, action, [reward]]))
                dream_episode.append(step)
            
            current_state = context_steps[-1][3]  # Last next_state becomes current state
            
            # Generate dream horizon steps
            for _ in range(self.dream_horizon):
                # Select action using current policy
                action = self.select_action(current_state, add_noise=False)
                
                # Predict next state using environment model
                sequence_tensor = torch.from_numpy(np.array([current_sequence], dtype=np.float32)).to(self.device)
                with torch.no_grad():
                    pred_next_state, pred_reward, pred_done = self.env_model(sequence_tensor)
                
                pred_next_state = pred_next_state.cpu().numpy().flatten()
                pred_reward = pred_reward.cpu().numpy().item()
                pred_done = pred_done.cpu().numpy().item() > 0.5
                
                # Add to dream episode
                dream_step = (current_state, action, pred_reward, pred_next_state, pred_done)
                dream_episode.append(dream_step)
                
                # Update sequence for next prediction
                current_sequence.append(np.concatenate([current_state, action, [pred_reward]]))
                if len(current_sequence) > self.context_length:
                    current_sequence.pop(0)
                
                current_state = pred_next_state
                
                if pred_done:
                    break
            
            dreams.append(dream_episode)
        
        return dreams
    
    def train_on_dreams(self, dreams):
        """Train actor-critic on dreamed experiences"""
        if not dreams:
            return 0
        
        # Flatten all dream experiences
        all_experiences = []
        for dream in dreams:
            all_experiences.extend(dream)
        
        if len(all_experiences) < 256:
            return 0
        
        # Sample batch from dream experiences
        batch_size = min(512, len(all_experiences))
        sampled_experiences = random.sample(all_experiences, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*sampled_experiences)
        
        # OPTIMIZED: Convert to numpy arrays first for performance
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(-1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(-1).to(self.device)
        
        # Train critic on dreamed data
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q1_next, q2_next, q3_next, var_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(torch.min(q1_next, q2_next), q3_next)
            target_q = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2, q3, var = self.critic(states, actions)
        
        # Compute critic losses
        q1_error = q1 - target_q
        q2_error = q2 - target_q
        q3_error = q3 - target_q
        
        critic_loss = self.rehe_loss(q1_error) + self.rehe_loss(q2_error) + self.rehe_loss(q3_error)
        
        # Add variance regularization
        q_mean = (q1 + q2 + q3) / 3
        q_var_actual = ((q1 - q_mean)**2 + (q2 - q_mean)**2 + (q3 - q_mean)**2) / 3
        var_loss = F.mse_loss(var, q_var_actual.detach())
        
        total_critic_loss = critic_loss + 0.1 * var_loss
        
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()
        
        # Train actor on dreamed data (less frequently)
        actor_loss = 0
        if self.total_it % self.policy_freq == 0:
            actor_actions = self.actor(states)
            q1_actor, q2_actor, q3_actor, _ = self.critic(states, actor_actions)
            q_actor = torch.min(torch.min(q1_actor, q2_actor), q3_actor)
            actor_loss = -q_actor.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss = actor_loss.item()
        
        return total_critic_loss.item()
    
    def train(self, batch_size=256):
        """Standard training on replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return 0, 0
        
        self.total_it += 1
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Train critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q1_next, q2_next, q3_next, var_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(torch.min(q1_next, q2_next), q3_next)
            target_q = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2, q3, var = self.critic(states, actions)
        
        q1_error = q1 - target_q
        q2_error = q2 - target_q
        q3_error = q3 - target_q
        
        critic_loss = self.rehe_loss(q1_error) + self.rehe_loss(q2_error) + self.rehe_loss(q3_error)
        
        # Variance regularization
        q_mean = (q1 + q2 + q3) / 3
        q_var_actual = ((q1 - q_mean)**2 + (q2 - q_mean)**2 + (q3 - q_mean)**2) / 3
        var_loss = F.mse_loss(var, q_var_actual.detach())
        
        total_critic_loss = critic_loss + 0.1 * var_loss
        
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()
        
        # Train actor
        actor_loss = 0
        if self.total_it % self.policy_freq == 0:
            actor_actions = self.actor(states)
            q1_actor, q2_actor, q3_actor, _ = self.critic(states, actor_actions)
            q_actor = torch.min(torch.min(q1_actor, q2_actor), q3_actor)
            actor_loss = -q_actor.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            actor_loss = actor_loss.item()
        
        return total_critic_loss.item(), actor_loss
    
    def save(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'env_model': self.env_model.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'env_model_optimizer': self.env_model_optimizer.state_dict(),
            'total_it': self.total_it,
            'config': self.config,
            'env_name': self.env_name
        }
        torch.save(checkpoint, filename)
        
    def load(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.env_model.load_state_dict(checkpoint['env_model'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.env_model_optimizer.load_state_dict(checkpoint['env_model_optimizer'])
        self.total_it = checkpoint['total_it']
        
        # Update target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
