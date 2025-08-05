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
    """Enhanced Environment model with uncertainty estimation and better accuracy"""
    def __init__(self, state_dim, action_dim, hidden_dim=512, num_layers=6, num_heads=8):
        super(FeedForwardTransformer, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim + 1  # +1 for reward
        self.hidden_dim = hidden_dim
        
        # Enhanced input embedding with better feature extraction
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Learnable positional encoding with better capacity
        self.pos_encoding = nn.Parameter(torch.randn(200, hidden_dim) * 0.02)  # Increased capacity
        
        # Enhanced transformer with proper masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, 
            batch_first=True,
            activation='gelu'  # Better activation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multiple prediction heads with uncertainty estimation
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, state_dim)
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Uncertainty estimation heads
        self.state_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, state_dim)
        )
        
        self.reward_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Model accuracy tracking
        self.register_buffer('prediction_errors', torch.zeros(1000))  # Rolling window of errors
        self.register_buffer('error_index', torch.tensor(0))
        self.accuracy_threshold = 0.1  # Minimum accuracy threshold for dreaming
        
    def forward(self, sequence, return_uncertainty=False):
        """
        Args:
            sequence: (batch_size, seq_len, state_dim + action_dim + reward_dim)
            return_uncertainty: Whether to return uncertainty estimates
        Returns:
            next_state, next_reward, next_done, [uncertainties if requested]
        """
        batch_size, seq_len, _ = sequence.shape
        
        # Enhanced input embedding
        embedded = self.input_embedding(sequence)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        embedded = embedded + pos_enc
        
        # Create causal mask to prevent information leakage
        mask = self.generate_square_subsequent_mask(seq_len).to(sequence.device)
        
        # Transform with attention
        transformed = self.transformer(embedded, mask=mask)
        
        # Use last timestep for prediction
        last_hidden = transformed[:, -1, :]
        
        # Predictions
        next_state = self.state_head(last_hidden)
        next_reward = self.reward_head(last_hidden)
        next_done = torch.sigmoid(self.done_head(last_hidden))
        
        if return_uncertainty:
            state_unc = F.softplus(self.state_uncertainty(last_hidden))
            reward_unc = F.softplus(self.reward_uncertainty(last_hidden))
            return next_state, next_reward, next_done, state_unc, reward_unc
        
        return next_state, next_reward, next_done
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for transformer"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def update_accuracy_tracking(self, pred_states, true_states, pred_rewards, true_rewards):
        """Update rolling accuracy metrics"""
        state_error = torch.mean((pred_states - true_states) ** 2).item()
        reward_error = torch.mean((pred_rewards - true_rewards) ** 2).item()
        total_error = state_error + reward_error
        
        # Update rolling window
        idx = self.error_index.item() % len(self.prediction_errors)
        self.prediction_errors[idx] = total_error
        self.error_index += 1
    
    def get_model_accuracy(self):
        """Get current model accuracy"""
        if self.error_index < len(self.prediction_errors):
            valid_errors = self.prediction_errors[:self.error_index]
        else:
            valid_errors = self.prediction_errors
        
        if len(valid_errors) == 0:
            return 0.0
        
        mean_error = torch.mean(valid_errors).item()
        # Convert error to accuracy score (0-1)
        accuracy = max(0.0, 1.0 - mean_error)
        return accuracy
    
    def is_accurate_enough_for_dreaming(self):
        """Check if model is accurate enough for reliable dreaming"""
        return self.get_model_accuracy() > self.accuracy_threshold

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
        
        # Variance estimation
        var = self.var_fourier1(sa)
        var = self.var_fourier2(var)
        var = F.softplus(self.var_output(var))
        
        return q1, q2, q3, var

# FIXED: Loss functions ensuring scalar outputs
class ReHE(nn.Module):
    """Rectified Huber Error Loss - Fixed to ensure scalar output"""
    def __init__(self):
        super(ReHE, self).__init__()
    
    def forward(self, error):
        # Ensure error is flattened and compute scalar loss
        error_flat = error.view(-1)
        a = torch.abs(error_flat).mean()
        huber_term = error_flat * torch.tanh(a - error_flat)
        return a + huber_term.mean()  # Guaranteed scalar

class ReHaE(nn.Module):
    """Rectified Huber Asymmetric Error Loss - Fixed to ensure scalar output"""
    def __init__(self):
        super(ReHaE, self).__init__()
    
    def forward(self, error):
        # Ensure error is flattened and compute scalar loss
        error_flat = error.view(-1)
        e = error_flat.mean()
        return torch.abs(e) + torch.tanh(e).mean()  # Guaranteed scalar

class DreamReplayBuffer:
    """Enhanced replay buffer with dream data management"""
    def __init__(self, capacity=100000, fade_factor=7.0, stall_penalty=0.07):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.fade_factor = fade_factor
        self.stall_penalty = stall_penalty
        self.episode_cache = []
        
        # Separate dream buffer to prevent contamination
        self.dream_buffer = deque(maxlen=capacity // 4)  # Smaller capacity for dreams
        self.dream_weight = 0.3  # Weight for dream data vs real data
        
    def add(self, state, action, reward, next_state, done):
        # Calculate stall penalty
        if len(self.buffer) > 0:
            last_state = self.buffer[-1][0]
            delta = np.linalg.norm(np.array(next_state) - np.array(last_state)) + 1e-8
            reward = reward - self.stall_penalty * math.log10(1.0/delta)
        
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.episode_cache.append(experience)
        
        if done:
            self.episode_cache = []
    
    def add_dream_experience(self, state, action, reward, next_state, done, uncertainty_weight=1.0):
        """Add dream experience with uncertainty weighting"""
        # Weight dream rewards by uncertainty
        weighted_reward = reward * uncertainty_weight
        experience = (state, action, weighted_reward, next_state, done)
        self.dream_buffer.append(experience)
    
    def sample(self, batch_size):
        # Sample from both real and dream buffers
        real_batch_size = int(batch_size * (1 - self.dream_weight))
        dream_batch_size = batch_size - real_batch_size
        
        # Sample from real buffer with fading memory
        real_experiences = []
        if len(self.buffer) > 0 and real_batch_size > 0:
            indices = np.arange(len(self.buffer))
            norm_indices = indices / (len(self.buffer) - 1) if len(self.buffer) > 1 else 0
            weights = np.tanh(self.fade_factor * norm_indices) ** 2
            weights = weights / weights.sum()
            
            sampled_indices = np.random.choice(indices, min(real_batch_size, len(self.buffer)), 
                                             p=weights, replace=True)
            real_experiences = [self.buffer[i] for i in sampled_indices]
        
        # Sample from dream buffer
        dream_experiences = []
        if len(self.dream_buffer) > 0 and dream_batch_size > 0:
            dream_experiences = random.sample(self.dream_buffer, 
                                            min(dream_batch_size, len(self.dream_buffer)))
        
        # Combine experiences
        all_experiences = real_experiences + dream_experiences
        
        if not all_experiences:
            return None, None, None, None, None
        
        states, actions, rewards, next_states, dones = zip(*all_experiences)
        
        # Convert to tensors efficiently
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1))
    
    def get_last_episode_steps(self, num_steps=7):
        """Get first num_steps from last episode for dreaming"""
        if len(self.episode_cache) < num_steps:
            return None
        return self.episode_cache[:num_steps]
    
    def clear_dream_buffer(self):
        """Clear dream buffer to prevent stale dream data"""
        self.dream_buffer.clear()
    
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

# Configuration dictionaries
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
        'actor_lr': 1e-4, 'critic_lr': 3e-4, 'env_model_lr': 5e-5,
        'dream_frequency': 5, 'dream_horizon': 40, 'noise_scale': 0.1,
        'buffer_capacity': 100000, 'dream_rollouts': 1000
    },
    'LunarLanderContinuous-v3': {
        'actor_lr': 3e-4, 'critic_lr': 1e-3, 'env_model_lr': 1e-4,
        'dream_frequency': 3, 'dream_horizon': 30, 'noise_scale': 0.15,
        'buffer_capacity': 50000, 'dream_rollouts': 500
    },
    'HalfCheetah-v3': {
        'actor_lr': 3e-4, 'critic_lr': 1e-3, 'env_model_lr': 1e-4,
        'dream_frequency': 3, 'dream_horizon': 30, 'noise_scale': 0.2,
        'buffer_capacity': 200000, 'dream_rollouts': 1000
    },
    'Humanoid-v3': {
        'actor_lr': 1e-4, 'critic_lr': 3e-4, 'env_model_lr': 3e-5,
        'dream_frequency': 10, 'dream_horizon': 50, 'noise_scale': 0.05,
        'buffer_capacity': 500000, 'dream_rollouts': 1500
    },
    'Ant-v3': {
        'actor_lr': 2e-4, 'critic_lr': 5e-4, 'env_model_lr': 8e-5,
        'dream_frequency': 4, 'dream_horizon': 35, 'noise_scale': 0.12,
        'buffer_capacity': 150000, 'dream_rollouts': 800
    }
}

EXPLORATION_CONFIGS = {
    'BipedalWalker-v3': {
        'exploration_episodes': 150, 'pure_random_ratio': 0.3,
        'exploration_noise_scale': 0.4, 'env_model_pretrain_steps': 500
    },
    'LunarLanderContinuous-v3': {
        'exploration_episodes': 100, 'pure_random_ratio': 0.4,
        'exploration_noise_scale': 0.5, 'env_model_pretrain_steps': 300
    },
    'HalfCheetah-v3': {
        'exploration_episodes': 200, 'pure_random_ratio': 0.2,
        'exploration_noise_scale': 0.3, 'env_model_pretrain_steps': 600
    },
    'Humanoid-v3': {
        'exploration_episodes': 300, 'pure_random_ratio': 0.1,
        'exploration_noise_scale': 0.2, 'env_model_pretrain_steps': 800
    },
    'Ant-v3': {
        'exploration_episodes': 180, 'pure_random_ratio': 0.25,
        'exploration_noise_scale': 0.35, 'env_model_pretrain_steps': 450
    }
}

class Config:
    def __init__(self, env_name):
        self.base_config = BASE_CONFIG.copy()
        self.env_specific = ENV_CONFIGS.get(env_name, ENV_CONFIGS['LunarLanderContinuous-v3'])
        self.exploration_config = EXPLORATION_CONFIGS.get(env_name, EXPLORATION_CONFIGS['LunarLanderContinuous-v3'])
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
            return self.env.action_space.sample()
        else:
            action = self.agent.select_action(state, add_noise=True)
            noise = self.agent.noise.sample(self.config['exploration_noise_scale'])
            return np.clip(action + noise, -self.agent.max_action, self.agent.max_action)

class Agent:
    def __init__(self, state_dim, action_dim, max_action, env_name='LunarLanderContinuous-v3', device='cuda'):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.env_name = env_name
        
        self.config = Config(env_name).get_config()
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action, self.config['hidden_dim']).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, self.config['hidden_dim']).to(device)
        self.critic = Critic(state_dim, action_dim, self.config['hidden_dim']).to(device)
        self.critic_target = Critic(state_dim, action_dim, self.config['hidden_dim']).to(device)
        
        # Enhanced Environment model
        self.env_model = FeedForwardTransformer(
            state_dim, action_dim,
            self.config['transformer_hidden'],
            self.config['transformer_layers'],
            self.config['transformer_heads']
        ).to(device)
        
        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers with improved settings
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config['critic_lr'])
        self.env_model_optimizer = optim.Adam(self.env_model.parameters(), lr=self.config['env_model_lr'])
        
        # FIXED: Loss functions with scalar outputs
        self.rehe_loss = ReHE()
        self.rehae_loss = ReHaE()
        
        # Enhanced replay buffer
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
        
        # Dream quality tracking
        self.last_dream_quality = 0.0
        self.dream_quality_threshold = 0.8
    
    def select_action(self, state, add_noise=True):
        # Fixed: Use torch.tensor() instead of torch.from_numpy() with dtype
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = self.noise.sample(self.current_noise_scale)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def update_noise_scale(self, episode, total_episodes, exploration_episodes):
        """Gradually reduce noise as training progresses"""
        if episode > exploration_episodes:
            training_progress = (episode - exploration_episodes) / (total_episodes - exploration_episodes)
            self.current_noise_scale = self.config['noise_scale'] * max(0.1, 1.0 - training_progress * 0.8)
        else:
            self.current_noise_scale = self.config['noise_scale'] * 2.0
    
    def train_env_model(self, batch_size=256, validation_split=0.1):
        """Enhanced environment model training with validation"""
        if len(self.replay_buffer) < batch_size * 10:
            return 0, 0
        
        sequences = []
        targets = []
        
        # Generate training sequences
        for _ in range(batch_size):
            max_start = len(self.replay_buffer.buffer) - self.context_length - 1
            if max_start <= 0:
                continue
            
            start_idx = random.randint(0, max_start)
            sequence = []
            
            for i in range(self.context_length):
                state, action, reward, next_state, done = self.replay_buffer.buffer[start_idx + i]
                sequence.append(np.concatenate([state, action, [reward]]))
            
            target_state, target_action, target_reward, target_next_state, target_done = self.replay_buffer.buffer[start_idx + self.context_length]
            
            sequences.append(sequence)
            targets.append([target_next_state, target_reward, float(target_done)])
        
        if len(sequences) == 0:
            return 0, 0
        
        # Split into training and validation
        val_size = int(len(sequences) * validation_split)
        train_sequences = sequences[val_size:]
        train_targets = targets[val_size:]
        val_sequences = sequences[:val_size]
        val_targets = targets[:val_size]
        
        # Training
        train_loss = 0
        if train_sequences:
            train_loss = self._train_env_model_batch(train_sequences, train_targets)
        
        # Validation
        val_loss = 0
        if val_sequences:
            with torch.no_grad():
                val_loss = self._validate_env_model_batch(val_sequences, val_targets)
        
        return train_loss, val_loss
    
    def _train_env_model_batch(self, sequences, targets):
        """Train environment model on a batch - Fixed tensor conversion"""
        # Convert to numpy arrays first, then to tensors for efficiency
        sequences_array = np.array(sequences, dtype=np.float32)
        target_states_array = np.array([t[0] for t in targets], dtype=np.float32)
        target_rewards_array = np.array([t[1] for t in targets], dtype=np.float32)
        target_dones_array = np.array([t[2] for t in targets], dtype=np.float32)
        
        # Convert to tensors efficiently
        sequences_tensor = torch.tensor(sequences_array).to(self.device)
        target_states = torch.tensor(target_states_array).to(self.device)
        target_rewards = torch.tensor(target_rewards_array).unsqueeze(-1).to(self.device)
        target_dones = torch.tensor(target_dones_array).unsqueeze(-1).to(self.device)
        
        pred_states, pred_rewards, pred_dones = self.env_model(sequences_tensor)
        
        # Compute losses - ensure scalars
        state_loss = F.mse_loss(pred_states, target_states)
        reward_loss = F.mse_loss(pred_rewards, target_rewards)
        done_loss = F.binary_cross_entropy(pred_dones, target_dones)
        
        total_loss = state_loss + reward_loss + done_loss
        
<<<<<<< Updated upstream
        # Backward pass with gradient clipping
=======
        # Verify scalar before backward
        if total_loss.dim() != 0:
            total_loss = total_loss.mean()
        
        # Update accuracy tracking
        self.env_model.update_accuracy_tracking(pred_states, target_states, pred_rewards, target_rewards)
        
        # Backward pass
>>>>>>> Stashed changes
        self.env_model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.env_model.parameters(), max_norm=1.0)
        self.env_model_optimizer.step()
        
        return total_loss.item()
    
    def _validate_env_model_batch(self, sequences, targets):
        """Validate environment model on a batch - Fixed tensor conversion"""
        # Convert to numpy arrays first, then to tensors for efficiency
        sequences_array = np.array(sequences, dtype=np.float32)
        target_states_array = np.array([t[0] for t in targets], dtype=np.float32)
        target_rewards_array = np.array([t[1] for t in targets], dtype=np.float32)
        target_dones_array = np.array([t[2] for t in targets], dtype=np.float32)
        
        # Convert to tensors efficiently
        sequences_tensor = torch.tensor(sequences_array).to(self.device)
        target_states = torch.tensor(target_states_array).to(self.device)
        target_rewards = torch.tensor(target_rewards_array).unsqueeze(-1).to(self.device)
        target_dones = torch.tensor(target_dones_array).unsqueeze(-1).to(self.device)
        
        pred_states, pred_rewards, pred_dones = self.env_model(sequences_tensor)
        
        # Compute validation losses
        state_loss = F.mse_loss(pred_states, target_states)
        reward_loss = F.mse_loss(pred_rewards, target_rewards)
        done_loss = F.binary_cross_entropy(pred_dones, target_dones)
        
        total_loss = state_loss + reward_loss + done_loss
        return total_loss.item()
    
    def generate_dreams(self):
        """Generate synthetic rollouts with quality assessment"""
        if not self.env_model.is_accurate_enough_for_dreaming():
            print(f"Environment model accuracy too low: {self.env_model.get_model_accuracy():.3f}")
            return []
        
        context_steps = self.replay_buffer.get_last_episode_steps(self.context_length)
        if context_steps is None:
            return []
        
        dreams = []
        successful_dreams = 0
        
        for _ in range(self.dream_rollouts):
            dream_episode = []
            
            # Initialize with context steps
            current_sequence = []
            for step in context_steps:
                state, action, reward, next_state, done = step
                current_sequence.append(np.concatenate([state, action, [reward]]))
                dream_episode.append(step)
            
            current_state = context_steps[-1][3]  # Last next_state becomes current state
            
            # Generate dream trajectory
            dream_successful = True
            for step in range(self.dream_horizon):
                # Select action using current policy
                action = self.select_action(current_state, add_noise=False)
                
                # Predict next state using environment model
                sequence_array = np.array([current_sequence], dtype=np.float32)
                sequence_tensor = torch.tensor(sequence_array).to(self.device)
                
                with torch.no_grad():
                    pred_next_state, pred_reward, pred_done, state_unc, reward_unc = \
                        self.env_model(sequence_tensor, return_uncertainty=True)
                    
                    pred_next_state = pred_next_state.cpu().numpy().flatten()
                    pred_reward = pred_reward.cpu().numpy().item()
                    pred_done = pred_done.cpu().numpy().item() > 0.5
                    
                    # Calculate uncertainty score
                    state_uncertainty = torch.mean(state_unc).cpu().numpy().item()
                    reward_uncertainty = reward_unc.cpu().numpy().item()
                    
                    # If uncertainty is too high, mark dream as unsuccessful
                    if state_uncertainty > 0.5 or reward_uncertainty > 0.3:
                        dream_successful = False
                        break
                
                # Create dream step
                dream_step = (current_state, action, pred_reward, pred_next_state, pred_done)
                dream_episode.append(dream_step)
                
                # Update sequence for next prediction
                current_sequence.append(np.concatenate([current_state, action, [pred_reward]]))
                if len(current_sequence) > self.context_length:
                    current_sequence.pop(0)
                
                current_state = pred_next_state
                
                if pred_done:
                    break
            
            if dream_successful and len(dream_episode) > self.context_length:
                dreams.append(dream_episode)
                successful_dreams += 1
        
        # Update dream quality tracking
        self.last_dream_quality = successful_dreams / self.dream_rollouts if self.dream_rollouts > 0 else 0
        
        return dreams
    
    def train_on_dreams(self, dreams):
        """Train actor-critic on dreamed experiences - Fixed scalar loss issue"""
        if not dreams or self.last_dream_quality < self.dream_quality_threshold:
            return 0
        
        # Clear old dream data
        self.replay_buffer.clear_dream_buffer()
        
        # Add dream experiences to dream buffer
        for dream in dreams:
            for step in dream[self.context_length:]:  # Skip context steps
                state, action, reward, next_state, done = step
                # Weight by dream quality
                uncertainty_weight = min(1.0, self.last_dream_quality * 1.5)
                self.replay_buffer.add_dream_experience(state, action, reward, next_state, done, 
                                                     uncertainty_weight)
        
<<<<<<< Updated upstream
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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
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
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            actor_loss = actor_loss.item()
=======
        # Train on mixed real and dream data
        total_critic_loss = 0
        actor_loss = 0
>>>>>>> Stashed changes
        
        for _ in range(10):  # Multiple training iterations on dream data
            batch = self.replay_buffer.sample(256)
            if batch[0] is None:
                continue
                
            states, actions, rewards, next_states, dones = batch
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
            
            # Ensure scalar losses
            critic_loss_1 = self.rehe_loss(q1_error)
            critic_loss_2 = self.rehe_loss(q2_error)
            critic_loss_3 = self.rehe_loss(q3_error)
            critic_loss = critic_loss_1 + critic_loss_2 + critic_loss_3
            
            # Variance regularization
            q_mean = (q1 + q2 + q3) / 3
            q_var_actual = ((q1 - q_mean)**2 + (q2 - q_mean)**2 + (q3 - q_mean)**2) / 3
            var_loss = F.mse_loss(var, q_var_actual.detach())
            
            total_critic_loss = critic_loss + 0.1 * var_loss
            
            # Ensure scalar before backward
            if total_critic_loss.dim() != 0:
                total_critic_loss = total_critic_loss.mean()
            
            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
            
            # Train actor less frequently
            if self.total_it % self.policy_freq == 0:
                actor_actions = self.actor(states)
                q1_actor, q2_actor, q3_actor, _ = self.critic(states, actor_actions)
                q_actor = torch.min(torch.min(q1_actor, q2_actor), q3_actor)
                actor_loss = -q_actor.mean()
                
                # Ensure scalar before backward
                if actor_loss.dim() != 0:
                    actor_loss = actor_loss.mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
                
                actor_loss = actor_loss.item()
        
        return total_critic_loss.item() if isinstance(total_critic_loss, torch.Tensor) else total_critic_loss
    
    def train(self, batch_size=256):
        """Standard training on replay buffer - Fixed scalar loss issue"""
        if len(self.replay_buffer) < batch_size:
            return 0, 0
        
        self.total_it += 1
        
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
        
        # Compute errors
        q1_error = q1 - target_q
        q2_error = q2 - target_q
        q3_error = q3 - target_q
        
        # Ensure losses are scalars
        critic_loss_1 = self.rehe_loss(q1_error)
        critic_loss_2 = self.rehe_loss(q2_error)  
        critic_loss_3 = self.rehe_loss(q3_error)
        
        # Sum to get total critic loss (should be scalar)
        critic_loss = critic_loss_1 + critic_loss_2 + critic_loss_3
        
        # Variance regularization - ensure scalar
        q_mean = (q1 + q2 + q3) / 3
        q_var_actual = ((q1 - q_mean)**2 + (q2 - q_mean)**2 + (q3 - q_mean)**2) / 3
        var_loss = F.mse_loss(var, q_var_actual.detach())  # This should be scalar
        
        # Total critic loss - ensure scalar
        total_critic_loss = critic_loss + 0.1 * var_loss
        
        # Verify scalar before backward
        if total_critic_loss.dim() != 0:
            total_critic_loss = total_critic_loss.mean()
        
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Train actor
        actor_loss = 0
        if self.total_it % self.policy_freq == 0:
            actor_actions = self.actor(states)
            q1_actor, q2_actor, q3_actor, _ = self.critic(states, actor_actions)
            q_actor = torch.min(torch.min(q1_actor, q2_actor), q3_actor)
            actor_loss = -q_actor.mean()  # Ensure scalar
            
            # Verify scalar before backward
            if actor_loss.dim() != 0:
                actor_loss = actor_loss.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            actor_loss = actor_loss.item()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
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
