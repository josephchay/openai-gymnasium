import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import time
import json
from pipeline import Agent, ExplorationStrategy


def train(env_name='LunarLanderContinuous-v3', max_episodes=2000, 
                   save_dir='./models', enable_exploration=True, resume_from=None):
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Environment setup
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    
    # Agent setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = Agent(state_dim, action_dim, max_action, env_name, device)
    exploration_strategy = ExplorationStrategy(env_name, agent, env)
    
    # Resume from checkpoint if specified
    start_episode = 0
    episode_rewards = []
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        agent.load(resume_from)
        # Load episode rewards if exists
        reward_file = resume_from.replace('.pth', '_rewards.json')
        if os.path.exists(reward_file):
            with open(reward_file, 'r') as f:
                episode_rewards = json.load(f)
            start_episode = len(episode_rewards)
    
    # Get environment-specific parameters
    exploration_episodes = agent.config['exploration_episodes'] if enable_exploration else 0
    dream_frequency = agent.config['dream_frequency']
    pretrain_steps = agent.config['env_model_pretrain_steps']
    
    print(f"Exploration episodes: {exploration_episodes}")
    print(f"Dream frequency: every {dream_frequency} episodes")
    print(f"Environment model pretrain steps: {pretrain_steps}")
    
    # Training metrics
    best_avg_reward = -float('inf')
    start_time = time.time()
    
    # Phase 1: Exploration Phase
    if enable_exploration and start_episode < exploration_episodes:
        print("=== EXPLORATION PHASE ===")
        
        for episode in range(start_episode, exploration_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done:
                # Get exploration action
                action = exploration_strategy.get_exploration_action(state, episode, exploration_episodes)
                
                # Environment step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store experience (no training yet)
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                print(f"Exploration Episode {episode + 1}/{exploration_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, Buffer: {len(agent.replay_buffer)}")
        
        print(f"Exploration completed. Buffer size: {len(agent.replay_buffer)}")
        
        # Pre-train environment model on exploration data
        print("Pre-training environment model on exploration data...")
        pretrain_losses = []
        for i in range(pretrain_steps):
            loss = agent.train_env_model(batch_size=128)
            if loss > 0:
                pretrain_losses.append(loss)
            if (i + 1) % 100 == 0:
                avg_loss = np.mean(pretrain_losses[-100:]) if pretrain_losses else 0
                print(f"Pre-train step {i + 1}/{pretrain_steps}, Avg Loss: {avg_loss:.4f}")
        
        print("Environment model pre-training completed!")
        agent.exploration_phase = False
        
        start_episode = exploration_episodes
    
    print("=== TRAINING PHASE ===")
    
    # Phase 2: Training Phase
    for episode in range(start_episode, max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        # Update noise scale based on progress
        agent.update_noise_scale(episode, max_episodes, exploration_episodes)
        
        while not done:
            # Select action with adaptive noise
            action = agent.select_action(state, add_noise=True)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Training (now that we have good data)
            if len(agent.replay_buffer) > 1000:
                critic_loss, actor_loss = agent.train()
                if steps % 5 == 0:  # Train env model less frequently
                    env_model_loss = agent.train_env_model()
        
        episode_rewards.append(episode_reward)
        
        # Dream training phase
        if (episode - exploration_episodes) % dream_frequency == 0 and episode > exploration_episodes + 10:
            print(f"Episode {episode}: Entering dream phase...")
            
            # Generate dreams
            dreams = agent.generate_dreams()
            
            if dreams:
                # Train on dreams multiple times
                dream_losses = []
                for _ in range(15):  # More dream training iterations
                    dream_loss = agent.train_on_dreams(dreams)
                    if dream_loss > 0:
                        dream_losses.append(dream_loss)
                
                avg_dream_loss = np.mean(dream_losses) if dream_losses else 0
                print(f"Dream phase completed. Generated {len(dreams)} rollouts. "
                      f"Avg dream loss: {avg_dream_loss:.4f}")
        
        # Logging and checkpointing
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            std_reward = np.std(episode_rewards[-100:])
            elapsed_time = time.time() - start_time
            
            print(f"Episode {episode}")
            print(f"Average Reward (last 100): {avg_reward:.2f} ± {std_reward:.2f}")
            print(f"Latest Reward: {episode_reward:.2f}")
            print(f"Buffer Size: {len(agent.replay_buffer)}")
            print(f"Current Noise Scale: {agent.current_noise_scale:.3f}")
            print(f"Time Elapsed: {elapsed_time/60:.1f} minutes")
            print("-" * 60)
            
            # Save training progress
            reward_file = os.path.join(save_dir, f'{env_name}_rewards.json')
            with open(reward_file, 'w') as f:
                json.dump(episode_rewards, f)
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_model_path = os.path.join(save_dir, f'{env_name}_best.pth')
                agent.save(best_model_path)
                print(f"🎉 New best model saved! Average reward: {best_avg_reward:.2f}")
            
            # Regular checkpoint
            checkpoint_path = os.path.join(save_dir, f'{env_name}_checkpoint_ep{episode}.pth')
            agent.save(checkpoint_path)
        
        # Early stopping for excellent performance
        if len(episode_rewards) >= 100:
            recent_avg = np.mean(episode_rewards[-100:])
            # Environment-specific early stopping thresholds
            thresholds = {
                'LunarLanderContinuous-v3': 200,
                'BipedalWalker-v3': 300,
                'HalfCheetah-v3': 3000,
                'Ant-v3': 4000,
                'Humanoid-v3': 5000
            }
            threshold = thresholds.get(env_name, 200)
            
            if recent_avg > threshold:
                print(f"🚀 Early stopping! Achieved average reward of {recent_avg:.2f}")
                break
    
    env.close()
    
    # Final save
    final_model_path = os.path.join(save_dir, f'{env_name}_final.pth')
    agent.save(final_model_path)
    
    return episode_rewards

def evaluate_agent(env_name, model_path, num_episodes=10, render=False):
    """Evaluate a trained agent"""
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(state_dim, action_dim, max_action, env_name, device)
    agent.load(model_path)
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Evaluating agent on {env_name} for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state, add_noise=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if render:
                time.sleep(0.01)  # Small delay for human viewing
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n📊 Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {min(episode_rewards):.2f}")
    print(f"Max Reward: {max(episode_rewards):.2f}")
    print(f"Average Episode Length: {avg_length:.1f}")
    
    env.close()
    return episode_rewards

def plot_training_results(rewards, env_name, save_dir='./plots'):
    """Plot training results"""
    try:
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 8))
        
        # Raw rewards
        plt.subplot(2, 2, 1)
        plt.plot(rewards, alpha=0.6, label='Episode Reward')
        plt.title('Raw Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Moving averages
        plt.subplot(2, 2, 2)
        window_sizes = [50, 100]
        colors = ['red', 'blue']
        
        for window, color in zip(window_sizes, colors):
            if len(rewards) >= window:
                moving_avg = [np.mean(rewards[i:i+window]) 
                             for i in range(len(rewards) - window + 1)]
                plt.plot(range(window-1, len(rewards)), moving_avg, 
                        color=color, linewidth=2, label=f'Moving Average ({window})')
        
        plt.title('Moving Averages')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Reward distribution
        plt.subplot(2, 2, 3)
        plt.hist(rewards, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(rewards), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(rewards):.2f}')
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning curve
        plt.subplot(2, 2, 4)
        if len(rewards) >= 100:
            learning_curve = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
            plt.plot(learning_curve, color='green', linewidth=2)
            plt.title('Learning Curve (100-episode average)')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Results - {env_name}', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, f'training_{env_name.replace("-", "_")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or Evaluate Agent')
    parser.add_argument("--env", default="LunarLanderContinuous-v3", 
                       help="Environment name")
    parser.add_argument("--episodes", type=int, default=2000,
                       help="Number of training episodes")
    parser.add_argument("--save_dir", default="./models",
                       help="Directory to save models")
    parser.add_argument("--no_exploration", action="store_true",
                       help="Skip exploration phase")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate a trained model")
    parser.add_argument("--model_path", type=str,
                       help="Path to model for evaluation")
    parser.add_argument("--eval_episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render during evaluation")
    parser.add_argument("--resume", type=str,
                       help="Resume training from checkpoint")
    parser.add_argument("--plot", action="store_true",
                       help="Generate training plots")
    
    args = parser.parse_args()
    
    if args.evaluate:
        if not args.model_path:
            print("❌ Please provide --model_path for evaluation")
            exit(1)
        evaluate_agent(args.env, args.model_path, args.eval_episodes, args.render)
    else:
        print(f"🚀 Starting training on {args.env}")
        enable_exploration = not args.no_exploration
        
        rewards = train(
            env_name=args.env, 
            max_episodes=args.episodes, 
            save_dir=args.save_dir,
            enable_exploration=enable_exploration,
            resume_from=args.resume
        )
        
        print("✅ Training completed!")
        
        # Generate plots if requested
        if args.plot:
            plot_training_results(rewards, args.env)
        
        # Final evaluation
        print("\n🔍 Running final evaluation...")
        best_model = os.path.join(args.save_dir, f'{args.env}_best.pth')
        if os.path.exists(best_model):
            evaluate_agent(args.env, best_model, num_episodes=5)
