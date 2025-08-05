import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import time
import json
import signal
import sys
import pickle
from datetime import datetime
from pathlib import Path
from pipeline import Agent, ExplorationStrategy

<<<<<<< Updated upstream
def train(env_name='LunarLanderContinuous-v3', max_episodes=2000, 
                   save_dir='./models', enable_exploration=True, resume_from=None):
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
=======
# Optional memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class GracefulStopper:
    """Handle graceful stopping via Ctrl+C"""
    def __init__(self):
        self.stop_training = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print('\n🛑 Graceful stop requested. Finishing current episode...')
        self.stop_training = True


def get_memory_usage():
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return None
    return None


def create_run_directory(base_models_dir, env_name, resume_run=None):
    """
    Create or find run directory with structure: /models/{env}/
    
    Args:
        base_models_dir: Base models directory (e.g., './models')
        env_name: Environment name
        resume_run: Specific run name to resume from or 'latest'
    
    Returns:
        run_dir: Path to run directory
        is_resume: Boolean indicating if this is a resume
    """
    env_dir = Path(base_models_dir) / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    
    if resume_run:
        # Resume from specific run or latest
        if resume_run == "latest":
            # Find latest run based on modification time
            run_dirs = [d for d in env_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
            if not run_dirs:
                raise ValueError(f"No existing runs found for {env_name}")
            # Sort by modification time and get the latest
            run_dirs.sort(key=lambda x: x.stat().st_mtime)
            run_dir = run_dirs[-1]
            print(f"📂 Resuming from latest run: {run_dir}")
        else:
            # Resume from specific run name
            run_dir = env_dir / resume_run
            if not run_dir.exists():
                available_runs = [d.name for d in env_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
                raise ValueError(f"Run '{resume_run}' not found. Available runs: {available_runs}")
            print(f"📂 Resuming from specified run: {run_dir}")
        return run_dir, True
    else:
        # Create new run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = env_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Starting new run: {run_dir}")
        return run_dir, False


def save_training_state(run_dir, agent, episode_rewards, episode, best_avg_reward, best_episode):
    """Save complete training state for resuming with enhanced error handling"""
    
    try:
        # Create backup directory if saving fails
        backup_dir = run_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        # Save model checkpoints
        checkpoint_path = run_dir / f"checkpoint_ep{episode}.pth"
        agent.save(str(checkpoint_path))
        
        # Save latest model with backup
        latest_path = run_dir / "latest.pth"
        latest_backup_path = backup_dir / "latest_backup.pth"
        
        # Save to backup first, then to main location
        agent.save(str(latest_backup_path))
        agent.save(str(latest_path))
        
        # Save episode rewards
        rewards_path = run_dir / "episode_rewards.json"
        rewards_backup_path = backup_dir / "episode_rewards_backup.json"
        
        with open(rewards_backup_path, 'w') as f:
            json.dump(episode_rewards, f)
        with open(rewards_path, 'w') as f:
            json.dump(episode_rewards, f)
        
        # Save training metadata with enhanced info
        metadata = {
            'current_episode': episode,
            'total_episodes': len(episode_rewards),
            'best_avg_reward': best_avg_reward,
            'best_episode': best_episode,
            'last_update': datetime.now().isoformat(),
            'env_name': agent.env_name,
            'env_model_accuracy': agent.env_model.get_model_accuracy(),
            'last_dream_quality': getattr(agent, 'last_dream_quality', 0.0),
            'exploration_phase': agent.exploration_phase,
            'total_training_iterations': agent.total_it,
            'current_noise_scale': agent.current_noise_scale,
            'dream_quality_threshold': agent.dream_quality_threshold,
            'context_length': agent.context_length
        }
        
        metadata_path = run_dir / "training_metadata.json"
        metadata_backup_path = backup_dir / "training_metadata_backup.json"
        
        with open(metadata_backup_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save replay buffer (this can be large, so save less frequently)
        if episode % 10 == 0 or episode < 100:  # Save more frequently early on
            buffer_path = run_dir / "replay_buffer.pkl"
            buffer_backup_path = backup_dir / "replay_buffer_backup.pkl"
            
            buffer_data = {
                'buffer': agent.replay_buffer,
                'current_episode': episode,
                'episode_rewards': episode_rewards,
                'best_avg_reward': best_avg_reward,
                'best_episode': best_episode
            }
            
            with open(buffer_backup_path, 'wb') as f:
                pickle.dump(buffer_data, f)
            with open(buffer_path, 'wb') as f:
                pickle.dump(buffer_data, f)
        
        print(f"💾 Training state saved at episode {episode}")
        
    except Exception as e:
        print(f"❌ Error saving training state: {e}")
        print("⚠️  Training will continue but state may not be recoverable")
        # Try to save at least the model
        try:
            emergency_path = run_dir / f"emergency_save_ep{episode}.pth"
            agent.save(str(emergency_path))
            print(f"🆘 Emergency model save successful: {emergency_path}")
        except Exception as e2:
            print(f"💥 Emergency save also failed: {e2}")


def load_training_state(run_dir, agent):
    """Load training state for resuming with enhanced error handling"""
    
    # Try to load latest model (with backup fallback)
    latest_path = run_dir / "latest.pth"
    backup_path = run_dir / "backup" / "latest_backup.pth"
    
    model_loaded = False
    if latest_path.exists():
        try:
            print("🔄 Loading latest model...")
            agent.load(str(latest_path))
            model_loaded = True
        except Exception as e:
            print(f"❌ Error loading latest model: {e}")
            if backup_path.exists():
                try:
                    print("🔄 Trying backup model...")
                    agent.load(str(backup_path))
                    model_loaded = True
                    print("✅ Backup model loaded successfully")
                except Exception as e2:
                    print(f"❌ Backup model also failed: {e2}")
    
    if not model_loaded:
        raise FileNotFoundError(f"No working model found in {run_dir}")
    
    # Load episode rewards
    rewards_path = run_dir / "episode_rewards.json"
    rewards_backup_path = run_dir / "backup" / "episode_rewards_backup.json"
    
    episode_rewards = []
    if rewards_path.exists():
        try:
            with open(rewards_path, 'r') as f:
                episode_rewards = json.load(f)
            print(f"📊 Loaded {len(episode_rewards)} episode rewards")
        except Exception as e:
            print(f"❌ Error loading rewards: {e}")
            if rewards_backup_path.exists():
                try:
                    with open(rewards_backup_path, 'r') as f:
                        episode_rewards = json.load(f)
                    print(f"📊 Loaded {len(episode_rewards)} episode rewards from backup")
                except Exception as e2:
                    print(f"❌ Backup rewards also failed: {e2}")
    
    # Load training metadata
    metadata_path = run_dir / "training_metadata.json"
    metadata_backup_path = run_dir / "backup" / "training_metadata_backup.json"
    
    start_episode = len(episode_rewards)
    best_avg_reward = -float('inf')
    best_episode = 0
    
    metadata_loaded = False
    for path in [metadata_path, metadata_backup_path]:
        if path.exists() and not metadata_loaded:
            try:
                with open(path, 'r') as f:
                    metadata = json.load(f)
                
                start_episode = metadata['current_episode'] + 1
                best_avg_reward = metadata.get('best_avg_reward', -float('inf'))
                best_episode = metadata.get('best_episode', 0)
                
                # Load enhanced metadata
                env_model_accuracy = metadata.get('env_model_accuracy', 0.0)
                last_dream_quality = metadata.get('last_dream_quality', 0.0)
                total_it = metadata.get('total_training_iterations', 0)
                
                # Restore agent state
                agent.total_it = total_it
                
                print(f"📈 Resume from episode {start_episode}, best avg reward: {best_avg_reward:.2f}")
                print(f"🧠 Environment model accuracy: {env_model_accuracy:.3f}")
                print(f"💭 Last dream quality: {last_dream_quality:.3f}")
                print(f"🔄 Total training iterations: {total_it}")
                
                metadata_loaded = True
                break
                
            except Exception as e:
                print(f"❌ Error loading metadata from {path}: {e}")
    
    if not metadata_loaded:
        print("⚠️  No metadata loaded, using defaults")
    
    # Load replay buffer
    buffer_path = run_dir / "replay_buffer.pkl"
    buffer_backup_path = run_dir / "backup" / "replay_buffer_backup.pkl"
    
    for path in [buffer_path, buffer_backup_path]:
        if path.exists():
            try:
                print(f"🧠 Loading replay buffer from {path.name}...")
                with open(path, 'rb') as f:
                    buffer_data = pickle.load(f)
                agent.replay_buffer = buffer_data['buffer']
                print(f"✅ Loaded replay buffer with {len(agent.replay_buffer)} experiences")
                break
            except Exception as e:
                print(f"❌ Error loading replay buffer from {path}: {e}")
    
    return episode_rewards, start_episode, best_avg_reward, best_episode


def train(env_name='LunarLanderContinuous-v3', max_episodes=None, 
          models_dir='./models', enable_exploration=True, 
          resume_run=None, log_frequency=1):
    # Create run directory structure: models/{env}/run_timestamp/
    run_dir, is_resume = create_run_directory(models_dir, env_name, resume_run)
    
    # Initialize graceful stopper
    stopper = GracefulStopper()
>>>>>>> Stashed changes
    
    # Environment setup
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"🌍 Environment: {env_name}")
    print(f"📏 State dimension: {state_dim}")
    print(f"🎯 Action dimension: {action_dim}")
    print(f"⚡ Max action: {max_action}")
    print(f"📝 Logging frequency: Every {log_frequency} episode(s)")
    print(f"📁 Run directory: {run_dir}")
    
    if max_episodes:
        print(f"🔄 Training will run for {max_episodes} episodes")
    else:
        print("🔄 Training will run INDEFINITELY until manually stopped (Ctrl+C)")
    
    # Agent setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    if PSUTIL_AVAILABLE:
        initial_memory = get_memory_usage()
        if initial_memory:
            print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
    
    agent = Agent(state_dim, action_dim, max_action, env_name, device)
    exploration_strategy = ExplorationStrategy(env_name, agent, env)
    
    # Load or initialize training state
    if is_resume:
        try:
            episode_rewards, start_episode, best_avg_reward, best_episode = load_training_state(run_dir, agent)
            print(f"✅ Successfully resumed training from episode {start_episode}")
        except Exception as e:
            print(f"❌ Error loading training state: {e}")
            print("🔄 Starting fresh training...")
            episode_rewards, start_episode, best_avg_reward, best_episode = [], 0, -float('inf'), 0
    else:
        episode_rewards, start_episode, best_avg_reward, best_episode = [], 0, -float('inf'), 0
    
    # Get environment-specific parameters
    exploration_episodes = agent.config['exploration_episodes'] if enable_exploration else 0
    dream_frequency = agent.config['dream_frequency']
    pretrain_steps = agent.config['env_model_pretrain_steps']
    
    print(f"🔍 Exploration episodes: {exploration_episodes}")
    print(f"💭 Dream frequency: every {dream_frequency} episodes")
    print(f"🧠 Environment model pretrain steps: {pretrain_steps}")
    print(f"🎯 Dream quality threshold: {agent.dream_quality_threshold}")
    
    # Training metrics
    start_time = time.time()
    last_save_time = time.time()
    save_interval = 300  # Save every 5 minutes
    
    # Phase 1: Exploration Phase (only if not resuming from post-exploration)
    if enable_exploration and start_episode < exploration_episodes:
        print("=== EXPLORATION PHASE ===")
        
        for episode in range(start_episode, exploration_episodes):
            if stopper.stop_training:
                print("🛑 Training stopped during exploration phase.")
                break
                
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            episode_start_time = time.time()
            
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
            episode_time = time.time() - episode_start_time
            
            # Per-episode exploration logging
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                print(f"Exploration Episode {episode + 1}/{exploration_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, Buffer: {len(agent.replay_buffer)}, "
                      f"Episode Time: {episode_time:.1f}s")
            
            # Save state every 50 episodes during exploration or every 5 minutes
            current_time = time.time()
            if (episode + 1) % 50 == 0 or (current_time - last_save_time) > save_interval:
                save_training_state(run_dir, agent, episode_rewards, episode, best_avg_reward, best_episode)
                last_save_time = current_time
        
        if not stopper.stop_training:
            print(f"✅ Exploration completed. Buffer size: {len(agent.replay_buffer)}")
            
            # Pre-train environment model on exploration data
            print("🧠 Pre-training environment model on exploration data...")
            pretrain_losses = []
            validation_losses = []
            pretrain_start_time = time.time()
            
            for i in range(pretrain_steps):
                if stopper.stop_training:
                    print("🛑 Training stopped during environment model pre-training.")
                    break
                
                step_start_time = time.time()
                
                # Updated to handle the new return format (train_loss, val_loss)
                try:
                    result = agent.train_env_model(batch_size=128)
                    if isinstance(result, tuple):
                        train_loss, val_loss = result
                        if train_loss > 0:
                            pretrain_losses.append(train_loss)
                        if val_loss > 0:
                            validation_losses.append(val_loss)
                    else:
                        # Fallback for single return value
                        if result > 0:
                            pretrain_losses.append(result)
                except Exception as e:
                    print(f"❌ Error in environment model training step {i+1}: {e}")
                    continue
                
                step_time = time.time() - step_start_time
                
                if (i + 1) % 100 == 0:
                    avg_train_loss = np.mean(pretrain_losses[-100:]) if pretrain_losses else 0
                    avg_val_loss = np.mean(validation_losses[-100:]) if validation_losses else 0
                    model_accuracy = agent.env_model.get_model_accuracy()
                    elapsed_time = time.time() - pretrain_start_time
                    
                    print(f"Pre-train step {i + 1}/{pretrain_steps}, "
                          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                          f"Accuracy: {model_accuracy:.3f}, Time: {elapsed_time:.1f}s, "
                          f"Step Time: {step_time:.3f}s")
            
            if not stopper.stop_training:
                final_accuracy = agent.env_model.get_model_accuracy()
                pretrain_time = time.time() - pretrain_start_time
                print(f"✅ Environment model pre-training completed! Final accuracy: {final_accuracy:.3f}")
                print(f"⏱️  Pre-training time: {pretrain_time/60:.1f} minutes")
                
                if final_accuracy < 0.05:
                    print("🚨 Critical: Environment model accuracy is very low. Dreams will be disabled.")
                elif final_accuracy < 0.1:
                    print("⚠️  Warning: Environment model accuracy is low. Dreams may be unreliable.")
                else:
                    print("✅ Environment model accuracy is sufficient for dreaming.")
                
                agent.exploration_phase = False
            
            start_episode = exploration_episodes
    
    # Phase 2: Training Phase
    if not stopper.stop_training:
        print("=== TRAINING PHASE ===")
        print("Episode format: Ep XXXX | R: reward | Avg: avg_reward | Buf: buffer_size | Noise: noise_scale | Steps: episode_steps")
        print("Press Ctrl+C to stop training gracefully and save the model.")
        
        episode = start_episode
        training_start_time = time.time()
        
        while not stopper.stop_training and (max_episodes is None or episode < max_episodes):
            episode_start_time = time.time()
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            # Track losses during episode
            episode_losses = {'critic': [], 'actor': [], 'env_model_train': [], 'env_model_val': []}
            
            # Update noise scale based on progress
            total_episodes = max_episodes if max_episodes else episode + 10000
            agent.update_noise_scale(episode, total_episodes, exploration_episodes)
            
            while not done and not stopper.stop_training:
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
                    try:
                        critic_loss, actor_loss = agent.train()
                        episode_losses['critic'].append(critic_loss)
                        episode_losses['actor'].append(actor_loss)
                    except Exception as e:
                        print(f"❗ Warning: Agent training error: {e}")
                    
                    if steps % 5 == 0:  # Train env model less frequently
                        try:
                            # Handle new return format
                            result = agent.train_env_model()
                            if isinstance(result, tuple):
                                train_loss, val_loss = result
                                if train_loss > 0:
                                    episode_losses['env_model_train'].append(train_loss)
                                if val_loss > 0:
                                    episode_losses['env_model_val'].append(val_loss)
                            else:
                                if result > 0:
                                    episode_losses['env_model_train'].append(result)
                        except Exception as e:
                            print(f"❗ Warning: Env model training error: {e}")
            
            if stopper.stop_training:
                break
                
            episode_rewards.append(episode_reward)
            episode_time = time.time() - episode_start_time
            
            # Calculate average losses for this episode
            avg_losses = {k: np.mean(v) if v else 0.0 for k, v in episode_losses.items()}
            
            # PER-EPISODE LOGGING based on log_frequency
            if episode % log_frequency == 0:
                avg_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
                
                if log_frequency == 1:
                    # Compact format for every episode
                    memory_info = ""
                    if PSUTIL_AVAILABLE:
                        memory = get_memory_usage()
                        if memory:
                            memory_info = f" | Mem: {memory:.0f}MB"
                    
                    print(f"Ep {episode:4d} | R: {episode_reward:7.2f} | Avg: {avg_reward:7.2f} | "
                          f"Buf: {len(agent.replay_buffer):5d} | Noise: {agent.current_noise_scale:.3f} | "
                          f"Steps: {steps:3d} | Acc: {agent.env_model.get_model_accuracy():.3f} | "
                          f"Time: {episode_time:.1f}s{memory_info}")
                else:
                    # Detailed format for less frequent logging
                    std_reward = np.std(episode_rewards[-min(100, len(episode_rewards)):])
                    elapsed_time = time.time() - start_time
                    model_accuracy = agent.env_model.get_model_accuracy()
                    dream_quality = getattr(agent, 'last_dream_quality', 0.0)
                    
                    print(f"Episode {episode}")
                    print(f"Average Reward (last {min(100, len(episode_rewards))}): {avg_reward:.2f} ± {std_reward:.2f}")
                    print(f"Latest Reward: {episode_reward:.2f}")
                    print(f"Buffer Size: {len(agent.replay_buffer)}")
                    print(f"Current Noise Scale: {agent.current_noise_scale:.3f}")
                    print(f"Environment Model Accuracy: {model_accuracy:.3f}")
                    print(f"Last Dream Quality: {dream_quality:.3f}")
                    print(f"Episode Time: {episode_time:.1f}s")
                    print(f"Total Time Elapsed: {elapsed_time/60:.1f} minutes")
                    print(f"Losses - Critic: {avg_losses['critic']:.4f}, Actor: {avg_losses['actor']:.4f}")
                    print(f"Env Model - Train: {avg_losses['env_model_train']:.4f}, Val: {avg_losses['env_model_val']:.4f}")
                    
                    if PSUTIL_AVAILABLE:
                        memory = get_memory_usage()
                        if memory:
                            print(f"Memory Usage: {memory:.1f} MB")
                    
                    print("-" * 60)
            
            # Detailed logging every 50 episodes regardless of log_frequency
            if episode % 50 == 0 and log_frequency == 1:
                avg_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
                std_reward = np.std(episode_rewards[-min(100, len(episode_rewards)):])
                elapsed_time = time.time() - start_time
                training_time = time.time() - training_start_time
                model_accuracy = agent.env_model.get_model_accuracy()
                dream_quality = getattr(agent, 'last_dream_quality', 0.0)
                
                print(f"\n=== Episode {episode} Detailed Report ===")
                print(f"Average Reward (last {min(100, len(episode_rewards))}): {avg_reward:.2f} ± {std_reward:.2f}")
                print(f"Latest Reward: {episode_reward:.2f}")
                print(f"Buffer Size: {len(agent.replay_buffer)}")
                print(f"Current Noise Scale: {agent.current_noise_scale:.3f}")
                print(f"Environment Model Accuracy: {model_accuracy:.3f}")
                print(f"Dream Quality: {dream_quality:.3f}")
                print(f"Total Time Elapsed: {elapsed_time/60:.1f} minutes")
                print(f"Training Phase Time: {training_time/60:.1f} minutes")
                print(f"Total Episodes Completed: {episode + 1}")
                print(f"Training Iterations: {agent.total_it}")
                print(f"Losses - Critic: {avg_losses['critic']:.4f}, Actor: {avg_losses['actor']:.4f}")
                print(f"Env Model - Train: {avg_losses['env_model_train']:.4f}, Val: {avg_losses['env_model_val']:.4f}")
                
                # Show best performance achieved so far
                if len(episode_rewards) >= 100:
                    best_100_avg = max([np.mean(episode_rewards[i:i+100]) for i in range(len(episode_rewards)-99)])
                    print(f"Best 100-episode average so far: {best_100_avg:.2f}")
                
                if PSUTIL_AVAILABLE:
                    memory = get_memory_usage()
                    if memory:
                        print(f"Memory Usage: {memory:.1f} MB")
                
                print("-" * 70)
            
            # Enhanced Dream training phase with quality checks
            if (episode - exploration_episodes) % dream_frequency == 0 and episode > exploration_episodes + 10:
                model_accuracy = agent.env_model.get_model_accuracy()
                
                if model_accuracy > 0.05:  # Minimum threshold for attempting dreams
                    print(f"💭 Episode {episode}: Entering dream phase (Model Accuracy: {model_accuracy:.3f})...")
                    
                    dream_start_time = time.time()
                    
                    try:
                        # Generate dreams
                        dreams = agent.generate_dreams()
                        
                        if dreams:
                            # Check dream generation success rate
                            success_rate = len(dreams) / agent.dream_rollouts
                            if success_rate < 0.1:
                                print(f"⚠️  Warning: Very low dream generation success rate: {len(dreams)}/{agent.dream_rollouts} ({success_rate:.1%})")
                            
                            # Train on dreams multiple times
                            dream_losses = []
                            for iteration in range(15):  # More dream training iterations
                                try:
                                    dream_loss = agent.train_on_dreams(dreams)
                                    if dream_loss > 0:
                                        dream_losses.append(dream_loss)
                                except Exception as e:
                                    print(f"❗ Warning: Dream training iteration {iteration+1} error: {e}")
                                    break
                            
                            avg_dream_loss = np.mean(dream_losses) if dream_losses else 0
                            dream_quality = getattr(agent, 'last_dream_quality', 0.0)
                            dream_time = time.time() - dream_start_time
                            
                            print(f"✨ Dream phase completed. Generated {len(dreams)} rollouts. "
                                  f"Avg dream loss: {avg_dream_loss:.4f}, Dream quality: {dream_quality:.3f}, "
                                  f"Time: {dream_time:.1f}s")
                            
                            if dream_quality < 0.3:
                                print("🚨 Critical: Very low dream quality detected. Consider more environment model training.")
                            elif dream_quality < 0.5:
                                print("⚠️  Warning: Low dream quality detected. Consider more environment model training.")
                            else:
                                print("✅ Good dream quality achieved.")
                        else:
                            print("⚠️  No dreams generated - model accuracy too low or insufficient context.")
                            
                    except Exception as e:
                        print(f"❗ Warning: Dream phase error: {e}")
                        print("⚠️  Continuing without dreams...")
                        
                else:
                    print(f"⚠️  Skipping dream phase - model accuracy too low: {model_accuracy:.3f}")
            
            # Save checkpoints and models every 50 episodes or every 5 minutes
            current_time = time.time()
            if episode % 50 == 0 or (current_time - last_save_time) > save_interval:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                
                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    try:
                        best_model_path = run_dir / "best_model.pth"
                        agent.save(str(best_model_path))
                        best_episode = episode
                        print(f"🎉 New best model saved! Average reward: {best_avg_reward:.2f}")
                    except Exception as e:
                        print(f"❌ Error saving best model: {e}")
                
                # Save training state
                save_training_state(run_dir, agent, episode_rewards, episode, best_avg_reward, best_episode)
                last_save_time = current_time
            
            # Performance milestone notifications (but don't stop!)
            if len(episode_rewards) >= 100:
                recent_avg = np.mean(episode_rewards[-100:])
                # Environment-specific performance milestones for notifications
                milestones = {
                    'LunarLanderContinuous-v3': [0, 50, 100, 150, 200, 250],
                    'BipedalWalker-v3': [100, 200, 300, 350],
                    'HalfCheetah-v3': [1000, 2000, 3000, 4000, 5000],
                    'Ant-v3': [1000, 2000, 3000, 4000, 5000],
                    'Humanoid-v3': [2000, 4000, 6000, 8000]
                }
                
                env_milestones = milestones.get(env_name, [0, 100, 200, 300])
                
                # Check if we've reached a new milestone
                for milestone in env_milestones:
                    if (recent_avg >= milestone and 
                        (episode == 0 or np.mean(episode_rewards[-200:-100]) < milestone)):
                        print(f"🎯 MILESTONE REACHED! Average reward: {recent_avg:.2f} >= {milestone}")
                        break
            
            episode += 1
    
    env.close()
    
    # Final save when training is stopped
    print("💾 Saving final training state...")
    try:
        final_model_path = run_dir / "final_model.pth"
        agent.save(str(final_model_path))
        save_training_state(run_dir, agent, episode_rewards, episode-1, best_avg_reward, best_episode)
        print("✅ Final save completed successfully")
    except Exception as e:
        print(f"❌ Error in final save: {e}")
    
    total_time = time.time() - start_time
    print(f"✅ Training completed after {len(episode_rewards)} episodes.")
    print(f"⏱️  Total training time: {total_time/3600:.1f} hours")
    print(f"📁 All files saved to: {run_dir}")
    
    # Create enhanced run summary
    try:
        final_accuracy = agent.env_model.get_model_accuracy()
        final_dream_quality = getattr(agent, 'last_dream_quality', 0.0)
        
        summary = {
            'environment': env_name,
            'total_episodes': len(episode_rewards),
            'final_average_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
            'best_average_reward': best_avg_reward,
            'best_episode': best_episode,
            'final_env_model_accuracy': final_accuracy,
            'final_dream_quality': final_dream_quality,
            'total_training_time_hours': total_time / 3600,
            'total_training_iterations': agent.total_it,
            'training_completed': datetime.now().isoformat(),
            'run_directory': str(run_dir),
            'device_used': str(device),
            'final_noise_scale': agent.current_noise_scale
        }
        
        summary_path = run_dir / "run_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"📊 Run summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error creating run summary: {e}")
    
    return episode_rewards


def evaluate_agent_from_run(run_dir, num_episodes=10, render=False):
    """Evaluate agent from a specific run directory with enhanced error handling"""
    
    # Load metadata to get environment name
    metadata_path = run_dir / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No training metadata found in {run_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    env_name = metadata['env_name']
    
    # Setup environment
    try:
        if render:
            env = gym.make(env_name, render_mode='human')
        else:
            env = gym.make(env_name)
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return []
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(state_dim, action_dim, max_action, env_name, device)
    
    # Load best model with fallback
    model_loaded = False
    best_model_path = run_dir / "best_model.pth"
    latest_model_path = run_dir / "latest.pth"
    final_model_path = run_dir / "final_model.pth"
    
    for model_path in [best_model_path, final_model_path, latest_model_path]:
        if model_path.exists():
            try:
                agent.load(str(model_path))
                print(f"✅ Loaded model from {model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"❌ Error loading {model_path}: {e}")
    
    if not model_loaded:
        print("❌ No working model found for evaluation")
        env.close()
        return []
    
    episode_rewards = []
    episode_lengths = []
    episode_times = []
    
    print(f"🧪 Evaluating agent on {env_name} for {num_episodes} episodes...")
    
    eval_start_time = time.time()
    
    for episode in range(num_episodes):
        try:
            episode_start_time = time.time()
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
            
            episode_time = time.time() - episode_start_time
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_times.append(episode_time)
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}, Time = {episode_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Error in evaluation episode {episode + 1}: {e}")
            continue
    
    total_eval_time = time.time() - eval_start_time
    
    if episode_rewards:
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        avg_time = np.mean(episode_times)
        
        print(f"\n📊 Evaluation Results:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Min Reward: {min(episode_rewards):.2f}")
        print(f"Max Reward: {max(episode_rewards):.2f}")
        print(f"Average Episode Length: {avg_length:.1f}")
        print(f"Average Episode Time: {avg_time:.1f}s")
        print(f"Total Evaluation Time: {total_eval_time:.1f}s")
        
        # Save evaluation results
        try:
            eval_results = {
                'num_episodes': len(episode_rewards),
                'average_reward': avg_reward,
                'std_reward': std_reward,
                'min_reward': min(episode_rewards),
                'max_reward': max(episode_rewards),
                'average_length': avg_length,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'evaluation_date': datetime.now().isoformat(),
                'total_evaluation_time': total_eval_time
            }
            
            eval_path = run_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            print(f"📝 Evaluation results saved to: {eval_path}")
            
        except Exception as e:
            print(f"❌ Error saving evaluation results: {e}")
    else:
        print("❌ No successful evaluation episodes")
    
    env.close()
    return episode_rewards


def list_runs(models_dir, env_name):
    """List all available runs for an environment with enhanced information"""
    env_dir = Path(models_dir) / env_name
    
    if not env_dir.exists():
        print(f"❌ No runs found for environment {env_name}")
        return []
    
    runs = []
    for run_dir in env_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith('run_'):
            metadata_path = run_dir / "training_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Calculate run size
                    total_size = sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    
                    runs.append({
                        'name': run_dir.name,
                        'path': str(run_dir),
                        'episodes': metadata.get('total_episodes', 'Unknown'),
                        'best_reward': metadata.get('best_avg_reward', 'Unknown'),
                        'last_update': metadata.get('last_update', 'Unknown'),
                        'env_model_accuracy': metadata.get('env_model_accuracy', 'Unknown'),
                        'dream_quality': metadata.get('last_dream_quality', 'Unknown'),
                        'training_iterations': metadata.get('total_training_iterations', 'Unknown'),
                        'size_mb': size_mb
                    })
                except Exception as e:
                    print(f"❌ Error reading metadata for {run_dir.name}: {e}")
    
    if not runs:
        print(f"❌ No valid runs found for environment {env_name}")
        return []
    
    # Sort by last update
    runs.sort(key=lambda x: x['last_update'], reverse=True)
    
    print(f"\n📋 Available runs for {env_name}:")
    print("-" * 140)
    print(f"{'Run Name':<20} {'Episodes':<10} {'Best Reward':<12} {'Model Acc':<10} {'Dream Qual':<10} "
          f"{'Iterations':<12} {'Size (MB)':<10} {'Last Update':<20}")
    print("-" * 140)
    
    for run in runs:
        episodes = str(run['episodes']) if run['episodes'] != 'Unknown' else 'Unknown'
        best_reward = f"{run['best_reward']:.2f}" if run['best_reward'] != 'Unknown' else 'Unknown'
        model_acc = f"{run['env_model_accuracy']:.3f}" if run['env_model_accuracy'] != 'Unknown' else 'Unknown'
        dream_qual = f"{run['dream_quality']:.3f}" if run['dream_quality'] != 'Unknown' else 'Unknown'
        iterations = str(run['training_iterations']) if run['training_iterations'] != 'Unknown' else 'Unknown'
        size = f"{run['size_mb']:.1f}" if run['size_mb'] else 'Unknown'
        last_update = run['last_update'][:19] if run['last_update'] != 'Unknown' else 'Unknown'
        
        print(f"{run['name']:<20} {episodes:<10} {best_reward:<12} {model_acc:<10} {dream_qual:<10} "
              f"{iterations:<12} {size:<10} {last_update:<20}")
    
    print("-" * 140)
    print(f"Total runs: {len(runs)}")
    total_size = sum(run['size_mb'] for run in runs if isinstance(run['size_mb'], float))
    print(f"Total size: {total_size:.1f} MB")
    
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Enhanced Symphony 3.0 Agent with FeedForward Transformer - Production Ready',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new training
  python train.py --env LunarLanderContinuous-v3
  
  # Resume from latest run
  python train.py --env LunarLanderContinuous-v3 --resume latest
  
  # Resume from specific run
  python train.py --env LunarLanderContinuous-v3 --resume run_20240802_143022
  
  # List available runs
  python train.py --env LunarLanderContinuous-v3 --list-runs
  
  # Evaluate latest run
  python train.py --env LunarLanderContinuous-v3 --evaluate latest --render
  
  # Train with episode limit and less frequent logging
  python train.py --env BipedalWalker-v3 --episodes 5000 --log-freq 10
  
  # Train without exploration phase
  python train.py --env HalfCheetah-v3 --no-exploration
        """
    )
    
    parser.add_argument("--env", default="LunarLanderContinuous-v3", 
                       help="Environment name")
    parser.add_argument("--episodes", type=int, default=None,
                       help="Number of training episodes (None = infinite)")
    parser.add_argument("--models-dir", default="./models",
                       help="Base models directory")
    parser.add_argument("--log-freq", type=int, default=1,
                       help="Logging frequency (1=every episode, 10=every 10 episodes)")
    parser.add_argument("--no-exploration", action="store_true",
                       help="Skip exploration phase")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from run (use 'latest' for most recent)")
    parser.add_argument("--evaluate", type=str, default=None,
                       help="Evaluate specific run (use 'latest' for most recent)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render during evaluation")
    parser.add_argument("--list-runs", action="store_true",
                       help="List all available runs for the environment")
    
    args = parser.parse_args()
    
    try:
        if args.list_runs:
            list_runs(args.models_dir, args.env)
        elif args.evaluate:
            # Find run directory for evaluation
            env_dir = Path(args.models_dir) / args.env
            if args.evaluate == "latest":
                run_dirs = [d for d in env_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
                if not run_dirs:
                    print(f"❌ No runs found for {args.env}")
                    sys.exit(1)
                run_dirs.sort(key=lambda x: x.stat().st_mtime)
                run_dir = run_dirs[-1]
            else:
                run_dir = env_dir / args.evaluate
                if not run_dir.exists():
                    print(f"❌ Run {args.evaluate} not found")
                    sys.exit(1)
            
            evaluate_agent_from_run(run_dir, args.eval_episodes, args.render)
        else:
            print(f"🚀 Starting Enhanced Symphony 3.0 training on {args.env}")
            if args.episodes:
                print(f"📊 Training for {args.episodes} episodes")
            else:
                print("🔄 Training indefinitely until stopped")
            
            enable_exploration = not args.no_exploration
            
            rewards = train(
                env_name=args.env, 
                max_episodes=args.episodes,
                models_dir=args.models_dir,
                enable_exploration=enable_exploration,
                resume_run=args.resume,
                log_frequency=args.log_freq
            )
            
            print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
        print("💾 Training state has been saved and can be resumed.")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print("💾 If training was in progress, state should be saved and resumable.")
        sys.exit(1)
