"""
Training Visualization Script for Run-Based Directory Structure

This script provides comprehensive training analytics for RL training
including episode rewards, learning curves, performance statistics, and model tracking.

Updated to work with the run-based directory structure: models/{env}/run_timestamp/

Features:
- Compatible with run-based directory structure
- Automatic latest run detection or specific run selection
- Best model performance tracking from training metadata
- Multiple visualization panels with customizable smoothing
- Training statistics summary
- Support for different environments

Usage:
    python visualise_training.py --env LunarLanderContinuous-v3                    # Latest run
    python visualise_training.py --env LunarLanderContinuous-v3 --run latest      # Latest run (explicit)
    python visualise_training.py --env BipedalWalker-v3 --run run_20250804_150000 # Specific run
    python visualise_training.py --env HalfCheetah-v3 --window 50                 # Custom smoothing
"""

import argparse
import json
import os
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def find_training_files(models_dir: Path, env_name: str, run_name: Optional[str] = None) -> Dict[str, Path]:
    """
    Find training files for the specified environment and run.
    Updated to work with run-based directory structure.
    
    Args:
        models_dir: Directory containing saved models
        env_name: Environment name (e.g., 'LunarLanderContinuous-v3')
        run_name: Specific run name, 'latest', or None for latest
        
    Returns:
        Dictionary with paths to all relevant files
        
    Raises:
        SystemExit: If no training files found
    """
    env_dir = models_dir / env_name
    
    if not env_dir.exists():
        available_envs = [d.name for d in models_dir.iterdir() if d.is_dir()]
        raise SystemExit(
            f"No training data found for environment '{env_name}'.\n"
            f"Available environments: {available_envs if available_envs else 'None'}\n"
            f"Looking in directory: {models_dir}"
        )
    
    # Find run directory
    if run_name is None or run_name == "latest":
        # Find latest run based on modification time
        run_dirs = [d for d in env_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        if not run_dirs:
            raise SystemExit(f"No runs found for environment '{env_name}'")
        run_dirs.sort(key=lambda x: x.stat().st_mtime)
        run_dir = run_dirs[-1]
        print(f"📂 Using latest run: {run_dir.name}")
    else:
        # Specific run name
        run_dir = env_dir / run_name
        if not run_dir.exists():
            available_runs = [d.name for d in env_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
            raise SystemExit(
                f"Run '{run_name}' not found.\n"
                f"Available runs: {available_runs if available_runs else 'None'}"
            )
        print(f"📂 Using specified run: {run_dir.name}")
    
    # Build file paths according to your current train.py structure
    files = {
        'run_dir': run_dir,
        'rewards_file': run_dir / 'episode_rewards.json',           # Updated filename
        'best_model': run_dir / 'best_model.pth',                  # Updated filename
        'final_model': run_dir / 'final_model.pth',                # Updated filename
        'latest_model': run_dir / 'latest.pth',                    # Add latest model
        'training_metadata': run_dir / 'training_metadata.json',   # Add metadata file
        'run_summary': run_dir / 'run_summary.json',              # Add run summary
        'checkpoints': list(run_dir.glob('checkpoint_ep*.pth')),   # Updated pattern
    }
    
    # Check if rewards file exists
    if not files['rewards_file'].exists():
        raise SystemExit(
            f"No episode rewards found in {run_dir}.\n"
            f"Expected file: episode_rewards.json"
        )
    
    # Sort checkpoints by episode number
    files['checkpoints'].sort(key=lambda x: int(x.stem.split('_ep')[-1]))
    
    # Find latest checkpoint
    if files['checkpoints']:
        files['latest_checkpoint'] = files['checkpoints'][-1]
    else:
        files['latest_checkpoint'] = None
    
    print(f"Found training data for {env_name}:")
    print(f"  Run directory: {run_dir.name}")
    print(f"  Rewards file: {files['rewards_file'].name}")
    print(f"  Training metadata: {'found' if files['training_metadata'].exists() else 'not found'}")
    print(f"  Checkpoints: {len(files['checkpoints'])} files")
    if files['best_model'].exists():
        print(f"  Best model: {files['best_model'].name}")
    if files['latest_checkpoint']:
        print(f"  Latest checkpoint: {files['latest_checkpoint'].name}")
    
    return files


def load_data(files: Dict[str, Path]) -> Dict[str, Any]:
    """
    Load training data from JSON rewards and PyTorch checkpoints.
    Updated to work with your current file structure.
    
    Args:
        files: Dictionary of file paths from find_training_files()
        
    Returns:
        Dictionary containing training data and metadata
    """
    # Load episode rewards
    print(f"Loading episode rewards from {files['rewards_file'].name}...")
    with files['rewards_file'].open('r') as f:
        episode_rewards = json.load(f)
    
    # Load metadata from training_metadata.json
    metadata = {}
    if files['training_metadata'].exists():
        print(f"Loading training metadata from {files['training_metadata'].name}...")
        with files['training_metadata'].open('r') as f:
            training_metadata = json.load(f)
            metadata.update({
                'env_name': training_metadata.get('env_name', 'Unknown'),
                'best_avg_reward': training_metadata.get('best_avg_reward', float('-inf')),
                'best_episode': training_metadata.get('best_episode', -1),
                'current_episode': training_metadata.get('current_episode', len(episode_rewards) - 1),
                'total_episodes': training_metadata.get('total_episodes', len(episode_rewards)),
                'last_update': training_metadata.get('last_update', 'Unknown')
            })
    
    # Load run summary if available
    if files['run_summary'].exists():
        print(f"Loading run summary from {files['run_summary'].name}...")
        with files['run_summary'].open('r') as f:
            run_summary = json.load(f)
            metadata.update({
                'final_average_reward': run_summary.get('final_average_reward', 0),
                'training_completed': run_summary.get('training_completed', 'Unknown')
            })
    
    # Try to load config from latest checkpoint
    if files['latest_checkpoint'] and files['latest_checkpoint'].exists():
        print(f"Loading config from {files['latest_checkpoint'].name}...")
        try:
            checkpoint = torch.load(files['latest_checkpoint'], map_location='cpu')
            if 'config' in checkpoint:
                metadata['config'] = checkpoint['config']
            if 'total_it' in checkpoint:
                metadata['total_iterations'] = checkpoint['total_it']
        except Exception as e:
            print(f"Warning: Could not load checkpoint metadata: {e}")
            metadata['config'] = {}
    
    # Calculate best model info from episode rewards if not in metadata
    if metadata.get('best_avg_reward', float('-inf')) == float('-inf'):
        best_episode = -1
        best_average_reward = float('-inf')
        
        if len(episode_rewards) >= 100:
            for i in range(99, len(episode_rewards)):
                avg_reward = np.mean(episode_rewards[i-99:i+1])
                if avg_reward > best_average_reward:
                    best_average_reward = avg_reward
                    best_episode = i
            
            metadata['best_avg_reward'] = best_average_reward
            metadata['best_episode'] = best_episode
    
    # Ensure required metadata exists
    metadata.setdefault('env_name', 'Unknown')
    metadata.setdefault('config', {})
    
    # Estimate episode lengths (your training doesn't save this)
    env_name = metadata.get('env_name', 'Unknown')
    estimated_lengths = estimate_episode_lengths(env_name, len(episode_rewards))
    
    return {
        'total_rewards': episode_rewards,
        'total_steps': estimated_lengths,
        'best_average_reward': metadata.get('best_avg_reward', float('-inf')),
        'best_episode': metadata.get('best_episode', -1),
        'current_episode': metadata.get('current_episode', len(episode_rewards) - 1),
        'env_name': metadata.get('env_name', 'Unknown'),
        'metadata': metadata,
        'num_checkpoints': len(files['checkpoints']),
        'run_directory': str(files['run_dir'])
    }


def estimate_episode_lengths(env_name: str, num_episodes: int) -> List[int]:
    """
    Estimate episode lengths for environments (since training doesn't save this).
    
    Args:
        env_name: Environment name
        num_episodes: Number of episodes to generate estimates for
        
    Returns:
        List of estimated episode lengths
    """
    # Typical episode length ranges for common environments
    length_ranges = {
        'LunarLanderContinuous-v3': (100, 400),
        'LunarLander-v2': (100, 400),
        'BipedalWalker-v3': (300, 1600),
        'BipedalWalkerHardcore-v3': (300, 1600),
        'HalfCheetah-v3': (1000, 1000),  # Fixed length
        'HalfCheetah-v4': (1000, 1000),  # Fixed length
        'Ant-v3': (1000, 1000),  # Fixed length
        'Ant-v4': (1000, 1000),  # Fixed length
        'Humanoid-v3': (1000, 1000),  # Fixed length
        'Humanoid-v4': (1000, 1000),  # Fixed length
        'Walker2d-v3': (1000, 1000),  # Fixed length
        'Walker2d-v4': (1000, 1000),  # Fixed length
    }
    
    # Default range for unknown environments
    default_range = (200, 800)
    
    # Get range for this environment
    length_range = length_ranges.get(env_name, default_range)
    min_len, max_len = length_range
    
    # Generate realistic episode lengths with some variation
    np.random.seed(42)  # For reproducibility
    
    if min_len == max_len:
        # Fixed length environments
        lengths = [min_len] * num_episodes
    else:
        # Variable length environments - start lower, gradually improve
        lengths = []
        for i in range(num_episodes):
            # Early episodes tend to be shorter (worse performance)
            # Later episodes tend to be longer (better performance)
            progress = min(1.0, i / 500.0)  # Reach max performance by episode 500
            
            # Linear interpolation with some randomness
            base_length = min_len + (max_len - min_len) * progress
            noise = np.random.normal(0, (max_len - min_len) * 0.1)
            
            length = int(max(min_len, min(max_len, base_length + noise)))
            lengths.append(length)
    
    return lengths


def rolling_mean(x: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Calculate rolling mean with proper edge handling.
    
    Args:
        x: Input array
        window: Rolling window size
        
    Returns:
        Array of rolling means (same length as input, with NaN for insufficient data)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size < window:
        return np.full_like(x, np.nan)
    
    # Calculate rolling mean
    result = np.full_like(x, np.nan)
    
    # Calculate rolling mean for positions where we have enough data
    for i in range(window-1, len(x)):
        result[i] = np.mean(x[i-window+1:i+1])
    
    return result


def calculate_training_stats(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate comprehensive training statistics.
    
    Args:
        data: Training data dictionary
        
    Returns:
        Dictionary of calculated statistics
    """
    rewards = np.asarray(data["total_rewards"], dtype=np.float32)
    steps = np.asarray(data["total_steps"], dtype=np.float32)
    
    stats = {}
    
    if len(rewards) > 0:
        stats['total_episodes'] = len(rewards)
        stats['latest_reward'] = rewards[-1]
        stats['max_reward'] = np.max(rewards)
        stats['min_reward'] = np.min(rewards)
        stats['mean_reward'] = np.mean(rewards)
        stats['std_reward'] = np.std(rewards)
        
        # Recent performance (last 100 episodes or all if less than 100)
        recent_count = min(100, len(rewards))
        recent_rewards = rewards[-recent_count:]
        stats['recent_mean_reward'] = np.mean(recent_rewards)
        stats['recent_std_reward'] = np.std(recent_rewards)
        stats['recent_count'] = recent_count
        
        # Learning progress indicators
        if len(rewards) >= 200:
            early_mean = np.mean(rewards[:100])
            late_mean = np.mean(rewards[-100:])
            stats['improvement'] = late_mean - early_mean
        else:
            stats['improvement'] = 0.0
    
    if len(steps) > 0:
        stats['mean_episode_length'] = np.mean(steps)
        stats['max_episode_length'] = np.max(steps)
        stats['min_episode_length'] = np.min(steps)
        
        recent_count = min(100, len(steps))
        stats['recent_mean_length'] = np.mean(steps[-recent_count:])
    
    # Best model info
    stats['best_average_reward'] = data.get('best_average_reward', float('-inf'))
    stats['best_episode'] = data.get('best_episode', -1)
    stats['current_episode'] = data.get('current_episode', len(rewards))
    stats['num_checkpoints'] = data.get('num_checkpoints', 0)
    
    return stats


def create_training_plots(data: Dict[str, Any], window: int = 100) -> plt.Figure:
    """
    Create comprehensive training visualization plots.
    
    Args:
        data: Training data dictionary
        window: Rolling mean window size
        
    Returns:
        Matplotlib figure object
    """
    rewards = np.asarray(data["total_rewards"], dtype=np.float32)
    steps = np.asarray(data["total_steps"], dtype=np.float32)
    
    # Calculate rolling means
    rewards_smooth = rolling_mean(rewards, window)
    steps_smooth = rolling_mean(steps, window)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    
    # Enhanced title with info
    env_name = data.get('env_name', 'Unknown Environment')
    config = data.get('metadata', {}).get('config', {})
    dream_freq = config.get('dream_frequency', 'N/A')
    run_dir = Path(data.get('run_directory', 'Unknown'))
    
    fig.suptitle(
        f"Training Progress - {env_name}\n"
        f"Episodes: {len(rewards)} | "
        f"Best Avg Reward: {data.get('best_average_reward', 'N/A'):.2f} "
        f"(Episode {data.get('best_episode', 'N/A')}) | "
        f"Dream Frequency: {dream_freq} | "
        f"Run: {run_dir.name}",
        fontsize=14, 
        weight="bold"
    )
    
    episodes = np.arange(len(rewards))
    
    # 1) Episode Returns (Raw + Smoothed)
    axs[0, 0].plot(episodes, rewards, alpha=0.3, color="#1f77b4", linewidth=0.5, label="Episode rewards")
    valid_smooth = ~np.isnan(rewards_smooth)
    if np.any(valid_smooth):
        axs[0, 0].plot(
            episodes[valid_smooth], 
            rewards_smooth[valid_smooth], 
            color="#ff7f0e", 
            linewidth=2.5, 
            label=f"{window}-episode mean"
        )
    
    # Mark best episode if available
    best_ep = data.get('best_episode', -1)
    if best_ep >= 0 and best_ep < len(rewards):
        axs[0, 0].axvline(x=best_ep, color='red', linestyle='--', alpha=0.7, 
                         label=f'Best model (ep {best_ep})')
        axs[0, 0].scatter(best_ep, rewards[best_ep], color='red', s=50, zorder=5)
    
    # Mark exploration phase end (assume first 100 episodes)
    exploration_episodes = config.get('exploration_episodes', 100)
    if exploration_episodes > 0 and exploration_episodes < len(rewards):
        axs[0, 0].axvline(x=exploration_episodes, color='green', linestyle=':', alpha=0.5, 
                         label=f'Training phase (ep {exploration_episodes})')
    
    axs[0, 0].set_ylabel("Episode Return")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()
    axs[0, 0].set_title("Episode Returns (Raw + Smoothed)")
    
    # 2) Learning Curve (Smoothed only, with confidence interval)
    if np.any(valid_smooth):
        axs[0, 1].plot(
            episodes[valid_smooth], 
            rewards_smooth[valid_smooth], 
            color="#2ca02c", 
            linewidth=3,
            label=f"{window}-episode mean"
        )
        
        # Add confidence interval if we have enough data
        if len(rewards) > window * 2:
            # Calculate rolling std for confidence interval
            rolling_std = []
            for i in range(window-1, len(rewards)):
                rolling_std.append(np.std(rewards[i-window+1:i+1]))
            rolling_std = np.array(rolling_std)
            
            valid_indices = episodes[valid_smooth]
            upper = rewards_smooth[valid_smooth] + rolling_std
            lower = rewards_smooth[valid_smooth] - rolling_std
            
            axs[0, 1].fill_between(valid_indices, upper, lower, alpha=0.2, color="#2ca02c")
    
    if best_ep >= 0 and best_ep < len(rewards):
        axs[0, 1].axvline(x=best_ep, color='red', linestyle='--', alpha=0.7)
        if best_ep < len(rewards_smooth) and not np.isnan(rewards_smooth[best_ep]):
            axs[0, 1].scatter(best_ep, rewards_smooth[best_ep], 
                             color='red', s=100, zorder=5, 
                             label=f'Best: {data.get("best_average_reward", 0):.2f}')
    
    axs[0, 1].set_ylabel("Episode Return (Smoothed)")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()
    axs[0, 1].set_title(f"Learning Curve ({window}-episode moving average)")
    
    # 3) Episode Length (Estimated)
    episodes_steps = np.arange(len(steps))
    axs[1, 0].plot(episodes_steps, steps, alpha=0.4, color="#d62728", linewidth=0.8, 
                   label="Estimated length")
    valid_smooth_steps = ~np.isnan(steps_smooth)
    if np.any(valid_smooth_steps):
        axs[1, 0].plot(
            episodes_steps[valid_smooth_steps], 
            steps_smooth[valid_smooth_steps], 
            color="#9467bd", 
            linewidth=2.5, 
            label=f"{window}-episode mean"
        )
    
    axs[1, 0].set_ylabel("Episode Length (Steps)")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()
    axs[1, 0].set_title("Episode Length (Estimated)")
    
    # Add note about estimation
    axs[1, 0].text(0.02, 0.98, "Note: Lengths estimated\n(not recorded in training)", 
                   transform=axs[1, 0].transAxes, fontsize=8, 
                   verticalalignment='top', alpha=0.7,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    # 4) Performance Analysis
    recent_rewards = rewards[-min(100, len(rewards)):]
    if len(recent_rewards) > 5:
        # Performance distribution
        axs[1, 1].hist(recent_rewards, bins=min(20, len(recent_rewards)//2), 
                       alpha=0.7, color="#17becf", edgecolor='black', density=True)
        
        # Statistical lines
        mean_val = np.mean(recent_rewards)
        median_val = np.median(recent_rewards)
        std_val = np.std(recent_rewards)
        
        axs[1, 1].axvline(mean_val, color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {mean_val:.1f}')
        axs[1, 1].axvline(median_val, color='orange', linestyle='--', linewidth=2,
                         label=f'Median: {median_val:.1f}')
        axs[1, 1].axvline(mean_val + std_val, color='red', linestyle=':', alpha=0.7,
                         label=f'Mean ± σ: {std_val:.1f}')
        axs[1, 1].axvline(mean_val - std_val, color='red', linestyle=':', alpha=0.7)
        
        axs[1, 1].set_xlabel("Episode Return")
        axs[1, 1].set_ylabel("Density")
        axs[1, 1].legend()
        axs[1, 1].set_title(f"Recent Performance Distribution\n(Last {len(recent_rewards)} episodes)")
    else:
        axs[1, 1].text(0.5, 0.5, "Insufficient data\nfor distribution analysis\n(need >5 episodes)", 
                      ha='center', va='center', transform=axs[1, 1].transAxes, 
                      fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        axs[1, 1].set_title("Performance Distribution")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def print_training_summary(stats: Dict[str, float], files: Dict[str, Path], data: Dict[str, Any]):
    """Print a comprehensive training summary to console."""
    print("\n" + "="*70)
    print(f"TRAINING SUMMARY")
    print("="*70)
    
    env_name = data.get('env_name', 'Unknown')
    config = data.get('metadata', {}).get('config', {})
    
    print(f"🌍 ENVIRONMENT: {env_name}")
    print(f"📁 RUN: {files['run_dir'].name}")
    
    print(f"\n⚙️  ALGORITHM CONFIG:")
    if config:
        print(f"   Dream Frequency: {config.get('dream_frequency', 'N/A')} episodes")
        print(f"   Dream Horizon: {config.get('dream_horizon', 'N/A')} steps")
        print(f"   Exploration Episodes: {config.get('exploration_episodes', 'N/A')}")
        print(f"   Actor LR: {config.get('actor_lr', 'N/A')}")
        print(f"   Critic LR: {config.get('critic_lr', 'N/A')}")
        print(f"   Env Model LR: {config.get('env_model_lr', 'N/A')}")
    else:
        print("   Configuration not available")
    
    print(f"\n📊 TRAINING PROGRESS:")
    print(f"   Total Episodes: {stats.get('total_episodes', 0)}")
    print(f"   Current Episode: {stats.get('current_episode', 0)}")
    print(f"   Saved Checkpoints: {stats.get('num_checkpoints', 0)}")
    
    print(f"\n🎯 EPISODE RETURNS:")
    print(f"   Latest Return: {stats.get('latest_reward', 0):.2f}")
    print(f"   Overall Mean: {stats.get('mean_reward', 0):.2f} ± {stats.get('std_reward', 0):.2f}")
    print(f"   Recent Mean (last {stats.get('recent_count', 0)}): {stats.get('recent_mean_reward', 0):.2f} ± {stats.get('recent_std_reward', 0):.2f}")
    print(f"   Range: [{stats.get('min_reward', 0):.2f}, {stats.get('max_reward', 0):.2f}]")
    
    improvement = stats.get('improvement', 0)
    if abs(improvement) > 0.1:
        direction = "📈" if improvement > 0 else "📉"
        print(f"   Learning Progress: {direction} {improvement:+.2f} (first 100 vs last 100)")
    
    print(f"\n📏 EPISODE LENGTHS (ESTIMATED):")
    print(f"   Mean Length: {stats.get('mean_episode_length', 0):.1f} steps")
    print(f"   Recent Mean: {stats.get('recent_mean_length', 0):.1f} steps")
    print(f"   Range: [{stats.get('min_episode_length', 0)}, {stats.get('max_episode_length', 0)}] steps")
    
    print(f"\n🏆 BEST MODEL:")
    if stats.get('best_episode', -1) >= 0:
        print(f"   Best Average Reward: {stats.get('best_average_reward', 0):.2f}")
        print(f"   Achieved at Episode: {stats.get('best_episode', 0)}")
        if files['best_model'].exists():
            print(f"   Model saved as: {files['best_model'].name}")
    else:
        print(f"   No best model recorded yet (need 100+ episodes for averaging)")
    
    print(f"\n📁 FILES:")
    print(f"   Run directory: {files['run_dir']}")
    print(f"   Rewards data: {files['rewards_file'].name}")
    if files['training_metadata'].exists():
        print(f"   Training metadata: {files['training_metadata'].name}")
    if files['latest_checkpoint']:
        print(f"   Latest checkpoint: {files['latest_checkpoint'].name}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training progress for run-based directory structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualise_training.py --env LunarLanderContinuous-v3                    # Latest run
  python visualise_training.py --env LunarLanderContinuous-v3 --run latest      # Latest run (explicit)
  python visualise_training.py --env BipedalWalker-v3 --run run_20250804_150000 # Specific run
  python visualise_training.py --env HalfCheetah-v3 --window 50 --save-only     # Custom smoothing, save only
        """
    )
    
    parser.add_argument(
        "--env",
        required=True,
        help="Environment name (e.g., LunarLanderContinuous-v3, BipedalWalker-v3)"
    )
    
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Directory containing saved models (default: ./models)"
    )
    
    parser.add_argument(
        "--run",
        help="Specific run name or 'latest' (default: latest)"
    )
    
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling mean window size (default: 100)"
    )
    
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save plot without showing interactive window"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing training summary"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Custom output directory for plots (default: same as run directory)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.window < 1:
        raise ValueError("Window size must be positive")
    
    # Setup directories
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise SystemExit(f"Models directory '{models_dir}' does not exist.")
    
    # Find training files
    try:
        files = find_training_files(models_dir, args.env, args.run)
    except SystemExit as e:
        raise e
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else files['run_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    try:
        data = load_data(files)
    except Exception as e:
        raise SystemExit(f"Error loading training data: {e}")
    
    # Calculate statistics
    stats = calculate_training_stats(data)
    
    # Print summary
    if not args.no_summary:
        print_training_summary(stats, files, data)
    
    # Create plots
    print(f"\nGenerating plots with {args.window}-episode smoothing window...")
    fig = create_training_plots(data, args.window)
    
    # Save plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"training_curves_{args.env}_{files['run_dir'].name}_{timestamp}.png"
    plot_path = output_dir / plot_filename
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to: {plot_path}")
    
    # Also save as latest
    latest_plot_path = output_dir / f"training_curves_{args.env}_latest.png"
    fig.savefig(latest_plot_path, dpi=300, bbox_inches='tight')
    
    # Show interactive plot
    if not args.save_only:
        print("Showing interactive plot... (close window to exit)")
        plt.show()
    else:
        plt.close(fig)
        print("Plot saved successfully (interactive display skipped)")


if __name__ == "__main__":
    main()
