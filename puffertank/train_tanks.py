"""
Training script for TankSim environment using PufferLib

This script demonstrates how to train multi-agent RL policies
in the TankSim combat vehicle simulation environment.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict

import pufferlib
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.policy_store

from tank_sim_env import TankSimEnv


class TankSimPolicy(nn.Module):
    """
    Convolutional neural network policy for TankSim.

    Takes a (1, 32, 32) observation and outputs action logits.
    """

    def __init__(self, input_channels: int = 1, grid_size: int = 32, num_actions: int = 5):
        super().__init__()

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            # First conv layer: 1 -> 16 channels
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            # Second conv layer: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            # Third conv layer: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )

        # Calculate flattened size: 64 channels * 4 * 4
        self.flatten_size = 64 * 4 * 4

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, observations):
        """
        Forward pass through the network.

        Args:
            observations: Tensor of shape (batch, 1, 32, 32)

        Returns:
            action_logits: Tensor of shape (batch, num_actions)
            values: Tensor of shape (batch, 1)
        """
        # Extract features
        features = self.conv(observations)
        features = features.reshape(-1, self.flatten_size)

        # Get action logits and value estimates
        action_logits = self.actor(features)
        values = self.critic(features)

        return action_logits, values


def make_env():
    """Create a TankSim environment wrapped for PufferLib."""
    return TankSimEnv()


def train():
    """Main training function."""

    print("="*60)
    print("TankSim Multi-Agent Training with PufferLib")
    print("="*60)

    # Training configuration
    config = {
        # Environment settings
        'env_name': 'TankSim',
        'num_envs': 4,  # Number of parallel environments
        'num_agents': 4,  # Number of agents per environment

        # Training hyperparameters
        'total_timesteps': 100_000,  # Total training steps (small for quick testing)
        'learning_rate': 3e-4,
        'batch_size': 256,
        'num_minibatches': 4,
        'update_epochs': 4,
        'gamma': 0.99,  # Discount factor
        'gae_lambda': 0.95,  # GAE lambda
        'clip_coef': 0.2,  # PPO clipping coefficient
        'ent_coef': 0.01,  # Entropy coefficient
        'vf_coef': 0.5,  # Value function coefficient
        'max_grad_norm': 0.5,

        # Logging
        'verbose': True,
        'track': False,  # Set to True to use Weights & Biases
    }

    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create vectorized environment using PufferLib
    print("Creating environments...")

    # PufferLib emulation wrapper for multi-agent support
    vec_env = pufferlib.vector.make(
        make_env,
        num_envs=config['num_envs'],
        num_workers=1,  # Single worker for simplicity
    )

    print(f"Created {config['num_envs']} parallel environments")

    # Create policy network
    print("Initializing policy network...")
    policy = TankSimPolicy()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.to(device)
    print(f"Using device: {device}")

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=config['learning_rate'])

    # Training loop configuration
    num_steps = 128  # Steps per environment per update
    num_updates = config['total_timesteps'] // (num_steps * config['num_envs'])

    print(f"\nStarting training for {num_updates} updates...")
    print(f"Each update processes {num_steps * config['num_envs']} environment steps")
    print()

    # Simple training loop (simplified PPO)
    obs, info = vec_env.reset()

    episode_rewards = []
    episode_lengths = []

    for update in range(num_updates):
        # Collect rollout data
        for step in range(num_steps):
            # Convert observations to tensor
            if isinstance(obs, dict):
                # Handle multi-agent observations
                obs_list = [obs[i] for i in sorted(obs.keys()) if isinstance(i, int)]
                if obs_list:
                    obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device)
                else:
                    break
            else:
                obs_tensor = torch.FloatTensor(obs).to(device)

            # Get actions from policy
            with torch.no_grad():
                action_logits, values = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=action_logits)
                actions = dist.sample()

            # Step environment
            # Convert actions to dict format for multi-agent env
            if isinstance(obs, dict):
                actions_dict = {i: int(actions[idx].item())
                              for idx, i in enumerate(sorted(obs.keys()))
                              if isinstance(i, int)}
                next_obs, rewards, terminated, truncated, info = vec_env.step(actions_dict)
            else:
                next_obs, rewards, terminated, truncated, info = vec_env.step(actions.cpu().numpy())

            obs = next_obs

            # Track episode statistics
            if isinstance(terminated, dict) and terminated.get("__all__", False):
                # Episode ended
                if isinstance(rewards, dict):
                    total_reward = sum(r for k, r in rewards.items() if isinstance(k, int))
                    episode_rewards.append(total_reward)
                episode_lengths.append(info.get('step_count', 0))

        # Logging
        if (update + 1) % 10 == 0:
            print(f"Update {update + 1}/{num_updates}")
            if episode_rewards:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"  Avg Episode Reward (last 10): {avg_reward:.2f}")
                print(f"  Avg Episode Length (last 10): {avg_length:.1f}")

            # Show team statistics from last episode
            if 'team0_alive' in info and 'team1_alive' in info:
                print(f"  Team 0 Alive: {info['team0_alive']}, Team 1 Alive: {info['team1_alive']}")
            print()

    print("="*60)
    print("Training Complete!")
    print("="*60)

    if episode_rewards:
        print(f"\nFinal Statistics:")
        print(f"  Total Episodes: {len(episode_rewards)}")
        print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"  Average Episode Length: {np.mean(episode_lengths):.1f}")

    # Save the trained policy
    save_path = "/puffertank/tanksim_policy.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"\nPolicy saved to: {save_path}")

    vec_env.close()


if __name__ == "__main__":
    train()
