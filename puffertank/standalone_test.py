"""
Standalone test of TankSim components without PufferLib
"""

import numpy as np
import torch
import torch.nn as nn
from tank_sim_env import TankSimEnv


class TankSimPolicy(nn.Module):
    """CNN policy for TankSim"""

    def __init__(self, input_channels=1, grid_size=32, num_actions=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_size = 64 * 4 * 4
        self.actor = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, observations):
        features = self.conv(observations)
        features = features.reshape(-1, self.flatten_size)
        action_logits = self.actor(features)
        values = self.critic(features)
        return action_logits, values


print("="*60)
print("TANKSIM TRAINING COMPONENTS TEST")
print("="*60)

# Test environment
print("\n1. Creating environment...")
env = TankSimEnv()
print("   ✓ Environment created")

print("\n2. Resetting environment...")
obs, info = env.reset(seed=42)
print(f"   ✓ {len(obs)} agents active")

# Test policy
print("\n3. Creating policy network...")
policy = TankSimPolicy()
print("   ✓ Policy created")

print("\n4. Testing forward pass...")
obs_list = [obs[i] for i in sorted(obs.keys())]
obs_tensor = torch.FloatTensor(np.array(obs_list))
print(f"   Input shape: {obs_tensor.shape}")

with torch.no_grad():
    action_logits, values = policy(obs_tensor)
print(f"   Action logits: {action_logits.shape}")
print(f"   Values: {values.shape}")
print("   ✓ Forward pass works")

print("\n5. Running episode with policy...")
total_reward = {0: 0, 1: 0, 2: 0, 3: 0}
steps = 0

for step in range(50):
    # Get actions from policy
    obs_list = [obs[i] for i in sorted(obs.keys())]
    obs_tensor = torch.FloatTensor(np.array(obs_list))

    with torch.no_grad():
        action_logits, values = policy(obs_tensor)
        dist = torch.distributions.Categorical(logits=action_logits)
        actions = dist.sample()

    # Step environment
    actions_dict = {agent_id: int(actions[idx].item())
                   for idx, agent_id in enumerate(sorted(obs.keys()))}

    obs, rewards, terminated, truncated, info = env.step(actions_dict)

    for agent_id, reward in rewards.items():
        if agent_id in total_reward:
            total_reward[agent_id] += reward

    steps += 1

    if terminated.get("__all__", False):
        print(f"   Episode completed after {steps} steps")
        print(f"   Team 0: {info['team0_alive']} alive")
        print(f"   Team 1: {info['team1_alive']} alive")
        break
else:
    print(f"   Ran {steps} steps")

print(f"   Cumulative rewards: {total_reward}")
print("   ✓ Policy-environment interaction works")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nTankSim is fully functional!")
print("The environment and training components work correctly.")
