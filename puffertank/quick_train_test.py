"""
Quick test of the training script - runs a minimal version to verify it works
"""

import numpy as np
import torch
import torch.nn as nn
from tank_sim_env import TankSimEnv


print("="*60)
print("QUICK TRAINING SCRIPT TEST")
print("="*60)

# Test 1: Environment creation
print("\n1. Creating environment...")
env = TankSimEnv()
print("   ✓ Environment created")

# Test 2: Reset
print("\n2. Testing reset...")
obs, info = env.reset(seed=42)
print(f"   ✓ Reset successful, {len(obs)} agents active")

# Test 3: Simple policy network
print("\n3. Creating policy network...")
from train_tanks import TankSimPolicy
policy = TankSimPolicy()
print("   ✓ Policy network created")

# Test 4: Forward pass
print("\n4. Testing policy forward pass...")
obs_tensor = torch.FloatTensor(np.array([obs[i] for i in sorted(obs.keys())]))
print(f"   Input shape: {obs_tensor.shape}")

with torch.no_grad():
    action_logits, values = policy(obs_tensor)
    print(f"   Action logits shape: {action_logits.shape}")
    print(f"   Values shape: {values.shape}")
    print("   ✓ Forward pass successful")

# Test 5: Sample actions
print("\n5. Sampling actions from policy...")
dist = torch.distributions.Categorical(logits=action_logits)
actions = dist.sample()
print(f"   Sampled actions: {actions.tolist()}")
print("   ✓ Action sampling works")

# Test 6: Run a few steps
print("\n6. Running 10 environment steps...")
for i in range(10):
    actions_dict = {agent_id: int(actions[idx].item())
                   for idx, agent_id in enumerate(sorted(obs.keys()))}
    obs, rewards, terminated, truncated, info = env.step(actions_dict)

    if terminated.get("__all__", False):
        print(f"   Episode ended at step {i+1}")
        break

    # Get new actions
    obs_tensor = torch.FloatTensor(np.array([obs[i] for i in sorted(obs.keys())]))
    with torch.no_grad():
        action_logits, values = policy(obs_tensor)
        dist = torch.distributions.Categorical(logits=action_logits)
        actions = dist.sample()

print(f"   ✓ Completed {i+1} steps")
print(f"   Final rewards: {rewards}")

print("\n" + "="*60)
print("✓ ALL TRAINING COMPONENTS WORKING!")
print("="*60)
print("\nThe full training script should work correctly.")
print("Note: Full training requires PufferLib which is installed in the Docker container.")
