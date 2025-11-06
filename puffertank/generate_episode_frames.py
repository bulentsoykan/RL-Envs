"""
Generate Episode Frames for TankSim

This script generates a sequence of frames showing a TankSim episode.
Perfect for creating animations or reviewing gameplay.
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from tank_sim_env import TankSimEnv
from visualize_tanksim import TankSimVisualizer, TankSimPolicy


def generate_episode(seed=None, max_steps=100, use_policy=False, output_dir="/tmp", prefix="tanksim"):
    """
    Generate frames for an episode.

    Args:
        seed: Random seed for reproducibility
        max_steps: Maximum number of steps
        use_policy: Whether to use policy or random actions
        output_dir: Directory to save frames
        prefix: Prefix for frame filenames
    """

    print("="*60)
    print("GENERATING TANKSIM EPISODE FRAMES")
    print("="*60)
    print(f"Seed: {seed}")
    print(f"Max steps: {max_steps}")
    print(f"Using policy: {use_policy}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")

    # Create environment
    env = TankSimEnv(grid_size=24, num_obstacles=15)
    visualizer = TankSimVisualizer(env)

    # Create policy if needed
    policy = None
    if use_policy:
        print("Loading policy...")
        policy = TankSimPolicy()

    # Reset environment
    obs, info = env.reset(seed=seed)

    # Save initial frame
    frame_path = f"{output_dir}/{prefix}_frame_000.png"
    visualizer.save_frame(obs, info, {}, 0, frame_path)

    print(f"Frame 000 saved")

    step = 0
    done = False

    while not done and step < max_steps:
        # Get actions
        if policy is not None:
            obs_list = [obs[i] for i in sorted(obs.keys())]
            obs_tensor = torch.FloatTensor(np.array(obs_list))
            with torch.no_grad():
                action_logits, _ = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=action_logits)
                actions = dist.sample()
            actions_dict = {agent_id: int(actions[idx].item())
                          for idx, agent_id in enumerate(sorted(obs.keys()))}
        else:
            # Random actions with bias towards movement and firing
            actions_dict = {agent_id: np.random.choice([1, 2, 3, 4], p=[0.3, 0.2, 0.2, 0.3])
                          for agent_id in obs.keys()}

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions_dict)
        step += 1

        # Save frame
        frame_path = f"{output_dir}/{prefix}_frame_{step:03d}.png"
        visualizer.save_frame(obs, info, rewards, step, frame_path)

        print(f"Frame {step:03d} saved - T0: {info['team0_alive']} alive, T1: {info['team1_alive']} alive", end='\r')

        done = terminated.get("__all__", False)

    print("\n")
    print("="*60)
    print("EPISODE COMPLETE")
    print("="*60)
    print(f"Total frames: {step + 1}")
    print(f"Team 0 alive: {info['team0_alive']}")
    print(f"Team 1 alive: {info['team1_alive']}")

    if info['team0_alive'] == 0:
        print("🏆 Winner: Team 1 (Red)")
    elif info['team1_alive'] == 0:
        print("🏆 Winner: Team 0 (Blue)")
    else:
        print("⚔️  Draw - both teams survived")

    print(f"\nFrames saved to: {output_dir}/{prefix}_frame_*.png")
    print("\nTo create an animation:")
    print(f"  GIF:   convert -delay 10 -loop 0 {output_dir}/{prefix}_frame_*.png {prefix}.gif")
    print(f"  MP4:   ffmpeg -framerate 10 -pattern_type glob -i '{output_dir}/{prefix}_frame_*.png' -c:v libx264 {prefix}.mp4")
    print("="*60)

    return step + 1


def main():
    parser = argparse.ArgumentParser(description='Generate TankSim episode frames')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps')
    parser.add_argument('--policy', action='store_true', help='Use policy instead of random actions')
    parser.add_argument('--output', type=str, default='/tmp', help='Output directory')
    parser.add_argument('--prefix', type=str, default='tanksim', help='Filename prefix')

    args = parser.parse_args()

    generate_episode(
        seed=args.seed,
        max_steps=args.max_steps,
        use_policy=args.policy,
        output_dir=args.output,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()
