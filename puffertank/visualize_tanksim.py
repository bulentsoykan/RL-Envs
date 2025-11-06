"""
TankSim Visualization Script

This script provides visualization for the TankSim environment using matplotlib.
You can watch episodes play out in real-time or save them as images/GIFs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
from tank_sim_env import TankSimEnv


class TankSimPolicy(nn.Module):
    """Simple CNN policy for controlling tanks"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten_size = 64 * 4 * 4
        self.actor = nn.Sequential(nn.Linear(self.flatten_size, 128), nn.ReLU(), nn.Linear(128, 5))
        self.critic = nn.Sequential(nn.Linear(self.flatten_size, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs):
        features = self.conv(obs).reshape(-1, self.flatten_size)
        return self.actor(features), self.critic(features)


class TankSimVisualizer:
    """Visualizer for TankSim environment"""

    # Color scheme
    COLORS = {
        'empty': '#FFFFFF',
        'obstacle': '#2C3E50',
        'team0': '#3498DB',  # Blue
        'team1': '#E74C3C',  # Red
        'team0_light': '#AED6F1',
        'team1_light': '#F5B7B1',
        'grid': '#ECF0F1',
    }

    # Direction arrows
    ARROWS = {
        0: '↑',  # North
        1: '→',  # East
        2: '↓',  # South
        3: '←',  # West
    }

    def __init__(self, env):
        self.env = env
        self.grid_size = env.grid_size

    def render_frame(self, ax, obs, info, rewards=None, step_num=0):
        """Render a single frame of the environment"""
        ax.clear()
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Flip y-axis so (0,0) is top-left

        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color=self.COLORS['grid'], linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color=self.COLORS['grid'], linewidth=0.5, alpha=0.3)

        # Draw obstacles
        for obs_x, obs_y in self.env.obstacles:
            rect = patches.Rectangle(
                (obs_y - 0.4, obs_x - 0.4), 0.8, 0.8,
                linewidth=1, edgecolor=self.COLORS['obstacle'],
                facecolor=self.COLORS['obstacle'], alpha=0.8
            )
            ax.add_patch(rect)

        # Draw tanks
        for tank in info['tanks']:
            if not tank['alive']:
                continue

            x, y = tank['x'], tank['y']
            team = tank['team']
            orientation = tank['orientation']
            health = tank['health']

            # Tank color
            color = self.COLORS['team0'] if team == 0 else self.COLORS['team1']
            light_color = self.COLORS['team0_light'] if team == 0 else self.COLORS['team1_light']

            # Draw tank body
            circle = patches.Circle(
                (y, x), 0.35,
                linewidth=2, edgecolor=color,
                facecolor=light_color, alpha=0.9
            )
            ax.add_patch(circle)

            # Draw orientation arrow
            arrow = self.ARROWS[orientation]
            ax.text(y, x, arrow, fontsize=20, ha='center', va='center',
                   color=color, weight='bold')

            # Draw health bar
            health_width = 0.6
            health_height = 0.08
            health_x = y - health_width / 2
            health_y = x - 0.55

            # Background (gray)
            health_bg = patches.Rectangle(
                (health_x, health_y), health_width, health_height,
                linewidth=0.5, edgecolor='black', facecolor='gray', alpha=0.5
            )
            ax.add_patch(health_bg)

            # Health bar (colored)
            health_pct = health / self.env.initial_health
            health_bar = patches.Rectangle(
                (health_x, health_y), health_width * health_pct, health_height,
                linewidth=0, facecolor=color, alpha=0.9
            )
            ax.add_patch(health_bar)

            # Agent ID label
            ax.text(y, x + 0.6, f"A{tank['agent_id']}", fontsize=8,
                   ha='center', va='top', color=color, weight='bold')

        # Title with game info
        title = f"Step {step_num} | "
        title += f"Team 0 (Blue): {info['team0_alive']} alive | "
        title += f"Team 1 (Red): {info['team1_alive']} alive"

        if rewards:
            total_rewards = {0: 0, 1: 0}
            for tank in info['tanks']:
                if tank['agent_id'] in rewards:
                    total_rewards[tank['team']] += rewards[tank['agent_id']]
            title += f"\nRewards - T0: {total_rewards[0]:.2f}, T1: {total_rewards[1]:.2f}"

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS['team0'],
                  markersize=10, label='Team 0 (Blue)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS['team1'],
                  markersize=10, label='Team 1 (Red)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=self.COLORS['obstacle'],
                  markersize=10, label='Obstacle'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    def save_frame(self, obs, info, rewards, step_num, filename):
        """Save a single frame to file"""
        fig, ax = plt.subplots(figsize=(10, 10))
        self.render_frame(ax, obs, info, rewards, step_num)
        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved frame to {filename}")

    def run_episode_visual(self, policy=None, seed=None, max_steps=200, save_path=None):
        """Run an episode with visualization"""
        obs, info = self.env.reset(seed=seed)

        fig, ax = plt.subplots(figsize=(12, 10))
        plt.ion()  # Interactive mode

        step = 0
        done = False

        frames = []  # Store frames for potential GIF creation

        while not done and step < max_steps:
            # Get actions
            if policy is not None:
                # Use policy
                obs_list = [obs[i] for i in sorted(obs.keys())]
                obs_tensor = torch.FloatTensor(np.array(obs_list))
                with torch.no_grad():
                    action_logits, _ = policy(obs_tensor)
                    dist = torch.distributions.Categorical(logits=action_logits)
                    actions = dist.sample()
                actions_dict = {agent_id: int(actions[idx].item())
                              for idx, agent_id in enumerate(sorted(obs.keys()))}
            else:
                # Random actions with bias
                actions_dict = {agent_id: np.random.choice([1, 2, 3, 4], p=[0.3, 0.2, 0.2, 0.3])
                              for agent_id in obs.keys()}

            # Step environment
            obs, rewards, terminated, truncated, info = self.env.step(actions_dict)

            # Render frame
            self.render_frame(ax, obs, info, rewards, step)
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)  # Pause for animation effect

            # Save frame if path provided
            if save_path:
                frame_path = f"{save_path}_frame_{step:03d}.png"
                plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                frames.append(frame_path)

            step += 1
            done = terminated.get("__all__", False)

        # Final frame
        plt.pause(2)  # Hold final frame

        # Print summary
        print("\n" + "="*60)
        print("EPISODE SUMMARY")
        print("="*60)
        print(f"Total steps: {step}")
        print(f"Team 0 alive: {info['team0_alive']}")
        print(f"Team 1 alive: {info['team1_alive']}")

        if info['team0_alive'] == 0:
            print("🏆 Winner: Team 1 (Red)")
        elif info['team1_alive'] == 0:
            print("🏆 Winner: Team 0 (Blue)")
        else:
            print("Draw - both teams survived")
        print("="*60)

        plt.ioff()
        plt.show()

        return frames


def demo_random_episode():
    """Demo: Run episode with random actions"""
    print("\n" + "="*60)
    print("DEMO 1: Random Actions Episode")
    print("="*60)
    print("Running episode with random actions...")
    print("Close the window to continue.\n")

    env = TankSimEnv(grid_size=24, num_obstacles=15)  # Smaller for faster demo
    visualizer = TankSimVisualizer(env)

    visualizer.run_episode_visual(policy=None, seed=42, max_steps=100)


def demo_policy_episode():
    """Demo: Run episode with trained policy"""
    print("\n" + "="*60)
    print("DEMO 2: Policy-Controlled Episode")
    print("="*60)
    print("Running episode with CNN policy...")
    print("Close the window to continue.\n")

    env = TankSimEnv(grid_size=24, num_obstacles=15)
    policy = TankSimPolicy()
    visualizer = TankSimVisualizer(env)

    visualizer.run_episode_visual(policy=policy, seed=123, max_steps=100)


def demo_save_frames():
    """Demo: Save frames to files"""
    print("\n" + "="*60)
    print("DEMO 3: Save Episode Frames")
    print("="*60)
    print("Running episode and saving frames...")
    print("Frames will be saved to /tmp/tanksim_*\n")

    env = TankSimEnv(grid_size=20, num_obstacles=10)
    visualizer = TankSimVisualizer(env)

    frames = visualizer.run_episode_visual(
        policy=None,
        seed=999,
        max_steps=50,
        save_path="/tmp/tanksim"
    )

    print(f"\nSaved {len(frames)} frames!")
    print("You can use these to create a GIF with:")
    print("  convert -delay 10 /tmp/tanksim_frame_*.png tanksim.gif")


def main():
    """Main demo menu"""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  TANKSIM VISUALIZATION DEMO".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)

    print("\nAvailable demos:")
    print("  1. Random Actions Episode (watch tanks move randomly)")
    print("  2. Policy-Controlled Episode (watch AI-controlled tanks)")
    print("  3. Save Episode Frames (save images to disk)")
    print("  4. Quick Test (save single frame)")
    print("  5. Run All Demos")

    choice = input("\nSelect demo (1-5) or press Enter for demo 1: ").strip()

    if not choice:
        choice = "1"

    if choice == "1":
        demo_random_episode()
    elif choice == "2":
        demo_policy_episode()
    elif choice == "3":
        demo_save_frames()
    elif choice == "4":
        # Quick test - just save one frame
        env = TankSimEnv()
        obs, info = env.reset(seed=42)
        visualizer = TankSimVisualizer(env)
        visualizer.save_frame(obs, info, {}, 0, "/tmp/tanksim_test.png")
        print("\n✓ Test frame saved to /tmp/tanksim_test.png")
    elif choice == "5":
        demo_random_episode()
        demo_policy_episode()
        demo_save_frames()
    else:
        print("Invalid choice, running demo 1...")
        demo_random_episode()

    print("\n✅ Visualization demo complete!")


if __name__ == "__main__":
    main()
