"""
TankSim: A Multi-Agent Combat Vehicle Simulation Environment

A Gymnasium-compatible environment for training multi-agent reinforcement learning
agents in a 2D grid-based combat simulation.
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional


class TankSimEnv(gymnasium.Env):
    """
    A 2D grid-based multi-agent tank combat environment.

    Features:
    - 32x32 grid world
    - 2 teams with 2 tanks each (4 agents total)
    - Turn-based movement and combat
    - Projectile-based combat system
    - Team-based rewards
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    # Direction mappings: North, East, South, West
    DIRECTIONS = {
        0: (-1, 0),  # North (up)
        1: (0, 1),   # East (right)
        2: (1, 0),   # South (down)
        3: (0, -1)   # West (left)
    }

    # Action definitions
    ACTION_IDLE = 0
    ACTION_MOVE_FORWARD = 1
    ACTION_TURN_RIGHT = 2
    ACTION_TURN_LEFT = 3
    ACTION_FIRE = 4

    # Grid cell values for observations
    CELL_EMPTY = 0
    CELL_OBSTACLE = 1
    CELL_ALLY = 2
    CELL_ENEMY = 3

    def __init__(self, grid_size: int = 32, num_obstacles: int = 12,
                 team_size: int = 2, initial_health: int = 100):
        """
        Initialize the TankSim environment.

        Args:
            grid_size: Size of the square grid (default: 32)
            num_obstacles: Number of obstacles to place (default: 12)
            team_size: Number of tanks per team (default: 2)
            initial_health: Starting health for each tank (default: 100)
        """
        super().__init__()

        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.team_size = team_size
        self.initial_health = initial_health
        self.num_agents = team_size * 2

        # Define observation and action spaces
        # Observation: Single-channel top-down view of the grid
        self.observation_space = spaces.Box(
            low=0, high=3,
            shape=(1, grid_size, grid_size),
            dtype=np.float32
        )

        # Action: Discrete action space with 5 actions
        self.action_space = spaces.Discrete(5)

        # Initialize environment state
        self.obstacles = None
        self.tanks = None
        self.step_count = 0
        self.max_steps = 500  # Episode length limit

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observations: Dictionary of observations for each agent
            info: Dictionary of additional information
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0

        # Initialize obstacle positions
        self._place_obstacles()

        # Initialize tank states
        self._place_tanks()

        # Get initial observations
        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def _place_obstacles(self):
        """Place random obstacles on the grid."""
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            self.obstacles.add((x, y))

    def _place_tanks(self):
        """
        Place tanks for both teams on opposite sides of the map.
        Team 0 starts on the left side, Team 1 on the right side.
        """
        self.tanks = []

        # Team 0 (left side)
        for i in range(self.team_size):
            placed = False
            while not placed:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size // 4)  # Left quarter
                if (x, y) not in self.obstacles and not self._position_occupied(x, y):
                    self.tanks.append({
                        'team': 0,
                        'agent_id': i,
                        'x': x,
                        'y': y,
                        'orientation': 1,  # Facing East (towards enemy)
                        'health': self.initial_health,
                        'alive': True
                    })
                    placed = True

        # Team 1 (right side)
        for i in range(self.team_size):
            placed = False
            while not placed:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(3 * self.grid_size // 4, self.grid_size)  # Right quarter
                if (x, y) not in self.obstacles and not self._position_occupied(x, y):
                    self.tanks.append({
                        'team': 1,
                        'agent_id': i + self.team_size,
                        'x': x,
                        'y': y,
                        'orientation': 3,  # Facing West (towards enemy)
                        'health': self.initial_health,
                        'alive': True
                    })
                    placed = True

    def _position_occupied(self, x: int, y: int) -> bool:
        """Check if a position is occupied by a tank."""
        for tank in self.tanks:
            if tank['alive'] and tank['x'] == x and tank['y'] == y:
                return True
        return False

    def _get_observations(self) -> Dict[int, np.ndarray]:
        """
        Generate observations for all agents.

        Returns:
            Dictionary mapping agent_id to observation array
        """
        observations = {}

        for tank in self.tanks:
            if tank['alive']:
                obs = self._generate_observation(tank)
                observations[tank['agent_id']] = obs

        return observations

    def _generate_observation(self, tank: Dict) -> np.ndarray:
        """
        Generate observation for a single tank.

        Args:
            tank: Tank state dictionary

        Returns:
            Observation array of shape (1, grid_size, grid_size)
        """
        grid = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)

        # Add obstacles
        for obs_x, obs_y in self.obstacles:
            grid[0, obs_x, obs_y] = self.CELL_OBSTACLE

        # Add tanks
        for other_tank in self.tanks:
            if not other_tank['alive']:
                continue

            x, y = other_tank['x'], other_tank['y']

            if other_tank['agent_id'] == tank['agent_id']:
                # Skip self (or mark as ally if desired)
                continue
            elif other_tank['team'] == tank['team']:
                # Ally tank
                grid[0, x, y] = self.CELL_ALLY
            else:
                # Enemy tank
                grid[0, x, y] = self.CELL_ENEMY

        return grid

    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one step of the environment.

        Args:
            actions: Dictionary mapping agent_id to action

        Returns:
            observations: New observations for each agent
            rewards: Rewards for each agent
            terminated: Whether episode is done for each agent
            truncated: Whether episode was truncated for each agent
            info: Additional information
        """
        self.step_count += 1

        # Initialize rewards for all agents
        rewards = {tank['agent_id']: -0.01 for tank in self.tanks if tank['alive']}

        # Track team rewards for distribution
        team_rewards = {0: 0.0, 1: 0.0}

        # Process actions for each agent
        for tank in self.tanks:
            if not tank['alive']:
                continue

            agent_id = tank['agent_id']
            action = actions.get(agent_id, self.ACTION_IDLE)

            if action == self.ACTION_MOVE_FORWARD:
                self._move_tank(tank)
            elif action == self.ACTION_TURN_RIGHT:
                tank['orientation'] = (tank['orientation'] + 1) % 4
            elif action == self.ACTION_TURN_LEFT:
                tank['orientation'] = (tank['orientation'] - 1) % 4
            elif action == self.ACTION_FIRE:
                hit_result = self._fire_projectile(tank)
                if hit_result is not None:
                    hit_tank, damage = hit_result
                    if hit_tank['team'] == tank['team']:
                        # Hit ally - negative reward
                        team_rewards[tank['team']] -= 1.0
                    else:
                        # Hit enemy - positive reward
                        team_rewards[tank['team']] += 1.0

        # Distribute team rewards to all living team members
        for tank in self.tanks:
            if tank['alive']:
                rewards[tank['agent_id']] += team_rewards[tank['team']]

        # Check for episode termination
        team0_alive = sum(1 for t in self.tanks if t['team'] == 0 and t['alive'])
        team1_alive = sum(1 for t in self.tanks if t['team'] == 1 and t['alive'])

        done = team0_alive == 0 or team1_alive == 0 or self.step_count >= self.max_steps

        # Award victory bonus
        if done and self.step_count < self.max_steps:
            winning_team = 0 if team1_alive == 0 else 1
            for tank in self.tanks:
                if tank['team'] == winning_team and tank['alive']:
                    rewards[tank['agent_id']] += 10.0

        # Generate observations
        observations = self._get_observations()

        # Create terminated and truncated dictionaries
        terminated = {tank['agent_id']: done for tank in self.tanks if tank['alive']}
        truncated = {tank['agent_id']: self.step_count >= self.max_steps for tank in self.tanks if tank['alive']}

        # Add special "__all__" key for multi-agent environments
        terminated["__all__"] = done
        truncated["__all__"] = self.step_count >= self.max_steps

        info = self._get_info()

        return observations, rewards, terminated, truncated, info

    def _move_tank(self, tank: Dict):
        """
        Move a tank forward in its current orientation.

        Args:
            tank: Tank state dictionary
        """
        dx, dy = self.DIRECTIONS[tank['orientation']]
        new_x = tank['x'] + dx
        new_y = tank['y'] + dy

        # Check boundaries
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            return

        # Check for obstacles
        if (new_x, new_y) in self.obstacles:
            return

        # Check for other tanks
        if self._position_occupied(new_x, new_y):
            return

        # Move is valid
        tank['x'] = new_x
        tank['y'] = new_y

    def _fire_projectile(self, tank: Dict) -> Optional[Tuple[Dict, int]]:
        """
        Fire a projectile from a tank.

        The projectile travels in a straight line until it hits a tank or obstacle.

        Args:
            tank: Tank state dictionary that is firing

        Returns:
            Tuple of (hit_tank, damage) if a tank was hit, None otherwise
        """
        dx, dy = self.DIRECTIONS[tank['orientation']]
        x, y = tank['x'], tank['y']

        # Travel along the direction until hitting something
        while True:
            x += dx
            y += dy

            # Check boundaries
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                return None

            # Check for obstacle
            if (x, y) in self.obstacles:
                return None

            # Check for tank hit
            for other_tank in self.tanks:
                if other_tank['alive'] and other_tank['x'] == x and other_tank['y'] == y:
                    if other_tank['agent_id'] != tank['agent_id']:
                        # Hit a tank (not self)
                        damage = 25
                        other_tank['health'] -= damage
                        if other_tank['health'] <= 0:
                            other_tank['alive'] = False
                        return (other_tank, damage)

            # Projectile range limit (prevent infinite loops)
            if abs(x - tank['x']) + abs(y - tank['y']) > self.grid_size:
                return None

    def _get_info(self) -> Dict:
        """Get additional information about the environment state."""
        team0_alive = sum(1 for t in self.tanks if t['team'] == 0 and t['alive'])
        team1_alive = sum(1 for t in self.tanks if t['team'] == 1 and t['alive'])

        return {
            'step_count': self.step_count,
            'team0_alive': team0_alive,
            'team1_alive': team1_alive,
            'tanks': [
                {
                    'agent_id': t['agent_id'],
                    'team': t['team'],
                    'x': t['x'],
                    'y': t['y'],
                    'orientation': t['orientation'],
                    'health': t['health'],
                    'alive': t['alive']
                } for t in self.tanks
            ]
        }

    def render(self):
        """Render the environment (optional - for future visualization)."""
        if self.render_mode == "rgb_array":
            # Could implement pygame rendering here
            pass
        return None

    def close(self):
        """Clean up resources."""
        pass
