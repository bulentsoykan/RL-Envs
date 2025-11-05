![figure](https://pufferai.github.io/source/resource/header.png)

# TankSim: A Combat Vehicle Simulation Environment

A GPU-accelerated development environment for multi-agent reinforcement learning featuring **TankSim**, a Gymnasium-compatible combat vehicle simulation built on PufferLib.

**TankSim** provides a 2D grid-based, multi-agent environment where teams of tanks engage in tactical combat. Train agents to coordinate, navigate obstacles, and outmaneuver opponents in real-time combat scenarios.

## Features

- **Multi-Agent Combat**: 2 teams of 2 tanks each (4 agents total)
- **Tactical Gameplay**: Movement, orientation, and projectile-based combat
- **Gymnasium Compatible**: Standard RL interface for easy integration
- **GPU-Accelerated Training**: Leverage CUDA for fast parallel environment execution
- **PufferLib Integration**: Built-in support for state-of-the-art RL algorithms

We **strongly** recommend VSCode DevContainers for local development.

### 1. Install Docker

Docker is a containerization technology to package code and its dependencies. Install and configure docker [Docker Desktop](https://docs.docker.com/desktop/) by following the official install instructions for your operationg system and then completing the post-installation steps. Linux users may also need to install [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide).

To test your docker installation: `docker run hello-world`.

### 2. Download This Repository

Windows users should clone to WSL

### 3. Open in VS Code

Install [Visual Studio Code](https://code.visualstudio.com/)
From the Extensions panel on the left: Dev Containers > Install
F1 > Dev Containers: Open Folder in Container > Select the TankSim folder

For more information see [Containerized Development](https://code.visualstudio.com/docs/devcontainers/containers)

### 4. Run the Simulation

Once inside the container, you can start training agents immediately with PufferLib:

```bash
python puffertank/train_tanks.py
```

This will:
- Initialize the TankSim multi-agent environment
- Create a convolutional neural network policy
- Train agents using parallel environment execution
- Save the trained policy to `/puffertank/tanksim_policy.pt`

### 5. Run without VS Code

```bash
bash docker.sh test
```

Comes pre-loaded with NeoVim config. Use `:PlugInstall` for code completion with SuperMaven.

## Environment Details

**Observation Space**: Box(1, 32, 32) - Top-down grid view where:
- 0 = Empty cell
- 1 = Obstacle
- 2 = Ally tank
- 3 = Enemy tank

**Action Space**: Discrete(5)
- 0 = Idle
- 1 = Move Forward
- 2 = Turn Right
- 3 = Turn Left
- 4 = Fire

**Rewards**:
- +1.0 for hitting an enemy tank
- -1.0 for hitting an ally
- -0.01 per step (encourages quick decisions)
- +10.0 for winning the battle
