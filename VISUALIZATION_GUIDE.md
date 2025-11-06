# TankSim Visualization Guide

This guide shows you how to visualize and watch TankSim episodes in action!

## Quick Start

### 1. Single Frame Test
Generate a single frame to test the visualization:

```bash
cd puffertank
python -c "
import matplotlib
matplotlib.use('Agg')
from tank_sim_env import TankSimEnv
from visualize_tanksim import TankSimVisualizer

env = TankSimEnv()
obs, info = env.reset(seed=42)
visualizer = TankSimVisualizer(env)
visualizer.save_frame(obs, info, {}, 0, 'tanksim_frame.png')
print('✓ Saved to tanksim_frame.png')
"
```

### 2. Generate Episode Sequence
Create a sequence of frames showing an episode:

```bash
cd puffertank
python generate_episode_frames.py
```

This will create numbered frames in `/tmp/` that you can view individually or combine into an animation.

### 3. Interactive Visualization (with display)
If you have a display/GUI environment (local machine, not headless server):

```bash
cd puffertank
python visualize_tanksim.py
```

Then select:
- Option 1: Random actions episode
- Option 2: Policy-controlled episode
- Option 3: Save frames to disk

## What You'll See

The visualization shows:

- **Grid**: 32x32 battlefield with light grid lines
- **Obstacles**: Dark gray squares blocking movement
- **Team 0 Tanks**: Blue circles with arrows showing orientation
- **Team 1 Tanks**: Red circles with arrows showing orientation
- **Health Bars**: Below each tank showing remaining health
- **Agent IDs**: Labels (A0, A1, A2, A3) identifying each tank
- **Game Status**: Step count, alive tanks per team, rewards

### Tank Indicators

- **Arrow Direction**: ↑ ↓ ← → shows which way the tank is facing
- **Health Bar**: Green/colored bar under tank (full = 100 HP)
- **Team Colors**:
  - Blue = Team 0 (starts on left side)
  - Red = Team 1 (starts on right side)

## Creating Animations

### Using ImageMagick (convert)
If you have ImageMagick installed:

```bash
# Create GIF animation
convert -delay 10 -loop 0 /tmp/tanksim_frame_*.png tanksim_episode.gif

# Create slower animation (20 = 2 seconds per frame)
convert -delay 20 -loop 0 /tmp/tanksim_frame_*.png tanksim_slow.gif
```

### Using FFmpeg
If you have FFmpeg installed:

```bash
# Create MP4 video
ffmpeg -framerate 10 -pattern_type glob -i '/tmp/tanksim_frame_*.png' \
  -c:v libx264 -pix_fmt yuv420p tanksim_episode.mp4
```

### Using Python (PIL/Pillow)
```python
from PIL import Image
import glob

frames = [Image.open(f) for f in sorted(glob.glob('/tmp/tanksim_frame_*.png'))]
frames[0].save('tanksim.gif', save_all=True, append_images=frames[1:],
               duration=100, loop=0)
```

## Docker Container Usage

If you're using the TankSim Docker container:

### Method 1: Mount volume to access images
```bash
docker run -it --gpus all \
  -v $(pwd)/output:/output \
  pufferai/TankSim:3.0 bash

# Inside container:
cd /puffertank
python generate_episode_frames.py --output /output
```

Then view the images in your `./output/` directory on your host machine.

### Method 2: Copy files out of container
```bash
# Run visualization inside container
docker exec -it TankSim bash -c "cd /puffertank && python generate_episode_frames.py"

# Copy frames out
docker cp TankSim:/tmp/tanksim_frame_000.png ./
docker cp TankSim:/tmp/tanksim_frame_001.png ./
# ... etc
```

### Method 3: Use X11 forwarding (Linux/Mac)
```bash
# On host, allow Docker to access display
xhost +local:docker

# Run container with display
docker run -it --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  pufferai/TankSim:3.0 bash

# Inside container, run interactive visualization
cd /puffertank
python visualize_tanksim.py
```

## Customization

### Change Grid Size
```python
env = TankSimEnv(grid_size=24, num_obstacles=15)  # Smaller battlefield
```

### Change Episode Length
```python
visualizer.run_episode_visual(max_steps=500)  # Longer episodes
```

### Change Visualization Speed
Edit the pause duration in `visualize_tanksim.py`:
```python
plt.pause(0.5)  # Slower (0.5 seconds per frame)
plt.pause(0.05)  # Faster (0.05 seconds per frame)
```

### Save High-Resolution Images
```python
visualizer.save_frame(obs, info, rewards, step, 'frame.png')
# Then in the save_frame method, change dpi:
plt.savefig(filename, dpi=300, bbox_inches='tight')  # High quality
```

## Troubleshooting

### "No display" error
If you get `_tkinter.TclError: no display name and no $DISPLAY environment variable`:
- You're in a headless environment
- Use `matplotlib.use('Agg')` before importing pyplot
- Use the frame-saving methods instead of interactive display

### "Module not found: matplotlib"
```bash
pip install matplotlib pillow
```

### Frames not showing up
- Check `/tmp/` directory: `ls -lh /tmp/tanksim*.png`
- Ensure you have write permissions to the output directory

### Animation looks choppy
- Reduce the delay between frames: `convert -delay 5 ...`
- Increase frame generation steps (capture more frames)

## Examples

### Generate a complete episode visualization:
```bash
cd /home/user/RL-Envs/puffertank
python generate_episode_frames.py --seed 42 --max-steps 100
ls /tmp/tanksim_frame_*.png | wc -l  # Count frames generated
```

### View frames using command line (if you have `feh`, `eog`, etc):
```bash
feh /tmp/tanksim_frame_*.png  # Image viewer with slideshow
eog /tmp/tanksim_frame_000.png  # GNOME image viewer
```

## Next Steps

1. **Run training** with `python train_tanks.py` in the Docker container
2. **Visualize trained agents** by loading the saved policy
3. **Compare** random vs trained agent performance
4. **Share** your GIFs/videos of epic tank battles!

## Need Help?

- Check the test scripts: `test_tanksim.py`, `comprehensive_test.py`
- Review the environment code: `tank_sim_env.py`
- See training example: `train_tanks.py`

Happy tank battling! 🎮🚀
