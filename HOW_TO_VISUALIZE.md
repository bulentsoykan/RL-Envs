# How to Visualize TankSim Episodes

## 🚀 Quick Start (Easiest)

From **anywhere** in the repository:

```bash
./visualize.sh
```

This generates 50 frames with seed 42 (default).

### Custom seed and steps:

```bash
./visualize.sh 123 30    # seed=123, 30 steps
./visualize.sh 999 100   # seed=999, 100 steps
```

## 📁 Where Are My Images?

All frames are saved to `/tmp/tanksim_frame_*.png`

View them:
```bash
ls /tmp/tanksim_frame_*.png
```

## 🎬 Create Animations

### Method 1: Using ImageMagick (if installed)
```bash
convert -delay 10 -loop 0 /tmp/tanksim_frame_*.png battle.gif
```

### Method 2: Using Python/Pillow
```bash
python3 << 'EOF'
from PIL import Image
import glob

frames = [Image.open(f) for f in sorted(glob.glob('/tmp/tanksim_frame_*.png'))]
frames[0].save('battle.gif', save_all=True, append_images=frames[1:],
               duration=100, loop=0)
print('✓ Saved as battle.gif')
EOF
```

### Method 3: Using FFmpeg (for MP4 video)
```bash
ffmpeg -framerate 10 -pattern_type glob -i '/tmp/tanksim_frame_*.png' \
  -c:v libx264 -pix_fmt yuv420p battle.mp4
```

## 🎨 What You'll See

Each frame shows:
- **Blue circles** with arrows = Team 0 tanks (start left side)
- **Red circles** with arrows = Team 1 tanks (start right side)
- **Arrows (↑ ↓ ← →)** = Direction tank is facing
- **Colored bar under tank** = Health (starts at 100 HP)
- **Dark gray squares** = Obstacles
- **A0, A1, A2, A3** = Agent ID labels
- **Top text** = Step count, alive count, rewards

## 🛠️ Advanced Usage

### Run from puffertank directory:
```bash
cd puffertank
python generate_episode_frames.py --seed 42 --max-steps 100
```

### Use a trained policy (instead of random):
```bash
cd puffertank
python generate_episode_frames.py --policy --max-steps 100
```

### Save to custom location:
```bash
cd puffertank
python generate_episode_frames.py --output ./my_episodes --prefix battle1
```

### All options:
```bash
python generate_episode_frames.py --help
```

## 📊 Interactive Visualization (with GUI)

If you have a display (not headless):

```bash
cd puffertank
python visualize_tanksim.py
```

Then select from the menu:
1. Random actions episode
2. Policy-controlled episode
3. Save frames to disk

## 🐳 Docker Usage

### Mount a volume to save images:
```bash
docker run -it --gpus all \
  -v $(pwd)/output:/output \
  pufferai/TankSim:3.0 bash

# Inside container:
cd /puffertank
python generate_episode_frames.py --output /output --max-steps 100
```

Then view images in your `./output/` directory on your host.

### Copy images from running container:
```bash
# Run visualization inside
docker exec TankSim bash -c "cd /puffertank && python generate_episode_frames.py"

# Copy out
docker cp TankSim:/tmp/tanksim_frame_000.png ./
docker cp TankSim:/tmp/tanksim_frame_001.png ./
# etc...
```

## 🧪 Test It Works

Quick test (generates just 5 frames):
```bash
cd /home/user/RL-Envs/puffertank
python generate_episode_frames.py --max-steps 5
ls -lh /tmp/tanksim_frame_*.png
```

## ❓ Troubleshooting

### "No such file or directory"
Make sure you're in the right directory:
```bash
cd /home/user/RL-Envs
./visualize.sh
```

Or use the full path:
```bash
/home/user/RL-Envs/visualize.sh
```

### "Module not found: matplotlib"
```bash
pip install matplotlib pillow
```

### Images not showing up
Check if they're there:
```bash
ls -lh /tmp/tanksim_frame_*.png
```

### Want to clear old frames
```bash
rm /tmp/tanksim_frame_*.png
rm /tmp/demo_frame_*.png
rm /tmp/test_episode_frame_*.png
```

## 📚 More Info

- Detailed guide: See `VISUALIZATION_GUIDE.md`
- Environment details: See `README.md`
- Test environment: `python test_tanksim.py`
- Full test suite: `python comprehensive_test.py`

## 🎮 Example Workflow

```bash
# 1. Generate episode
cd /home/user/RL-Envs
./visualize.sh 42 30

# 2. Create GIF
python3 -c "from PIL import Image; import glob; frames=[Image.open(f) for f in sorted(glob.glob('/tmp/tanksim_frame_*.png'))]; frames[0].save('my_battle.gif', save_all=True, append_images=frames[1:], duration=100, loop=0); print('✓ Saved as my_battle.gif')"

# 3. View GIF
# (copy my_battle.gif to your local machine and open it)
```

Enjoy watching your tanks battle! 🎉
