#!/bin/bash
# Quick visualization script for TankSim
# Run this from anywhere in the repository

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/puffertank"

echo "========================================"
echo "  TankSim Visualization Quick Start"
echo "========================================"
echo

# Default values
SEED=${1:-42}
STEPS=${2:-50}

echo "Generating episode..."
echo "  Seed: $SEED"
echo "  Max steps: $STEPS"
echo "  Output: /tmp/tanksim_frame_*.png"
echo

python generate_episode_frames.py --seed "$SEED" --max-steps "$STEPS"

echo
echo "========================================"
echo "  Frames saved!"
echo "========================================"
echo
echo "View frames:"
echo "  ls /tmp/tanksim_frame_*.png"
echo
echo "Create GIF (requires ImageMagick):"
echo "  convert -delay 10 -loop 0 /tmp/tanksim_frame_*.png tanksim.gif"
echo
echo "Create GIF (using Python):"
echo "  python3 -c 'from PIL import Image; import glob; frames=[Image.open(f) for f in sorted(glob.glob(\"/tmp/tanksim_frame_*.png\"))]; frames[0].save(\"battle.gif\", save_all=True, append_images=frames[1:], duration=100, loop=0)'"
echo
