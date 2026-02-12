"""
Run Trajectory Visualisation
=============================
This script visualises a random (or specific) session from your bee trajectory data,
showing both the original and interpolated data.

Edit the settings below, then click the Run button (▶️) in VS Code.
"""

import os

# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# SETTINGS - Edit these to match your setup
# ============================================================

INPUT_DIR = './bee_data'      # Folder containing your xlsx files
OUTPUT_DIR = './output'       # Folder for output plots (set to None to display interactively)
FPS = 30.0                    # Target frame rate for interpolation

# Optional: specify a particular file and/or session
BEE_FILE = None               # Set to a filename like 'bee001.xlsx' to use a specific file, or None for random
SESSION_INDEX = None          # Set to a number like 0, 1, 2 to pick a specific session, or None for random
SEED = None                   # Set to a number like 42 for reproducible random selection, or None for truly random

# ============================================================
# Run the visualisation (no need to edit below this line)
# ============================================================

from visualise_trajectory import visualise_random_session

print("="*60)
print("TRAJECTORY VISUALISATION")
print("="*60)
print(f"Input directory:  {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Target FPS:       {FPS}")
print(f"Bee file:         {BEE_FILE if BEE_FILE else 'Random'}")
print(f"Session index:    {SESSION_INDEX if SESSION_INDEX is not None else 'Random'}")
print(f"Random seed:      {SEED if SEED else 'None (truly random)'}")
print("="*60)
print()

visualise_random_session(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    target_fps=FPS,
    bee_file=BEE_FILE,
    session_index=SESSION_INDEX,
    seed=SEED
)