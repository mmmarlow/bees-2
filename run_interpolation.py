"""
Run Trajectory Interpolation
=============================
This script interpolates your bee trajectory data using cubic splines.

IMPORTANT: Run the autocorrelation analysis first to determine the best gap threshold!

Edit the settings below, then click the Run button (▶️) in VS Code.
"""

import os

# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# SETTINGS - Edit these to match your setup
# ============================================================

INPUT_DIR = './bee_data'                      # Folder containing your xlsx files
OUTPUT_FILE = './output/trajectories.csv'     # Path for the output CSV file
GAP_THRESHOLD = 0.5                           # Gap threshold in seconds (SET THIS BASED ON AUTOCORRELATION ANALYSIS!)
FPS = 30.0                                    # Target frame rate for interpolation

# ============================================================
# Run the interpolation (no need to edit below this line)
# ============================================================

from interpolate_trajectories import interpolate_all_bees

print("="*60)
print("TRAJECTORY INTERPOLATION")
print("="*60)
print(f"Input directory:  {INPUT_DIR}")
print(f"Output file:      {OUTPUT_FILE}")
print(f"Gap threshold:    {GAP_THRESHOLD} seconds")
print(f"Target FPS:       {FPS}")
print("="*60)
print()

interpolate_all_bees(
    input_dir=INPUT_DIR,
    output_file=OUTPUT_FILE,
    gap_threshold=GAP_THRESHOLD,
    target_fps=FPS,
    verbose=True
)

print()
print(f"Done! Output saved to: {OUTPUT_FILE}")