"""
Run Autocorrelation Analysis
=============================
This script runs the autocorrelation analysis on your bee trajectory data.
Edit the settings below, then click the Run button (▶️) in VS Code.
"""

import os

# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# SETTINGS - Edit these to match your setup
# ============================================================

INPUT_DIR = './bee_data'          # Folder containing your xlsx files
OUTPUT_DIR = './output'           # Folder for output files (will be created if it doesn't exist)
GAP_THRESHOLD = 1.0               # Gap threshold in seconds for splitting sessions during analysis
MAX_LAG = 5.0                     # Maximum lag in seconds for autocorrelation computation
FPS = 30.0                        # Target frame rate for interpolation

# ============================================================
# Run the analysis (no need to edit below this line)
# ============================================================

from autocorrelation_analysis import analyse_autocorrelation

print("="*60)
print("AUTOCORRELATION ANALYSIS")
print("="*60)
print(f"Input directory:  {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Gap threshold:    {GAP_THRESHOLD} seconds")
print(f"Max lag:          {MAX_LAG} seconds")
print(f"Target FPS:       {FPS}")
print("="*60)
print()

analyse_autocorrelation(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    gap_threshold_for_analysis=GAP_THRESHOLD,
    target_dt=1/FPS,
    max_lag_seconds=MAX_LAG
)

print()
print("Done! Check the output folder for:")
print("  - autocorrelation_analysis.png (plots)")
print("  - autocorrelation_summary.csv (decay times)")
print("  - autocorrelation_data.csv (full data)")