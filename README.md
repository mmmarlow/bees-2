# Bee Trajectory Preprocessing Pipeline

## Background

### The Problem

The raw bee trajectory data has two major problems:

1. **Gaps in the data**: There are gaps of various sizes in the video frame sequence, resulting from problems tracking the 3D coordinates of the bees.
2. **Variable frame rate**: The time between consecutive frames is not constant.

These problems impact the ability of a neural network (NN) to interpret and learn information about the trajectories, as any pair of consecutive frames do not necessarily correspond to the same period of elapsed time.

### The Solution: Interpolation

Rather than providing timestamps as additional inputs to the NN (which increases dimensionality and training time), we preprocess the data using **cubic spline interpolation**. This approach:

1. Estimates coordinates at consistent time intervals (1/30 seconds), eliminating the need for timestamp inputs
2. Fills in small gaps, providing more usable training data
3. Produces smooth trajectories that better represent continuous bee movement

**Trade-off**: We are training on synthesised data, which may not perfectly reflect true bee behaviour. However, with careful preprocessing (using a principled gap threshold), the advantages outweigh the disadvantages.

### Why Interpolate Coordinates First?

We interpolate the raw (x, y, z) coordinates rather than derived variables (velocity, acceleration, etc.) because:

- Small interpolation errors accumulate when reconstructing trajectories from derived variables
- Interpolating coordinates first, then computing derived variables, ensures the preprocessed data corresponds as accurately as possible to actual bee trajectories

### Determining the Gap Threshold

Not all gaps should be interpolated — large gaps produce unreliable estimates because the spline has no information about what happened during that time.

We use **temporal autocorrelation analysis** to determine an appropriate threshold:

- Autocorrelation measures how similar a bee's position is to its position at earlier times
- As the time lag increases, autocorrelation decays (past positions become less predictive)
- The timescale at which autocorrelation decays tells us: beyond what gap duration does interpolation become unreliable?

A gap threshold around the **25-50% autocorrelation decay time** is typically reasonable — past positions are still somewhat predictive within this interval.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────┐   │
│  │  Raw Data    │    │  Autocorrelation │    │  Choose Gap Threshold   │   │
│  │  (52 xlsx)   │───▶│  Analysis        │───▶│  from Decay Times       │   │
│  └──────────────┘    └──────────────────┘    └───────────┬─────────────┘   │
│                                                          │                  │
│                                                          ▼                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────┐   │
│  │  Preprocessed│    │  Cubic Spline    │    │  Apply Threshold        │   │
│  │  Data (CSV)  │◀───│  Interpolation   │◀───│  to Split Trajectories  │   │
│  └──────────────┘    └──────────────────┘    └─────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Each Script Does

| Script | Purpose |
|--------|---------|
| `autocorrelation_analysis.py` | Analyses how quickly positional predictability decays over time |
| `interpolate_trajectories.py` | Performs cubic spline interpolation with your chosen gap threshold |
| `visualise_trajectory.py` | Visualises original vs interpolated data for quality checking |

### Launcher Scripts (for VS Code Run Button)

| Script | What it runs |
|--------|--------------|
| `run_autocorrelation.py` | Runs autocorrelation analysis with editable settings |
| `run_interpolation.py` | Runs interpolation with editable gap threshold |
| `run_visualisation.py` | Visualises a random or specific session |

---

## Requirements

Install Python packages:

```
python -m pip install numpy pandas scipy matplotlib openpyxl
```

---

## Folder Structure

```
bee-trajectory-preprocessing/
├── autocorrelation_analysis.py
├── interpolate_trajectories.py
├── visualise_trajectory.py
├── run_autocorrelation.py
├── run_interpolation.py
├── run_visualisation.py
├── README.md
├── bee_data/
│   ├── [your 52 xlsx files]
│   └── ...
└── output/
    ├── autocorrelation_analysis.png
    ├── autocorrelation_summary.csv
    ├── autocorrelation_data.csv
    └── trajectories.csv
```

---

## Step-by-Step Workflow

### Step 1: Prepare Your Data

1. Place all 52 bee xlsx files in the `bee_data/` folder
2. Remove the 2 irrelevant files
3. Ensure each file has the correct columns:
   - Column G: Bee ID
   - Column AT: Timestamp
   - Columns AU, AV, AW: x, y, z coordinates

### Step 2: Run Autocorrelation Analysis

**Using VS Code (recommended):**
1. Open `run_autocorrelation.py`
2. Click the Run button (▶️)

**Using terminal:**
```
python autocorrelation_analysis.py -i ./bee_data -o ./output
```

**What it does internally:**
1. Loads each bee's xlsx file
2. Detects timestamp resets (separate recording sessions)
3. Splits sessions at gaps > 1 second
4. Interpolates each segment to regular 1/30s intervals
5. Computes autocorrelation on the interpolated data
6. Averages across all segments and bees
7. Outputs plots and decay time statistics

**Output files:**
- `autocorrelation_analysis.png` — Plot showing autocorrelation decay
- `autocorrelation_summary.csv` — Summary statistics including decay times
- `autocorrelation_data.csv` — Full autocorrelation data

### Step 3: Choose Your Gap Threshold

Open `autocorrelation_analysis.png` and look at the decay curves.

The terminal output will show:
```
DECAY TIMES (when autocorrelation drops below threshold):
------------------------------------------------------------
Threshold       X (s)        Y (s)        Z (s)
50%             0.XXX        0.XXX        0.XXX
25%             0.XXX        0.XXX        0.XXX
10%             0.XXX        0.XXX        0.XXX

RECOMMENDATION:
Average decay time to 50% autocorrelation: X.XX seconds
Average decay time to 25% autocorrelation: X.XX seconds
Suggested range: X.XXs to X.XXs
```

**Choose a threshold** between the 50% and 25% decay times.

### Step 4: Run Interpolation

**Using VS Code:**
1. Open `run_interpolation.py`
2. Edit the `GAP_THRESHOLD` value to your chosen threshold
3. Click the Run button (▶️)

**Using terminal:**
```
python interpolate_trajectories.py -i ./bee_data -o ./output/trajectories.csv -g 0.5
```

**Output:** `trajectories.csv` with columns:
- `bee_id`: Identifier for each bee
- `segment_id`: Unique identifier for each continuous segment
- `time`: Timestamp in seconds (regular 1/30s intervals)
- `x`, `y`, `z`: Interpolated coordinates

### Step 5: Visualise and Verify (Optional)

Inspect random sessions to verify the interpolation looks reasonable:

**Using VS Code:**
1. Open `run_visualisation.py`
2. Click the Run button (▶️)
3. Check the output plot

**Using terminal:**
```
python visualise_trajectory.py -i ./bee_data -o ./output
```

---

## Data Format Details

### Input xlsx Files
- Column G: Bee ID (ID_No)
- Column AT: Timestamp
- Column AU: x coordinate
- Column AV: y coordinate
- Column AW: z coordinate
- First row contains headers

### Timestamp Resets
The scripts automatically detect and handle timestamp resets that occur when data from multiple video recordings are appended in a single file. Resets are detected when the timestamp decreases by more than 0.5 seconds.

### Gap Handling
- **Gaps ≤ threshold**: Interpolated using cubic spline
- **Gaps > threshold**: Trajectory split into separate segments (each gets a unique `segment_id`)

---

## Notes for Dissertation

The gap threshold should be justified based on the autocorrelation analysis. In your methods section, you can write something like:

> "The gap threshold for interpolation was determined empirically using temporal autocorrelation analysis. Autocorrelation of the (x, y, z) coordinates was computed across all bee trajectories after interpolating to regular 1/30 second intervals. The autocorrelation decayed to 50% at approximately [X] seconds and to 25% at approximately [Y] seconds. Based on this analysis, a gap threshold of [Z] seconds was chosen, corresponding to the timescale at which past positions remain moderately predictive of current position. Gaps shorter than this threshold were filled using cubic spline interpolation; gaps longer than this threshold resulted in the trajectory being split into separate segments."

Include the `autocorrelation_analysis.png` plot in your methods or supplementary materials.

---

## Troubleshooting

### "pip is not recognized"
Use `python -m pip` instead:
```
python -m pip install numpy pandas scipy matplotlib openpyxl
```

### "python is not recognized"
- Ensure Python is installed: https://www.python.org/downloads/
- Tick "Add Python to PATH" during installation
- Restart VS Code after installation

### Script runs but no output files
- Check that `bee_data/` folder contains your xlsx files
- Check that `output/` folder exists (create it if not)

### Memory errors with large files
- Close other applications
- Process fewer bees at a time by moving some xlsx files temporarily