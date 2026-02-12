# bees-2

# Bee Trajectory Preprocessing Pipeline

This pipeline preprocesses bee trajectory data for neural network training by:
1. Analysing temporal autocorrelation to determine an appropriate gap threshold
2. Interpolating trajectories to regular time intervals using cubic splines

## Requirements

**Bash (Linux/Mac):**
```bash
pip install numpy pandas scipy matplotlib openpyxl
```

**Command Prompt (Windows):**
```cmd
pip install numpy pandas scipy matplotlib openpyxl
```

## Files

- `autocorrelation_analysis.py` - Analyses autocorrelation to help choose gap threshold
- `interpolate_trajectories.py` - Performs cubic spline interpolation
- `visualise_trajectory.py` - Visualises a random or specific session with original and interpolated data

## Workflow

### Step 1: Prepare your data

Place all 52 bee xlsx files in a single directory. Remove the 2 irrelevant files.

### Step 2: Run autocorrelation analysis

**Bash (Linux/Mac):**
```bash
python autocorrelation_analysis.py \
    --input_dir /path/to/xlsx/files \
    --output_dir /path/to/output \
    --gap_threshold 1.0 \
    --max_lag 5.0
```

**Command Prompt (Windows):**
```cmd
python autocorrelation_analysis.py ^
    --input_dir C:\path\to\xlsx\files ^
    --output_dir C:\path\to\output ^
    --gap_threshold 1.0 ^
    --max_lag 5.0
```

Or on a single line:
```cmd
python autocorrelation_analysis.py --input_dir C:\path\to\xlsx\files --output_dir C:\path\to\output --gap_threshold 1.0 --max_lag 5.0
```

**Arguments:**
- `--input_dir`, `-i`: Directory containing the xlsx files (required)
- `--output_dir`, `-o`: Directory for output files (required)
- `--gap_threshold`, `-g`: Gap threshold for splitting sessions during analysis (default: 1.0s)
- `--max_lag`, `-m`: Maximum lag for autocorrelation computation (default: 5.0s)
- `--fps`: Target frame rate for interpolation (default: 30.0)

**Output files:**
- `autocorrelation_analysis.png` - Visualisation of autocorrelation decay
- `autocorrelation_summary.csv` - Summary statistics including decay times
- `autocorrelation_data.csv` - Full autocorrelation data for further analysis

**Interpreting results:**

The script reports decay times at 50%, 25%, and 10% autocorrelation thresholds. 
- **50% decay time**: Past positions still moderately predictive
- **25% decay time**: Past positions weakly predictive
- **10% decay time**: Past positions nearly uninformative

Choose a gap threshold based on how much autocorrelation you want to preserve. A threshold around the 25-50% decay time is typically reasonable.

### Step 3: Run interpolation

After reviewing the autocorrelation analysis, run the interpolation with your chosen threshold:

**Bash (Linux/Mac):**
```bash
python interpolate_trajectories.py \
    --input_dir /path/to/xlsx/files \
    --output_file /path/to/output/trajectories.csv \
    --gap_threshold 0.5
```

**Command Prompt (Windows):**
```cmd
python interpolate_trajectories.py ^
    --input_dir C:\path\to\xlsx\files ^
    --output_file C:\path\to\output\trajectories.csv ^
    --gap_threshold 0.5
```

Or on a single line:
```cmd
python interpolate_trajectories.py --input_dir C:\path\to\xlsx\files --output_file C:\path\to\output\trajectories.csv --gap_threshold 0.5
```

**Arguments:**
- `--input_dir`, `-i`: Directory containing the xlsx files (required)
- `--output_file`, `-o`: Path to output CSV file (required)
- `--gap_threshold`, `-g`: Gap threshold in seconds (required) - gaps longer than this create segment breaks
- `--fps`: Target frame rate for interpolation (default: 30.0)
- `--quiet`, `-q`: Suppress progress output

**Output CSV columns:**
- `bee_id`: Identifier for each bee (from column G in xlsx)
- `segment_id`: Unique identifier for each continuous segment (format: `{bee_id}_seg{number}`)
- `time`: Timestamp in seconds (monotonically increasing)
- `x`: Interpolated x coordinate
- `y`: Interpolated y coordinate
- `z`: Interpolated z coordinate

## Example workflow

**Bash (Linux/Mac):**
```bash
# 1. Run autocorrelation analysis
python autocorrelation_analysis.py -i ./raw_data -o ./analysis_output

# 2. Review the output plots and summary
# Suppose the analysis suggests decay times around 0.3-0.6 seconds

# 3. Run interpolation with chosen threshold (e.g., 0.5s)
python interpolate_trajectories.py -i ./raw_data -o ./preprocessed/trajectories.csv -g 0.5
```

**Command Prompt (Windows):**
```cmd
REM 1. Run autocorrelation analysis
python autocorrelation_analysis.py -i .\raw_data -o .\analysis_output

REM 2. Review the output plots and summary
REM Suppose the analysis suggests decay times around 0.3-0.6 seconds

REM 3. Run interpolation with chosen threshold (e.g., 0.5s)
python interpolate_trajectories.py -i .\raw_data -o .\preprocessed\trajectories.csv -g 0.5
```

## Visualising trajectories

Use `visualise_trajectory.py` to inspect a random (or specific) session with both original and interpolated data.

**Bash (Linux/Mac):**
```bash
# Random session, display interactively
python visualise_trajectory.py -i ./bee_data

# Random session, save to file
python visualise_trajectory.py -i ./bee_data -o ./plots

# Specific file and session
python visualise_trajectory.py -i ./bee_data -o ./plots --bee_file bee001.xlsx --session_index 2

# With random seed for reproducibility
python visualise_trajectory.py -i ./bee_data -o ./plots --seed 42
```

**Command Prompt (Windows):**
```cmd
REM Random session, display interactively
python visualise_trajectory.py -i .\bee_data

REM Random session, save to file
python visualise_trajectory.py -i .\bee_data -o .\plots

REM Specific file and session
python visualise_trajectory.py -i .\bee_data -o .\plots --bee_file bee001.xlsx --session_index 2

REM With random seed for reproducibility
python visualise_trajectory.py -i .\bee_data -o .\plots --seed 42
```

**Arguments:**
- `--input_dir`, `-i`: Directory containing the xlsx files (required)
- `--output_dir`, `-o`: Directory for output plots (if not specified, displays interactively)
- `--fps`: Target frame rate for interpolation (default: 30.0)
- `--bee_file`: Specific xlsx file to use (if not specified, random selection)
- `--session_index`: Specific session index within the file (if not specified, random selection)
- `--seed`: Random seed for reproducibility

**Output:**
A figure with 4 subplots:
1. 3D trajectory with original points (blue) and interpolated line (red)
2. X coordinate over time
3. Y coordinate over time
4. Z coordinate over time

## Data format notes

### Input xlsx files
- Column G: Bee ID (ID_No)
- Column AT: Timestamp
- Column AU: x coordinate
- Column AV: y coordinate  
- Column AW: z coordinate
- First row contains headers

### Timestamp resets
The scripts automatically detect and handle timestamp resets that occur when data from multiple video recordings are appended in a single file. Resets are detected when the timestamp decreases by more than 0.5 seconds.

### Gap handling
- Gaps ≤ threshold: Interpolated using cubic spline
- Gaps > threshold: Trajectory split into separate segments

## Notes for dissertation

The gap threshold should be justified based on the autocorrelation analysis. In your methods section, you can report:

1. The autocorrelation decay profile (include the plot)
2. The decay times at various thresholds
3. Your chosen gap threshold and the rationale

This provides a principled, data-driven justification for your preprocessing choices rather than using an arbitrary threshold.