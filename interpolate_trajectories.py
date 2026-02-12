#!/usr/bin/env python3
"""
Bee Trajectory Interpolation Script
====================================
This script preprocesses bee trajectory data by:
1. Detecting timestamp resets (new video recordings)
2. Splitting data at large gaps
3. Interpolating to regular time intervals using cubic splines
4. Outputting a combined CSV with all bees

Usage:
    python interpolate_trajectories.py --input_dir /path/to/xlsx/files --output_file output.csv --gap_threshold 0.5

The gap threshold should be informed by the autocorrelation analysis.
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def load_bee_data(filepath: str) -> Tuple[np.ndarray, str]:
    """
    Load bee trajectory data from an Excel file.
    
    Parameters:
        filepath: Path to the xlsx file
        
    Returns:
        Tuple of (numpy array with columns [time, x, y, z], bee_id string)
    """
    # Load only the required columns: G (ID_No), AT (time), AU (x), AV (y), AW (z)
    # Column G is ID_No, columns AT:AW are time, x, y, z
    df = pd.read_excel(filepath, usecols='G,AT:AW', skiprows=0)
    
    # Extract bee ID from column G (first non-null value)
    bee_id_col = df.iloc[:, 0]
    bee_id = str(bee_id_col.dropna().iloc[0]) if not bee_id_col.dropna().empty else Path(filepath).stem
    
    # Get time and coordinates (columns AT:AW, which are now columns 1-4 after loading)
    data = df.iloc[:, 1:5].to_numpy()
    
    # Remove any rows with NaN values
    valid_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_mask]
    
    return data, bee_id


def detect_timestamp_resets(timestamps: np.ndarray, reset_threshold: float = -0.5) -> List[int]:
    """
    Detect indices where timestamp resets occur (new video recording appended).
    
    A reset is detected when the timestamp decreases significantly.
    
    Parameters:
        timestamps: Array of timestamps
        reset_threshold: If time difference is below this, it's a reset (default: -0.5s)
        
    Returns:
        List of boundary indices (including 0 and len(timestamps))
    """
    time_diffs = np.diff(timestamps)
    reset_indices = np.where(time_diffs < reset_threshold)[0] + 1
    return [0] + list(reset_indices) + [len(timestamps)]


def make_timestamps_monotonic(data: np.ndarray) -> np.ndarray:
    """
    Process timestamps to handle resets by making them monotonically increasing.
    
    When a reset is detected, subsequent timestamps are offset to continue
    from where the previous session ended.
    
    Parameters:
        data: numpy array with columns [time, x, y, z]
        
    Returns:
        data array with adjusted timestamps (modified in place)
    """
    timestamps = data[:, 0].copy()
    reset_indices = detect_timestamp_resets(timestamps)
    
    # Make timestamps monotonic by adding offsets after each reset
    cumulative_offset = 0
    for i in range(len(reset_indices) - 1):
        start_idx = reset_indices[i]
        end_idx = reset_indices[i + 1]
        
        if i > 0:
            # Add a small gap (0.1s) between sessions to mark the boundary
            cumulative_offset = timestamps[start_idx - 1] + 0.1
        
        timestamps[start_idx:end_idx] += cumulative_offset
    
    data[:, 0] = timestamps
    return data


def split_by_gaps(data: np.ndarray, gap_threshold: float) -> List[np.ndarray]:
    """
    Split data into segments based on gaps exceeding the threshold.
    
    Parameters:
        data: numpy array with columns [time, x, y, z]
        gap_threshold: Maximum gap in seconds before splitting
        
    Returns:
        List of numpy arrays, one per continuous segment
    """
    timestamps = data[:, 0]
    time_diffs = np.diff(timestamps)
    gap_indices = np.where(time_diffs > gap_threshold)[0] + 1
    
    split_indices = [0] + list(gap_indices) + [len(timestamps)]
    
    segments = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        if end_idx - start_idx >= 3:  # Need at least 3 points for cubic spline
            segments.append(data[start_idx:end_idx])
    
    return segments


def interpolate_segment(segment: np.ndarray, target_dt: float = 1/30) -> Optional[np.ndarray]:
    """
    Interpolate a segment to regular time intervals using cubic spline.
    
    Parameters:
        segment: numpy array with columns [time, x, y, z]
        target_dt: Target time interval in seconds (default: 1/30s for 30fps)
        
    Returns:
        Interpolated numpy array with columns [time, x, y, z], or None if too short
    """
    timestamps = segment[:, 0]
    
    if len(timestamps) < 3:
        return None
    
    # Create query times at regular intervals
    t_start = timestamps[0]
    t_end = timestamps[-1]
    
    # Generate query times
    qtime = np.arange(t_start, t_end + target_dt/2, target_dt)
    
    # Ensure we don't extrapolate beyond the data
    qtime = qtime[qtime <= t_end]
    
    if len(qtime) < 2:
        return None
    
    try:
        # Create cubic splines for x, y, z
        xspline = CubicSpline(timestamps, segment[:, 1])
        yspline = CubicSpline(timestamps, segment[:, 2])
        zspline = CubicSpline(timestamps, segment[:, 3])
        
        # Query the splines
        qx = xspline(qtime)
        qy = yspline(qtime)
        qz = zspline(qtime)
        
        return np.column_stack([qtime, qx, qy, qz])
    
    except Exception as e:
        print(f"Warning: Interpolation failed for segment: {e}")
        return None


def process_bee_file(
    filepath: str,
    gap_threshold: float,
    target_dt: float
) -> Tuple[List[np.ndarray], str]:
    """
    Process a single bee file: load, handle resets, split by gaps, interpolate.
    
    Parameters:
        filepath: Path to the xlsx file
        gap_threshold: Maximum gap in seconds before splitting
        target_dt: Target time interval for interpolation
        
    Returns:
        Tuple of (list of interpolated segments, bee_id)
    """
    # Load data
    data, bee_id = load_bee_data(filepath)
    
    if len(data) < 3:
        return [], bee_id
    
    # Make timestamps monotonic (handle resets)
    data = make_timestamps_monotonic(data)
    
    # Split by gaps
    segments = split_by_gaps(data, gap_threshold)
    
    # Interpolate each segment
    interpolated_segments = []
    for segment in segments:
        interpolated = interpolate_segment(segment, target_dt)
        if interpolated is not None:
            interpolated_segments.append(interpolated)
    
    return interpolated_segments, bee_id


def interpolate_all_bees(
    input_dir: str,
    output_file: str,
    gap_threshold: float = 0.5,
    target_fps: float = 30.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process all bee files and create a combined output CSV.
    
    Parameters:
        input_dir: Directory containing xlsx files
        output_file: Path to output CSV file
        gap_threshold: Maximum gap in seconds before splitting
        target_fps: Target frame rate for interpolation
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with all interpolated data
    """
    target_dt = 1.0 / target_fps
    
    # Find all xlsx files
    xlsx_files = sorted(Path(input_dir).glob('*.xlsx'))
    
    if verbose:
        print(f"Found {len(xlsx_files)} xlsx files")
        print(f"Gap threshold: {gap_threshold} seconds")
        print(f"Target frame rate: {target_fps} fps (dt = {target_dt:.6f}s)")
        print()
    
    all_data = []
    stats = {
        'total_segments': 0,
        'total_frames': 0,
        'total_duration': 0,
        'bees_processed': 0,
    }
    
    for file_idx, filepath in enumerate(xlsx_files):
        if verbose:
            print(f"Processing {filepath.name} ({file_idx + 1}/{len(xlsx_files)})", end='')
        
        try:
            segments, bee_id = process_bee_file(str(filepath), gap_threshold, target_dt)
            
            if verbose:
                print(f" - {len(segments)} segments", end='')
            
            for seg_idx, segment in enumerate(segments):
                # Create DataFrame for this segment
                n_frames = len(segment)
                seg_df = pd.DataFrame({
                    'bee_id': bee_id,
                    'segment_id': f"{bee_id}_seg{seg_idx:03d}",
                    'time': segment[:, 0],
                    'x': segment[:, 1],
                    'y': segment[:, 2],
                    'z': segment[:, 3],
                })
                all_data.append(seg_df)
                
                stats['total_segments'] += 1
                stats['total_frames'] += n_frames
                stats['total_duration'] += segment[-1, 0] - segment[0, 0]
            
            stats['bees_processed'] += 1
            
            if verbose:
                print()
                
        except Exception as e:
            if verbose:
                print(f" - ERROR: {e}")
            continue
    
    if len(all_data) == 0:
        print("No valid data processed!")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    
    if verbose:
        print()
        print("=" * 60)
        print("INTERPOLATION SUMMARY")
        print("=" * 60)
        print(f"Bees processed: {stats['bees_processed']}")
        print(f"Total segments: {stats['total_segments']}")
        print(f"Total frames: {stats['total_frames']:,}")
        print(f"Total duration: {stats['total_duration']:.1f} seconds ({stats['total_duration']/60:.1f} minutes)")
        print(f"Average frames per segment: {stats['total_frames']/max(stats['total_segments'],1):.1f}")
        print(f"Average segment duration: {stats['total_duration']/max(stats['total_segments'],1):.2f} seconds")
        print()
        print(f"Output file: {output_file}")
        print(f"Output shape: {combined_df.shape[0]:,} rows × {combined_df.shape[1]} columns")
        print(f"Columns: {list(combined_df.columns)}")
        print("=" * 60)
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description='Interpolate bee trajectory data using cubic splines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with 0.5 second gap threshold
    python interpolate_trajectories.py -i ./bee_data -o ./output/trajectories.csv -g 0.5
    
    # Use 1 second threshold at 25 fps
    python interpolate_trajectories.py -i ./bee_data -o ./output/trajectories.csv -g 1.0 --fps 25
        """
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Directory containing bee trajectory xlsx files'
    )
    parser.add_argument(
        '--output_file', '-o',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--gap_threshold', '-g',
        type=float,
        required=True,
        help='Gap threshold (seconds). Gaps longer than this will split the trajectory into segments. '
             'Should be informed by autocorrelation analysis.'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Target frame rate for interpolation (default: 30.0)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    interpolate_all_bees(
        input_dir=args.input_dir,
        output_file=args.output_file,
        gap_threshold=args.gap_threshold,
        target_fps=args.fps,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
