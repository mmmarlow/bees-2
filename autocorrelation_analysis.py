#!/usr/bin/env python3
"""
Autocorrelation Analysis for Bee Trajectory Data
=================================================
This script analyses temporal autocorrelation of bee trajectory coordinates
to help determine an appropriate gap threshold for interpolation.

Usage:
    python autocorrelation_analysis.py --input_dir /path/to/xlsx/files --output_dir /path/to/output

Output:
    - Autocorrelation plots for x, y, z coordinates
    - Summary statistics CSV
    - Recommendations for gap threshold based on autocorrelation decay
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


def load_bee_data(filepath: str) -> np.ndarray:
    """
    Load bee trajectory data from an Excel file.
    
    Parameters:
        filepath: Path to the xlsx file
        
    Returns:
        numpy array with columns [time, x, y, z]
    """
    # Load only the required columns: G (ID_No), AT (time), AU (x), AV (y), AW (z)
    # We only need time and coordinates for this analysis
    df = pd.read_excel(filepath, usecols='AT:AW', skiprows=0)
    data = df.to_numpy()
    
    # Remove any rows with NaN values
    data = data[~np.isnan(data).any(axis=1)]
    
    return data


def detect_timestamp_resets(timestamps: np.ndarray, reset_threshold: float = -0.5) -> List[int]:
    """
    Detect indices where timestamp resets occur (new video recording appended).
    
    A reset is detected when the timestamp decreases significantly.
    
    Parameters:
        timestamps: Array of timestamps
        reset_threshold: If time difference is below this, it's a reset (default: -0.5s)
        
    Returns:
        List of indices where resets occur
    """
    time_diffs = np.diff(timestamps)
    reset_indices = np.where(time_diffs < reset_threshold)[0] + 1
    return [0] + list(reset_indices) + [len(timestamps)]


def split_into_sessions(data: np.ndarray) -> List[np.ndarray]:
    """
    Split data into separate recording sessions based on timestamp resets.
    
    Parameters:
        data: numpy array with columns [time, x, y, z]
        
    Returns:
        List of numpy arrays, one per session
    """
    timestamps = data[:, 0]
    reset_indices = detect_timestamp_resets(timestamps)
    
    sessions = []
    for i in range(len(reset_indices) - 1):
        start_idx = reset_indices[i]
        end_idx = reset_indices[i + 1]
        if end_idx - start_idx > 10:  # Only keep sessions with more than 10 points
            sessions.append(data[start_idx:end_idx])
    
    return sessions


def split_session_by_gaps(session: np.ndarray, gap_threshold: float = 1.0) -> List[np.ndarray]:
    """
    Split a session into segments based on gaps in the data.
    
    Parameters:
        session: numpy array with columns [time, x, y, z]
        gap_threshold: Maximum gap in seconds before splitting (default: 1.0s)
        
    Returns:
        List of numpy arrays, one per continuous segment
    """
    timestamps = session[:, 0]
    time_diffs = np.diff(timestamps)
    gap_indices = np.where(time_diffs > gap_threshold)[0] + 1
    
    split_indices = [0] + list(gap_indices) + [len(timestamps)]
    
    segments = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        if end_idx - start_idx > 10:  # Only keep segments with more than 10 points
            segments.append(session[start_idx:end_idx])
    
    return segments


def interpolate_segment(segment: np.ndarray, target_dt: float = 1/30) -> np.ndarray:
    """
    Interpolate a segment to regular time intervals using cubic spline.
    
    Parameters:
        segment: numpy array with columns [time, x, y, z]
        target_dt: Target time interval in seconds (default: 1/30s for 30fps)
        
    Returns:
        Interpolated numpy array with columns [time, x, y, z]
    """
    timestamps = segment[:, 0]
    
    # Create query times at regular intervals
    t_start = timestamps[0]
    t_end = timestamps[-1]
    qtime = np.arange(t_start, t_end, target_dt)
    
    if len(qtime) < 10:
        return None
    
    # Create cubic splines for x, y, z
    try:
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


def compute_autocorrelation(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute the autocorrelation of a signal up to max_lag.
    
    Parameters:
        signal: 1D array of values
        max_lag: Maximum lag to compute
        
    Returns:
        Array of autocorrelation values for lags 0 to max_lag
    """
    n = len(signal)
    if n < max_lag + 1:
        max_lag = n - 1
    
    # Normalise the signal
    signal = signal - np.mean(signal)
    variance = np.var(signal)
    
    if variance == 0:
        return np.zeros(max_lag + 1)
    
    autocorr = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if n - lag > 0:
            autocorr[lag] = np.sum(signal[:n-lag] * signal[lag:]) / ((n - lag) * variance)
    
    return autocorr


def analyse_autocorrelation(
    input_dir: str,
    output_dir: str,
    gap_threshold_for_analysis: float = 1.0,
    target_dt: float = 1/30,
    max_lag_seconds: float = 5.0
) -> Dict:
    """
    Main function to analyse autocorrelation across all bee files.
    
    Parameters:
        input_dir: Directory containing xlsx files
        output_dir: Directory for output files
        gap_threshold_for_analysis: Gap threshold for splitting sessions (seconds)
        target_dt: Target time interval for interpolation (seconds)
        max_lag_seconds: Maximum lag to compute autocorrelation (seconds)
        
    Returns:
        Dictionary with summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all xlsx files
    xlsx_files = list(Path(input_dir).glob('*.xlsx'))
    print(f"Found {len(xlsx_files)} xlsx files")
    
    max_lag = int(max_lag_seconds / target_dt)
    all_autocorr_x = []
    all_autocorr_y = []
    all_autocorr_z = []
    
    segment_durations = []
    
    for file_idx, filepath in enumerate(xlsx_files):
        print(f"Processing {filepath.name} ({file_idx + 1}/{len(xlsx_files)})")
        
        try:
            data = load_bee_data(str(filepath))
            sessions = split_into_sessions(data)
            
            for session in sessions:
                segments = split_session_by_gaps(session, gap_threshold_for_analysis)
                
                for segment in segments:
                    interpolated = interpolate_segment(segment, target_dt)
                    
                    if interpolated is not None and len(interpolated) > max_lag:
                        segment_duration = interpolated[-1, 0] - interpolated[0, 0]
                        segment_durations.append(segment_duration)
                        
                        # Compute autocorrelation for x, y, z
                        ac_x = compute_autocorrelation(interpolated[:, 1], max_lag)
                        ac_y = compute_autocorrelation(interpolated[:, 2], max_lag)
                        ac_z = compute_autocorrelation(interpolated[:, 3], max_lag)
                        
                        all_autocorr_x.append(ac_x)
                        all_autocorr_y.append(ac_y)
                        all_autocorr_z.append(ac_z)
                        
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
            continue
    
    print(f"\nAnalysed {len(all_autocorr_x)} segments from {len(xlsx_files)} bees")
    
    if len(all_autocorr_x) == 0:
        print("No valid segments found for analysis!")
        return {}
    
    # Compute mean autocorrelation across all segments
    all_autocorr_x = np.array(all_autocorr_x)
    all_autocorr_y = np.array(all_autocorr_y)
    all_autocorr_z = np.array(all_autocorr_z)
    
    mean_ac_x = np.mean(all_autocorr_x, axis=0)
    mean_ac_y = np.mean(all_autocorr_y, axis=0)
    mean_ac_z = np.mean(all_autocorr_z, axis=0)
    
    std_ac_x = np.std(all_autocorr_x, axis=0)
    std_ac_y = np.std(all_autocorr_y, axis=0)
    std_ac_z = np.std(all_autocorr_z, axis=0)
    
    # Time lags in seconds
    lags_seconds = np.arange(max_lag + 1) * target_dt
    
    # Find decay times (when autocorrelation drops below threshold)
    def find_decay_time(autocorr: np.ndarray, threshold: float = 0.5) -> float:
        below_threshold = np.where(autocorr < threshold)[0]
        if len(below_threshold) > 0:
            return lags_seconds[below_threshold[0]]
        return lags_seconds[-1]
    
    decay_time_x_50 = find_decay_time(mean_ac_x, 0.5)
    decay_time_y_50 = find_decay_time(mean_ac_y, 0.5)
    decay_time_z_50 = find_decay_time(mean_ac_z, 0.5)
    
    decay_time_x_25 = find_decay_time(mean_ac_x, 0.25)
    decay_time_y_25 = find_decay_time(mean_ac_y, 0.25)
    decay_time_z_25 = find_decay_time(mean_ac_z, 0.25)
    
    decay_time_x_10 = find_decay_time(mean_ac_x, 0.1)
    decay_time_y_10 = find_decay_time(mean_ac_y, 0.1)
    decay_time_z_10 = find_decay_time(mean_ac_z, 0.1)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean autocorrelation with confidence bands
    ax1 = axes[0, 0]
    ax1.plot(lags_seconds, mean_ac_x, 'b-', label='x', linewidth=2)
    ax1.plot(lags_seconds, mean_ac_y, 'g-', label='y', linewidth=2)
    ax1.plot(lags_seconds, mean_ac_z, 'r-', label='z', linewidth=2)
    ax1.fill_between(lags_seconds, mean_ac_x - std_ac_x, mean_ac_x + std_ac_x, alpha=0.2, color='blue')
    ax1.fill_between(lags_seconds, mean_ac_y - std_ac_y, mean_ac_y + std_ac_y, alpha=0.2, color='green')
    ax1.fill_between(lags_seconds, mean_ac_z - std_ac_z, mean_ac_z + std_ac_z, alpha=0.2, color='red')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='25% threshold')
    ax1.axhline(y=0.1, color='gray', linestyle='-.', alpha=0.5, label='10% threshold')
    ax1.set_xlabel('Lag (seconds)')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('Mean Autocorrelation with ±1 SD')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max_lag_seconds])
    
    # Plot 2: Zoomed view (first 2 seconds)
    ax2 = axes[0, 1]
    zoom_idx = int(2.0 / target_dt)
    ax2.plot(lags_seconds[:zoom_idx], mean_ac_x[:zoom_idx], 'b-', label='x', linewidth=2)
    ax2.plot(lags_seconds[:zoom_idx], mean_ac_y[:zoom_idx], 'g-', label='y', linewidth=2)
    ax2.plot(lags_seconds[:zoom_idx], mean_ac_z[:zoom_idx], 'r-', label='z', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=0.1, color='gray', linestyle='-.', alpha=0.5)
    ax2.set_xlabel('Lag (seconds)')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('Autocorrelation (Zoomed: 0-2 seconds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined autocorrelation (average of x, y, z)
    ax3 = axes[1, 0]
    mean_ac_combined = (mean_ac_x + mean_ac_y + mean_ac_z) / 3
    std_ac_combined = np.sqrt((std_ac_x**2 + std_ac_y**2 + std_ac_z**2) / 9)
    ax3.plot(lags_seconds, mean_ac_combined, 'k-', linewidth=2, label='Mean (x,y,z)')
    ax3.fill_between(lags_seconds, mean_ac_combined - std_ac_combined, 
                     mean_ac_combined + std_ac_combined, alpha=0.3, color='gray')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50%')
    ax3.axhline(y=0.25, color='orange', linestyle=':', alpha=0.7, label='25%')
    ax3.axhline(y=0.1, color='orange', linestyle='-.', alpha=0.7, label='10%')
    ax3.set_xlabel('Lag (seconds)')
    ax3.set_ylabel('Autocorrelation')
    ax3.set_title('Combined Autocorrelation (Mean of x, y, z)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, max_lag_seconds])
    
    # Plot 4: Segment duration histogram
    ax4 = axes[1, 1]
    ax4.hist(segment_durations, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=np.median(segment_durations), color='red', linestyle='--', 
                label=f'Median: {np.median(segment_durations):.2f}s')
    ax4.set_xlabel('Segment Duration (seconds)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Distribution of Segment Durations (n={len(segment_durations)})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'autocorrelation_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    summary = {
        'num_bees': len(xlsx_files),
        'num_segments_analysed': len(all_autocorr_x),
        'total_segment_duration_seconds': np.sum(segment_durations),
        'mean_segment_duration_seconds': np.mean(segment_durations),
        'median_segment_duration_seconds': np.median(segment_durations),
        'decay_time_x_50pct': decay_time_x_50,
        'decay_time_y_50pct': decay_time_y_50,
        'decay_time_z_50pct': decay_time_z_50,
        'decay_time_x_25pct': decay_time_x_25,
        'decay_time_y_25pct': decay_time_y_25,
        'decay_time_z_25pct': decay_time_z_25,
        'decay_time_x_10pct': decay_time_x_10,
        'decay_time_y_10pct': decay_time_y_10,
        'decay_time_z_10pct': decay_time_z_10,
        'gap_threshold_used_for_analysis': gap_threshold_for_analysis,
        'interpolation_dt': target_dt,
        'max_lag_seconds': max_lag_seconds,
    }
    
    # Save summary to CSV
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'autocorrelation_summary.csv'), index=False)
    
    # Save autocorrelation data for further analysis
    autocorr_df = pd.DataFrame({
        'lag_seconds': lags_seconds,
        'mean_autocorr_x': mean_ac_x,
        'mean_autocorr_y': mean_ac_y,
        'mean_autocorr_z': mean_ac_z,
        'std_autocorr_x': std_ac_x,
        'std_autocorr_y': std_ac_y,
        'std_autocorr_z': std_ac_z,
        'mean_autocorr_combined': mean_ac_combined,
    })
    autocorr_df.to_csv(os.path.join(output_dir, 'autocorrelation_data.csv'), index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("AUTOCORRELATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Number of bees processed: {len(xlsx_files)}")
    print(f"Number of segments analysed: {len(all_autocorr_x)}")
    print(f"Total data duration: {np.sum(segment_durations):.1f} seconds")
    print(f"Mean segment duration: {np.mean(segment_durations):.2f} seconds")
    print(f"Median segment duration: {np.median(segment_durations):.2f} seconds")
    print()
    print("DECAY TIMES (when autocorrelation drops below threshold):")
    print("-" * 60)
    print(f"{'Threshold':<15} {'X (s)':<12} {'Y (s)':<12} {'Z (s)':<12}")
    print(f"{'50%':<15} {decay_time_x_50:<12.3f} {decay_time_y_50:<12.3f} {decay_time_z_50:<12.3f}")
    print(f"{'25%':<15} {decay_time_x_25:<12.3f} {decay_time_y_25:<12.3f} {decay_time_z_25:<12.3f}")
    print(f"{'10%':<15} {decay_time_x_10:<12.3f} {decay_time_y_10:<12.3f} {decay_time_z_10:<12.3f}")
    print()
    print("RECOMMENDATION:")
    print("-" * 60)
    avg_decay_50 = (decay_time_x_50 + decay_time_y_50 + decay_time_z_50) / 3
    avg_decay_25 = (decay_time_x_25 + decay_time_y_25 + decay_time_z_25) / 3
    print(f"Average decay time to 50% autocorrelation: {avg_decay_50:.3f} seconds")
    print(f"Average decay time to 25% autocorrelation: {avg_decay_25:.3f} seconds")
    print()
    print("Consider using a gap threshold around the 25-50% decay time.")
    print(f"Suggested range: {avg_decay_50:.2f}s to {avg_decay_25:.2f}s")
    print()
    print(f"Output saved to: {output_dir}")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Analyse temporal autocorrelation of bee trajectory data'
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Directory containing bee trajectory xlsx files'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help='Directory for output files'
    )
    parser.add_argument(
        '--gap_threshold', '-g',
        type=float,
        default=1.0,
        help='Gap threshold (seconds) for splitting sessions during analysis (default: 1.0)'
    )
    parser.add_argument(
        '--max_lag', '-m',
        type=float,
        default=5.0,
        help='Maximum lag (seconds) for autocorrelation computation (default: 5.0)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Target frame rate for interpolation (default: 30.0)'
    )
    
    args = parser.parse_args()
    
    analyse_autocorrelation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gap_threshold_for_analysis=args.gap_threshold,
        target_dt=1.0/args.fps,
        max_lag_seconds=args.max_lag
    )


if __name__ == '__main__':
    main()
