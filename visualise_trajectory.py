#!/usr/bin/env python3
"""
Bee Trajectory Visualisation Script
====================================
This script selects a random session from the bee trajectory data and plots:
1. 3D trajectory (original and interpolated)
2. X coordinate over time
3. Y coordinate over time
4. Z coordinate over time

Usage:
    python visualise_trajectory.py --input_dir /path/to/xlsx/files --output_dir /path/to/output --gap_threshold 0.5

Optional:
    --bee_file: Specify a particular xlsx file instead of random selection
    --session_index: Specify a particular session index within the file
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
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
    
    Parameters:
        timestamps: Array of timestamps
        reset_threshold: If time difference is below this, it's a reset (default: -0.5s)
        
    Returns:
        List of boundary indices (including 0 and len(timestamps))
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


def interpolate_session(session: np.ndarray, target_dt: float = 1/30) -> Optional[np.ndarray]:
    """
    Interpolate a session to regular time intervals using cubic spline.
    
    Parameters:
        session: numpy array with columns [time, x, y, z]
        target_dt: Target time interval in seconds (default: 1/30s for 30fps)
        
    Returns:
        Interpolated numpy array with columns [time, x, y, z], or None if too short
    """
    timestamps = session[:, 0]
    
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
        xspline = CubicSpline(timestamps, session[:, 1])
        yspline = CubicSpline(timestamps, session[:, 2])
        zspline = CubicSpline(timestamps, session[:, 3])
        
        # Query the splines
        qx = xspline(qtime)
        qy = yspline(qtime)
        qz = zspline(qtime)
        
        return np.column_stack([qtime, qx, qy, qz])
    
    except Exception as e:
        print(f"Warning: Interpolation failed for session: {e}")
        return None


def plot_trajectory(
    original: np.ndarray,
    interpolated: np.ndarray,
    bee_id: str,
    session_index: int,
    output_path: Optional[str] = None
):
    """
    Plot the trajectory with 4 subplots: 3D view, X vs time, Y vs time, Z vs time.
    
    Parameters:
        original: Original data array [time, x, y, z]
        interpolated: Interpolated data array [time, x, y, z]
        bee_id: Bee identifier for title
        session_index: Session number for title
        output_path: Path to save the figure (if None, displays interactively)
    """
    fig = plt.figure(figsize=(16, 4))
    
    # Plot 1: 3D Trajectory
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.plot(interpolated[:, 1], interpolated[:, 2], interpolated[:, 3], 
             'r-', linewidth=1, label='Interpolated')
    ax1.scatter(original[:, 1], original[:, 2], original[:, 3], 
                c='blue', s=10, alpha=0.7, label='Original')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'3D Trajectory - Session {session_index}')
    ax1.legend(loc='upper left', fontsize=8)
    
    # Plot 2: X Coordinate over Time
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(interpolated[:, 0], interpolated[:, 1], 'r-', linewidth=1, label='Interpolated')
    ax2.scatter(original[:, 0], original[:, 1], c='blue', s=15, alpha=0.7, label='Original')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('X (mm)')
    ax2.set_title('X Coordinate over Time')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Y Coordinate over Time
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(interpolated[:, 0], interpolated[:, 2], 'r-', linewidth=1, label='Interpolated')
    ax3.scatter(original[:, 0], original[:, 2], c='blue', s=15, alpha=0.7, label='Original')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_title('Y Coordinate over Time')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Z Coordinate over Time
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(interpolated[:, 0], interpolated[:, 3], 'r-', linewidth=1, label='Interpolated')
    ax4.scatter(original[:, 0], original[:, 3], c='blue', s=15, alpha=0.7, label='Original')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('Z Coordinate over Time')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Bee {bee_id} - Session {session_index}', fontsize=12, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualise_random_session(
    input_dir: str,
    output_dir: Optional[str] = None,
    target_fps: float = 30.0,
    bee_file: Optional[str] = None,
    session_index: Optional[int] = None,
    seed: Optional[int] = None
):
    """
    Select a random session and create visualisation.
    
    Parameters:
        input_dir: Directory containing xlsx files
        output_dir: Directory for output files (if None, displays interactively)
        target_fps: Target frame rate for interpolation
        bee_file: Specific xlsx file to use (if None, random selection)
        session_index: Specific session index to use (if None, random selection)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    target_dt = 1.0 / target_fps
    
    # Find all xlsx files
    xlsx_files = sorted(Path(input_dir).glob('*.xlsx'))
    
    if len(xlsx_files) == 0:
        print(f"No xlsx files found in {input_dir}")
        return
    
    print(f"Found {len(xlsx_files)} xlsx files")
    
    # Select bee file
    if bee_file:
        filepath = Path(bee_file)
        if not filepath.exists():
            filepath = Path(input_dir) / bee_file
        if not filepath.exists():
            print(f"File not found: {bee_file}")
            return
    else:
        filepath = random.choice(xlsx_files)
    
    print(f"Selected file: {filepath.name}")
    
    # Load data
    data, bee_id = load_bee_data(str(filepath))
    print(f"Bee ID: {bee_id}")
    print(f"Total data points: {len(data)}")
    
    # Split into sessions
    sessions = split_into_sessions(data)
    print(f"Number of sessions: {len(sessions)}")
    
    if len(sessions) == 0:
        print("No valid sessions found in this file")
        return
    
    # Select session
    if session_index is not None:
        if session_index < 0 or session_index >= len(sessions):
            print(f"Session index {session_index} out of range (0-{len(sessions)-1})")
            return
        selected_session_idx = session_index
    else:
        selected_session_idx = random.randint(0, len(sessions) - 1)
    
    session = sessions[selected_session_idx]
    print(f"Selected session {selected_session_idx}: {len(session)} data points, "
          f"duration {session[-1, 0] - session[0, 0]:.2f}s")
    
    # Interpolate
    interpolated = interpolate_session(session, target_dt)
    
    if interpolated is None:
        print("Failed to interpolate session")
        return
    
    print(f"Interpolated to {len(interpolated)} points")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"trajectory_{bee_id}_session{selected_session_idx}.png")
    else:
        output_path = None
    
    # Plot
    plot_trajectory(session, interpolated, bee_id, selected_session_idx, output_path)
    
    # Print summary
    print()
    print("=" * 50)
    print("VISUALISATION SUMMARY")
    print("=" * 50)
    print(f"Bee ID: {bee_id}")
    print(f"Session: {selected_session_idx}")
    print(f"Original points: {len(session)}")
    print(f"Interpolated points: {len(interpolated)}")
    print(f"Duration: {session[-1, 0] - session[0, 0]:.2f} seconds")
    print(f"Original avg frame rate: {len(session) / (session[-1, 0] - session[0, 0]):.1f} fps")
    print(f"Interpolated frame rate: {target_fps} fps")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Visualise bee trajectory data with original and interpolated plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Random session, display interactively
    python visualise_trajectory.py -i ./bee_data
    
    # Random session, save to file
    python visualise_trajectory.py -i ./bee_data -o ./plots
    
    # Specific file and session
    python visualise_trajectory.py -i ./bee_data -o ./plots --bee_file bee001.xlsx --session_index 2
    
    # With random seed for reproducibility
    python visualise_trajectory.py -i ./bee_data -o ./plots --seed 42
        """
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
        default=None,
        help='Directory for output files (if not specified, displays interactively)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Target frame rate for interpolation (default: 30.0)'
    )
    parser.add_argument(
        '--bee_file',
        type=str,
        default=None,
        help='Specific xlsx file to use (if not specified, random selection)'
    )
    parser.add_argument(
        '--session_index',
        type=int,
        default=None,
        help='Specific session index to use (if not specified, random selection)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    visualise_random_session(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_fps=args.fps,
        bee_file=args.bee_file,
        session_index=args.session_index,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
