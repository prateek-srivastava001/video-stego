#!/usr/bin/env python3
"""
Frame Utilization Analysis for Video Steganography

Analyzes how frames are selected and utilized for embedding data.
Visualizes the frame selection pattern and efficiency metrics.
"""

import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import START_FRAME, STEP
from utils.utils import get_frame_indices


def analyze_frame_utilization(video_path: str, max_message_length: int = 1000):
    """Analyze frame utilization patterns for a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    # Calculate frame indices for different message lengths
    message_lengths = [10, 50, 100, 250, 500, 1000]
    frame_patterns = {}
    
    for msg_len in message_lengths:
        if msg_len <= max_message_length:
            # Convert characters to bits (8 bits per char + 32 bit header)
            total_bits = (msg_len * 8) + 32
            frame_indices = get_frame_indices(total_bits, START_FRAME, STEP)
            
            # Only keep frames that exist in the video
            valid_indices = [idx for idx in frame_indices if idx < total_frames]
            
            frame_patterns[msg_len] = {
                'required_bits': total_bits,
                'frame_indices': valid_indices,
                'frames_used': len(valid_indices),
                'max_frame': max(valid_indices) if valid_indices else 0,
                'span_duration': (max(valid_indices) / fps) if valid_indices and fps > 0 else 0
            }
    
    # Calculate overall utilization metrics
    max_possible_frames = max(0, (total_frames - START_FRAME + STEP - 1) // STEP)
    
    utilization_data = {
        'video_path': video_path,
        'total_frames': total_frames,
        'duration': duration,
        'fps': fps,
        'start_frame': START_FRAME,
        'step': STEP,
        'max_possible_embedding_frames': max_possible_frames,
        'overall_utilization_rate': max_possible_frames / total_frames if total_frames > 0 else 0,
        'frame_patterns': frame_patterns
    }
    
    return utilization_data


def create_frame_pattern_visualization(utilization_data):
    """Create visualizations of frame utilization patterns."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    total_frames = utilization_data['total_frames']
    fps = utilization_data['fps']
    duration = utilization_data['duration']
    
    # Plot 1: Frame Selection Pattern
    ax1.set_title('Frame Selection Pattern (First 200 Frames)')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Selected for Embedding')
    
    # Show frame selection pattern for first 200 frames
    frame_range = min(200, total_frames)
    frames = np.arange(frame_range)
    selected = np.zeros(frame_range)
    
    # Mark selected frames
    for i in range(frame_range):
        if i >= START_FRAME and (i - START_FRAME) % STEP == 0:
            selected[i] = 1
    
    ax1.bar(frames, selected, width=1, alpha=0.7, color='blue', 
           label=f'Selected (START={START_FRAME}, STEP={STEP})')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Utilization vs Message Length
    ax2.set_title('Frame Utilization vs Message Length')
    ax2.set_xlabel('Message Length (characters)')
    ax2.set_ylabel('Frames Used')
    
    msg_lengths = []
    frames_used = []
    
    for msg_len, pattern in utilization_data['frame_patterns'].items():
        msg_lengths.append(msg_len)
        frames_used.append(pattern['frames_used'])
    
    ax2.plot(msg_lengths, frames_used, 'bo-', linewidth=2, markersize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add efficiency line
    ax2_twin = ax2.twinx()
    efficiency = [frames / total_frames * 100 for frames in frames_used]
    ax2_twin.plot(msg_lengths, efficiency, 'r^--', linewidth=2, markersize=6, 
                 label='Efficiency %')
    ax2_twin.set_ylabel('Frame Efficiency (%)', color='red')
    ax2_twin.legend(loc='upper right')
    
    # Plot 3: Temporal Distribution
    ax3.set_title('Temporal Distribution of Embedded Data')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Message Length (characters)')
    
    # Show how different message lengths span across time
    for msg_len, pattern in utilization_data['frame_patterns'].items():
        if pattern['frame_indices']:
            time_points = [idx / fps for idx in pattern['frame_indices']]
            ax3.scatter(time_points, [msg_len] * len(time_points), 
                       alpha=0.6, s=20, label=f'{msg_len} chars')
    
    ax3.set_xlim(0, duration)
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: Frame Utilization Efficiency
    ax4.set_title('Frame Utilization Efficiency Analysis')
    ax4.set_xlabel('Video Duration (seconds)')
    ax4.set_ylabel('Utilization Rate (%)')
    
    # Theoretical utilization for different video lengths
    durations = np.linspace(10, duration * 2, 50)
    utilization_rates = []
    
    for dur in durations:
        frames_in_duration = int(dur * fps)
        if frames_in_duration > START_FRAME:
            max_embedding_frames = (frames_in_duration - START_FRAME) // STEP
            utilization = max_embedding_frames / frames_in_duration * 100
        else:
            utilization = 0
        utilization_rates.append(utilization)
    
    ax4.plot(durations, utilization_rates, 'g-', linewidth=2, 
            label=f'Theoretical (STEP={STEP})')
    
    # Mark actual video
    actual_utilization = utilization_data['overall_utilization_rate'] * 100
    ax4.scatter([duration], [actual_utilization], color='red', s=100, 
               marker='*', label='Actual Video', zorder=5)
    
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(0, max(utilization_rates) * 1.1)
    
    plt.tight_layout()
    return fig


def create_frame_timeline_plot(utilization_data, msg_length=100):
    """Create a detailed timeline showing frame selection for a specific message."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    total_frames = utilization_data['total_frames']
    fps = utilization_data['fps']
    duration = utilization_data['duration']
    
    if msg_length not in utilization_data['frame_patterns']:
        print(f"Message length {msg_length} not analyzed")
        return fig
    
    pattern = utilization_data['frame_patterns'][msg_length]
    frame_indices = pattern['frame_indices']
    
    # Plot 1: Frame selection timeline (full video)
    ax1.set_title(f'Frame Selection Timeline - {msg_length} Character Message')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Frame Number')
    
    # All frames
    all_times = np.arange(total_frames) / fps
    ax1.plot(all_times, np.arange(total_frames), 'lightgray', alpha=0.5, 
            linewidth=1, label='All Frames')
    
    # Selected frames
    if frame_indices:
        selected_times = [idx / fps for idx in frame_indices]
        ax1.scatter(selected_times, frame_indices, color='red', s=30, 
                   alpha=0.8, label=f'Selected Frames ({len(frame_indices)})')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, duration)
    
    # Plot 2: Embedding density over time
    ax2.set_title('Embedding Density Over Time')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Embeddings per Second')
    
    # Calculate embedding density in time windows
    window_size = 10  # seconds
    time_windows = np.arange(0, duration, window_size)
    densities = []
    
    for t in time_windows:
        window_start = t
        window_end = t + window_size
        count = sum(1 for idx in frame_indices 
                   if window_start <= idx / fps < window_end)
        density = count / window_size
        densities.append(density)
    
    ax2.bar(time_windows, densities, width=window_size*0.8, alpha=0.7, 
           color='blue', label=f'{window_size}s windows')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, duration)
    
    plt.tight_layout()
    return fig


def print_utilization_report(utilization_data):
    """Print detailed frame utilization analysis."""
    print("\n" + "="*60)
    print("FRAME UTILIZATION ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nVideo Information:")
    print(f"  File: {os.path.basename(utilization_data['video_path'])}")
    print(f"  Total Frames: {utilization_data['total_frames']:,}")
    print(f"  Duration: {utilization_data['duration']:.2f} seconds")
    print(f"  Frame Rate: {utilization_data['fps']:.2f} FPS")
    
    print(f"\nEmbedding Parameters:")
    print(f"  Start Frame: {utilization_data['start_frame']}")
    print(f"  Step Size: {utilization_data['step']}")
    print(f"  Maximum Embedding Frames: {utilization_data['max_possible_embedding_frames']:,}")
    print(f"  Overall Utilization Rate: {utilization_data['overall_utilization_rate']*100:.2f}%")
    
    print(f"\nFrame Usage by Message Length:")
    print(f"  {'Length':<8} {'Frames':<8} {'Span':<10} {'Efficiency':<12} {'Max Frame':<10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")
    
    for msg_len, pattern in utilization_data['frame_patterns'].items():
        efficiency = (pattern['frames_used'] / utilization_data['total_frames']) * 100
        span_pct = (pattern['span_duration'] / utilization_data['duration']) * 100 if utilization_data['duration'] > 0 else 0
        
        print(f"  {msg_len:<8} {pattern['frames_used']:<8} "
              f"{span_pct:<9.1f}% {efficiency:<11.2f}% {pattern['max_frame']:<10}")
    
    print(f"\nUtilization Efficiency Analysis:")
    total_frames = utilization_data['total_frames']
    theoretical_max = (total_frames - START_FRAME) // STEP if total_frames > START_FRAME else 0
    actual_max = utilization_data['max_possible_embedding_frames']
    
    print(f"  Theoretical Maximum Frames: {theoretical_max:,}")
    print(f"  Actual Maximum Frames: {actual_max:,}")
    print(f"  Frame Selection Efficiency: {(actual_max/theoretical_max)*100:.1f}%" if theoretical_max > 0 else "N/A")
    
    # Frame spacing analysis
    avg_spacing = STEP / utilization_data['fps'] if utilization_data['fps'] > 0 else 0
    print(f"  Average Time Between Embeddings: {avg_spacing:.3f} seconds")
    print(f"  Embeddings Per Second: {1/avg_spacing:.2f}" if avg_spacing > 0 else "N/A")


def main():
    """Main function to run frame utilization analysis."""
    # Check for input video
    default_video = "videos/sample.mp4"
    video_path = None
    
    if os.path.exists(default_video):
        video_path = default_video
    else:
        # Look for any video files
        if os.path.exists("videos"):
            for file in os.listdir("videos"):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join("videos", file)
                    break
    
    if not video_path:
        print("Error: No video files found.")
        print("Place a video file in 'videos/sample.mp4' or any video in 'videos/' directory.")
        return
    
    print(f"Analyzing frame utilization for: {video_path}")
    
    # Analyze frame utilization
    utilization_data = analyze_frame_utilization(video_path)
    
    # Print report
    print_utilization_report(utilization_data)
    
    # Create visualizations
    print("\nGenerating frame utilization plots...")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Main utilization plots
    fig1 = create_frame_pattern_visualization(utilization_data)
    fig1.savefig("results/frame_utilization_analysis.png", dpi=300, bbox_inches='tight')
    print("Frame utilization plots saved to: results/frame_utilization_analysis.png")
    plt.close(fig1)
    
    # Timeline plot for specific message length
    fig2 = create_frame_timeline_plot(utilization_data, msg_length=100)
    fig2.savefig("results/frame_timeline_100chars.png", dpi=300, bbox_inches='tight')
    print("Frame timeline plot saved to: results/frame_timeline_100chars.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
