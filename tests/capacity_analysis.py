#!/usr/bin/env python3
"""
Capacity Analysis for Video Steganography

Analyzes the theoretical and practical capacity limits of the video steganography system.
Generates graphs and metrics for different video parameters.
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


def analyze_video_capacity(video_path: str):
    """Analyze the capacity of a specific video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    # Calculate capacity metrics
    max_available_frames = max(0, (total_frames - START_FRAME + STEP - 1) // STEP)
    max_capacity_bits = max_available_frames - 32  # Subtract header bits
    max_capacity_chars = max(0, max_capacity_bits // 8)
    
    # Frame utilization
    used_frames = min(max_available_frames, total_frames)
    utilization_rate = used_frames / total_frames if total_frames > 0 else 0
    
    return {
        'video_path': video_path,
        'total_frames': total_frames,
        'duration_seconds': duration,
        'fps': fps,
        'resolution': (width, height),
        'max_available_frames': max_available_frames,
        'max_capacity_bits': max_capacity_bits,
        'max_capacity_chars': max_capacity_chars,
        'utilization_rate': utilization_rate,
        'bits_per_second': max_capacity_bits / duration if duration > 0 else 0,
        'chars_per_second': max_capacity_chars / duration if duration > 0 else 0
    }


def generate_theoretical_capacity_data():
    """Generate theoretical capacity data for different video parameters."""
    # Common video durations (in seconds)
    durations = np.array([10, 30, 60, 120, 300, 600, 1800, 3600])
    
    # Common frame rates
    fps_values = [24, 25, 30, 60]
    
    results = {}
    
    for fps in fps_values:
        capacities_chars = []
        capacities_bits = []
        
        for duration in durations:
            total_frames = int(duration * fps)
            max_available_frames = max(0, (total_frames - START_FRAME + STEP - 1) // STEP)
            max_capacity_bits = max(0, max_available_frames - 32)
            max_capacity_chars = max(0, max_capacity_bits // 8)
            
            capacities_chars.append(max_capacity_chars)
            capacities_bits.append(max_capacity_bits)
        
        results[fps] = {
            'durations': durations,
            'capacities_chars': np.array(capacities_chars),
            'capacities_bits': np.array(capacities_bits)
        }
    
    return results


def plot_capacity_analysis(theoretical_data, actual_video_data=None):
    """Create capacity analysis plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Capacity vs Duration for different FPS
    ax1.set_title('Video Capacity vs Duration (Different Frame Rates)')
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_ylabel('Maximum Message Capacity (characters)')
    ax1.grid(True, alpha=0.3)
    
    for fps, data in theoretical_data.items():
        ax1.plot(data['durations'], data['capacities_chars'], 
                marker='o', label=f'{fps} FPS', linewidth=2)
    
    if actual_video_data:
        ax1.scatter([actual_video_data['duration_seconds']], 
                   [actual_video_data['max_capacity_chars']], 
                   color='red', s=100, marker='*', 
                   label='Actual Video', zorder=5)
    
    ax1.legend()
    ax1.set_xlim(0, max(theoretical_data[24]['durations']) * 1.1)
    
    # Plot 2: Capacity in bits vs Duration
    ax2.set_title('Bit Capacity vs Duration')
    ax2.set_xlabel('Duration (seconds)')
    ax2.set_ylabel('Maximum Bit Capacity')
    ax2.grid(True, alpha=0.3)
    
    # Use 30 FPS as reference
    ref_data = theoretical_data[30]
    ax2.plot(ref_data['durations'], ref_data['capacities_bits'], 
            'b-o', linewidth=2, label='30 FPS')
    
    if actual_video_data:
        ax2.scatter([actual_video_data['duration_seconds']], 
                   [actual_video_data['max_capacity_bits']], 
                   color='red', s=100, marker='*', 
                   label='Actual Video', zorder=5)
    
    ax2.legend()
    
    # Plot 3: Capacity Rate (chars per second)
    ax3.set_title('Message Capacity Rate')
    ax3.set_xlabel('Duration (seconds)')
    ax3.set_ylabel('Characters per Second')
    ax3.grid(True, alpha=0.3)
    
    for fps, data in theoretical_data.items():
        chars_per_sec = data['capacities_chars'] / data['durations']
        ax3.plot(data['durations'], chars_per_sec, 
                marker='s', label=f'{fps} FPS', linewidth=2)
    
    if actual_video_data:
        ax3.scatter([actual_video_data['duration_seconds']], 
                   [actual_video_data['chars_per_second']], 
                   color='red', s=100, marker='*', 
                   label='Actual Video', zorder=5)
    
    ax3.legend()
    
    # Plot 4: Frame Utilization Analysis
    ax4.set_title('Frame Utilization vs Video Length')
    ax4.set_xlabel('Total Frames')
    ax4.set_ylabel('Utilization Rate (%)')
    ax4.grid(True, alpha=0.3)
    
    total_frames_range = np.arange(100, 10000, 100)
    utilization_rates = []
    
    for total_frames in total_frames_range:
        max_available_frames = max(0, (total_frames - START_FRAME + STEP - 1) // STEP)
        utilization = max_available_frames / total_frames if total_frames > 0 else 0
        utilization_rates.append(utilization * 100)
    
    ax4.plot(total_frames_range, utilization_rates, 'g-', linewidth=2, 
            label=f'STEP={STEP}, START={START_FRAME}')
    
    if actual_video_data:
        ax4.scatter([actual_video_data['total_frames']], 
                   [actual_video_data['utilization_rate'] * 100], 
                   color='red', s=100, marker='*', 
                   label='Actual Video', zorder=5)
    
    ax4.legend()
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig


def print_capacity_report(video_data):
    """Print a detailed capacity analysis report."""
    print("\n" + "="*60)
    print("VIDEO CAPACITY ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nVideo Information:")
    print(f"  File: {os.path.basename(video_data['video_path'])}")
    print(f"  Duration: {video_data['duration_seconds']:.2f} seconds")
    print(f"  Total Frames: {video_data['total_frames']:,}")
    print(f"  Frame Rate: {video_data['fps']:.2f} FPS")
    print(f"  Resolution: {video_data['resolution'][0]}x{video_data['resolution'][1]}")
    
    print(f"\nCapacity Analysis:")
    print(f"  Available Embedding Frames: {video_data['max_available_frames']:,}")
    print(f"  Frame Utilization Rate: {video_data['utilization_rate']*100:.1f}%")
    print(f"  Maximum Bit Capacity: {video_data['max_capacity_bits']:,} bits")
    print(f"  Maximum Character Capacity: {video_data['max_capacity_chars']:,} characters")
    
    print(f"\nCapacity Rates:")
    print(f"  Bits per Second: {video_data['bits_per_second']:.2f}")
    print(f"  Characters per Second: {video_data['chars_per_second']:.2f}")
    
    print(f"\nEmbedding Parameters:")
    print(f"  Start Frame: {START_FRAME}")
    print(f"  Step Size: {STEP}")
    print(f"  Header Overhead: 32 bits (4 characters)")
    
    # Practical examples
    print(f"\nPractical Examples:")
    example_messages = [
        ("Short message", 50),
        ("Tweet-length", 280),
        ("Paragraph", 500),
        ("Full page", 2000),
        ("Document", 5000)
    ]
    
    for name, chars in example_messages:
        if chars <= video_data['max_capacity_chars']:
            percentage = (chars / video_data['max_capacity_chars']) * 100
            print(f"  ✓ {name} ({chars} chars): {percentage:.1f}% of capacity")
        else:
            print(f"  ✗ {name} ({chars} chars): Exceeds capacity")


def main():
    """Main function to run capacity analysis."""
    # Check for input video
    default_video = "videos/sample.mp4"
    video_paths = []
    
    if os.path.exists(default_video):
        video_paths.append(default_video)
    
    # Look for any video files in videos directory
    if os.path.exists("videos"):
        for file in os.listdir("videos"):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                path = os.path.join("videos", file)
                if path not in video_paths:
                    video_paths.append(path)
    
    if not video_paths:
        print("No video files found. Creating theoretical analysis only.")
        print("Place a video file in 'videos/sample.mp4' for actual video analysis.")
        actual_data = None
    else:
        print(f"Analyzing video: {video_paths[0]}")
        actual_data = analyze_video_capacity(video_paths[0])
        print_capacity_report(actual_data)
    
    # Generate theoretical data
    print("\nGenerating theoretical capacity curves...")
    theoretical_data = generate_theoretical_capacity_data()
    
    # Create plots
    print("Creating capacity analysis plots...")
    fig = plot_capacity_analysis(theoretical_data, actual_data)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save plots
    output_path = "results/capacity_analysis_plots.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)


if __name__ == "__main__":
    main()
