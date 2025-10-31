#!/usr/bin/env python3
"""
Improvement Demonstration Analysis

This script creates specific demonstrations showing the key benefits of each improvement:
1. Adaptive Threshold: Better robustness against compression artifacts
2. Multi-Channel: Higher capacity utilization and better data distribution

Creates before/after graphs and metrics that clearly show improvement benefits.
"""

import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import subprocess

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.embed import embed_message_simple
from utils.extract import extract_message_simple
from utils.embed_adaptive import embed_message_adaptive, extract_message_adaptive
from utils.embed_multichannel import embed_message_multichannel, extract_message_multichannel
from tests.quality_impact import calculate_psnr, calculate_ssim
from config import VAL_ZERO, VAL_ONE, THRESH


def create_synthetic_video_with_capacity(width=640, height=480, fps=30, duration_sec=10, filename="videos/demo_video.mp4"):
    """Create a synthetic video with sufficient capacity for demonstrations."""
    os.makedirs("videos", exist_ok=True)
    
    # Create video with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = int(fps * duration_sec)
    
    print(f"Creating demonstration video: {total_frames} frames, {width}x{height}")
    
    for frame_idx in range(total_frames):
        # Create a frame with varied content
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some pattern to make it realistic
        gradient_x = np.linspace(0, 255, width)
        gradient_y = np.linspace(0, 255, height)
        xx, yy = np.meshgrid(gradient_x, gradient_y)
        
        # Create different patterns in different regions
        frame[:, :, 0] = (xx + frame_idx * 2) % 256  # Blue channel with motion
        frame[:, :, 1] = (yy + frame_idx) % 256      # Green channel
        frame[:, :, 2] = ((xx + yy) / 2 + frame_idx * 0.5) % 256  # Red channel
        
        # Add some noise for realism
        noise = np.random.normal(0, 10, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()
    
    # Convert to proper format with ffmpeg
    temp_file = filename.replace('.mp4', '_temp.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-i', filename,
        '-c:v', 'libx264', '-crf', '23',
        '-pix_fmt', 'yuv420p', temp_file
    ], capture_output=True)
    
    if os.path.exists(temp_file):
        os.replace(temp_file, filename)
    
    print(f"Created video: {filename}")
    return filename


def demonstrate_adaptive_threshold_robustness():
    """Demonstrate how adaptive threshold improves robustness against compression."""
    print("\n" + "="*60)
    print("ADAPTIVE THRESHOLD ROBUSTNESS DEMONSTRATION")
    print("="*60)
    
    # Create test video if it doesn't exist
    demo_video = "videos/demo_video.mp4"
    if not os.path.exists(demo_video):
        create_synthetic_video_with_capacity()
    
    test_message = "Adaptive threshold test - compression robustness analysis!"
    
    # Test with different compression levels
    compression_levels = [15, 20, 25, 30, 35]  # CRF values (lower = better quality)
    
    results = {
        'compression_levels': compression_levels,
        'original_accuracy': [],
        'adaptive_accuracy': [],
        'original_psnr': [],
        'adaptive_psnr': []
    }
    
    for crf in compression_levels:
        print(f"\nTesting with compression level CRF={crf}")
        
        try:
            # Test original system
            original_output = f"videos/original_crf{crf}.mp4"
            embed_message_simple(demo_video, original_output, test_message)
            
            # Apply additional compression
            compressed_original = f"videos/original_crf{crf}_compressed.mp4"
            subprocess.run([
                'ffmpeg', '-y', '-i', original_output,
                '-c:v', 'libx264', '-crf', str(crf),
                '-pix_fmt', 'yuv420p', compressed_original
            ], capture_output=True)
            
            try:
                extracted_original = extract_message_simple(compressed_original)
                original_accuracy = 1.0 if extracted_original == test_message else 0.0
            except:
                original_accuracy = 0.0
            
            # Calculate PSNR
            original_psnr = calculate_video_psnr(demo_video, compressed_original)
            
            # Test adaptive system
            adaptive_output = f"videos/adaptive_crf{crf}.mp4"
            embed_message_adaptive(demo_video, adaptive_output, test_message)
            
            # Apply additional compression
            compressed_adaptive = f"videos/adaptive_crf{crf}_compressed.mp4"
            subprocess.run([
                'ffmpeg', '-y', '-i', adaptive_output,
                '-c:v', 'libx264', '-crf', str(crf),
                '-pix_fmt', 'yuv420p', compressed_adaptive
            ], capture_output=True)
            
            try:
                extracted_adaptive = extract_message_adaptive(compressed_adaptive)
                adaptive_accuracy = 1.0 if extracted_adaptive == test_message else 0.0
            except:
                adaptive_accuracy = 0.0
            
            # Calculate PSNR
            adaptive_psnr = calculate_video_psnr(demo_video, compressed_adaptive)
            
            results['original_accuracy'].append(original_accuracy * 100)
            results['adaptive_accuracy'].append(adaptive_accuracy * 100)
            results['original_psnr'].append(original_psnr)
            results['adaptive_psnr'].append(adaptive_psnr)
            
            print(f"  Original accuracy: {original_accuracy*100:.0f}%, PSNR: {original_psnr:.1f}")
            print(f"  Adaptive accuracy: {adaptive_accuracy*100:.0f}%, PSNR: {adaptive_psnr:.1f}")
            
            # Clean up intermediate files
            for f in [original_output, compressed_original, adaptive_output, compressed_adaptive]:
                if os.path.exists(f):
                    os.remove(f)
                # Remove metadata files
                metadata_file = f.replace('.mp4', '_adaptive_metadata.txt')
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                    
        except Exception as e:
            print(f"  Error at CRF {crf}: {e}")
            results['original_accuracy'].append(0)
            results['adaptive_accuracy'].append(0)
            results['original_psnr'].append(0)
            results['adaptive_psnr'].append(0)
    
    return results


def demonstrate_multichannel_capacity():
    """Demonstrate how multi-channel embedding increases capacity."""
    print("\n" + "="*60)
    print("MULTI-CHANNEL CAPACITY DEMONSTRATION")
    print("="*60)
    
    # Create test video if it doesn't exist
    demo_video = "videos/demo_video.mp4"
    if not os.path.exists(demo_video):
        create_synthetic_video_with_capacity()
    
    # Test different message lengths to show capacity improvement
    message_lengths = [50, 100, 200, 300, 400]
    
    results = {
        'message_lengths': message_lengths,
        'original_success': [],
        'multichannel_success': [],
        'original_capacity': [],
        'multichannel_capacity': []
    }
    
    # Calculate theoretical capacities
    cap = cv2.VideoCapture(demo_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    from config import START_FRAME, STEP
    max_available_frames = max(0, (total_frames - START_FRAME + STEP - 1) // STEP)
    original_capacity = max(0, (max_available_frames - 32) // 8)  # Single channel
    multichannel_capacity = max(0, ((max_available_frames * 3) - 32) // 8)  # Three channels
    
    print(f"Video capacity - Original: {original_capacity} chars, Multi-channel: {multichannel_capacity} chars")
    
    for length in message_lengths:
        print(f"\nTesting with {length} character message...")
        
        test_message = "X" * length
        
        # Test original system
        try:
            original_output = f"videos/original_len{length}.mp4"
            embed_message_simple(demo_video, original_output, test_message)
            extracted_original = extract_message_simple(original_output)
            original_success = 1 if extracted_original == test_message else 0
            os.remove(original_output)
        except Exception as e:
            print(f"  Original system failed: {e}")
            original_success = 0
        
        # Test multi-channel system
        try:
            multichannel_output = f"videos/multichannel_len{length}.mp4"
            embed_message_multichannel(demo_video, multichannel_output, test_message)
            extracted_multichannel = extract_message_multichannel(multichannel_output)
            multichannel_success = 1 if extracted_multichannel == test_message else 0
            os.remove(multichannel_output)
            # Remove metadata
            metadata_file = multichannel_output.replace('.mp4', '_multichannel_metadata.txt')
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
        except Exception as e:
            print(f"  Multi-channel system failed: {e}")
            multichannel_success = 0
        
        results['original_success'].append(original_success)
        results['multichannel_success'].append(multichannel_success)
        results['original_capacity'].append(original_capacity)
        results['multichannel_capacity'].append(multichannel_capacity)
        
        print(f"  Original: {'✓' if original_success else '✗'}")
        print(f"  Multi-channel: {'✓' if multichannel_success else '✗'}")
    
    return results


def calculate_video_psnr(original_path, modified_path, sample_frames=20):
    """Calculate average PSNR between two videos."""
    cap1 = cv2.VideoCapture(original_path)
    cap2 = cv2.VideoCapture(modified_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        cap1.release()
        cap2.release()
        return 0
    
    total_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), 
                      int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    if total_frames == 0:
        cap1.release()
        cap2.release()
        return 0
    
    sample_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
    psnr_values = []
    
    for frame_idx in sample_indices:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            psnr = calculate_psnr(frame1, frame2)
            if not np.isinf(psnr):
                psnr_values.append(psnr)
    
    cap1.release()
    cap2.release()
    
    return np.mean(psnr_values) if psnr_values else 0


def create_improvement_graphs(adaptive_results, multichannel_results):
    """Create comprehensive improvement demonstration graphs."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Adaptive Threshold - Compression Robustness
    ax1.plot(adaptive_results['compression_levels'], adaptive_results['original_accuracy'], 
             'b-o', linewidth=3, markersize=8, label='Original System')
    ax1.plot(adaptive_results['compression_levels'], adaptive_results['adaptive_accuracy'], 
             'r-o', linewidth=3, markersize=8, label='Adaptive Threshold')
    
    ax1.set_xlabel('Compression Level (CRF)')
    ax1.set_ylabel('Extraction Accuracy (%)')
    ax1.set_title('IMPROVEMENT 1: Adaptive Threshold Compression Robustness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)
    
    # Add improvement annotations
    for i, (crf, orig, adapt) in enumerate(zip(adaptive_results['compression_levels'], 
                                              adaptive_results['original_accuracy'],
                                              adaptive_results['adaptive_accuracy'])):
        if adapt > orig:
            improvement = adapt - orig
            ax1.annotate(f'+{improvement:.0f}%', 
                        xy=(crf, adapt), xytext=(crf, adapt + 5),
                        ha='center', fontsize=10, color='green', weight='bold')
    
    # 2. Adaptive Threshold - Quality Comparison
    ax2.plot(adaptive_results['compression_levels'], adaptive_results['original_psnr'], 
             'b-s', linewidth=3, markersize=8, label='Original System')
    ax2.plot(adaptive_results['compression_levels'], adaptive_results['adaptive_psnr'], 
             'r-s', linewidth=3, markersize=8, label='Adaptive Threshold')
    
    ax2.set_xlabel('Compression Level (CRF)')
    ax2.set_ylabel('Video Quality (PSNR dB)')
    ax2.set_title('Quality Impact: Adaptive vs Original')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Multi-Channel - Capacity Improvement
    message_lengths = multichannel_results['message_lengths']
    
    # Create capacity visualization
    original_success_rate = []
    multichannel_success_rate = []
    
    for i, length in enumerate(message_lengths):
        orig_rate = multichannel_results['original_success'][i] * 100
        multi_rate = multichannel_results['multichannel_success'][i] * 100
        original_success_rate.append(orig_rate)
        multichannel_success_rate.append(multi_rate)
    
    x = np.arange(len(message_lengths))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, original_success_rate, width, 
                    label='Original System', color='lightblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, multichannel_success_rate, width, 
                    label='Multi-Channel', color='lightcoral', alpha=0.8)
    
    ax3.set_xlabel('Message Length (characters)')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('IMPROVEMENT 2: Multi-Channel Capacity Enhancement')
    ax3.set_xticks(x)
    ax3.set_xticklabels(message_lengths)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 110)
    
    # Add success/failure labels
    for i, (orig, multi) in enumerate(zip(original_success_rate, multichannel_success_rate)):
        if orig > 0:
            ax3.text(i - width/2, orig + 2, '✓', ha='center', va='bottom', 
                    fontsize=14, color='green', weight='bold')
        else:
            ax3.text(i - width/2, 5, '✗', ha='center', va='bottom', 
                    fontsize=14, color='red', weight='bold')
            
        if multi > 0:
            ax3.text(i + width/2, multi + 2, '✓', ha='center', va='bottom', 
                    fontsize=14, color='green', weight='bold')
        else:
            ax3.text(i + width/2, 5, '✗', ha='center', va='bottom', 
                    fontsize=14, color='red', weight='bold')
    
    # 4. Capacity Comparison Chart
    capacity_data = {
        'Method': ['Original\nSystem', 'Multi-Channel\nSystem'],
        'Capacity': [multichannel_results['original_capacity'][0], multichannel_results['multichannel_capacity'][0]]
    }
    
    colors = ['lightblue', 'lightcoral']
    bars = ax4.bar(capacity_data['Method'], capacity_data['Capacity'], color=colors, alpha=0.8)
    
    ax4.set_ylabel('Maximum Capacity (characters)')
    ax4.set_title('Theoretical Capacity Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add capacity improvement annotation
    improvement_factor = capacity_data['Capacity'][1] / capacity_data['Capacity'][0] if capacity_data['Capacity'][0] > 0 else 0
    ax4.annotate(f'{improvement_factor:.1f}x\nImprovement', 
                xy=(1, capacity_data['Capacity'][1]), 
                xytext=(0.5, capacity_data['Capacity'][1] * 0.8),
                ha='center', va='center', fontsize=12, color='green', weight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Add value labels on bars
    for bar, value in zip(bars, capacity_data['Capacity']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(value)}', ha='center', va='bottom', fontsize=12, weight='bold')
    
    plt.tight_layout()
    return fig


def print_improvement_summary(adaptive_results, multichannel_results):
    """Print a summary of the improvements demonstrated."""
    print("\n" + "="*80)
    print("IMPROVEMENT DEMONSTRATION SUMMARY")
    print("="*80)
    
    print("\n🎯 IMPROVEMENT 1: ADAPTIVE THRESHOLD EMBEDDING")
    print("-" * 60)
    print("Purpose: Better robustness against compression artifacts")
    print("Method: Analyzes block content to choose optimal embedding values")
    
    # Calculate improvement metrics
    avg_original_accuracy = np.mean(adaptive_results['original_accuracy'])
    avg_adaptive_accuracy = np.mean(adaptive_results['adaptive_accuracy'])
    accuracy_improvement = avg_adaptive_accuracy - avg_original_accuracy
    
    avg_original_psnr = np.mean([p for p in adaptive_results['original_psnr'] if p > 0])
    avg_adaptive_psnr = np.mean([p for p in adaptive_results['adaptive_psnr'] if p > 0])
    quality_improvement = avg_adaptive_psnr - avg_original_psnr
    
    print(f"\nResults:")
    print(f"  • Average accuracy improvement: +{accuracy_improvement:.1f}%")
    print(f"    (Original: {avg_original_accuracy:.1f}% → Adaptive: {avg_adaptive_accuracy:.1f}%)")
    print(f"  • Video quality impact: {quality_improvement:+.1f} dB PSNR")
    print(f"    (Better compression resilience)")
    
    # Find best improvement case
    max_improvement_idx = np.argmax(np.array(adaptive_results['adaptive_accuracy']) - np.array(adaptive_results['original_accuracy']))
    best_crf = adaptive_results['compression_levels'][max_improvement_idx]
    best_improvement = adaptive_results['adaptive_accuracy'][max_improvement_idx] - adaptive_results['original_accuracy'][max_improvement_idx]
    
    print(f"  • Best improvement: +{best_improvement:.0f}% at CRF {best_crf}")
    
    print("\n🚀 IMPROVEMENT 2: MULTI-CHANNEL EMBEDDING")
    print("-" * 60)
    print("Purpose: Increase embedding capacity and improve data distribution")
    print("Method: Distributes data across RGB channels instead of just blue channel")
    
    # Calculate capacity improvement
    original_capacity = multichannel_results['original_capacity'][0]
    multichannel_capacity = multichannel_results['multichannel_capacity'][0]
    capacity_improvement = multichannel_capacity - original_capacity
    capacity_multiplier = multichannel_capacity / original_capacity if original_capacity > 0 else 0
    
    # Calculate success rate improvement
    original_success_count = sum(multichannel_results['original_success'])
    multichannel_success_count = sum(multichannel_results['multichannel_success'])
    
    print(f"\nResults:")
    print(f"  • Capacity improvement: +{capacity_improvement} characters")
    print(f"    (Original: {original_capacity} → Multi-channel: {multichannel_capacity})")
    print(f"  • Capacity multiplier: {capacity_multiplier:.1f}x increase")
    print(f"  • Success rate: {multichannel_success_count}/{len(multichannel_results['message_lengths'])} vs {original_success_count}/{len(multichannel_results['message_lengths'])}")
    
    # Find capacity breakthrough
    breakthrough_messages = []
    for i, length in enumerate(multichannel_results['message_lengths']):
        if multichannel_results['multichannel_success'][i] and not multichannel_results['original_success'][i]:
            breakthrough_messages.append(length)
    
    if breakthrough_messages:
        print(f"  • Enabled embedding for {len(breakthrough_messages)} message sizes that failed with original system")
        print(f"    (Message lengths: {breakthrough_messages})")
    
    print("\n📊 OVERALL IMPACT")
    print("-" * 60)
    print("1. Adaptive Threshold: Maintains reliability under compression stress")
    print("2. Multi-Channel: Significantly expands embedding capacity")
    print("3. Combined benefits: More robust and higher-capacity steganography system")


def main():
    """Main function to run improvement demonstrations."""
    print("="*80)
    print("VIDEO STEGANOGRAPHY IMPROVEMENT DEMONSTRATION")
    print("="*80)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Demonstration 1: Adaptive Threshold Robustness
    adaptive_results = demonstrate_adaptive_threshold_robustness()
    
    # Demonstration 2: Multi-Channel Capacity
    multichannel_results = demonstrate_multichannel_capacity()
    
    # Create comprehensive graphs
    print(f"\n{'='*60}")
    print("GENERATING IMPROVEMENT DEMONSTRATION GRAPHS")
    print(f"{'='*60}")
    
    fig = create_improvement_graphs(adaptive_results, multichannel_results)
    
    # Save graphs
    output_path = "results/improvement_demonstration.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Improvement demonstration graphs saved to: {output_path}")
    
    # Print comprehensive summary
    print_improvement_summary(adaptive_results, multichannel_results)
    
    plt.close(fig)
    
    print(f"\n{'='*60}")
    print("IMPROVEMENT DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Check {output_path} for detailed before/after comparison graphs!")


if __name__ == "__main__":
    main()

