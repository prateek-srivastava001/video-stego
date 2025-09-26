#!/usr/bin/env python3
"""
Improvement Comparison Analysis

Comprehensive analysis comparing the original system with two improvements:
1. Adaptive Threshold Embedding - Better robustness against compression
2. Multi-Channel Embedding - Increased capacity and better distribution

Generates detailed metrics and before/after comparison graphs.
"""

import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import subprocess
import time
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.embed import embed_message_simple
from utils.extract import extract_message_simple
from utils.embed_adaptive import embed_message_adaptive, extract_message_adaptive
from utils.embed_multichannel import embed_message_multichannel, extract_message_multichannel
from tests.quality_impact import calculate_psnr, calculate_ssim


def generate_test_messages():
    """Generate test messages of various lengths."""
    return {
        'short': "Hi!",  # 3 characters - should work with any video
        'medium': "Hello World!",  # 12 characters - reasonable size
        'capacity_test': "X" * 10  # 10 characters for capacity testing
    }


def measure_embedding_performance(embed_func, extract_func, video_path: str, message: str, method_name: str) -> Dict:
    """Measure performance metrics for a specific embedding method."""
    print(f"\n--- Testing {method_name} ---")
    
    # Generate unique output path
    timestamp = int(time.time() * 1000)
    output_path = f"videos/test_{method_name.lower().replace(' ', '_')}_{timestamp}.mp4"
    
    try:
        # Measure embedding time
        start_time = time.time()
        embed_func(video_path, output_path, message)
        embed_time = time.time() - start_time
        
        # Measure extraction time
        start_time = time.time()
        extracted_message = extract_func(output_path)
        extract_time = time.time() - start_time
        
        # Calculate accuracy
        accuracy = 1.0 if extracted_message == message else 0.0
        if accuracy == 0:
            # Calculate character-level accuracy for partial matches
            min_len = min(len(message), len(extracted_message))
            if min_len > 0:
                correct_chars = sum(1 for i in range(min_len) if message[i] == extracted_message[i])
                accuracy = correct_chars / len(message)
        
        # Get file sizes
        original_size = os.path.getsize(video_path)
        output_size = os.path.getsize(output_path)
        size_ratio = output_size / original_size
        
        # Calculate capacity metrics
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        bits_embedded = len(message) * 8 + 32  # message + header
        capacity_efficiency = bits_embedded / total_frames if total_frames > 0 else 0
        
        # Calculate quality metrics (sample a few frames)
        quality_metrics = calculate_video_quality_impact(video_path, output_path)
        
        results = {
            'method': method_name,
            'message_length': len(message),
            'bits_embedded': bits_embedded,
            'embed_time': embed_time,
            'extract_time': extract_time,
            'total_time': embed_time + extract_time,
            'accuracy': accuracy,
            'extracted_length': len(extracted_message),
            'original_size_mb': original_size / (1024 * 1024),
            'output_size_mb': output_size / (1024 * 1024),
            'size_ratio': size_ratio,
            'capacity_efficiency': capacity_efficiency,
            'throughput_chars_per_sec': len(message) / (embed_time + extract_time) if (embed_time + extract_time) > 0 else 0,
            'avg_psnr': quality_metrics['avg_psnr'],
            'avg_ssim': quality_metrics['avg_ssim'],
            'quality_degradation': quality_metrics['quality_degradation'],
            'output_path': output_path
        }
        
        print(f"✓ Success: {accuracy*100:.1f}% accuracy, {embed_time+extract_time:.2f}s total time")
        
        return results
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return {
            'method': method_name,
            'message_length': len(message),
            'error': str(e),
            'accuracy': 0.0,
            'embed_time': float('inf'),
            'extract_time': float('inf'),
            'total_time': float('inf')
        }


def calculate_video_quality_impact(original_path: str, stego_path: str, sample_frames: int = 10) -> Dict:
    """Calculate quality impact by comparing sample frames."""
    cap_orig = cv2.VideoCapture(original_path)
    cap_stego = cv2.VideoCapture(stego_path)
    
    if not cap_orig.isOpened() or not cap_stego.isOpened():
        return {'avg_psnr': 0, 'avg_ssim': 0, 'quality_degradation': 'severe'}
    
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
    
    psnr_values = []
    ssim_values = []
    
    for frame_idx in sample_indices:
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_stego.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret_orig, frame_orig = cap_orig.read()
        ret_stego, frame_stego = cap_stego.read()
        
        if ret_orig and ret_stego:
            psnr = calculate_psnr(frame_orig, frame_stego)
            ssim = calculate_ssim(frame_orig, frame_stego)
            
            if not np.isinf(psnr):
                psnr_values.append(psnr)
            ssim_values.append(ssim)
    
    cap_orig.release()
    cap_stego.release()
    
    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0
    
    # Determine quality degradation level
    if avg_psnr > 40:
        quality_degradation = 'negligible'
    elif avg_psnr > 30:
        quality_degradation = 'minor'
    elif avg_psnr > 20:
        quality_degradation = 'noticeable'
    else:
        quality_degradation = 'significant'
    
    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'quality_degradation': quality_degradation
    }


def create_comparison_graphs(results_data: List[Dict]):
    """Create comprehensive comparison graphs."""
    fig = plt.figure(figsize=(20, 16))
    
    # Prepare data
    methods = list(set(r['method'] for r in results_data if 'error' not in r))
    message_lengths = sorted(list(set(r['message_length'] for r in results_data if 'error' not in r)))
    
    # Colors for different methods
    colors = {'Original System': '#1f77b4', 'Adaptive Threshold': '#ff7f0e', 'Multi-Channel': '#2ca02c'}
    
    # 1. Performance Comparison (Time)
    ax1 = plt.subplot(3, 3, 1)
    for method in methods:
        method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
        lengths = [r['message_length'] for r in method_data]
        times = [r['total_time'] for r in method_data]
        ax1.plot(lengths, times, 'o-', label=method, color=colors.get(method, 'gray'), linewidth=2, markersize=6)
    
    ax1.set_xlabel('Message Length (characters)')
    ax1.set_ylabel('Total Time (seconds)')
    ax1.set_title('Performance: Processing Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Comparison
    ax2 = plt.subplot(3, 3, 2)
    for method in methods:
        method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
        lengths = [r['message_length'] for r in method_data]
        accuracies = [r['accuracy'] * 100 for r in method_data]
        ax2.plot(lengths, accuracies, 'o-', label=method, color=colors.get(method, 'gray'), linewidth=2, markersize=6)
    
    ax2.set_xlabel('Message Length (characters)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Reliability: Extraction Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)
    
    # 3. Capacity Efficiency
    ax3 = plt.subplot(3, 3, 3)
    for method in methods:
        method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
        lengths = [r['message_length'] for r in method_data]
        efficiency = [r['capacity_efficiency'] for r in method_data]
        ax3.plot(lengths, efficiency, 'o-', label=method, color=colors.get(method, 'gray'), linewidth=2, markersize=6)
    
    ax3.set_xlabel('Message Length (characters)')
    ax3.set_ylabel('Bits per Frame')
    ax3.set_title('Capacity: Embedding Efficiency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Quality Impact (PSNR)
    ax4 = plt.subplot(3, 3, 4)
    for method in methods:
        method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
        lengths = [r['message_length'] for r in method_data]
        psnr = [r['avg_psnr'] for r in method_data]
        ax4.plot(lengths, psnr, 'o-', label=method, color=colors.get(method, 'gray'), linewidth=2, markersize=6)
    
    ax4.set_xlabel('Message Length (characters)')
    ax4.set_ylabel('PSNR (dB)')
    ax4.set_title('Quality: Peak Signal-to-Noise Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Good Quality Threshold')
    
    # 5. Quality Impact (SSIM)
    ax5 = plt.subplot(3, 3, 5)
    for method in methods:
        method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
        lengths = [r['message_length'] for r in method_data]
        ssim = [r['avg_ssim'] for r in method_data]
        ax5.plot(lengths, ssim, 'o-', label=method, color=colors.get(method, 'gray'), linewidth=2, markersize=6)
    
    ax5.set_xlabel('Message Length (characters)')
    ax5.set_ylabel('SSIM Score')
    ax5.set_title('Quality: Structural Similarity Index')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    
    # 6. Throughput Comparison
    ax6 = plt.subplot(3, 3, 6)
    for method in methods:
        method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
        lengths = [r['message_length'] for r in method_data]
        throughput = [r['throughput_chars_per_sec'] for r in method_data]
        ax6.plot(lengths, throughput, 'o-', label=method, color=colors.get(method, 'gray'), linewidth=2, markersize=6)
    
    ax6.set_xlabel('Message Length (characters)')
    ax6.set_ylabel('Characters per Second')
    ax6.set_title('Throughput: Processing Speed')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. File Size Impact
    ax7 = plt.subplot(3, 3, 7)
    for method in methods:
        method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
        lengths = [r['message_length'] for r in method_data]
        size_ratios = [r['size_ratio'] for r in method_data]
        ax7.plot(lengths, size_ratios, 'o-', label=method, color=colors.get(method, 'gray'), linewidth=2, markersize=6)
    
    ax7.set_xlabel('Message Length (characters)')
    ax7.set_ylabel('Output/Input Size Ratio')
    ax7.set_title('Storage: File Size Impact')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No Size Change')
    
    # 8. Overall Performance Radar Chart
    ax8 = plt.subplot(3, 3, 8, projection='polar')
    metrics = ['Speed', 'Accuracy', 'Quality', 'Efficiency']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    
    successful_methods = [method for method in methods 
                         if any(r['method'] == method and 'error' not in r for r in results_data)]
    
    if successful_methods:
        for method in successful_methods:
            method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
            if not method_data:
                continue
                
            # Normalize metrics (higher is better)
            avg_data = {
                'speed': 1 / np.mean([r['total_time'] for r in method_data]) if method_data else 0,
                'accuracy': np.mean([r['accuracy'] for r in method_data]) if method_data else 0,
                'quality': np.mean([r['avg_ssim'] for r in method_data]) if method_data else 0,
                'efficiency': np.mean([r['capacity_efficiency'] for r in method_data]) if method_data else 0
            }
            
            # Normalize to 0-1 scale for radar chart
            all_successful_data = [r for r in results_data if 'error' not in r]
            max_speed = max([1 / r['total_time'] for r in all_successful_data], default=1)
            max_efficiency = max([r['capacity_efficiency'] for r in all_successful_data], default=1)
            
            values = [
                avg_data['speed'] / max_speed if max_speed > 0 else 0,
                avg_data['accuracy'],
                avg_data['quality'],
                avg_data['efficiency'] / max_efficiency if max_efficiency > 0 else 0
            ]
            
            values += values[:1]  # Complete the circle
            plot_angles = angles + angles[:1]
            
            ax8.plot(plot_angles, values, 'o-', linewidth=2, label=method, color=colors.get(method, 'gray'))
            ax8.fill(plot_angles, values, alpha=0.25, color=colors.get(method, 'gray'))
    
    ax8.set_xticks(angles)
    ax8.set_xticklabels(metrics)
    ax8.set_ylim(0, 1)
    ax8.set_title('Overall Performance Comparison')
    if successful_methods:
        ax8.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # 9. Summary Statistics Bar Chart
    ax9 = plt.subplot(3, 3, 9)
    
    summary_metrics = {}
    for method in methods:
        method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
        if method_data:
            summary_metrics[method] = {
                'avg_accuracy': np.mean([r['accuracy'] for r in method_data]) * 100,
                'avg_psnr': np.mean([r['avg_psnr'] for r in method_data]),
                'avg_time': np.mean([r['total_time'] for r in method_data])
            }
    
    x = np.arange(len(methods))
    width = 0.25
    
    accuracies = [summary_metrics[m]['avg_accuracy'] for m in methods]
    psnrs = [summary_metrics[m]['avg_psnr'] for m in methods]
    times = [summary_metrics[m]['avg_time'] for m in methods]
    
    # Normalize for comparison
    max_psnr = max(psnrs) if psnrs else 1
    max_time = max(times) if times else 1
    
    ax9.bar(x - width, accuracies, width, label='Accuracy (%)', color='lightblue')
    ax9.bar(x, [p/max_psnr*100 for p in psnrs], width, label='PSNR (normalized)', color='lightgreen')
    ax9.bar(x + width, [100-t/max_time*100 for t in times], width, label='Speed (normalized)', color='lightcoral')
    
    ax9.set_xlabel('Methods')
    ax9.set_ylabel('Performance Score')
    ax9.set_title('Average Performance Summary')
    ax9.set_xticks(x)
    ax9.set_xticklabels(methods, rotation=45, ha='right')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_detailed_comparison_report(results_data: List[Dict]):
    """Print a detailed comparison report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE IMPROVEMENT COMPARISON REPORT")
    print("="*80)
    
    methods = list(set(r['method'] for r in results_data if 'error' not in r))
    
    for method in methods:
        method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
        if not method_data:
            continue
            
        print(f"\n{method} Performance:")
        print("-" * 40)
        
        avg_accuracy = np.mean([r['accuracy'] for r in method_data]) * 100
        avg_time = np.mean([r['total_time'] for r in method_data])
        avg_psnr = np.mean([r['avg_psnr'] for r in method_data])
        avg_ssim = np.mean([r['avg_ssim'] for r in method_data])
        avg_throughput = np.mean([r['throughput_chars_per_sec'] for r in method_data])
        
        print(f"  Average Accuracy: {avg_accuracy:.1f}%")
        print(f"  Average Processing Time: {avg_time:.2f} seconds")
        print(f"  Average PSNR: {avg_psnr:.1f} dB")
        print(f"  Average SSIM: {avg_ssim:.3f}")
        print(f"  Average Throughput: {avg_throughput:.1f} chars/sec")
        
        # Test different message lengths
        for length_category in ['short', 'medium', 'long']:
            category_data = [r for r in method_data 
                           if (length_category == 'short' and r['message_length'] < 100) or
                              (length_category == 'medium' and 100 <= r['message_length'] < 300) or
                              (length_category == 'long' and r['message_length'] >= 300)]
            
            if category_data:
                cat_accuracy = np.mean([r['accuracy'] for r in category_data]) * 100
                print(f"    {length_category.capitalize()} messages: {cat_accuracy:.1f}% accuracy")
    
    # Comparison analysis
    print(f"\n{'IMPROVEMENT ANALYSIS'}")
    print("-" * 40)
    
    if len(methods) >= 2:
        original_data = [r for r in results_data if r['method'] == 'Original System' and 'error' not in r]
        
        for method in methods:
            if method == 'Original System':
                continue
                
            method_data = [r for r in results_data if r['method'] == method and 'error' not in r]
            
            if original_data and method_data:
                # Calculate improvements
                orig_accuracy = np.mean([r['accuracy'] for r in original_data]) * 100
                new_accuracy = np.mean([r['accuracy'] for r in method_data]) * 100
                accuracy_improvement = new_accuracy - orig_accuracy
                
                orig_psnr = np.mean([r['avg_psnr'] for r in original_data])
                new_psnr = np.mean([r['avg_psnr'] for r in method_data])
                psnr_improvement = new_psnr - orig_psnr
                
                orig_time = np.mean([r['total_time'] for r in original_data])
                new_time = np.mean([r['total_time'] for r in method_data])
                time_change = ((new_time - orig_time) / orig_time) * 100
                
                print(f"\n{method} vs Original System:")
                print(f"  Accuracy: {accuracy_improvement:+.1f}% ({new_accuracy:.1f}% vs {orig_accuracy:.1f}%)")
                print(f"  Quality (PSNR): {psnr_improvement:+.1f} dB ({new_psnr:.1f} vs {orig_psnr:.1f})")
                print(f"  Processing Time: {time_change:+.1f}% change ({new_time:.2f}s vs {orig_time:.2f}s)")


def main():
    """Main function to run improvement comparison analysis."""
    print("="*80)
    print("VIDEO STEGANOGRAPHY IMPROVEMENT COMPARISON")
    print("="*80)
    
    # Check for input video
    video_paths = []
    default_video = "videos/sample.mp4"
    
    if os.path.exists(default_video):
        video_paths.append(default_video)
    
    # Look for any video files in videos directory
    if os.path.exists("videos"):
        for file in os.listdir("videos"):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) and not file.startswith('test_'):
                path = os.path.join("videos", file)
                if path not in video_paths:
                    video_paths.append(path)
    
    if not video_paths:
        print("Error: No video files found.")
        print("Place a video file in 'videos/sample.mp4' for analysis.")
        return 1
    
    video_path = video_paths[0]
    print(f"Using video: {video_path}")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Generate test messages
    test_messages = generate_test_messages()
    
    # Test methods
    methods = [
        ('Original System', embed_message_simple, extract_message_simple),
        ('Adaptive Threshold', embed_message_adaptive, extract_message_adaptive),
        ('Multi-Channel', embed_message_multichannel, extract_message_multichannel)
    ]
    
    # Run comprehensive tests
    all_results = []
    
    for method_name, embed_func, extract_func in methods:
        print(f"\n{'='*60}")
        print(f"TESTING {method_name.upper()}")
        print(f"{'='*60}")
        
        for message_name, message in test_messages.items():
            print(f"\nTesting with {message_name} message ({len(message)} characters)...")
            
            try:
                result = measure_embedding_performance(
                    embed_func, extract_func, video_path, message, method_name
                )
                result['message_category'] = message_name
                all_results.append(result)
                
            except Exception as e:
                print(f"Error testing {method_name} with {message_name}: {e}")
                all_results.append({
                    'method': method_name,
                    'message_category': message_name,
                    'message_length': len(message),
                    'error': str(e),
                    'accuracy': 0.0
                })
    
    # Generate comparison graphs
    print(f"\n{'='*60}")
    print("GENERATING COMPARISON GRAPHS")
    print(f"{'='*60}")
    
    fig = create_comparison_graphs(all_results)
    
    # Save graphs
    output_path = "results/improvement_comparison_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison graphs saved to: {output_path}")
    
    # Print detailed report
    print_detailed_comparison_report(all_results)
    
    # Clean up test files
    print(f"\n{'='*60}")
    print("CLEANING UP TEST FILES")
    print(f"{'='*60}")
    
    cleanup_count = 0
    for result in all_results:
        if 'output_path' in result and os.path.exists(result['output_path']):
            try:
                os.remove(result['output_path'])
                # Also remove metadata files
                metadata_files = [
                    result['output_path'].replace('.mp4', '_adaptive_metadata.txt'),
                    result['output_path'].replace('.mp4', '_multichannel_metadata.txt')
                ]
                for metadata_file in metadata_files:
                    if os.path.exists(metadata_file):
                        os.remove(metadata_file)
                cleanup_count += 1
            except Exception as e:
                print(f"Warning: Could not clean up {result['output_path']}: {e}")
    
    print(f"Cleaned up {cleanup_count} test files")
    
    plt.close(fig)
    print(f"\n{'='*60}")
    print("IMPROVEMENT COMPARISON COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
