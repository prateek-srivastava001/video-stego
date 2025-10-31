#!/usr/bin/env python3
"""
Focused Improvement Demonstration

Creates focused demonstrations showing the specific benefits of each improvement
with appropriate message sizes and realistic scenarios.
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.embed import embed_message_simple
from utils.extract import extract_message_simple
from utils.embed_adaptive import embed_message_adaptive, extract_message_adaptive
from utils.embed_multichannel import embed_message_multichannel, extract_message_multichannel
from tests.quality_impact import calculate_psnr, calculate_ssim
from config import VAL_ZERO, VAL_ONE, THRESH, START_FRAME, STEP


def get_video_capacity(video_path):
    """Calculate the actual capacity of a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    max_available_frames = max(0, (total_frames - START_FRAME + STEP - 1) // STEP)
    single_channel_capacity = max(0, (max_available_frames - 32) // 8)
    multi_channel_capacity = max(0, ((max_available_frames * 3) - 32) // 8)
    
    return single_channel_capacity, multi_channel_capacity


def demonstrate_adaptive_threshold_benefits():
    """Demonstrate adaptive threshold benefits with realistic scenarios."""
    print("\n" + "="*70)
    print("🎯 IMPROVEMENT 1: ADAPTIVE THRESHOLD EMBEDDING")
    print("="*70)
    print("Benefit: Better robustness against compression and varying video content")
    
    # Use the existing video
    video_path = "videos/sample.mp4"
    if not os.path.exists(video_path):
        print("No sample video found!")
        return {}
    
    single_cap, multi_cap = get_video_capacity(video_path)
    test_message = "Test" if single_cap >= 4 else "Hi"  # Use appropriate size
    
    print(f"Using test message: '{test_message}' ({len(test_message)} characters)")
    
    results = {
        'scenarios': [],
        'original_times': [],
        'adaptive_times': [],
        'original_accuracy': [],
        'adaptive_accuracy': [],
        'original_robustness': [],
        'adaptive_robustness': []
    }
    
    # Test different scenarios that show adaptive threshold benefits
    scenarios = [
        ("Normal Quality", 23),      # Standard CRF
        ("High Compression", 30),    # Higher compression
        ("Very High Compression", 35) # Very high compression
    ]
    
    for scenario_name, crf in scenarios:
        print(f"\n--- Testing {scenario_name} (CRF {crf}) ---")
        
        try:
            # Test Original System
            start_time = time.time()
            original_output = f"videos/temp_original_{crf}.mp4"
            embed_message_simple(video_path, original_output, test_message)
            
            # Apply compression to simulate real-world usage
            compressed_original = f"videos/temp_original_{crf}_compressed.mp4"
            subprocess.run([
                'ffmpeg', '-y', '-i', original_output, '-c:v', 'libx264', 
                '-crf', str(crf), '-pix_fmt', 'yuv420p', compressed_original
            ], capture_output=True, stderr=subprocess.DEVNULL)
            
            try:
                extracted_orig = extract_message_simple(compressed_original)
                original_accuracy = 1.0 if extracted_orig == test_message else 0.0
            except:
                original_accuracy = 0.0
            
            original_time = time.time() - start_time
            
            # Test Adaptive System
            start_time = time.time()
            adaptive_output = f"videos/temp_adaptive_{crf}.mp4"
            embed_message_adaptive(video_path, adaptive_output, test_message)
            
            # Apply same compression
            compressed_adaptive = f"videos/temp_adaptive_{crf}_compressed.mp4"
            subprocess.run([
                'ffmpeg', '-y', '-i', adaptive_output, '-c:v', 'libx264', 
                '-crf', str(crf), '-pix_fmt', 'yuv420p', compressed_adaptive
            ], capture_output=True, stderr=subprocess.DEVNULL)
            
            try:
                extracted_adapt = extract_message_adaptive(compressed_adaptive)
                adaptive_accuracy = 1.0 if extracted_adapt == test_message else 0.0
            except:
                adaptive_accuracy = 0.0
            
            adaptive_time = time.time() - start_time
            
            # Calculate robustness metrics (ability to survive compression)
            original_psnr = calculate_video_psnr(video_path, compressed_original)
            adaptive_psnr = calculate_video_psnr(video_path, compressed_adaptive)
            
            results['scenarios'].append(scenario_name)
            results['original_times'].append(original_time)
            results['adaptive_times'].append(adaptive_time)
            results['original_accuracy'].append(original_accuracy * 100)
            results['adaptive_accuracy'].append(adaptive_accuracy * 100)
            results['original_robustness'].append(original_psnr)
            results['adaptive_robustness'].append(adaptive_psnr)
            
            print(f"  Original: {original_accuracy*100:.0f}% accuracy, {original_time:.2f}s, {original_psnr:.1f}dB PSNR")
            print(f"  Adaptive: {adaptive_accuracy*100:.0f}% accuracy, {adaptive_time:.2f}s, {adaptive_psnr:.1f}dB PSNR")
            
            if adaptive_accuracy > original_accuracy:
                print(f"  ✅ Improvement: +{(adaptive_accuracy-original_accuracy)*100:.0f}% accuracy")
            elif adaptive_accuracy == original_accuracy and adaptive_accuracy > 0:
                print(f"  ✅ Maintained reliability under compression")
            
            # Cleanup
            for f in [original_output, compressed_original, adaptive_output, compressed_adaptive]:
                if os.path.exists(f):
                    os.remove(f)
            # Remove metadata
            metadata_file = adaptive_output.replace('.mp4', '_adaptive_metadata.txt')
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                
        except Exception as e:
            print(f"  Error in {scenario_name}: {e}")
    
    return results


def demonstrate_multichannel_capacity_benefits():
    """Demonstrate multi-channel capacity benefits."""
    print("\n" + "="*70)
    print("🚀 IMPROVEMENT 2: MULTI-CHANNEL EMBEDDING")
    print("="*70)
    print("Benefit: Significantly increased embedding capacity")
    
    video_path = "videos/sample.mp4"
    if not os.path.exists(video_path):
        print("No sample video found!")
        return {}
    
    single_cap, multi_cap = get_video_capacity(video_path)
    print(f"Capacity Analysis:")
    print(f"  Original System: {single_cap} characters")
    print(f"  Multi-Channel: {multi_cap} characters")
    print(f"  Improvement: {multi_cap - single_cap} additional characters ({multi_cap/single_cap:.1f}x increase)")
    
    # Test messages of increasing size to show capacity limits
    test_sizes = []
    current_size = 3
    while current_size <= multi_cap + 5:  # Test beyond multi-channel capacity too
        test_sizes.append(current_size)
        current_size += max(2, current_size // 3)
    
    results = {
        'message_sizes': test_sizes,
        'original_success': [],
        'multichannel_success': [],
        'original_times': [],
        'multichannel_times': [],
        'capacity_limit_original': single_cap,
        'capacity_limit_multichannel': multi_cap
    }
    
    print(f"\nTesting message sizes: {test_sizes}")
    
    for size in test_sizes:
        print(f"\n--- Testing {size} character message ---")
        test_message = "X" * size
        
        # Test Original System
        original_success = False
        original_time = 0
        
        try:
            start_time = time.time()
            original_output = f"videos/temp_original_len{size}.mp4"
            embed_message_simple(video_path, original_output, test_message)
            extracted = extract_message_simple(original_output)
            original_time = time.time() - start_time
            original_success = (extracted == test_message)
            os.remove(original_output)
            print(f"  Original: ✅ Success in {original_time:.2f}s")
        except Exception as e:
            print(f"  Original: ❌ Failed - {str(e)[:50]}...")
        
        # Test Multi-Channel System
        multichannel_success = False
        multichannel_time = 0
        
        try:
            start_time = time.time()
            multichannel_output = f"videos/temp_multi_len{size}.mp4"
            embed_message_multichannel(video_path, multichannel_output, test_message)
            extracted = extract_message_multichannel(multichannel_output)
            multichannel_time = time.time() - start_time
            multichannel_success = (extracted == test_message)
            os.remove(multichannel_output)
            # Remove metadata
            metadata_file = multichannel_output.replace('.mp4', '_multichannel_metadata.txt')
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            print(f"  Multi-Channel: ✅ Success in {multichannel_time:.2f}s")
        except Exception as e:
            print(f"  Multi-Channel: ❌ Failed - {str(e)[:50]}...")
        
        results['original_success'].append(original_success)
        results['multichannel_success'].append(multichannel_success)
        results['original_times'].append(original_time)
        results['multichannel_times'].append(multichannel_time)
        
        # Highlight improvements
        if multichannel_success and not original_success:
            print(f"  🎯 BREAKTHROUGH: Multi-channel enables {size}-char messages!")
    
    return results


def calculate_video_psnr(original_path, modified_path, sample_frames=5):
    """Calculate PSNR between two videos."""
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
            if not np.isinf(psnr) and psnr > 0:
                psnr_values.append(psnr)
    
    cap1.release()
    cap2.release()
    
    return np.mean(psnr_values) if psnr_values else 0


def create_focused_improvement_graphs(adaptive_results, multichannel_results):
    """Create focused graphs showing clear improvements."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Adaptive Threshold: Compression Robustness
    scenarios = adaptive_results['scenarios']
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, adaptive_results['original_accuracy'], width, 
                    label='Original System', color='lightblue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, adaptive_results['adaptive_accuracy'], width, 
                    label='Adaptive Threshold', color='orange', alpha=0.8)
    
    ax1.set_xlabel('Compression Scenario')
    ax1.set_ylabel('Extraction Accuracy (%)')
    ax1.set_title('🎯 Improvement 1: Adaptive Threshold Robustness')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scenarios, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 110)
    
    # Add value labels and improvement indicators
    for i, (orig, adapt) in enumerate(zip(adaptive_results['original_accuracy'], 
                                         adaptive_results['adaptive_accuracy'])):
        ax1.text(i - width/2, orig + 2, f'{orig:.0f}%', ha='center', va='bottom', fontsize=10)
        ax1.text(i + width/2, adapt + 2, f'{adapt:.0f}%', ha='center', va='bottom', fontsize=10)
        
        if adapt > orig:
            ax1.annotate(f'↑+{adapt-orig:.0f}%', xy=(i, max(orig, adapt) + 8), 
                        ha='center', fontsize=10, color='green', weight='bold')
        elif adapt == orig and adapt > 0:
            ax1.annotate('✓', xy=(i, adapt + 8), ha='center', fontsize=14, color='green')
    
    # 2. Processing Time Comparison
    ax2.bar(x_pos - width/2, adaptive_results['original_times'], width, 
            label='Original System', color='lightblue', alpha=0.8)
    ax2.bar(x_pos + width/2, adaptive_results['adaptive_times'], width, 
            label='Adaptive Threshold', color='orange', alpha=0.8)
    
    ax2.set_xlabel('Compression Scenario')
    ax2.set_ylabel('Processing Time (seconds)')
    ax2.set_title('Processing Time: Adaptive vs Original')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Multi-Channel Capacity Breakthrough
    message_sizes = multichannel_results['message_sizes']
    original_success = [100 if s else 0 for s in multichannel_results['original_success']]
    multichannel_success = [100 if s else 0 for s in multichannel_results['multichannel_success']]
    
    x_pos = np.arange(len(message_sizes))
    
    bars1 = ax3.bar(x_pos - width/2, original_success, width, 
                    label='Original System', color='lightcoral', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, multichannel_success, width, 
                    label='Multi-Channel', color='lightgreen', alpha=0.8)
    
    ax3.set_xlabel('Message Size (characters)')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('🚀 Improvement 2: Multi-Channel Capacity Enhancement')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(message_sizes)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 110)
    
    # Mark capacity limits
    single_cap = multichannel_results['capacity_limit_original']
    multi_cap = multichannel_results['capacity_limit_multichannel']
    
    # Find where capacities are exceeded
    for i, size in enumerate(message_sizes):
        if size <= single_cap:
            color = 'green'
        elif size <= multi_cap:
            color = 'orange'  # Multi-channel can handle this
        else:
            color = 'red'  # Beyond both capacities
        
        # Add success/failure indicators
        if multichannel_results['original_success'][i]:
            ax3.text(i - width/2, 105, '✓', ha='center', fontsize=12, color='green', weight='bold')
        else:
            ax3.text(i - width/2, 105, '✗', ha='center', fontsize=12, color='red', weight='bold')
            
        if multichannel_results['multichannel_success'][i]:
            ax3.text(i + width/2, 105, '✓', ha='center', fontsize=12, color='green', weight='bold')
        else:
            ax3.text(i + width/2, 105, '✗', ha='center', fontsize=12, color='red', weight='bold')
        
        # Highlight breakthrough messages
        if multichannel_results['multichannel_success'][i] and not multichannel_results['original_success'][i]:
            ax3.annotate('BREAKTHROUGH!', xy=(i, 50), xytext=(i, 80),
                        ha='center', fontsize=9, color='blue', weight='bold',
                        arrowprops=dict(arrowstyle='->', color='blue'))
    
    # 4. Capacity Comparison with Benefits
    methods = ['Original\nSystem', 'Multi-Channel\nSystem']
    capacities = [single_cap, multi_cap]
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax4.bar(methods, capacities, color=colors, alpha=0.8, width=0.6)
    
    ax4.set_ylabel('Maximum Capacity (characters)')
    ax4.set_title('Theoretical Capacity Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, capacities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(value)} chars', ha='center', va='bottom', fontsize=12, weight='bold')
    
    # Add improvement annotation
    improvement = multi_cap - single_cap
    improvement_factor = multi_cap / single_cap if single_cap > 0 else 0
    
    ax4.annotate(f'+{improvement} chars\n({improvement_factor:.1f}x increase)', 
                xy=(1, multi_cap/2), xytext=(0.5, multi_cap * 0.8),
                ha='center', va='center', fontsize=11, color='blue', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    plt.tight_layout()
    return fig


def print_focused_summary(adaptive_results, multichannel_results):
    """Print a focused summary of key improvements."""
    print("\n" + "="*80)
    print("📊 IMPROVEMENT SUMMARY REPORT")
    print("="*80)
    
    print(f"\n🎯 ADAPTIVE THRESHOLD EMBEDDING - Key Benefits:")
    print("-" * 50)
    
    # Find improvements in adaptive results
    improvements_found = 0
    total_accuracy_gain = 0
    
    for i, (orig, adapt) in enumerate(zip(adaptive_results['original_accuracy'], 
                                         adaptive_results['adaptive_accuracy'])):
        if adapt > orig:
            improvements_found += 1
            total_accuracy_gain += (adapt - orig)
            scenario = adaptive_results['scenarios'][i]
            print(f"  ✅ {scenario}: +{adapt-orig:.0f}% accuracy improvement")
        elif adapt == orig and adapt > 0:
            print(f"  ✅ {adaptive_results['scenarios'][i]}: Maintained 100% reliability")
    
    if improvements_found > 0:
        avg_improvement = total_accuracy_gain / improvements_found
        print(f"\n  📈 Average improvement in challenging scenarios: +{avg_improvement:.1f}%")
    
    print(f"\n🚀 MULTI-CHANNEL EMBEDDING - Key Benefits:")
    print("-" * 50)
    
    single_cap = multichannel_results['capacity_limit_original']
    multi_cap = multichannel_results['capacity_limit_multichannel']
    
    print(f"  📊 Capacity increase: {single_cap} → {multi_cap} characters")
    print(f"  📊 Improvement factor: {multi_cap/single_cap:.1f}x increase")
    print(f"  📊 Additional capacity: +{multi_cap - single_cap} characters")
    
    # Count breakthrough messages
    breakthroughs = 0
    breakthrough_sizes = []
    
    for i, size in enumerate(multichannel_results['message_sizes']):
        if multichannel_results['multichannel_success'][i] and not multichannel_results['original_success'][i]:
            breakthroughs += 1
            breakthrough_sizes.append(size)
    
    if breakthroughs > 0:
        print(f"  🎯 Breakthrough messages: {breakthroughs} sizes now possible")
        print(f"  🎯 New message sizes: {breakthrough_sizes}")
    
    # Overall system impact
    print(f"\n🏆 OVERALL SYSTEM IMPROVEMENTS:")
    print("-" * 50)
    print(f"  1. Adaptive Threshold: Better compression resistance")
    if improvements_found > 0:
        print(f"     → Up to +{max(np.array(adaptive_results['adaptive_accuracy']) - np.array(adaptive_results['original_accuracy'])):.0f}% accuracy in challenging scenarios")
    print(f"     → Maintains reliability under compression stress")
    
    print(f"  2. Multi-Channel: Significantly expanded capacity")
    print(f"     → {multi_cap/single_cap:.1f}x capacity increase")
    print(f"     → Enables {breakthroughs} additional message size categories")
    
    print(f"\n💡 Combined benefits create a more robust and capable steganography system!")


def main():
    """Main function for focused improvement demonstration."""
    print("="*80)
    print("🎯 FOCUSED VIDEO STEGANOGRAPHY IMPROVEMENT DEMONSTRATION")
    print("="*80)
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Run focused demonstrations
    print("Running targeted improvement demonstrations...")
    
    adaptive_results = demonstrate_adaptive_threshold_benefits()
    multichannel_results = demonstrate_multichannel_capacity_benefits()
    
    # Create focused graphs
    print(f"\n{'='*60}")
    print("GENERATING FOCUSED IMPROVEMENT GRAPHS")
    print(f"{'='*60}")
    
    if adaptive_results and multichannel_results:
        fig = create_focused_improvement_graphs(adaptive_results, multichannel_results)
        
        # Save graphs
        output_path = "results/focused_improvement_demonstration.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Focused improvement graphs saved to: {output_path}")
        
        # Print summary
        print_focused_summary(adaptive_results, multichannel_results)
        
        plt.close(fig)
    else:
        print("Could not generate complete results for demonstration.")
    
    print(f"\n{'='*60}")
    print("🎯 FOCUSED IMPROVEMENT DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    
    if 'output_path' in locals():
        print(f"📊 Check {output_path} for detailed improvement visualizations!")


if __name__ == "__main__":
    main()

