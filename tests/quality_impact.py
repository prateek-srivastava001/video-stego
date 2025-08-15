#!/usr/bin/env python3
"""
Quality Impact Analysis for Video Steganography

Analyzes the visual quality impact of embedding hidden messages in videos.
Calculates PSNR, SSIM, and other quality metrics.
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
from config import BLOCK_SIZE, BLOCK_Y, BLOCK_X, VAL_ZERO, VAL_ONE


def calculate_psnr(original, modified):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(original, modified):
    """Calculate Structural Similarity Index between two images."""
    # Convert to grayscale if needed
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if len(modified.shape) == 3:
        modified = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
    
    # Constants for SSIM calculation
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    original = original.astype(np.float64)
    modified = modified.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(modified, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(original ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(modified ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original * modified, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


def analyze_single_frame_impact(frame, message_bits, bit_index):
    """Analyze the quality impact of embedding a single bit in a frame."""
    original_frame = frame.copy()
    modified_frame = frame.copy()
    
    # Apply embedding to the frame
    y0 = min(BLOCK_Y, max(0, frame.shape[0] - BLOCK_SIZE))
    x0 = min(BLOCK_X, max(0, frame.shape[1] - BLOCK_SIZE))
    
    if bit_index < len(message_bits):
        bit = message_bits[bit_index]
        modified_frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0] = VAL_ONE if bit == 1 else VAL_ZERO
    
    # Calculate quality metrics
    psnr = calculate_psnr(original_frame, modified_frame)
    ssim = calculate_ssim(original_frame, modified_frame)
    
    # Calculate local impact in embedding region
    original_block = original_frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0]
    modified_block = modified_frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0]
    local_mse = np.mean((original_block.astype(np.float64) - modified_block.astype(np.float64)) ** 2)
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'local_mse': local_mse,
        'bit_value': message_bits[bit_index] if bit_index < len(message_bits) else None,
        'original_mean': np.mean(original_block),
        'modified_mean': np.mean(modified_block)
    }


def analyze_video_quality_impact(video_path, messages):
    """Analyze quality impact for different message lengths."""
    results = {}
    
    for msg_length, message in messages.items():
        print(f"Analyzing quality impact for {msg_length}-character message...")
        
        # Create temporary output
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_output = tmp_file.name
        
        try:
            # Embed message
            embed_message_simple(video_path, temp_output, message)
            
            # Analyze quality
            quality_metrics = compare_video_quality(video_path, temp_output)
            quality_metrics['message_length'] = msg_length
            quality_metrics['message'] = message
            
            # Test extraction accuracy
            try:
                extracted = extract_message_simple(temp_output)
                quality_metrics['extraction_success'] = (extracted == message)
                quality_metrics['extracted_message'] = extracted
            except Exception as e:
                quality_metrics['extraction_success'] = False
                quality_metrics['extraction_error'] = str(e)
            
            results[msg_length] = quality_metrics
            
        except Exception as e:
            results[msg_length] = {'error': str(e)}
        finally:
            # Clean up
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    return results


def compare_video_quality(original_path, modified_path):
    """Compare quality between original and modified videos."""
    cap_orig = cv2.VideoCapture(original_path)
    cap_mod = cv2.VideoCapture(modified_path)
    
    if not cap_orig.isOpened() or not cap_mod.isOpened():
        raise RuntimeError("Cannot open videos for comparison")
    
    frame_metrics = []
    frame_count = 0
    
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_mod, frame_mod = cap_mod.read()
        
        if not ret_orig or not ret_mod:
            break
        
        # Calculate frame-level metrics
        psnr = calculate_psnr(frame_orig, frame_mod)
        ssim = calculate_ssim(frame_orig, frame_mod)
        
        frame_metrics.append({
            'frame': frame_count,
            'psnr': psnr,
            'ssim': ssim
        })
        
        frame_count += 1
        
        # Limit analysis to avoid excessive computation
        if frame_count >= 100:
            break
    
    cap_orig.release()
    cap_mod.release()
    
    # Calculate aggregate metrics
    psnr_values = [m['psnr'] for m in frame_metrics if m['psnr'] != float('inf')]
    ssim_values = [m['ssim'] for m in frame_metrics]
    
    # File size comparison
    orig_size = os.path.getsize(original_path)
    mod_size = os.path.getsize(modified_path)
    size_ratio = mod_size / orig_size if orig_size > 0 else 0
    
    return {
        'frame_count_analyzed': frame_count,
        'avg_psnr': np.mean(psnr_values) if psnr_values else 0,
        'min_psnr': np.min(psnr_values) if psnr_values else 0,
        'max_psnr': np.max(psnr_values) if psnr_values else 0,
        'std_psnr': np.std(psnr_values) if psnr_values else 0,
        'avg_ssim': np.mean(ssim_values) if ssim_values else 0,
        'min_ssim': np.min(ssim_values) if ssim_values else 0,
        'max_ssim': np.max(ssim_values) if ssim_values else 0,
        'std_ssim': np.std(ssim_values) if ssim_values else 0,
        'original_size_mb': orig_size / (1024 * 1024),
        'modified_size_mb': mod_size / (1024 * 1024),
        'size_ratio': size_ratio,
        'frame_metrics': frame_metrics
    }


def create_quality_analysis_plots(quality_results):
    """Create quality analysis visualizations."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    msg_lengths = list(quality_results.keys())
    avg_psnr = [quality_results[length].get('avg_psnr', 0) for length in msg_lengths]
    avg_ssim = [quality_results[length].get('avg_ssim', 0) for length in msg_lengths]
    size_ratios = [quality_results[length].get('size_ratio', 1) for length in msg_lengths]
    extraction_success = [quality_results[length].get('extraction_success', False) for length in msg_lengths]
    
    # Plot 1: PSNR vs Message Length
    ax1.set_title('PSNR vs Message Length')
    ax1.set_xlabel('Message Length (characters)')
    ax1.set_ylabel('Average PSNR (dB)')
    ax1.plot(msg_lengths, avg_psnr, 'bo-', linewidth=2, markersize=8)
    ax1.grid(True, alpha=0.3)
    
    # Add quality thresholds
    ax1.axhline(y=30, color='r', linestyle='--', alpha=0.7, label='Good Quality (30 dB)')
    ax1.axhline(y=40, color='g', linestyle='--', alpha=0.7, label='Excellent Quality (40 dB)')
    ax1.legend()
    
    # Plot 2: SSIM vs Message Length
    ax2.set_title('SSIM vs Message Length')
    ax2.set_xlabel('Message Length (characters)')
    ax2.set_ylabel('Average SSIM')
    ax2.plot(msg_lengths, avg_ssim, 'go-', linewidth=2, markersize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add SSIM quality thresholds
    ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Good Quality (0.8)')
    ax2.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Excellent Quality (0.9)')
    ax2.legend()
    
    # Plot 3: File Size Impact
    ax3.set_title('File Size Impact')
    ax3.set_xlabel('Message Length (characters)')
    ax3.set_ylabel('Size Ratio (Modified/Original)')
    ax3.plot(msg_lengths, size_ratios, 'ro-', linewidth=2, markersize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='k', linestyle='-', alpha=0.5, label='No Size Change')
    ax3.legend()
    
    # Plot 4: Extraction Success Rate
    ax4.set_title('Message Extraction Success')
    ax4.set_xlabel('Message Length (characters)')
    ax4.set_ylabel('Extraction Success')
    
    success_values = [1 if success else 0 for success in extraction_success]
    colors = ['green' if success else 'red' for success in extraction_success]
    bars = ax4.bar(msg_lengths, success_values, color=colors, alpha=0.7)
    ax4.set_ylim(0, 1.2)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Failed', 'Success'])
    ax4.grid(True, alpha=0.3)
    
    # Add success rate labels
    for i, (length, success) in enumerate(zip(msg_lengths, success_values)):
        ax4.text(length, success + 0.05, '✓' if success else '✗', 
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_detailed_quality_plot(quality_results, msg_length):
    """Create detailed quality analysis for a specific message length."""
    if msg_length not in quality_results:
        print(f"No quality data for {msg_length}-character message")
        return None
    
    data = quality_results[msg_length]
    frame_metrics = data.get('frame_metrics', [])
    
    if not frame_metrics:
        print(f"No frame-level data for {msg_length}-character message")
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    frames = [m['frame'] for m in frame_metrics]
    psnr_values = [m['psnr'] for m in frame_metrics]
    ssim_values = [m['ssim'] for m in frame_metrics]
    
    # Plot 1: Frame-by-frame PSNR
    ax1.set_title(f'Frame-by-Frame Quality Analysis - {msg_length} Characters')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('PSNR (dB)')
    ax1.plot(frames, psnr_values, 'b-', linewidth=1, alpha=0.7)
    ax1.grid(True, alpha=0.3)
    
    # Highlight frames with quality drops
    avg_psnr = np.mean(psnr_values)
    threshold = avg_psnr - 2 * np.std(psnr_values)
    low_quality_frames = [(f, p) for f, p in zip(frames, psnr_values) if p < threshold]
    
    if low_quality_frames:
        low_frames, low_psnr = zip(*low_quality_frames)
        ax1.scatter(low_frames, low_psnr, color='red', s=30, alpha=0.8, 
                   label=f'Quality Drops ({len(low_quality_frames)})')
        ax1.legend()
    
    # Plot 2: Frame-by-frame SSIM
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('SSIM')
    ax2.plot(frames, ssim_values, 'g-', linewidth=1, alpha=0.7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def print_quality_report(quality_results):
    """Print detailed quality analysis report."""
    print("\n" + "="*60)
    print("VIDEO QUALITY IMPACT ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nQuality Metrics Summary:")
    print(f"  {'Length':<8} {'PSNR':<8} {'SSIM':<8} {'Size Ratio':<12} {'Extraction':<12}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*12}")
    
    for msg_length, data in quality_results.items():
        if 'error' in data:
            print(f"  {msg_length:<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<12} {'ERROR':<12}")
            continue
        
        psnr = data.get('avg_psnr', 0)
        ssim = data.get('avg_ssim', 0)
        size_ratio = data.get('size_ratio', 1)
        extraction = '✓' if data.get('extraction_success', False) else '✗'
        
        print(f"  {msg_length:<8} {psnr:<7.1f}  {ssim:<7.3f}  {size_ratio:<11.3f}  {extraction:<12}")
    
    print(f"\nDetailed Analysis:")
    
    for msg_length, data in quality_results.items():
        if 'error' in data:
            print(f"\n  {msg_length}-character message: ERROR - {data['error']}")
            continue
        
        print(f"\n  {msg_length}-character message:")
        print(f"    PSNR: {data.get('avg_psnr', 0):.2f} ± {data.get('std_psnr', 0):.2f} dB")
        print(f"    SSIM: {data.get('avg_ssim', 0):.3f} ± {data.get('std_ssim', 0):.3f}")
        print(f"    File Size: {data.get('original_size_mb', 0):.2f} → {data.get('modified_size_mb', 0):.2f} MB")
        print(f"    Size Change: {((data.get('size_ratio', 1) - 1) * 100):+.1f}%")
        print(f"    Extraction: {'Success' if data.get('extraction_success', False) else 'Failed'}")
        
        if not data.get('extraction_success', False) and 'extraction_error' in data:
            print(f"    Error: {data['extraction_error']}")
    
    print(f"\nQuality Assessment:")
    avg_psnr = np.mean([data.get('avg_psnr', 0) for data in quality_results.values() if 'error' not in data])
    avg_ssim = np.mean([data.get('avg_ssim', 0) for data in quality_results.values() if 'error' not in data])
    
    print(f"  Overall Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Overall Average SSIM: {avg_ssim:.3f}")
    
    if avg_psnr >= 40:
        print("  Quality Assessment: Excellent (PSNR ≥ 40 dB)")
    elif avg_psnr >= 30:
        print("  Quality Assessment: Good (PSNR ≥ 30 dB)")
    elif avg_psnr >= 20:
        print("  Quality Assessment: Acceptable (PSNR ≥ 20 dB)")
    else:
        print("  Quality Assessment: Poor (PSNR < 20 dB)")


def main():
    """Main function to run quality impact analysis."""
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
    
    print(f"Analyzing quality impact for: {video_path}")
    
    # Test messages of different lengths
    test_messages = {
        10: "Short test",
        50: "This is a medium length test message for analysis",
        100: "This is a longer test message that will be used to analyze the quality impact of video steganography",
        250: "This is an even longer test message that spans multiple lines and contains more text to thoroughly test the quality impact of the video steganography system. It includes various characters and punctuation marks to ensure comprehensive testing of the embedding and extraction process."
    }
    
    print("Starting quality impact analysis...")
    print("This may take several minutes depending on video size...")
    
    # Analyze quality impact
    quality_results = analyze_video_quality_impact(video_path, test_messages)
    
    # Print report
    print_quality_report(quality_results)
    
    # Create visualizations
    print("\nGenerating quality analysis plots...")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Main quality plots
    fig1 = create_quality_analysis_plots(quality_results)
    fig1.savefig("results/quality_impact_analysis.png", dpi=300, bbox_inches='tight')
    print("Quality analysis plots saved to: results/quality_impact_analysis.png")
    plt.close(fig1)
    
    # Detailed plot for 100-character message
    fig2 = create_detailed_quality_plot(quality_results, 100)
    if fig2:
        fig2.savefig("results/detailed_quality_100chars.png", dpi=300, bbox_inches='tight')
        print("Detailed quality plot saved to: results/detailed_quality_100chars.png")
        plt.close(fig2)


if __name__ == "__main__":
    main()
