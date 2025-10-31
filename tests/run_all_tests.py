#!/usr/bin/env python3
"""
Main Test Runner for Video Steganography Analysis

Runs all analysis tests and generates comprehensive reports and visualizations.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def check_video_files():
    """Check for available video files."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    # Check videos directory
    videos_dir = Path("videos")
    if videos_dir.exists():
        for file in videos_dir.iterdir():
            if file.suffix.lower() in video_extensions:
                video_files.append(str(file))
    
    return video_files


def run_test_script(script_name, description):
    """Run a specific test script and capture its output."""
    script_path = Path("tests") / script_name
    
    if not script_path.exists():
        print(f"  ✗ {script_name} not found")
        return False
    
    print(f"  → Running {description}...")
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"  ✓ {description} completed ({duration:.1f}s)")
            return True
        else:
            print(f"  ✗ {description} failed ({duration:.1f}s)")
            if result.stderr:
                print(f"    Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ {description} timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"  ✗ {description} error: {e}")
        return False


def main():
    """Main function to run all tests."""
    print("="*60)
    print("VIDEO STEGANOGRAPHY - COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Check working directory
    if not Path("main.py").exists():
        print("Error: Run this script from the project root directory")
        return 1
    
    # Check video files
    print("\n2. Checking video files...")
    video_files = check_video_files()
    if not video_files:
        print("  ⚠ No video files found in 'videos/' directory")
        print("  Place a video file (e.g., 'videos/sample.mp4') for analysis")
        print("  Continuing with theoretical analysis only...")
    else:
        print(f"  ✓ Found {len(video_files)} video file(s):")
        for video in video_files:
            file_size = Path(video).stat().st_size / (1024 * 1024)  # MB
            print(f"    - {video} ({file_size:.1f} MB)")
    
    # Create tests directory if it doesn't exist
    test_dir = Path("tests")
    test_dir.mkdir(exist_ok=True)
    
    # Run analysis tests
    print("\n4. Running analysis tests...")
    test_results = {}
    
    tests = [
        ("capacity_analysis.py", "Capacity Analysis"),
        ("frame_utilization.py", "Frame Utilization Analysis"),
        ("quality_impact.py", "Quality Impact Analysis"),
    ]
    
    for script, description in tests:
        test_results[script] = run_test_script(script, description)
    
    # Final report
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    successful_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\nTest Results: {successful_tests}/{total_tests} successful")
    
    for script, success in test_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {script}")
    
    print(f"\nGenerated Files:")
    plot_files = list(Path("results").glob("*.png"))
    if plot_files:
        for plot in sorted(plot_files):
            print(f"  - {plot}")
    else:
        print("  (No plots generated - check for errors above)")
    
    print(f"\nNext Steps:")
    print(f"  1. Review generated plots in the 'results/' directory")
    print(f"  2. Use the analysis data for your writeup")
    print(f"  3. All plots are saved as PNG files without popup windows")
    
    if not video_files:
        print(f"  4. Add video files to 'videos/' directory for complete analysis")
    
    return 0 if successful_tests == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
