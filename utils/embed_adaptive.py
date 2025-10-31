#!/usr/bin/env python3
"""
Adaptive Threshold Embedding for Video Steganography

This module implements an adaptive embedding strategy that analyzes the original
frame content to choose optimal embedding values that are more robust against
compression and less detectable.
"""

import cv2
import os
import tempfile
import subprocess
import numpy as np
from config import START_FRAME, STEP, BLOCK_SIZE, BLOCK_Y, BLOCK_X, THRESH
from utils.utils import text_to_bits, int_to_32bits, get_frame_indices


def analyze_block_characteristics(block):
    """Analyze a block to determine optimal embedding values."""
    mean_val = np.mean(block)
    std_val = np.std(block)
    
    # Choose embedding values based on block characteristics
    if mean_val < 85:  # Dark region
        val_zero = max(16, int(mean_val - std_val))
        val_one = min(255, int(mean_val + std_val + 40))
    elif mean_val > 170:  # Bright region
        val_zero = max(0, int(mean_val - std_val - 40))
        val_one = min(239, int(mean_val + std_val))
    else:  # Mid-tone region
        val_zero = max(16, int(mean_val - 60))
        val_one = min(239, int(mean_val + 60))
    
    # Ensure minimum separation for reliable extraction
    if val_one - val_zero < 80:
        center = (val_zero + val_one) // 2
        val_zero = max(16, center - 40)
        val_one = min(239, center + 40)
    
    return val_zero, val_one


def embed_message_adaptive(in_path: str, out_path: str, message: str):
    """Embed message using adaptive threshold values."""
    msg_bits = text_to_bits(message)
    header_bits = int_to_32bits(len(msg_bits))
    payload_bits = header_bits + msg_bits
    frame_indices = get_frame_indices(len(payload_bits), START_FRAME, STEP)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {in_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_indices[-1] >= total_frames:
        cap.release()
        max_available_frames = max(0, (total_frames - START_FRAME + STEP - 1) // STEP)
        max_capacity_bits = max_available_frames - 32
        max_capacity_chars = max(0, max_capacity_bits // 8)
        
        raise ValueError(
            f"Text too long for this video. "
            f"Your message has {len(message)} characters ({len(payload_bits)} bits), "
            f"but this video can only fit approximately {max_capacity_chars} characters "
            f"({max_capacity_bits} bits after header). "
            f"Video has {total_frames} frames, allowing {max_available_frames} embedding positions."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video = os.path.join(temp_dir, "temp_uncompressed.avi")
        
        fourcc = cv2.VideoWriter.fourcc(*'MJPG') 
        temp_out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        if not temp_out.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create temporary video writer")

        y0 = min(BLOCK_Y, max(0, height - BLOCK_SIZE))
        x0 = min(BLOCK_X, max(0, width  - BLOCK_SIZE))

        bit_ptr = 0
        frame_id = 0
        targets = set(frame_indices)
        
        # Store adaptive values for later extraction
        adaptive_values = {}

        print(f"Processing {total_frames} frames with adaptive embedding...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id in targets and bit_ptr < len(payload_bits):
                bit = payload_bits[bit_ptr]
                
                # Analyze the original block
                original_block = frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0].copy()
                val_zero, val_one = analyze_block_characteristics(original_block)
                
                # Store values for extraction
                adaptive_values[frame_id] = {'val_zero': val_zero, 'val_one': val_one, 'original_mean': np.mean(original_block)}
                
                # Apply adaptive embedding
                frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0] = val_one if bit == 1 else val_zero
                bit_ptr += 1
                
            temp_out.write(frame)
            frame_id += 1

        cap.release()
        temp_out.release()
        
        # Save adaptive values for extraction
        metadata_file = out_path.replace('.mp4', '_adaptive_metadata.txt')
        with open(metadata_file, 'w') as f:
            for frame_id, values in adaptive_values.items():
                f.write(f"{frame_id},{values['val_zero']},{values['val_one']},{values['original_mean']:.2f}\n")
        
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', in_path
            ], capture_output=True, text=True, check=True)
            import json
            probe_data = json.loads(result.stdout)
            original_bitrate = int(probe_data['format'].get('bit_rate', '2500000'))
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            original_bitrate = 2500000

        print(f"Compressing with FFmpeg (target bitrate: {original_bitrate // 1000}k)...")
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_video,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-b:v', f'{original_bitrate}',
                '-maxrate', f'{int(original_bitrate * 1.2)}',
                '-bufsize', f'{int(original_bitrate * 2)}',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                out_path
            ], check=True, capture_output=True)
            
            print(f"Successfully embedded {len(msg_bits)} bits (+32 header) into {out_path}")
            print(f"Adaptive metadata saved to {metadata_file}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg compression failed: {e.stderr.decode() if e.stderr else str(e)}")


def extract_message_adaptive(stego_path: str) -> str:
    """Extract message using adaptive threshold values."""
    metadata_file = stego_path.replace('.mp4', '_adaptive_metadata.txt')
    
    # Load adaptive values
    adaptive_values = {}
    try:
        with open(metadata_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                val_zero = int(parts[1])
                val_one = int(parts[2])
                original_mean = float(parts[3])
                adaptive_values[frame_id] = {
                    'val_zero': val_zero, 
                    'val_one': val_one, 
                    'threshold': (val_zero + val_one) // 2,
                    'original_mean': original_mean
                }
    except FileNotFoundError:
        raise RuntimeError(f"Adaptive metadata file not found: {metadata_file}")

    cap = cv2.VideoCapture(stego_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stego video: {stego_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Extracting from video: {total_frames} frames, {width}x{height}")

    y0 = min(BLOCK_Y, max(0, height - BLOCK_SIZE))
    x0 = min(BLOCK_X, max(0, width  - BLOCK_SIZE))

    # Read header bits (32 bits for message length)
    header_indices = get_frame_indices(32, START_FRAME, STEP)
    header_bits = []
    frame_id = 0
    targets = set(header_indices)

    print("Reading header bits with adaptive thresholds...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in targets:
            mean_blue = float(frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0].mean())
            
            # Use adaptive threshold if available, otherwise use default
            if frame_id in adaptive_values:
                threshold = adaptive_values[frame_id]['threshold']
            else:
                threshold = THRESH
                
            header_bits.append(1 if mean_blue > threshold else 0)
            if len(header_bits) >= 32:
                break
        frame_id += 1

    if len(header_bits) < 32:
        cap.release()
        raise RuntimeError("Could not read complete header from video")

    from utils.utils import bits_to_int32, bits_to_text
    msg_bits_len = bits_to_int32(header_bits)
    print(f"Message length: {msg_bits_len} bits")
    
    if msg_bits_len <= 0 or msg_bits_len > 100000:
        cap.release()
        raise RuntimeError(f"Invalid message length: {msg_bits_len}")

    # Read message bits
    data_indices = get_frame_indices(32 + msg_bits_len, START_FRAME, STEP)[32:]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    msg_bits = []
    frame_id = 0
    targets = set(data_indices)

    print("Reading message bits with adaptive thresholds...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in targets:
            mean_blue = float(frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0].mean())
            
            # Use adaptive threshold if available, otherwise use default
            if frame_id in adaptive_values:
                threshold = adaptive_values[frame_id]['threshold']
            else:
                threshold = THRESH
                
            msg_bits.append(1 if mean_blue > threshold else 0)
            if len(msg_bits) >= msg_bits_len:
                break
        frame_id += 1

    cap.release()
    
    if len(msg_bits) < msg_bits_len:
        raise RuntimeError(f"Could not read complete message: got {len(msg_bits)} bits, expected {msg_bits_len}")
    
    return bits_to_text(msg_bits)

