#!/usr/bin/env python3
"""
Multi-Channel Embedding for Video Steganography

This module implements a multi-channel embedding strategy that distributes
data across RGB channels, increasing capacity and improving data distribution
to reduce detectability.
"""

import cv2
import os
import tempfile
import subprocess
import numpy as np
from config import START_FRAME, STEP, BLOCK_SIZE, BLOCK_Y, BLOCK_X, VAL_ZERO, VAL_ONE, THRESH
from utils.utils import text_to_bits, int_to_32bits, get_frame_indices


def get_channel_strategy(frame_shape, bit_count):
    """Determine optimal channel distribution strategy."""
    height, width = frame_shape[:2]
    
    # Calculate available embedding positions per channel
    max_frames = ((height * width) - START_FRAME + STEP - 1) // STEP
    
    # Distribute bits across channels for better capacity
    if bit_count <= max_frames:
        # Single channel is sufficient - use blue for compatibility
        return {'channels': [0], 'distribution': [bit_count]}
    elif bit_count <= max_frames * 2:
        # Use two channels - blue and green
        half = bit_count // 2
        remainder = bit_count % 2
        return {'channels': [0, 1], 'distribution': [half + remainder, half]}
    else:
        # Use all three channels
        third = bit_count // 3
        remainder = bit_count % 3
        if remainder == 0:
            return {'channels': [0, 1, 2], 'distribution': [third, third, third]}
        elif remainder == 1:
            return {'channels': [0, 1, 2], 'distribution': [third + 1, third, third]}
        else:
            return {'channels': [0, 1, 2], 'distribution': [third + 1, third + 1, third]}


def distribute_bits_across_channels(payload_bits, strategy):
    """Distribute payload bits across multiple channels."""
    channels = strategy['channels']
    distribution = strategy['distribution']
    
    channel_bits = {}
    bit_index = 0
    
    for i, channel in enumerate(channels):
        bit_count = distribution[i]
        channel_bits[channel] = payload_bits[bit_index:bit_index + bit_count]
        bit_index += bit_count
    
    return channel_bits


def embed_message_multichannel(in_path: str, out_path: str, message: str):
    """Embed message using multi-channel distribution."""
    msg_bits = text_to_bits(message)
    header_bits = int_to_32bits(len(msg_bits))
    payload_bits = header_bits + msg_bits
    
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {in_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate capacity with multi-channel support
    max_available_frames = max(0, (total_frames - START_FRAME + STEP - 1) // STEP)
    max_capacity_bits_single = max_available_frames - 32
    max_capacity_bits_multi = (max_available_frames * 3) - 32  # 3 channels
    
    if len(payload_bits) > max_capacity_bits_multi:
        cap.release()
        max_capacity_chars_single = max(0, max_capacity_bits_single // 8)
        max_capacity_chars_multi = max(0, max_capacity_bits_multi // 8)
        
        raise ValueError(
            f"Text too long for this video. "
            f"Your message has {len(message)} characters ({len(payload_bits)} bits), "
            f"but this video can fit approximately {max_capacity_chars_single} characters "
            f"(single-channel) or {max_capacity_chars_multi} characters (multi-channel). "
            f"Video has {total_frames} frames, allowing {max_available_frames} embedding positions."
        )

    # Determine optimal channel strategy
    strategy = get_channel_strategy((height, width), len(payload_bits))
    channel_bits = distribute_bits_across_channels(payload_bits, strategy)
    
    print(f"Using multi-channel strategy: {len(strategy['channels'])} channels")
    print(f"Bit distribution: {dict(zip(strategy['channels'], strategy['distribution']))}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video = os.path.join(temp_dir, "temp_uncompressed.avi")
        
        fourcc = cv2.VideoWriter.fourcc(*'MJPG') 
        temp_out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        if not temp_out.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create temporary video writer")

        y0 = min(BLOCK_Y, max(0, height - BLOCK_SIZE))
        x0 = min(BLOCK_X, max(0, width  - BLOCK_SIZE))

        # Track bit pointers for each channel
        bit_ptrs = {channel: 0 for channel in strategy['channels']}
        frame_id = 0
        
        # Calculate frame indices for each channel
        channel_targets = {}
        for channel in strategy['channels']:
            bit_count = strategy['distribution'][strategy['channels'].index(channel)]
            indices = get_frame_indices(bit_count, START_FRAME, STEP)
            channel_targets[channel] = set(indices)

        print(f"Processing {total_frames} frames with multi-channel embedding...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process each channel
            for channel in strategy['channels']:
                if frame_id in channel_targets[channel] and bit_ptrs[channel] < len(channel_bits[channel]):
                    bit = channel_bits[channel][bit_ptrs[channel]]
                    
                    # Modify the specific channel in the block
                    frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, channel] = VAL_ONE if bit == 1 else VAL_ZERO
                    bit_ptrs[channel] += 1
                    
            temp_out.write(frame)
            frame_id += 1

        cap.release()
        temp_out.release()
        
        # Save strategy metadata for extraction
        metadata_file = out_path.replace('.mp4', '_multichannel_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"channels:{','.join(map(str, strategy['channels']))}\n")
            f.write(f"distribution:{','.join(map(str, strategy['distribution']))}\n")
            f.write(f"total_bits:{len(payload_bits)}\n")
        
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
            print(f"Multi-channel metadata saved to {metadata_file}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg compression failed: {e.stderr.decode() if e.stderr else str(e)}")


def extract_message_multichannel(stego_path: str) -> str:
    """Extract message using multi-channel distribution."""
    metadata_file = stego_path.replace('.mp4', '_multichannel_metadata.txt')
    
    # Load strategy metadata
    try:
        with open(metadata_file, 'r') as f:
            lines = f.readlines()
            channels = list(map(int, lines[0].split(':')[1].strip().split(',')))
            distribution = list(map(int, lines[1].split(':')[1].strip().split(',')))
            total_bits = int(lines[2].split(':')[1].strip())
    except FileNotFoundError:
        raise RuntimeError(f"Multi-channel metadata file not found: {metadata_file}")

    cap = cv2.VideoCapture(stego_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stego video: {stego_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Extracting from video: {total_frames} frames, {width}x{height}")
    print(f"Using channels: {channels}, distribution: {distribution}")

    y0 = min(BLOCK_Y, max(0, height - BLOCK_SIZE))
    x0 = min(BLOCK_X, max(0, width  - BLOCK_SIZE))

    # Extract bits from each channel
    all_extracted_bits = []
    
    for i, channel in enumerate(channels):
        bit_count = distribution[i]
        frame_indices = get_frame_indices(bit_count, START_FRAME, STEP)
        targets = set(frame_indices)
        
        channel_bits = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_id = 0

        print(f"Reading {bit_count} bits from channel {channel}...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id in targets:
                mean_channel = float(frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, channel].mean())
                channel_bits.append(1 if mean_channel > THRESH else 0)
                if len(channel_bits) >= bit_count:
                    break
            frame_id += 1
        
        if len(channel_bits) < bit_count:
            cap.release()
            raise RuntimeError(f"Could not read complete data from channel {channel}: got {len(channel_bits)} bits, expected {bit_count}")
        
        all_extracted_bits.extend(channel_bits)

    cap.release()
    
    # Reconstruct message
    if len(all_extracted_bits) < 32:
        raise RuntimeError("Could not read complete header from video")
    
    from utils.utils import bits_to_int32, bits_to_text
    header_bits = all_extracted_bits[:32]
    msg_bits_len = bits_to_int32(header_bits)
    print(f"Message length: {msg_bits_len} bits")
    
    if msg_bits_len <= 0 or msg_bits_len > 100000:
        raise RuntimeError(f"Invalid message length: {msg_bits_len}")

    msg_bits = all_extracted_bits[32:32 + msg_bits_len]
    
    if len(msg_bits) < msg_bits_len:
        raise RuntimeError(f"Could not read complete message: got {len(msg_bits)} bits, expected {msg_bits_len}")
    
    return bits_to_text(msg_bits)

