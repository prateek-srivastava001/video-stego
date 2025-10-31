import cv2
import os
import json
import tempfile
import subprocess
from config import START_FRAME, STEP, BLOCK_SIZE, BLOCK_Y, BLOCK_X, VAL_ZERO, VAL_ONE
from utils.utils import text_to_bits, int_to_32bits, get_frame_indices
from utils.crypto import encrypt_message

def embed_message_simple(in_path: str, out_path: str, message: str, password: str = None):
    if password:
        print("Encrypting message with password...")
        message = encrypt_message(message, password)
    
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
        
        # Calculate maximum capacity
        # Maximum frames available for embedding (considering START_FRAME and STEP)
        max_available_frames = max(0, (total_frames - START_FRAME + STEP - 1) // STEP)
        
        # Each text character requires 8 bits, plus 32 bits for header
        # So max_chars = (max_available_frames - 32) / 8
        max_capacity_bits = max_available_frames - 32  # Subtract header bits
        max_capacity_chars = max(0, max_capacity_bits // 8)
        
        # Calculate what we actually need
        required_chars = len(message)
        required_bits = len(payload_bits)
        
        raise ValueError(
            f"Text too long for this video. "
            f"Your message has {required_chars} characters ({required_bits} bits), "
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

        print(f"Processing {total_frames} frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id in targets and bit_ptr < len(payload_bits):
                bit = payload_bits[bit_ptr]
                # Modify the blue channel in the specified block
                frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0] = VAL_ONE if bit == 1 else VAL_ZERO
                bit_ptr += 1
            temp_out.write(frame)
            frame_id += 1

        cap.release()
        temp_out.release()
        
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', in_path
            ], capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            original_bitrate = int(probe_data['format'].get('bit_rate', '2500000'))
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            original_bitrate = 2500000  # Default 2.5 Mbps
        
        print(f"Compressing with FFmpeg (target bitrate: {original_bitrate // 1000}k)...")
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_video,
                '-c:v', 'libx264',         # H.264 codec
                '-preset', 'medium',       # Balance speed/compression
                '-crf', '23',             # Good quality constant rate factor
                '-b:v', f'{original_bitrate}',  # Match original bitrate
                '-maxrate', f'{int(original_bitrate * 1.2)}',  # Allow 20% variance
                '-bufsize', f'{int(original_bitrate * 2)}',    # Buffer size
                '-pix_fmt', 'yuv420p',    # Standard pixel format
                '-movflags', '+faststart', # Optimize for streaming
                out_path
            ], check=True, capture_output=True)
            
            print(f"Successfully embedded {len(msg_bits)} bits (+32 header) into {out_path}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg compression failed: {e.stderr.decode() if e.stderr else str(e)}")