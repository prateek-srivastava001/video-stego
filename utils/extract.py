import cv2
from config import START_FRAME, STEP, BLOCK_SIZE, BLOCK_Y, BLOCK_X, THRESH
from utils.utils import bits_to_text, bits_to_int32, get_frame_indices

def extract_message_simple(stego_path: str) -> str:
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

    print("Reading header bits...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in targets:
            mean_blue = float(frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0].mean())
            header_bits.append(1 if mean_blue > THRESH else 0)
            if len(header_bits) >= 32:
                break
        frame_id += 1

    if len(header_bits) < 32:
        cap.release()
        raise RuntimeError("Could not read complete header from video")

    msg_bits_len = bits_to_int32(header_bits)
    print(f"Message length: {msg_bits_len} bits")
    
    if msg_bits_len <= 0 or msg_bits_len > 100000:  # Sanity check
        cap.release()
        raise RuntimeError(f"Invalid message length: {msg_bits_len}")

    # Read message bits
    data_indices = get_frame_indices(32 + msg_bits_len, START_FRAME, STEP)[32:]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    msg_bits = []
    frame_id = 0
    targets = set(data_indices)

    print("Reading message bits...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in targets:
            mean_blue = float(frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0].mean())
            msg_bits.append(1 if mean_blue > THRESH else 0)
            if len(msg_bits) >= msg_bits_len:
                break
        frame_id += 1

    cap.release()
    
    if len(msg_bits) < msg_bits_len:
        raise RuntimeError(f"Could not read complete message: got {len(msg_bits)} bits, expected {msg_bits_len}")
    
    return bits_to_text(msg_bits)