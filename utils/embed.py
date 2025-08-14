import cv2
from config import START_FRAME, STEP, BLOCK_SIZE, BLOCK_Y, BLOCK_X, VAL_ZERO, VAL_ONE
from utils.utils import text_to_bits, int_to_32bits, get_frame_indices

def embed_message_simple(in_path: str, out_path: str, message: str):
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
        raise ValueError(
            f"Not enough frames. Need frame index {frame_indices[-1]}, "
            f"but video has only {total_frames} frames."
        )

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    y0 = min(BLOCK_Y, max(0, height - BLOCK_SIZE))
    x0 = min(BLOCK_X, max(0, width  - BLOCK_SIZE))

    bit_ptr = 0
    frame_id = 0
    targets = set(frame_indices)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in targets:
            bit = payload_bits[bit_ptr]
            frame[y0:y0+BLOCK_SIZE, x0:x0+BLOCK_SIZE, 0] = VAL_ONE if bit == 1 else VAL_ZERO
            bit_ptr += 1
        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print(f"Embedded {len(msg_bits)} bits (+32 header) into {out_path}")