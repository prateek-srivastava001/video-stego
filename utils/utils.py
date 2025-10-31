def get_frame_indices(bit_count, start, step):
    return [start + i * step for i in range(bit_count)]

def text_to_bits(text: str):
    data = text.encode("utf-8")
    bits = []
    for byte in data:
        bits.extend([(byte >> j) & 1 for j in range(7, -1, -1)])
    return bits

def bits_to_text(bits):
    if len(bits) % 8 != 0:
        bits = bits[:len(bits) - (len(bits) % 8)]
    out = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for j in range(8):
            b = (b << 1) | (bits[i+j] & 1)
        out.append(b)
    return out.decode("utf-8", errors="ignore")

def int_to_32bits(n: int):
    return [(n >> j) & 1 for j in range(31, -1, -1)]

def bits_to_int32(bits32):
    val = 0
    for b in bits32:
        val = (val << 1) | (b & 1)
    return val