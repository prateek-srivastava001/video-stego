import hashlib

def generate_keystream(password: str, length: int) -> bytes:
    keystream = b''
    seed = password.encode('utf-8')
    counter = 0
    
    while len(keystream) < length:
        hash_input = seed + counter.to_bytes(4, 'big')
        keystream += hashlib.sha256(hash_input).digest()
        counter += 1
    
    return keystream[:length]


def encrypt_message(message: str, password: str) -> str:
    message_bytes = message.encode('utf-8')
    keystream = generate_keystream(password, len(message_bytes))
    encrypted_bytes = bytes(m ^ k for m, k in zip(message_bytes, keystream))
    return encrypted_bytes.decode('latin-1')


def decrypt_message(encrypted_data: str, password: str) -> str:
    try:
        encrypted_bytes = encrypted_data.encode('latin-1')
        keystream = generate_keystream(password, len(encrypted_bytes))
        decrypted_bytes = bytes(e ^ k for e, k in zip(encrypted_bytes, keystream))
        result = decrypted_bytes.decode('utf-8')
        if not all(c.isprintable() or c.isspace() for c in result):
            raise ValueError("Decrypted data contains non-printable characters")        
        return result
        
    except UnicodeDecodeError:
        raise ValueError("Decryption failed: incorrect password")
    except Exception as e:
        raise ValueError(f"Decryption failed: incorrect password or corrupted data") from e