from config import DEFAULT_INPUT_VIDEO, DEFAULT_OUTPUT_VIDEO
from utils.embed import embed_message_simple
from utils.extract import extract_message_simple

if __name__ == "__main__":
    message = "hellloooo mai hu doraemon"
    embed_message_simple(DEFAULT_INPUT_VIDEO, DEFAULT_OUTPUT_VIDEO, message)
    recovered = extract_message_simple(DEFAULT_OUTPUT_VIDEO)
    print("Extracted message:", repr(recovered))