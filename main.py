#!/usr/bin/env python3
"""
Video Steganography CLI Tool

A command-line interface for embedding and extracting hidden messages in video files.
"""

import argparse
import sys
import os
import getpass
from pathlib import Path
from config import DEFAULT_INPUT_VIDEO, DEFAULT_OUTPUT_VIDEO
from utils.embed import embed_message_simple
from utils.extract import extract_message_simple


def get_input_with_default(prompt: str, default: str) -> str:
    user_input = input(f"{prompt} [{default}]: ").strip()
    return user_input if user_input else default


def validate_file_exists(filepath: str, file_type: str = "file") -> None:
    if not os.path.exists(filepath):
        print(f"Error: {file_type} '{filepath}' does not exist.")
        sys.exit(1)


def ensure_output_dir(filepath: str) -> None:
    output_dir = Path(filepath).parent
    output_dir.mkdir(parents=True, exist_ok=True)


def cmd_embed(args) -> None:
    if args.yes:
        input_video = DEFAULT_INPUT_VIDEO
    else:
        input_video = get_input_with_default("Enter input video path", DEFAULT_INPUT_VIDEO)
    
    validate_file_exists(input_video, "Input video")
    
    if args.yes:
        output_video = DEFAULT_OUTPUT_VIDEO
    else:
        output_video = get_input_with_default("Enter output video path", DEFAULT_OUTPUT_VIDEO)
    
    ensure_output_dir(output_video)
    
    # Determine message
    if args.message:
        message = args.message
    else:
        message = input("Enter the message to embed: ")
    
    if not message:
        print("Error: Message cannot be empty.")
        sys.exit(1)
    
    # Get password if requested
    password = None
    if args.password:
        password = getpass.getpass("Enter password for encryption: ")
        if not password:
            print("Error: Password cannot be empty when -p flag is used.")
            sys.exit(1)
        # Confirm password
        password_confirm = getpass.getpass("Confirm password: ")
        if password != password_confirm:
            print("Error: Passwords do not match.")
            sys.exit(1)
    
    try:
        print(f"Embedding message into '{input_video}' -> '{output_video}'...")
        embed_message_simple(input_video, output_video, message, password)
        if password:
            print(f"Successfully embedded encrypted message into '{output_video}'")
        else:
            print(f"Successfully embedded message into '{output_video}'")
    except Exception as e:
        print(f"Error during embedding: {e}")
        sys.exit(1)


def cmd_recover(args) -> None:
    if args.yes:
        stego_video = DEFAULT_OUTPUT_VIDEO
    else:
        stego_video = get_input_with_default("Enter path to video with hidden message", DEFAULT_OUTPUT_VIDEO)
    
    validate_file_exists(stego_video, "Stego video")
    
    password = None
    if args.password:
        password = getpass.getpass("Enter password for decryption: ")
        if not password:
            print("Error: Password cannot be empty when -p flag is used.")
            sys.exit(1)
    
    try:
        print(f"Extracting message from '{stego_video}'...")
        recovered_message = extract_message_simple(stego_video, password)
        print("Successfully extracted message:")
        print(f"Message: {repr(recovered_message)}")
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Video Steganography CLI Tool - Hide and extract messages in video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s embed                          # Interactive embed with prompts
  %(prog)s -y -m "secret message" embed   # Embed with defaults and specified message
  %(prog)s -p -m "secret message" embed   # Embed with password protection
  %(prog)s recover                        # Interactive recovery with prompts
  %(prog)s -y recover                     # Recover with default paths
  %(prog)s -p recover                     # Recover with password decryption
  
Default paths:
  Input video:  {input}
  Output video: {output}
        """.format(input=DEFAULT_INPUT_VIDEO, output=DEFAULT_OUTPUT_VIDEO)
    )
    
    # Global flags
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Use default paths without prompting')
    parser.add_argument('-m', '--message', type=str,
                       help='Message to embed (only for embed command)')
    parser.add_argument('-p', '--password', action='store_true',
                       help='Enable password protection (encrypt/decrypt message)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    embed_parser = subparsers.add_parser('embed', help='Embed a message into a video')
    embed_parser.set_defaults(func=cmd_embed)
    
    recover_parser = subparsers.add_parser('recover', help='Extract a hidden message from a video')
    recover_parser.set_defaults(func=cmd_recover)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.message and args.command != 'embed':
        print("Error: -m/--message flag can only be used with the 'embed' command.")
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()