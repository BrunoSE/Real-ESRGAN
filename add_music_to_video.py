#!/usr/bin/env python3
"""
Add background music to videos by replacing audio with a random section from a music file.

For Instagram Reels: extracts a random segment from your music that matches video length.
"""

import argparse
import glob
import os
import random
import subprocess
import sys
from pathlib import Path

try:
    import ffmpeg
except ImportError:
    print("Error: ffmpeg-python is not installed")
    print("Install it with: pip install ffmpeg-python")
    sys.exit(1)


MUSIC_DIR = 'inputs/music'
DEFAULT_MUSIC = 'inputs/music/top_2026.mp3'


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def get_duration(file_path):
    """Get duration of video or audio file in seconds"""
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe['format']['duration'])
    except Exception as e:
        print(f"{Colors.RED}Error probing {file_path}: {e}{Colors.END}")
        return None


def add_random_music_section(video_path, music_path, output_path, fade_duration=0.5):
    """
    Replace video audio with a random section from music file.

    Args:
        video_path: Input video file
        music_path: Music file (mp3, wav, etc.)
        output_path: Output video file
        fade_duration: Fade in/out duration in seconds (default: 0.5)

    Returns:
        bool: True if successful, False otherwise
    """
    # Get durations
    video_duration = get_duration(video_path)
    music_duration = get_duration(music_path)

    if video_duration is None or music_duration is None:
        return False

    print(f"  Video duration: {video_duration:.1f}s")
    print(f"  Music duration: {music_duration:.1f}s")

    # Check if music is long enough
    if music_duration < video_duration:
        print(f"  {Colors.YELLOW}⚠ Music shorter than video - will use entire music file{Colors.END}")
        start_time = 0
        extract_duration = music_duration
    else:
        # Pick random start time
        max_start = music_duration - video_duration
        start_time = random.uniform(0, max_start)
        extract_duration = video_duration
        print(f"  {Colors.BLUE}→ Random section: {start_time:.1f}s to {start_time + extract_duration:.1f}s{Colors.END}")

    # Build ffmpeg command
    # -ss before -i is faster (input seeking)
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-ss', str(start_time),
        '-t', str(extract_duration),
        '-i', music_path,
        '-map', '0:v',  # Video from first input
        '-map', '1:a',  # Audio from second input (music)
        '-c:v', 'copy',  # Copy video without re-encoding (fast!)
    ]

    # Add audio filters for fade in/out
    if fade_duration > 0 and fade_duration < extract_duration / 2:
        audio_filter = (
            f"afade=t=in:st=0:d={fade_duration},"
            f"afade=t=out:st={extract_duration - fade_duration}:d={fade_duration}"
        )
        cmd.extend(['-af', audio_filter])

    # Audio encoding
    cmd.extend([
        '-c:a', 'aac',
        '-b:a', '192k',
        '-ar', '48000',
        '-shortest',  # End when shortest stream ends
        output_path
    ])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{Colors.RED}FFmpeg error: {result.stderr[-500:]}{Colors.END}")
            return False
        return True
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.END}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Add background music to videos (random section matching video duration)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: uses inputs/music/top_2026.mp3
  uv run python add_music_to_video.py results/*.mp4 -o final/

  # Use a different file from inputs/music/ folder
  uv run python add_music_to_video.py results/*.mp4 -o final/ -m other_song.mp3

  # Single video
  uv run python add_music_to_video.py results/video_instagram.mp4 -o final/

  # No fade in/out
  uv run python add_music_to_video.py results/*.mp4 -o final/ --no-fade

  # Longer fade (2 seconds)
  uv run python add_music_to_video.py results/*.mp4 -o final/ --fade 2.0
        """
    )

    parser.add_argument(
        'videos',
        nargs='+',
        help='Input video file(s) or pattern (e.g., results/*.mp4)'
    )
    parser.add_argument(
        '-m', '--music',
        type=str,
        default=None,
        help=f'Music filename from inputs/music/ (default: top_2026.mp3). E.g. -m other_song.mp3'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output video file or directory'
    )
    parser.add_argument(
        '--fade',
        type=float,
        default=0.5,
        help='Fade in/out duration in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--no-fade',
        action='store_true',
        help='Disable fade in/out'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_music',
        help='Suffix for output files when output is a directory (default: _music)'
    )

    args = parser.parse_args()

    # Resolve music file path - always looks in MUSIC_DIR
    if args.music is None:
        music_path = DEFAULT_MUSIC
    else:
        # Assume filename is in MUSIC_DIR (strip any dir prefix the user may have included)
        music_path = os.path.join(MUSIC_DIR, os.path.basename(args.music))

    print(f"{Colors.BLUE}Using music: {music_path}{Colors.END}\n")

    if not os.path.isfile(music_path):
        print(f"{Colors.RED}Error: Music file not found: {music_path}{Colors.END}")
        print(f"{Colors.YELLOW}Place your music files in {MUSIC_DIR}/{Colors.END}")
        return 1

    args.music = music_path

    # Expand wildcards
    video_files = []
    for pattern in args.videos:
        if '*' in pattern or '?' in pattern:
            video_files.extend(glob.glob(pattern))
        else:
            video_files.append(pattern)

    # Filter to existing files
    video_files = [f for f in video_files if os.path.isfile(f)]

    if not video_files:
        print(f"{Colors.RED}Error: No video files found{Colors.END}")
        return 1

    # Determine if output is directory or file
    if len(video_files) > 1 or os.path.isdir(args.output):
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        single_output = False
    else:
        single_output = True
        output_dir = None

    # Set fade duration
    fade_duration = 0 if args.no_fade else args.fade

    print(f"{Colors.BOLD}Adding music to {len(video_files)} video(s){Colors.END}")
    print(f"Music: {args.music}")
    if fade_duration > 0:
        print(f"Fade: {fade_duration}s in/out")
    print()

    # Process videos
    success_count = 0
    for i, video_file in enumerate(video_files, 1):
        filename = os.path.basename(video_file)
        name_without_ext = os.path.splitext(filename)[0]

        # Determine output path
        if single_output:
            output_path = args.output
        else:
            output_filename = f"{name_without_ext}{args.suffix}.mp4"
            output_path = os.path.join(output_dir, output_filename)

        print(f"{Colors.BOLD}[{i}/{len(video_files)}] {filename}{Colors.END}")

        if add_random_music_section(video_file, args.music, output_path, fade_duration):
            print(f"{Colors.GREEN}✓ Saved: {output_path}{Colors.END}\n")
            success_count += 1
        else:
            print(f"{Colors.RED}✗ Failed{Colors.END}\n")

    # Summary
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}Summary:{Colors.END}")
    print(f"  {Colors.GREEN}✓ Successfully processed: {success_count}/{len(video_files)}{Colors.END}")
    if success_count < len(video_files):
        print(f"  {Colors.RED}✗ Failed: {len(video_files) - success_count}{Colors.END}")

    return 0 if success_count == len(video_files) else 1


if __name__ == '__main__':
    sys.exit(main())
