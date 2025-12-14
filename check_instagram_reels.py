#!/usr/bin/env python3
"""
Check if videos comply with Instagram Reels specifications.

Instagram Reels Requirements:
- Resolution: 1080 x 1920 pixels (vertical)
- Aspect Ratio: 9:16 (full screen)
- Frame Rate: 30 fps is ideal
- File Format: MP4 or MOV
- Codec: H.264 or H.265 (HEVC)
"""

import argparse
import glob
import os
import sys
from pathlib import Path

try:
    import ffmpeg
except ImportError:
    print("Error: ffmpeg-python is not installed")
    print("Install it with: uv pip install ffmpeg-python")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def check_video_compliance(video_path):
    """
    Check if a video meets Instagram Reels specifications.

    Returns:
        dict: Compliance information and recommendations
    """
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        return {
            'valid': False,
            'error': f"Could not analyze video: {str(e)}"
        }

    # Get video stream info
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    if not video_streams:
        return {
            'valid': False,
            'error': "No video stream found"
        }

    video_info = video_streams[0]

    # Extract video properties
    width = int(video_info.get('width', 0))
    height = int(video_info.get('height', 0))
    codec_name = video_info.get('codec_name', '').lower()

    # Get frame rate
    fps_str = video_info.get('r_frame_rate', '0/1')
    fps_parts = fps_str.split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 0

    # Get file extension
    file_ext = os.path.splitext(video_path)[1].lower()

    # Check each requirement
    results = {
        'file': video_path,
        'width': width,
        'height': height,
        'fps': round(fps, 2),
        'codec': codec_name,
        'format': file_ext,
        'checks': {},
        'compliant': True,
        'warnings': [],
        'errors': []
    }

    # 1. Check Resolution (1080 x 1920)
    if width == 1080 and height == 1920:
        results['checks']['resolution'] = '✓ Perfect'
    else:
        results['checks']['resolution'] = f'✗ {width}x{height} (should be 1080x1920)'
        results['errors'].append(f"Resolution is {width}x{height}, should be 1080x1920")
        results['compliant'] = False

    # 2. Check Aspect Ratio (9:16)
    aspect_ratio = width / height if height > 0 else 0
    expected_ratio = 9 / 16
    if abs(aspect_ratio - expected_ratio) < 0.01:
        results['checks']['aspect_ratio'] = '✓ 9:16'
    else:
        current_ratio = f"{width}:{height}"
        results['checks']['aspect_ratio'] = f'✗ {current_ratio} (should be 9:16)'
        results['warnings'].append(f"Aspect ratio is {current_ratio}, Instagram prefers 9:16")
        results['compliant'] = False

    # 3. Check Frame Rate (30 fps ideal)
    if abs(fps - 30) < 0.1:
        results['checks']['fps'] = '✓ 30 fps'
    elif 23 <= fps <= 60:
        results['checks']['fps'] = f'⚠ {fps} fps (30 fps recommended)'
        results['warnings'].append(f"Frame rate is {fps} fps, 30 fps is ideal to avoid compression")
    else:
        results['checks']['fps'] = f'✗ {fps} fps (should be ~30 fps)'
        results['errors'].append(f"Frame rate is {fps} fps, should be around 30 fps")
        results['compliant'] = False

    # 4. Check File Format (MP4 or MOV)
    if file_ext in ['.mp4', '.mov']:
        results['checks']['format'] = f'✓ {file_ext.upper()}'
    else:
        results['checks']['format'] = f'✗ {file_ext.upper()} (should be MP4 or MOV)'
        results['errors'].append(f"Format is {file_ext}, should be MP4 or MOV")
        results['compliant'] = False

    # 5. Check Codec (H.264 or H.265)
    if codec_name in ['h264', 'hevc', 'h265']:
        codec_display = 'H.264' if codec_name == 'h264' else 'H.265'
        results['checks']['codec'] = f'✓ {codec_display}'
    else:
        results['checks']['codec'] = f'✗ {codec_name.upper()} (should be H.264 or H.265)'
        results['errors'].append(f"Codec is {codec_name}, should be H.264 or H.265")
        results['compliant'] = False

    return results


def print_video_report(results):
    """Print a formatted report for a single video"""
    filename = os.path.basename(results['file'])

    # Print header
    if results.get('error'):
        print(f"\n{Colors.RED}✗ {filename}{Colors.END}")
        print(f"  {results['error']}")
        return

    # Print filename with compliance status
    if results['compliant']:
        print(f"\n{Colors.GREEN}✓ {filename}{Colors.END}")
    elif results['warnings'] and not results['errors']:
        print(f"\n{Colors.YELLOW}⚠ {filename}{Colors.END}")
    else:
        print(f"\n{Colors.RED}✗ {filename}{Colors.END}")

    # Print specs
    print(f"  Resolution: {results['width']}x{results['height']}")
    print(f"  Frame Rate: {results['fps']} fps")
    print(f"  Codec: {results['codec'].upper()}")
    print(f"  Format: {results['format'].upper()}")

    # Print check results
    print(f"\n  {Colors.BOLD}Instagram Reels Compliance:{Colors.END}")
    for check, status in results['checks'].items():
        if '✓' in status:
            print(f"    {Colors.GREEN}{status}{Colors.END} - {check.replace('_', ' ').title()}")
        elif '⚠' in status:
            print(f"    {Colors.YELLOW}{status}{Colors.END} - {check.replace('_', ' ').title()}")
        else:
            print(f"    {Colors.RED}{status}{Colors.END} - {check.replace('_', ' ').title()}")

    # Print warnings
    if results['warnings']:
        print(f"\n  {Colors.YELLOW}Warnings:{Colors.END}")
        for warning in results['warnings']:
            print(f"    • {warning}")

    # Print errors
    if results['errors']:
        print(f"\n  {Colors.RED}Issues:{Colors.END}")
        for error in results['errors']:
            print(f"    • {error}")


def main():
    parser = argparse.ArgumentParser(
        description='Check if videos comply with Instagram Reels specifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Instagram Reels Specifications:
  Resolution:    1080 x 1920 pixels (vertical)
  Aspect Ratio:  9:16 (full screen)
  Frame Rate:    30 fps (ideal to avoid compression)
  File Format:   MP4 or MOV
  Codec:         H.264 or H.265 (HEVC)

Examples:
  # Check a single video
  uv run python check_instagram_reels.py video.mp4

  # Check all videos in a folder
  uv run python check_instagram_reels.py inputs/*.mp4

  # Check with summary
  uv run python check_instagram_reels.py inputs/*.mp4 --summary
        """
    )

    parser.add_argument(
        'videos',
        nargs='+',
        help='Video file(s) to check (supports wildcards)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary statistics at the end'
    )

    args = parser.parse_args()

    # Expand wildcards
    video_files = []
    for pattern in args.videos:
        if '*' in pattern or '?' in pattern:
            video_files.extend(glob.glob(pattern))
        else:
            video_files.append(pattern)

    # Remove duplicates and filter existing files
    video_files = [f for f in set(video_files) if os.path.isfile(f)]

    if not video_files:
        print(f"{Colors.RED}Error: No video files found{Colors.END}")
        return 1

    # Check each video
    results_list = []
    for video_file in sorted(video_files):
        results = check_video_compliance(video_file)
        results_list.append(results)
        print_video_report(results)

    # Print summary
    if args.summary and len(results_list) > 1:
        compliant = sum(1 for r in results_list if r.get('compliant', False))
        warnings_only = sum(1 for r in results_list if r.get('warnings') and not r.get('errors'))
        non_compliant = sum(1 for r in results_list if r.get('errors'))

        print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"  Total videos checked: {len(results_list)}")
        print(f"  {Colors.GREEN}✓ Fully compliant: {compliant}{Colors.END}")
        print(f"  {Colors.YELLOW}⚠ Minor warnings: {warnings_only}{Colors.END}")
        print(f"  {Colors.RED}✗ Non-compliant: {non_compliant}{Colors.END}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
