#!/usr/bin/env python3
"""
Convert videos to Instagram Reels specifications.

This script processes videos to meet Instagram Reels requirements:
- Resolution: 1080 x 1920 pixels (vertical)
- Aspect Ratio: 9:16 (full screen)
- Frame Rate: 30 fps
- File Format: MP4
- Codec: H.264

Features:
- Upscales low-resolution videos using Real-ESRGAN (optional)
- Downscales high-resolution videos using ffmpeg
- Converts aspect ratios (crops or pads as needed)
- Adjusts frame rate to 30 fps
- Re-encodes to H.264 in MP4 container
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

try:
    import ffmpeg
except ImportError:
    print("Error: ffmpeg-python is not installed")
    print("Install it with: uv pip install ffmpeg-python")
    sys.exit(1)


# Instagram Reels specifications
INSTAGRAM_WIDTH = 1080
INSTAGRAM_HEIGHT = 1920
INSTAGRAM_FPS = 30
INSTAGRAM_ASPECT = 9 / 16  # width/height


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def get_video_info(video_path):
    """Extract video metadata"""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')

        width = int(video_stream['width'])
        height = int(video_stream['height'])

        # Get FPS
        fps_str = video_stream.get('r_frame_rate', '30/1')
        fps_parts = fps_str.split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30

        # Check for audio
        has_audio = any(s['codec_type'] == 'audio' for s in probe['streams'])

        return {
            'width': width,
            'height': height,
            'fps': fps,
            'has_audio': has_audio
        }
    except Exception as e:
        print(f"{Colors.RED}Error analyzing {video_path}: {e}{Colors.END}")
        return None


def needs_upscaling(width, height):
    """Check if video needs AI upscaling"""
    return width < INSTAGRAM_WIDTH or height < INSTAGRAM_HEIGHT


def calculate_scaling_strategy(width, height):
    """
    Determine how to scale the video to Instagram specs.

    Returns:
        dict: Scaling strategy including method and parameters
    """
    current_aspect = width / height
    target_aspect = INSTAGRAM_ASPECT

    strategy = {
        'needs_upscale': needs_upscaling(width, height),
        'needs_downscale': width > INSTAGRAM_WIDTH or height > INSTAGRAM_HEIGHT,
        'needs_crop': False,
        'needs_pad': False,
        'target_width': INSTAGRAM_WIDTH,
        'target_height': INSTAGRAM_HEIGHT,
        'upscale_factor': 1.0
    }

    # Calculate optimal upscale factor if upscaling is needed
    if strategy['needs_upscale']:
        # Calculate scale factor based on the dimension that needs more scaling
        width_scale = INSTAGRAM_WIDTH / width
        height_scale = INSTAGRAM_HEIGHT / height
        # Use the larger scale to ensure we meet minimum dimensions
        strategy['upscale_factor'] = max(width_scale, height_scale)
        # Cap at 4x (Real-ESRGAN maximum)
        strategy['upscale_factor'] = min(strategy['upscale_factor'], 4.0)
        # Round to 1 decimal place for practical purposes
        strategy['upscale_factor'] = round(strategy['upscale_factor'], 1)

    # Check if aspect ratio matches
    if abs(current_aspect - target_aspect) > 0.01:
        # Aspect ratio is different - need to crop or pad
        if current_aspect > target_aspect:
            # Video is wider - need to crop width or pad height
            strategy['needs_crop'] = True
            strategy['crop_direction'] = 'horizontal'
        else:
            # Video is taller - need to crop height or pad width
            strategy['needs_crop'] = True
            strategy['crop_direction'] = 'vertical'

    return strategy


def upscale_with_realesrgan(input_path, output_path, scale_factor=4.0, model='realesr-general-x4v3'):
    """Upscale video using Real-ESRGAN with calculated scale factor"""
    print(f"  {Colors.BLUE}Upscaling with Real-ESRGAN ({model}, {scale_factor}x)...{Colors.END}")

    cmd = [
        'python', 'inference_realesrgan_video.py',
        '-i', input_path,
        '-o', os.path.dirname(output_path),
        '-n', model,
        '-s', str(scale_factor),  # Use calculated scale factor
        '--suffix', 'upscaled'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{Colors.RED}Upscaling failed: {result.stderr}{Colors.END}")
            return False
        return True
    except Exception as e:
        print(f"{Colors.RED}Upscaling error: {e}{Colors.END}")
        return False


def convert_with_ffmpeg(input_path, output_path, strategy, adjust_fps=True, audio=True):
    """
    Convert video to Instagram Reels specs using ffmpeg.

    Handles: resolution, aspect ratio, fps, codec conversion
    """
    print(f"  {Colors.BLUE}Converting with ffmpeg...{Colors.END}")

    try:
        # Build ffmpeg filter chain
        filters = []

        # Scale to target resolution while maintaining aspect ratio
        # We'll use scale and crop/pad to get exact 1080x1920

        # First, scale to fit within bounds
        scale_filter = f"scale='min({INSTAGRAM_WIDTH},iw)':'min({INSTAGRAM_HEIGHT},ih)':force_original_aspect_ratio=decrease"
        filters.append(scale_filter)

        # Then pad to exact size (centers the video)
        pad_filter = f"pad={INSTAGRAM_WIDTH}:{INSTAGRAM_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
        filters.append(pad_filter)

        # Adjust FPS if needed
        if adjust_fps:
            filters.append(f"fps={INSTAGRAM_FPS}")

        filter_chain = ','.join(filters)

        # Build ffmpeg command
        input_stream = ffmpeg.input(input_path)

        output_kwargs = {
            'vcodec': 'libx264',
            'preset': 'medium',
            'crf': 23,
            'pix_fmt': 'yuv420p',
            'vf': filter_chain
        }

        # Add audio if present
        if audio:
            output_kwargs['acodec'] = 'aac'
            output_kwargs['audio_bitrate'] = '128k'

        output_stream = ffmpeg.output(input_stream, output_path, **output_kwargs)

        # Run ffmpeg
        ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
        return True

    except ffmpeg.Error as e:
        print(f"{Colors.RED}FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Conversion error: {e}{Colors.END}")
        return False


def process_video(input_path, output_dir, use_upscaling=True, model='realesr-general-x4v3'):
    """
    Process a single video to Instagram Reels specs.

    Args:
        input_path: Path to input video
        output_dir: Directory for output video
        use_upscaling: Whether to use Real-ESRGAN for upscaling
        model: Real-ESRGAN model to use
    """
    filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{name_without_ext}_instagram.mp4")

    print(f"\n{Colors.BOLD}Processing: {filename}{Colors.END}")

    # Get video info
    info = get_video_info(input_path)
    if not info:
        return False

    print(f"  Current: {info['width']}x{info['height']}, {info['fps']:.1f} fps")
    print(f"  Target:  {INSTAGRAM_WIDTH}x{INSTAGRAM_HEIGHT}, {INSTAGRAM_FPS} fps")

    # Determine strategy
    strategy = calculate_scaling_strategy(info['width'], info['height'])

    # Process based on strategy
    temp_upscaled = None

    if strategy['needs_upscale'] and use_upscaling:
        # Need AI upscaling
        print(f"  {Colors.YELLOW}→ AI Upscaling required (scale: {strategy['upscale_factor']}x){Colors.END}")

        temp_dir = os.path.join(output_dir, '.temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Upscale with Real-ESRGAN using calculated scale factor
        if not upscale_with_realesrgan(input_path, temp_dir, strategy['upscale_factor'], model):
            print(f"{Colors.RED}✗ Upscaling failed, falling back to ffmpeg{Colors.END}")
            # Fall through to ffmpeg conversion
        else:
            # Find the upscaled file
            temp_upscaled = os.path.join(temp_dir, f"{name_without_ext}_upscaled.mp4")
            if os.path.exists(temp_upscaled):
                input_path = temp_upscaled
                print(f"{Colors.GREEN}✓ Upscaling complete{Colors.END}")

    elif strategy['needs_downscale']:
        print(f"  {Colors.YELLOW}→ Downscaling required{Colors.END}")

    # Final conversion with ffmpeg (handles resolution, aspect ratio, fps, codec)
    success = convert_with_ffmpeg(
        input_path,
        output_path,
        strategy,
        adjust_fps=True,
        audio=info['has_audio']
    )

    # Cleanup temp files
    if temp_upscaled and os.path.exists(temp_upscaled):
        os.remove(temp_upscaled)

    if success:
        print(f"{Colors.GREEN}✓ Saved: {output_path}{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}✗ Conversion failed{Colors.END}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert videos to Instagram Reels specifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Instagram Reels Target Specs:
  Resolution:    1080 x 1920 pixels (vertical)
  Aspect Ratio:  9:16 (full screen)
  Frame Rate:    30 fps
  File Format:   MP4
  Codec:         H.264

Examples:
  # Convert all videos in inputs folder
  uv run python convert_to_instagram_reels.py inputs/*.mp4 -o instagram_ready

  # Convert without AI upscaling (faster)
  uv run python convert_to_instagram_reels.py inputs/*.mp4 -o output --no-upscale

  # Use anime model for upscaling
  uv run python convert_to_instagram_reels.py inputs/*.mp4 -o output -n realesr-animevideov3

  # Convert single file
  uv run python convert_to_instagram_reels.py video.mp4 -o output
        """
    )

    parser.add_argument(
        'inputs',
        nargs='+',
        help='Input video file(s) or pattern (e.g., inputs/*.mp4)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='instagram_ready',
        help='Output directory (default: instagram_ready)'
    )
    parser.add_argument(
        '--no-upscale',
        action='store_true',
        help='Disable AI upscaling (use ffmpeg only, faster but lower quality)'
    )
    parser.add_argument(
        '-n', '--model_name',
        type=str,
        default='realesr-general-x4v3',
        choices=['realesr-general-x4v3', 'realesr-animevideov3', 'RealESRGAN_x4plus', 'RealESRGAN_x2plus'],
        help='Real-ESRGAN model for upscaling (default: realesr-general-x4v3 for real videos)'
    )

    args = parser.parse_args()

    # Expand wildcards
    video_files = []
    for pattern in args.inputs:
        if '*' in pattern or '?' in pattern:
            video_files.extend(glob.glob(pattern))
        else:
            video_files.append(pattern)

    # Filter to existing files
    video_files = [f for f in video_files if os.path.isfile(f)]

    if not video_files:
        print(f"{Colors.RED}Error: No video files found{Colors.END}")
        return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"{Colors.BOLD}Converting {len(video_files)} video(s) to Instagram Reels format{Colors.END}")
    print(f"Output directory: {args.output}")
    print(f"AI Upscaling: {'Enabled' if not args.no_upscale else 'Disabled'}")
    if not args.no_upscale:
        print(f"Upscale Model: {args.model_name}")

    # Process each video
    success_count = 0
    for video_file in video_files:
        if process_video(video_file, args.output, use_upscaling=not args.no_upscale, model=args.model_name):
            success_count += 1

    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}Summary:{Colors.END}")
    print(f"  {Colors.GREEN}✓ Successfully converted: {success_count}/{len(video_files)}{Colors.END}")
    if success_count < len(video_files):
        print(f"  {Colors.RED}✗ Failed: {len(video_files) - success_count}{Colors.END}")

    return 0 if success_count == len(video_files) else 1


if __name__ == '__main__':
    sys.exit(main())
