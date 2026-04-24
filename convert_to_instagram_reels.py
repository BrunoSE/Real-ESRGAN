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
import shutil
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

# Default Real-ESRGAN model for live-action video upscaling
# RealESRGAN_x4plus: best quality for faces per research (GAN-trained)
DEFAULT_MODEL = 'RealESRGAN_x4plus'

# Always upscale at 4x and downscale with Lanczos for richer detail
# (produces better results than direct smaller-factor upscaling)
UPSCALE_FACTOR = 4.0


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

    # Always upscale at 4x then downscale with Lanczos (better detail than direct scaling)
    if strategy['needs_upscale']:
        strategy['upscale_factor'] = UPSCALE_FACTOR

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


def upscale_with_realesrgan(input_path, output_dir, scale_factor=4.0, model=DEFAULT_MODEL, tile_size=0):
    """Upscale video using Real-ESRGAN with calculated scale factor"""
    tile_msg = f"tile={tile_size}" if tile_size > 0 else "no tiling"
    print(f"  {Colors.BLUE}Upscaling with Real-ESRGAN ({model}, {scale_factor}x, {tile_msg})...{Colors.END}")

    cmd = [
        'python', 'inference_realesrgan_video.py',
        '-i', input_path,
        '-o', output_dir,  # Use output_dir directly, no dirname needed
        '-n', model,
        '-s', str(scale_factor),  # Use calculated scale factor
        '--fp32',  # Use full precision (better for Apple Silicon MPS)
        '--suffix', 'upscaled'
    ]

    # Only add tile argument if tiling is enabled
    if tile_size > 0:
        cmd.extend(['--tile', str(tile_size)])

    try:
        # Don't capture output - let Real-ESRGAN progress show in real-time
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"{Colors.RED}Upscaling failed (exit code: {result.returncode}){Colors.END}")
            return False
        return True
    except Exception as e:
        print(f"{Colors.RED}Upscaling error: {e}{Colors.END}")
        return False


def convert_with_ffmpeg(input_path, output_path, strategy, adjust_fps=True, audio=True, blur_bg=False):
    """
    Convert video to Instagram Reels specs using ffmpeg.

    Handles: resolution, aspect ratio, fps, codec conversion
    Uses Lanczos for high-quality downscaling.

    Args:
        blur_bg: If True, horizontal videos use blurred-background fill
                 instead of black letterbox bars
    """
    print(f"  {Colors.BLUE}Converting with ffmpeg...{Colors.END}")

    try:
        # Get current video dimensions and fps
        probe = ffmpeg.probe(input_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        current_width = int(video_stream['width'])
        current_height = int(video_stream['height'])

        # Get current FPS
        fps_str = video_stream.get('r_frame_rate', '30/1')
        fps_parts = fps_str.split('/')
        current_fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30

        # Detect horizontal vs vertical videos
        is_horizontal = current_width > current_height

        # Build filter chain - use complex filter for blur-bg, simple filter otherwise
        complex_filter = None
        filters = []

        if is_horizontal:
            # Horizontal video: scale by width
            scale_factor = INSTAGRAM_WIDTH / current_width
            scaled_width = INSTAGRAM_WIDTH
            scaled_height = int(current_height * scale_factor)

            if blur_bg:
                # Blurred-background fill: popular Reels style
                # Background: full-frame blurred and cropped to 1080x1920
                # Foreground: scaled video centered on top
                complex_filter = (
                    f"[0:v]split[fg][bg];"
                    f"[bg]scale={INSTAGRAM_WIDTH}:{INSTAGRAM_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
                    f"crop={INSTAGRAM_WIDTH}:{INSTAGRAM_HEIGHT},boxblur=20[bg];"
                    f"[fg]scale={scaled_width}:{scaled_height}:flags=lanczos[fg];"
                    f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
                )
                print(f"  {Colors.YELLOW}→ Horizontal video: scaling to {scaled_width}x{scaled_height} with blurred background{Colors.END}")
            else:
                # Black letterbox bars (default)
                scale_filter = f"scale={scaled_width}:{scaled_height}:flags=lanczos"
                filters.append(scale_filter)
                pad_filter = f"pad={INSTAGRAM_WIDTH}:{INSTAGRAM_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
                filters.append(pad_filter)
                print(f"  {Colors.YELLOW}→ Horizontal video: scaling to {scaled_width}x{scaled_height}, adding letterboxing{Colors.END}")
        else:
            # Vertical video: scale and crop
            width_scale = INSTAGRAM_WIDTH / current_width
            height_scale = INSTAGRAM_HEIGHT / current_height

            # Use the MAXIMUM scale factor to ensure both dimensions are covered
            scale_factor = max(width_scale, height_scale)
            scaled_width = int(current_width * scale_factor)
            scaled_height = int(current_height * scale_factor)

            # Scale with Lanczos for better quality
            scale_filter = f"scale={scaled_width}:{scaled_height}:flags=lanczos"
            filters.append(scale_filter)

            # Crop to exact size (removes any excess, centers the crop)
            crop_filter = f"crop={INSTAGRAM_WIDTH}:{INSTAGRAM_HEIGHT}"
            filters.append(crop_filter)

            print(f"  {Colors.YELLOW}→ Vertical video: scaling to {scaled_width}x{scaled_height}, cropping to fit{Colors.END}")

        # Determine fps filter
        fps_filter = None
        if adjust_fps:
            if current_fps < INSTAGRAM_FPS:
                fps_filter = f"fps={INSTAGRAM_FPS}"
                print(f"  {Colors.YELLOW}→ Upsampling fps from {current_fps:.1f} to {INSTAGRAM_FPS}{Colors.END}")
            elif current_fps > 60:
                fps_filter = "fps=60"
                print(f"  {Colors.YELLOW}→ Downsampling fps from {current_fps:.1f} to 60 (Instagram max){Colors.END}")
            else:
                print(f"  {Colors.GREEN}→ Keeping original fps: {current_fps:.1f}{Colors.END}")

        # Build ffmpeg command with research-recommended encoding settings
        # CRF 18 + preset slow + faststart + maxrate 12M preserves detail through
        # Instagram's re-encoding pipeline
        cmd = ['ffmpeg', '-y', '-i', input_path]

        if complex_filter is not None:
            # Complex filter chain (blur-bg uses filter_complex)
            if fps_filter:
                complex_filter = f"{complex_filter},{fps_filter}"
            cmd.extend(['-filter_complex', complex_filter])
        else:
            # Simple filter chain
            if fps_filter:
                filters.append(fps_filter)
            filter_chain = ','.join(filters)
            cmd.extend(['-vf', filter_chain])

        # Video encoding - research-recommended settings for Instagram Reels
        cmd.extend([
            '-c:v', 'libx264',
            '-profile:v', 'main',
            '-level:v', '4.0',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-maxrate', '12M',
            '-bufsize', '20M',
            '-preset', 'slow',
            '-movflags', '+faststart',
        ])

        # Audio encoding
        if audio:
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '48000',
            ])
        else:
            cmd.append('-an')

        cmd.append(output_path)

        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{Colors.RED}FFmpeg error: {result.stderr[-500:]}{Colors.END}")
            return False
        return True

    except ffmpeg.Error as e:
        print(f"{Colors.RED}FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Conversion error: {e}{Colors.END}")
        return False


def process_video(input_path, output_dir, use_upscaling=True, model=DEFAULT_MODEL, tile_size=0, blur_bg=False):
    """
    Process a single video to Instagram Reels specs.

    Args:
        input_path: Path to input video
        output_dir: Directory for output video
        use_upscaling: Whether to use Real-ESRGAN for upscaling
        model: Real-ESRGAN model to use
        tile_size: Tile size for Real-ESRGAN processing (memory management)
        blur_bg: For horizontal videos, use blurred-background fill instead of black bars
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
        if not upscale_with_realesrgan(input_path, temp_dir, strategy['upscale_factor'], model, tile_size):
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
        audio=info['has_audio'],
        blur_bg=blur_bg
    )

    # Cleanup temp directory
    temp_dir = os.path.join(output_dir, '.temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

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
  # Convert all videos in inputs folder (default: RealESRGAN_x4plus, black bars for horizontal)
  uv run python convert_to_instagram_reels.py inputs/*.mp4 -o instagram_ready

  # Use blurred-background fill for horizontal videos (popular Reels style)
  uv run python convert_to_instagram_reels.py inputs/*.mp4 -o output --blur-bg

  # Convert without AI upscaling (faster)
  uv run python convert_to_instagram_reels.py inputs/*.mp4 -o output --no-upscale

  # Use general model instead (compact, faster)
  uv run python convert_to_instagram_reels.py inputs/*.mp4 -o output -n realesr-general-x4v3

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
        default=DEFAULT_MODEL,
        choices=['RealESRGAN_x4plus', 'realesr-general-x4v3', 'realesr-animevideov3', 'RealESRGAN_x2plus'],
        help=f'Real-ESRGAN model for upscaling (default: {DEFAULT_MODEL} - best for faces per research)'
    )
    parser.add_argument(
        '--tile',
        type=int,
        default=0,
        help='Tile size for Real-ESRGAN (default: 0=no tiling for best quality, use 512+ if OOM errors occur)'
    )
    parser.add_argument(
        '--blur-bg',
        action='store_true',
        help='For horizontal videos, use blurred-background fill instead of black letterbox bars'
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
        tile_info = f"no tiling (best quality)" if args.tile == 0 else f"tile={args.tile}"
        print(f"Settings: {tile_info}, fp32 precision, always 4x (downscale with Lanczos)")
    print(f"Horizontal videos: {'blurred-background fill' if args.blur_bg else 'black letterbox bars'}")

    # Process each video
    success_count = 0
    for video_file in video_files:
        if process_video(
            video_file,
            args.output,
            use_upscaling=not args.no_upscale,
            model=args.model_name,
            tile_size=args.tile,
            blur_bg=args.blur_bg,
        ):
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
