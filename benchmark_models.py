#!/usr/bin/env python3
"""
Benchmark Real-ESRGAN models for video upscaling.

This script runs multiple Real-ESRGAN models on the same video(s)
to compare quality and performance.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import ffmpeg
except ImportError:
    print("Error: ffmpeg-python is not installed")
    print("Install it with: uv pip install ffmpeg-python")
    sys.exit(1)


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def get_model_files(weights_dir='weights'):
    """Find all .pth model files in weights directory"""
    pattern = os.path.join(weights_dir, '*.pth')
    models = glob.glob(pattern)
    return sorted(models)


def get_model_name_from_path(model_path):
    """Extract model name from .pth file path"""
    basename = os.path.basename(model_path)
    # Remove .pth extension
    return os.path.splitext(basename)[0]


def get_video_info(video_path):
    """Get video resolution and fps"""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps_str = video_stream.get('r_frame_rate', '30/1')
        fps_parts = fps_str.split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30
        return width, height, fps
    except:
        return None, None, None


def upscale_video(input_path, output_path, model_path, scale=2.0, tile_size=0):
    """
    Upscale video using Real-ESRGAN with custom model.

    Returns:
        tuple: (success: bool, elapsed_time: float)
    """
    model_name = get_model_name_from_path(model_path)

    print(f"  {Colors.BLUE}Running model: {model_name}{Colors.END}")
    print(f"  Scale: {scale}x, Tile: {tile_size if tile_size > 0 else 'None'}")

    # Get video name without extension for temp output
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    temp_dir = os.path.join(os.path.dirname(output_path), '.temp')
    os.makedirs(temp_dir, exist_ok=True)

    cmd = [
        'python', 'inference_realesrgan_video.py',
        '-i', input_path,
        '-o', temp_dir,
        '-n', model_name,  # Use model name (without .pth)
        '-s', str(scale),
        '--fp32',
        '--suffix', 'out'
    ]

    if tile_size > 0:
        cmd.extend(['--tile', str(tile_size)])

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"{Colors.RED}✗ Failed (exit code: {result.returncode}){Colors.END}")
            return False, elapsed

        # Find and move the output file
        temp_output = os.path.join(temp_dir, f'{video_name}_out.mp4')
        if os.path.exists(temp_output):
            shutil.move(temp_output, output_path)
            print(f"{Colors.GREEN}✓ Completed in {elapsed:.1f}s{Colors.END}")
            return True, elapsed
        else:
            print(f"{Colors.RED}✗ Output file not found{Colors.END}")
            return False, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")
        return False, elapsed
    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def benchmark_models(video_files, model_files, output_dir, scale=2.0, tile_size=0):
    """
    Benchmark all models on all videos.

    Returns:
        dict: Results keyed by (model_name, video_name)
    """
    results = {}
    total_runs = len(video_files) * len(model_files)
    current_run = 0

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        width, height, fps = get_video_info(video_path)

        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}Video: {video_name}{Colors.END}")
        print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}")

        for model_path in model_files:
            current_run += 1
            model_name = get_model_name_from_path(model_path)

            print(f"\n{Colors.CYAN}[{current_run}/{total_runs}] Model: {model_name}{Colors.END}")

            # Create output filename with model prefix
            output_filename = f"{model_name}_{video_name}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            # Run benchmark
            success, elapsed = upscale_video(
                video_path,
                output_path,
                model_path,
                scale=scale,
                tile_size=tile_size
            )

            # Store results
            results[(model_name, video_name)] = {
                'success': success,
                'time': elapsed,
                'output': output_path if success else None,
                'input_resolution': (width, height),
                'input_fps': fps
            }

    return results


def print_summary(results, video_files, model_files):
    """Print benchmark summary table"""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}BENCHMARK SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")

    # Get unique video and model names
    video_names = [os.path.splitext(os.path.basename(v))[0] for v in video_files]
    model_names = [get_model_name_from_path(m) for m in model_files]

    # Print results per video
    for video_name in video_names:
        print(f"\n{Colors.BOLD}Video: {video_name}{Colors.END}")
        print(f"{Colors.BOLD}{'-'*70}{Colors.END}")

        # Table header
        print(f"{'Model':<40} {'Status':<12} {'Time (s)':<12}")
        print(f"{'-'*70}")

        # Results for each model
        for model_name in model_names:
            key = (model_name, video_name)
            if key in results:
                result = results[key]
                status = f"{Colors.GREEN}✓ Success{Colors.END}" if result['success'] else f"{Colors.RED}✗ Failed{Colors.END}"
                time_str = f"{result['time']:.1f}s"

                # Truncate long model names
                display_name = model_name[:37] + '...' if len(model_name) > 40 else model_name
                print(f"{display_name:<40} {status:<20} {time_str:<12}")

    # Overall statistics
    print(f"\n{Colors.BOLD}OVERALL STATISTICS{Colors.END}")
    print(f"{Colors.BOLD}{'-'*70}{Colors.END}")

    for model_name in model_names:
        model_results = [r for k, r in results.items() if k[0] == model_name]

        total = len(model_results)
        successful = sum(1 for r in model_results if r['success'])
        total_time = sum(r['time'] for r in model_results)
        avg_time = total_time / total if total > 0 else 0

        print(f"\n{Colors.CYAN}{model_name}{Colors.END}")
        print(f"  Success Rate: {successful}/{total} ({100*successful/total:.0f}%)")
        print(f"  Total Time: {total_time:.1f}s")
        print(f"  Avg Time per Video: {avg_time:.1f}s")

    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"\n{Colors.YELLOW}TIP: Compare output quality by opening videos side-by-side{Colors.END}")
    print(f"{Colors.YELLOW}     Videos are saved with model name prefix in: {os.path.abspath('results')}{Colors.END}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Real-ESRGAN models on videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all models in weights/ on all videos in inputs/
  uv run python benchmark_models.py

  # Benchmark with custom scale factor
  uv run python benchmark_models.py --scale 2.5

  # Benchmark with tiling (for memory management)
  uv run python benchmark_models.py --tile 512

  # Benchmark specific videos
  uv run python benchmark_models.py --input "inputs/video1.mp4" "inputs/video2.mp4"

  # Use custom weights directory
  uv run python benchmark_models.py --weights-dir /path/to/models
        """
    )

    parser.add_argument(
        '--input',
        nargs='+',
        default=['inputs/*.mp4'],
        help='Input video file(s) or pattern (default: inputs/*.mp4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    parser.add_argument(
        '--weights-dir',
        type=str,
        default='weights',
        help='Directory containing .pth model files (default: weights)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=2.0,
        help='Upscale factor (default: 2.0)'
    )
    parser.add_argument(
        '--tile',
        type=int,
        default=0,
        help='Tile size (default: 0 = no tiling)'
    )

    args = parser.parse_args()

    # Find model files
    model_files = get_model_files(args.weights_dir)
    if not model_files:
        print(f"{Colors.RED}Error: No .pth model files found in {args.weights_dir}/{Colors.END}")
        return 1

    # Expand video file patterns
    video_files = []
    for pattern in args.input:
        if '*' in pattern or '?' in pattern:
            video_files.extend(glob.glob(pattern))
        else:
            video_files.append(pattern)

    video_files = [f for f in video_files if os.path.isfile(f)]

    if not video_files:
        print(f"{Colors.RED}Error: No video files found{Colors.END}")
        return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Print benchmark configuration
    print(f"{Colors.BOLD}BENCHMARK CONFIGURATION{Colors.END}")
    print(f"{'='*70}")
    print(f"Models to test: {len(model_files)}")
    for model in model_files:
        print(f"  • {get_model_name_from_path(model)}")
    print(f"\nVideos to process: {len(video_files)}")
    for video in video_files:
        print(f"  • {os.path.basename(video)}")
    print(f"\nScale factor: {args.scale}x")
    print(f"Tile size: {args.tile if args.tile > 0 else 'None (best quality)'}")
    print(f"Output directory: {args.output}")
    print(f"Total runs: {len(model_files) * len(video_files)}")
    print(f"{'='*70}\n")

    # Run benchmark
    results = benchmark_models(
        video_files,
        model_files,
        args.output,
        scale=args.scale,
        tile_size=args.tile
    )

    # Print summary
    print_summary(results, video_files, model_files)

    return 0


if __name__ == '__main__':
    sys.exit(main())
