#!/usr/bin/env python3
"""
Quick benchmark for the two models that had errors.
Tests: realesr-general-wdn-x4v3 and 4x_NMKD-Superscale-SP_178000_G
"""

import glob
import os
import shutil
import subprocess
import sys
import time

# Models to test
MODELS = [
    'realesr-general-wdn-x4v3',
    '4x_NMKD-Superscale-SP_178000_G'
]

SCALE = 2.7
TILE = 0
INPUT_PATTERN = 'inputs/video/*.mp4'
OUTPUT_DIR = 'results'


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def run_model(video_path, model_name):
    """Run single model on video"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{model_name}_{video_name}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\n{Colors.CYAN}Testing: {model_name}{Colors.END}")
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  Scale: {SCALE}x, Tile: {TILE if TILE > 0 else 'None'}")

    cmd = [
        'python', 'inference_realesrgan_video.py',
        '-i', video_path,
        '-o', OUTPUT_DIR,
        '-n', model_name,
        '-s', str(SCALE),
        '--fp32',
        '--suffix', f'{model_name}_out'
    ]

    if TILE > 0:
        cmd.extend(['--tile', str(TILE)])

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    # Rename output to match expected name
    temp_output = os.path.join(OUTPUT_DIR, f'{video_name}_{model_name}_out.mp4')
    if os.path.exists(temp_output):
        shutil.move(temp_output, output_path)

    if result.returncode == 0:
        print(f"{Colors.GREEN}✓ Success in {elapsed:.1f}s{Colors.END}")
        return True, elapsed
    else:
        print(f"{Colors.RED}✗ Failed (exit code: {result.returncode}){Colors.END}")
        return False, elapsed


def main():
    # Find videos
    videos = glob.glob(INPUT_PATTERN)
    if not videos:
        print(f"{Colors.RED}No videos found matching {INPUT_PATTERN}{Colors.END}")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}QUICK BENCHMARK - Two Models{Colors.END}")
    print(f"{'='*70}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Videos: {len(videos)}")
    print(f"Scale: {SCALE}x")
    print(f"Tile: {TILE if TILE > 0 else 'None'}")
    print(f"{'='*70}\n")

    results = {}
    total = len(MODELS) * len(videos)
    current = 0

    for video in videos:
        for model in MODELS:
            current += 1
            print(f"\n{Colors.BOLD}[{current}/{total}]{Colors.END}")
            success, elapsed = run_model(video, model)
            results[(model, os.path.basename(video))] = (success, elapsed)

    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print(f"{'='*70}\n")

    for model in MODELS:
        model_results = [(k, v) for k, v in results.items() if k[0] == model]
        successes = sum(1 for _, (s, _) in model_results if s)
        total_time = sum(t for _, (_, t) in model_results)

        print(f"{Colors.CYAN}{model}{Colors.END}")
        print(f"  Success: {successes}/{len(model_results)}")
        print(f"  Total time: {total_time:.1f}s")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
