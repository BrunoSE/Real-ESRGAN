#!/bin/bash

for number in {105..138}; do
    echo "Processing video $number..."
    uv run python inference_realesrgan_video.py -i inputs/video/${number}.mp4 -o results -n realesr-general-x4v3 
    echo "Sleeping for 10 seconds..."
    sleep 10
done

echo "All videos processed!"