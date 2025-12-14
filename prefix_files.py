#!/usr/bin/env python3
"""Prefix filenames with sequential numbers based on creation/modification date."""

import os
import sys
from pathlib import Path


def prefix_files(folder_path: str, start_num: int = 100):
    folder = Path(folder_path)
    
    if not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    # Get all files (not directories) and sort by modification time
    files = [f for f in folder.iterdir() if f.is_file()]
    files.sort(key=lambda f: f.stat().st_mtime)
    
    # Rename with prefix
    for i, file in enumerate(files, start=start_num):
        new_name = f"{i}{Path(file.name).suffix}"
        new_path = file.parent / new_name
        file.rename(new_path)
        print(f"{file.name} -> {new_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prefix_files.py <folder_path> [start_number]")
        sys.exit(1)
    
    folder = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    prefix_files(folder, start)