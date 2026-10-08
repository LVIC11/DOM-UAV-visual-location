#!/usr/bin/env python3
"""Rename images in a directory to their timestamps from metadata.csv.

Usage:
  python rename_by_timestamp.py /path/to/query
"""

import csv
import os
import sys

img_dir = sys.argv[1] if len(sys.argv) > 1 else "."
csv_path = os.path.join(img_dir, "metadata.csv")

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        old_name = row['filename']
        ts = float(row['timestamp'])
        new_name = f"{ts:.6f}.jpg"

        old_path = os.path.join(img_dir, old_name)
        new_path = os.path.join(img_dir, new_name)

        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"  {old_name} → {new_name}")
        else:
            print(f"  SKIP: {old_name} not found")

print("Done.")
