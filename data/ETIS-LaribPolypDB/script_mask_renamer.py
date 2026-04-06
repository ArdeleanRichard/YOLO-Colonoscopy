#!/usr/bin/env python3
"""
Rename p<number>.tif or p<number>.tiff -> <number>.tif/.tiff for all files in ./masks/
No dry-run, no prompts, will overwrite existing targets.
"""

from pathlib import Path
import re
import os
import sys

MASKS = Path("./masks")
PAT = re.compile(r"^p(\d+)(\.(tif|tiff))$", re.IGNORECASE)

if not MASKS.exists() or not MASKS.is_dir():
    print(f"Error: folder {MASKS} does not exist or is not a directory.", file=sys.stderr)
    sys.exit(1)

renamed = 0
for p in MASKS.iterdir():
    if not p.is_file():
        continue
    m = PAT.match(p.name)
    if not m:
        continue
    number = m.group(1)
    suffix = m.group(2)          # keeps the original extension (including dot and case)
    target = p.with_name(f"{number}{suffix}")
    try:
        # os.replace will overwrite target if it exists (atomic on most OSes)
        os.replace(p, target)
        print(f"Renamed: {p.name} -> {target.name}")
        renamed += 1
    except Exception as e:
        print(f"Error renaming {p.name}: {e}", file=sys.stderr)

print(f"Done. Renamed {renamed} file(s).")