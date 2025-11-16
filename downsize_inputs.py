"""
Helper script to downsize input images from 256x256 to 64x64 (So super resolution makes sense, we will upsample 4x to 256x256)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image

dataset_root = Path("Dataset")

splits = ["train", "validation", "test"] # Only the input data of all splits

for split in splits:
    d = dataset_root / split
    if not d.exists():
        print(f"{split}: MISSING -> {d}")
        continue
    exts = {".png", ".jpg", ".jpeg"}

    for i, p in enumerate(sorted(d.iterdir())):
        if not p.is_file() or p.suffix.lower() not in exts:
            print(f"FAILED to process: {str(p)}")
            continue

        img_in = cv2.imread(str(p))
        img_in = cv2.imread(str(p))
        if img_in is None:
            print(f"Failed to read: {p}")
            continue

        img_in_resized = cv2.resize(img_in, (64, 64), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(p), img_in_resized)

    print(f"{split}: Processed {i+1} images.")