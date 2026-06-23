#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split a YOLO dataset into train/val folders."""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split YOLO dataset train images into val.")
    parser.add_argument("--root", default="dataset", help="Dataset root.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    train_images = root / "images" / "train"
    train_labels = root / "labels" / "train"
    val_images = root / "images" / "val"
    val_labels = root / "labels" / "val"
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    images = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(train_images.glob(suffix))
    images = sorted(images)
    random.Random(args.seed).shuffle(images)

    n_val = int(len(images) * args.val_ratio)
    mover = shutil.copy2 if args.copy else shutil.move
    for image in images[:n_val]:
        mover(str(image), str(val_images / image.name))
        label = train_labels / f"{image.stem}.txt"
        if label.exists():
            mover(str(label), str(val_labels / label.name))

    action = "copied" if args.copy else "moved"
    print(f"{action} {n_val} image(s) to {val_images}")


if __name__ == "__main__":
    main()
