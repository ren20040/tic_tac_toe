#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a YOLO detector for tic-tac-toe pieces."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO piece detector.")
    parser.add_argument("--data", default="data.yaml", help="YOLO dataset yaml.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu, 0, 0,1, etc.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="tic_tac_toe_yolo")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    model = YOLO(args.model)
    kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "workers": args.workers,
        "patience": args.patience,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    results = model.train(**kwargs)
    save_dir = getattr(results, "save_dir", None) or getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is None:
        print("training finished")
        return
    print(f"training finished: {save_dir}")
    print(f"best weights: {Path(save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
