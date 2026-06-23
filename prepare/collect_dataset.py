#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect image frames for YOLO tic-tac-toe piece training.

Controls:
  s: save current frame
  a: toggle auto-save
  q/esc: quit
"""

import argparse
import time
from pathlib import Path

import cv2
from utils.config_loader import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Collect YOLO training images.")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml.")
    parser.add_argument("--source", default=None, help="Camera index, video path, image topic, or ROS topic.")
    parser.add_argument("--out", default=None, help="Output image directory.")
    parser.add_argument("--prefix", default="", help="Optional filename prefix.")
    parser.add_argument("--auto-save", action="store_true", help="Enable interval/change based saving.")
    parser.add_argument("--interval", type=float, default=None, help="Auto-save interval in seconds.")
    parser.add_argument("--threshold", type=float, default=None, help="Mean pixel difference threshold.")
    parser.add_argument("--no-window", action="store_true", help="Do not show preview window.")
    parser.add_argument("--ros", action="store_true", help="Read frames from a ROS sensor_msgs/Image topic.")
    return parser.parse_args()


def next_index(save_dir, prefix):
    max_idx = -1
    for path in save_dir.glob(f"{prefix}*.jpg"):
        stem = path.stem[len(prefix) :] if prefix and path.stem.startswith(prefix) else path.stem
        if stem.isdigit():
            max_idx = max(max_idx, int(stem))
    return max_idx + 1


def open_cv_source(source):
    if source is None:
        source = 0
    elif str(source).isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def changed(prev, current, threshold):
    if prev is None:
        return True
    prev_small = cv2.resize(prev, (160, 120), interpolation=cv2.INTER_AREA)
    current_small = cv2.resize(current, (160, 120), interpolation=cv2.INTER_AREA)
    return cv2.absdiff(prev_small, current_small).mean() > threshold


def draw_hud(frame, save_dir, count, auto_save):
    view = frame.copy()
    text = f"s:save  a:auto({auto_save})  q:quit  saved:{count}  dir:{save_dir}"
    cv2.putText(view, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return view


def save_frame(frame, save_dir, prefix, idx):
    filename = save_dir / f"{prefix}{idx:06d}.jpg"
    cv2.imwrite(str(filename), frame)
    print(f"saved {filename}")
    return idx + 1


def run_cv_collection(args, cfg):
    save_dir = Path(args.out or cfg["data"]["collect_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    auto_save = bool(args.auto_save or cfg["data"].get("auto_save", False))
    interval = args.interval if args.interval is not None else float(cfg["data"].get("save_interval", 2.0))
    threshold = args.threshold if args.threshold is not None else float(cfg["data"].get("image_change_threshold", 15))
    show_window = bool(cfg["data"].get("show_window", True)) and not args.no_window

    source = args.source or cfg.get("camera", {}).get("source", 0)
    cap = open_cv_source(source)
    idx = next_index(save_dir, args.prefix)
    saved_count = 0
    last_save = 0.0
    prev_saved = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No more frames from source.")
                break

            now = time.time()
            if auto_save and now - last_save >= interval and changed(prev_saved, frame, threshold):
                idx = save_frame(frame, save_dir, args.prefix, idx)
                saved_count += 1
                prev_saved = frame.copy()
                last_save = now

            if show_window:
                cv2.imshow("YOLO dataset collector", draw_hud(frame, save_dir, saved_count, auto_save))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    idx = save_frame(frame, save_dir, args.prefix, idx)
                    saved_count += 1
                    prev_saved = frame.copy()
                    last_save = now
                if key == ord("a"):
                    auto_save = not auto_save
            elif auto_save:
                time.sleep(0.01)
            else:
                idx = save_frame(frame, save_dir, args.prefix, idx)
                saved_count += 1
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


class RosCollector:
    def __init__(self, args, cfg):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.rospy = rospy
        self.bridge = CvBridge()
        self.frame = None
        self.cfg = cfg
        self.args = args
        self.save_dir = Path(args.out or cfg["data"]["collect_path"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.idx = next_index(self.save_dir, args.prefix)
        self.saved_count = 0
        self.auto_save = bool(args.auto_save or cfg["data"].get("auto_save", False))
        self.interval = args.interval if args.interval is not None else float(cfg["data"].get("save_interval", 2.0))
        self.threshold = args.threshold if args.threshold is not None else float(cfg["data"].get("image_change_threshold", 15))
        self.show_window = bool(cfg["data"].get("show_window", True)) and not args.no_window
        self.last_save = 0.0
        self.prev_saved = None

        topic = args.source or cfg["camera"]["topic"]
        rospy.init_node("yolo_dataset_collector", anonymous=True)
        rospy.Subscriber(topic, Image, self.callback, queue_size=1)
        print(f"listening on ROS topic: {topic}")

    def callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def run(self):
        rate = self.rospy.Rate(30)
        while not self.rospy.is_shutdown():
            if self.frame is None:
                rate.sleep()
                continue

            frame = self.frame.copy()
            now = time.time()
            if self.auto_save and now - self.last_save >= self.interval and changed(self.prev_saved, frame, self.threshold):
                self.idx = save_frame(frame, self.save_dir, self.args.prefix, self.idx)
                self.saved_count += 1
                self.prev_saved = frame.copy()
                self.last_save = now

            if self.show_window:
                cv2.imshow("YOLO dataset collector", draw_hud(frame, self.save_dir, self.saved_count, self.auto_save))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    self.idx = save_frame(frame, self.save_dir, self.args.prefix, self.idx)
                    self.saved_count += 1
                    self.prev_saved = frame.copy()
                    self.last_save = now
                if key == ord("a"):
                    self.auto_save = not self.auto_save
            rate.sleep()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.ros:
        RosCollector(args, cfg).run()
    else:
        run_cv_collection(args, cfg)


if __name__ == "__main__":
    main()
