#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SAM/FastSAM ROS controller for the tic-tac-toe robot flow.

This entry point reuses the existing game engine and pick/place controller, but
replaces the YOLO vision backend with Segment Anything masks plus HSV color
classification for yellow and blue pieces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

try:
    from ultralytics import FastSAM, SAM
except ImportError:  # pragma: no cover - gives a clear runtime error on robots.
    FastSAM = None
    SAM = None

from scripts.main_control import TicTacToeMainController
from utils.config_loader import load_config, resolve_project_path


BoardState = List[int]
BoxXYXY = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SAM ROS controller."""
    parser = argparse.ArgumentParser(description="Run SAM tic-tac-toe robot control.")
    parser.add_argument("--config", default="config/config.yaml", help="Config path.")
    parser.add_argument("--once", action="store_true", help="Execute at most one robot move.")
    parser.add_argument("--dry-run", action="store_true", help="Plan moves without moving the arm.")
    return parser.parse_args()


def _as_int_triplet(values: Sequence[int], name: str) -> Tuple[int, int, int]:
    """Convert a sequence to a validated HSV integer triplet.

    Args:
        values: Sequence expected to contain exactly three HSV channel values.
        name: Configuration field name used in the validation error message.

    Returns:
        Tuple[int, int, int]: The converted ``(h, s, v)`` triplet.

    Raises:
        ValueError: ``values`` does not contain exactly three items.
    """
    out = tuple(int(v) for v in values)
    if len(out) != 3:
        raise ValueError(f"{name} must contain three HSV values, got {values}")
    return out


def _hsv_between(hsv: np.ndarray, lower: Sequence[int], upper: Sequence[int]) -> np.ndarray:
    """Build a boolean mask for HSV pixels inside an inclusive range.

    Args:
        hsv: OpenCV HSV image array.
        lower: Inclusive lower HSV bound ``[h, s, v]``.
        upper: Inclusive upper HSV bound ``[h, s, v]``.

    Returns:
        np.ndarray: Boolean mask where ``True`` marks pixels inside the range.
    """
    lower_arr = np.array(lower, dtype=np.uint8)
    upper_arr = np.array(upper, dtype=np.uint8)
    return cv2.inRange(hsv, lower_arr, upper_arr) > 0


def _resolve_model_path(model_path: str) -> str:
    """Use a project-local file when present, otherwise let Ultralytics download by name."""
    candidate = resolve_project_path(model_path)
    if candidate.exists():
        return str(candidate)
    return str(model_path)


class TicTacToeSAMVisionState:
    """Subscribe to camera images and keep the latest SAM-derived board state."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize SAM/FastSAM and the camera subscriber."""
        self.cfg = load_config(config_path)
        self.sam_cfg = self.cfg.get("sam", {})
        self.vision_cfg = self.cfg.get("vision", {})
        self.game_cfg = self.cfg.get("game", {})

        self.camera_topic = self.cfg["camera"]["topic"]
        self.model_type = str(self.sam_cfg.get("model_type", "fastsam")).lower()
        default_model = (
            "FastSAM-s.pt"
            if self.model_type in ("fastsam", "fast_sam", "fast")
            else "sam_b.pt"
        )
        self.model_path = _resolve_model_path(
            str(self.sam_cfg.get("model_path", default_model))
        )

        self.conf = float(self.sam_cfg.get("conf", 0.25))
        self.iou = float(self.sam_cfg.get("iou", 0.7))
        self.imgsz = self.sam_cfg.get("imgsz", 640)
        self.device = self.sam_cfg.get("device", self.cfg.get("yolo", {}).get("device", 0))
        self.mask_threshold = float(self.sam_cfg.get("mask_threshold", 0.5))

        self.show_window = bool(
            self.sam_cfg.get("show_window", self.vision_cfg.get("show_window", False))
        )
        self.window_name = str(self.sam_cfg.get("window_name", "tic_tac_toe_sam_vision"))
        self.row_ratios = [
            float(v) for v in self.vision_cfg.get("row_ratios", [1.0, 1.5, 2.0])
        ]
        if len(self.row_ratios) != 3 or sum(self.row_ratios) <= 0:
            raise ValueError("vision.row_ratios must contain three positive values")
        self.trapezoid_top_width_ratio = float(
            self.vision_cfg.get("trapezoid_top_width_ratio", 2.0 / 3.0)
        )

        self.yellow_value = int(self.game_cfg.get("yellow_value", 1))
        self.blue_value = int(self.game_cfg.get("blue_value", 2))
        self.yellow_hsv_lower = _as_int_triplet(
            self.sam_cfg.get("yellow_hsv_lower", [15, 45, 50]),
            "sam.yellow_hsv_lower",
        )
        self.yellow_hsv_upper = _as_int_triplet(
            self.sam_cfg.get("yellow_hsv_upper", [40, 255, 255]),
            "sam.yellow_hsv_upper",
        )
        self.blue_hsv_lower = _as_int_triplet(
            self.sam_cfg.get("blue_hsv_lower", [90, 45, 40]),
            "sam.blue_hsv_lower",
        )
        self.blue_hsv_upper = _as_int_triplet(
            self.sam_cfg.get("blue_hsv_upper", [135, 255, 255]),
            "sam.blue_hsv_upper",
        )
        self.min_color_ratio = float(self.sam_cfg.get("min_color_ratio", 0.18))
        self.min_piece_area_ratio = float(self.sam_cfg.get("min_piece_area_ratio", 0.02))
        self.max_piece_area_ratio = float(self.sam_cfg.get("max_piece_area_ratio", 0.75))
        self.board_box_config = self.sam_cfg.get("board_box")

        self.bridge = CvBridge()
        self.model = self._load_model()
        self.latest_board_state: Optional[BoardState] = None

        self.sub = rospy.Subscriber(
            self.camera_topic,
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("[TicTacToeSAMVisionState] initialized")
        rospy.loginfo("[TicTacToeSAMVisionState] camera_topic: %s", self.camera_topic)
        rospy.loginfo("[TicTacToeSAMVisionState] model_type: %s", self.model_type)
        rospy.loginfo("[TicTacToeSAMVisionState] model_path: %s", self.model_path)

    def _load_model(self):
        """Load the configured Ultralytics SAM implementation."""
        if self.model_type in ("fastsam", "fast_sam", "fast"):
            if FastSAM is None:
                raise RuntimeError(
                    "ultralytics FastSAM is not available. Install/upgrade ultralytics."
                )
            return FastSAM(self.model_path)
        if self.model_type in ("sam", "segment_anything"):
            if SAM is None:
                raise RuntimeError("ultralytics SAM is not available. Install/upgrade ultralytics.")
            return SAM(self.model_path)
        raise ValueError("sam.model_type must be 'fastsam' or 'sam'")

    def _image_callback(self, msg: Image):
        """Handle one ROS image message from the camera topic."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr("[TicTacToeSAMVisionState] cv_bridge error: %s", exc)
            return

        board_state, debug_image, _ = self.detect_from_cv_image(frame)
        self.latest_board_state = board_state

        if self.show_window and debug_image is not None:
            cv2.imshow(self.window_name, debug_image)
            cv2.waitKey(1)

    def detect_from_cv_image(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[BoardState], np.ndarray, BoxXYXY]:
        """Run SAM/FastSAM segmentation and convert masks to a 0~8 board state."""
        debug_image = frame.copy()
        board_box = self._get_board_box(frame)
        board_state: BoardState = [0] * 9
        best_scores = [-1.0] * 9

        masks = self._predict_masks(frame)
        if not masks and self.board_box_config is None:
            self._draw_board_grid(debug_image, board_box)
            return None, debug_image, board_box

        classified_by_cell = {}
        for mask in masks:
            candidate = self._classify_piece_mask(frame, mask, board_box)
            if candidate is None:
                continue

            index = int(candidate["index"])
            score = float(candidate["score"])
            if score <= best_scores[index]:
                continue

            best_scores[index] = score
            board_state[index] = int(candidate["piece_value"])
            classified_by_cell[index] = (mask, candidate)

        self._draw_classified_masks(debug_image, classified_by_cell.values())
        self._draw_board_grid(debug_image, board_box)
        return board_state, debug_image, board_box

    def _predict_masks(self, frame: np.ndarray) -> List[np.ndarray]:
        """Return binary masks from one SAM/FastSAM prediction."""
        kwargs = {
            "source": frame,
            "verbose": False,
            "conf": self.conf,
            "iou": self.iou,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz

        try:
            results = self.model.predict(**kwargs)
        except TypeError:
            kwargs.pop("conf", None)
            kwargs.pop("iou", None)
            kwargs.pop("imgsz", None)
            results = self.model.predict(**kwargs)

        if not results:
            return []

        result = results[0]
        masks_obj = getattr(result, "masks", None)
        if masks_obj is None or getattr(masks_obj, "data", None) is None:
            return []

        data = masks_obj.data
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        else:
            data = np.asarray(data)

        frame_h, frame_w = frame.shape[:2]
        masks: List[np.ndarray] = []
        for raw_mask in data:
            mask = np.asarray(raw_mask) > self.mask_threshold
            if mask.shape[:2] != (frame_h, frame_w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            masks.append(mask)
        return masks

    def _get_board_box(self, frame: np.ndarray) -> BoxXYXY:
        """Read a configured board box or use the whole frame as the board area."""
        frame_h, frame_w = frame.shape[:2]
        raw_box = self.board_box_config
        if raw_box is None:
            return 0, 0, frame_w - 1, frame_h - 1

        if len(raw_box) != 4:
            raise ValueError("sam.board_box must be [x1, y1, x2, y2] or null")

        values = [float(v) for v in raw_box]
        if max(values) <= 1.0:
            x1, y1, x2, y2 = (
                int(values[0] * frame_w),
                int(values[1] * frame_h),
                int(values[2] * frame_w),
                int(values[3] * frame_h),
            )
        else:
            x1, y1, x2, y2 = (int(v) for v in values)

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid sam.board_box: {raw_box}")
        return x1, y1, x2, y2

    def _classify_piece_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        board_box: BoxXYXY,
    ) -> Optional[Dict[str, object]]:
        """Classify one mask as yellow/blue and map it to a board cell."""
        bx1, by1, bx2, by2 = board_box
        board_mask = np.zeros(mask.shape, dtype=bool)
        board_mask[by1:by2 + 1, bx1:bx2 + 1] = True
        clipped = mask & board_mask

        ys, xs = np.nonzero(clipped)
        if len(xs) == 0:
            return None

        cell_area = max(1.0, ((bx2 - bx1) * (by2 - by1)) / 9.0)
        area = float(len(xs))
        area_ratio = area / cell_area
        if area_ratio < self.min_piece_area_ratio or area_ratio > self.max_piece_area_ratio:
            return None

        cx = float(xs.mean())
        cy = float(ys.mean())
        index = self._point_to_cell(cx, cy, board_box)
        if index is None:
            return None

        pixels = frame[clipped]
        if len(pixels) == 0:
            return None

        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        yellow_ratio = float(_hsv_between(hsv, self.yellow_hsv_lower, self.yellow_hsv_upper).mean())
        blue_ratio = float(_hsv_between(hsv, self.blue_hsv_lower, self.blue_hsv_upper).mean())

        if yellow_ratio >= blue_ratio:
            color_ratio = yellow_ratio
            piece_value = self.yellow_value
            color_name = "yellow_piece"
        else:
            color_ratio = blue_ratio
            piece_value = self.blue_value
            color_name = "blue_piece"

        if color_ratio < self.min_color_ratio:
            return None

        return {
            "index": index,
            "piece_value": piece_value,
            "color_name": color_name,
            "score": color_ratio * min(area_ratio, 1.0),
            "centroid": (int(cx), int(cy)),
        }

    def _point_to_cell(self, cx: float, cy: float, board_box: BoxXYXY) -> Optional[int]:
        """Map a point in the board trapezoid to a 0~8 cell index."""
        bx1, by1, bx2, by2 = board_box
        board_w = bx2 - bx1
        board_h = by2 - by1
        if board_w <= 0 or board_h <= 0:
            return None
        if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
            return None

        t = max(0.0, min(1.0, (cy - by1) / float(board_h)))
        bottom_width = float(board_w)
        top_width = bottom_width * self.trapezoid_top_width_ratio
        top_left_x = bx1 + (bottom_width - top_width) / 2.0
        top_right_x = top_left_x + top_width
        left_x = top_left_x + (float(bx1) - top_left_x) * t
        right_x = top_right_x + (float(bx2) - top_right_x) * t
        current_width = right_x - left_x
        if current_width <= 0:
            return None

        relative_x = (cx - left_x) / current_width
        if relative_x < -0.05 or relative_x > 1.05:
            return None
        relative_x = max(0.0, min(1.0, relative_x))
        col = max(0, min(2, int(relative_x * 3.0)))

        row_total = sum(self.row_ratios)
        row_bound_1 = self.row_ratios[0] / row_total
        row_bound_2 = (self.row_ratios[0] + self.row_ratios[1]) / row_total
        if t < row_bound_1:
            row = 0
        elif t < row_bound_2:
            row = 1
        else:
            row = 2
        return row * 3 + col

    def _draw_classified_masks(self, image: np.ndarray, classified_masks):
        """Overlay accepted piece masks and labels on a debug image."""
        overlay = image.copy()
        for mask, candidate in classified_masks:
            color_name = candidate["color_name"]
            draw_color = (0, 255, 255) if color_name == "yellow_piece" else (255, 0, 0)
            overlay[mask] = draw_color
            cx, cy = candidate["centroid"]
            cv2.circle(image, (cx, cy), 4, draw_color, -1)
            cv2.putText(
                image,
                f"{color_name}:{candidate['index']}",
                (max(0, cx - 40), max(15, cy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                draw_color,
                2,
            )
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, dst=image)

    def _draw_board_grid(self, image: np.ndarray, board_box: BoxXYXY):
        """Draw the configured board trapezoid, grid, and cell IDs."""
        bx1, by1, bx2, by2 = board_box
        board_w = bx2 - bx1
        board_h = by2 - by1
        if board_w <= 0 or board_h <= 0:
            return

        bottom_width = float(board_w)
        top_width = bottom_width * self.trapezoid_top_width_ratio
        top_left_x = bx1 + (bottom_width - top_width) / 2.0
        top_right_x = top_left_x + top_width

        def edge_at(t: float):
            """Return the left edge, right edge, and y coordinate at row ratio t."""
            left_x = top_left_x + (float(bx1) - top_left_x) * t
            right_x = top_right_x + (float(bx2) - top_right_x) * t
            y = by1 + board_h * t
            return left_x, right_x, y

        outline = np.array(
            [
                [int(top_left_x), int(by1)],
                [int(top_right_x), int(by1)],
                [int(bx2), int(by2)],
                [int(bx1), int(by2)],
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [outline], True, (0, 255, 0), 2)
        cv2.putText(
            image,
            "sam_board",
            (bx1, max(0, by1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        for i in range(1, 3):
            fraction = i / 3.0
            top_x = top_left_x + top_width * fraction
            bottom_x = bx1 + bottom_width * fraction
            cv2.line(image, (int(top_x), int(by1)), (int(bottom_x), int(by2)), (255, 255, 0), 2)

        row_total = sum(self.row_ratios)
        row_bounds = [
            0.0,
            self.row_ratios[0] / row_total,
            (self.row_ratios[0] + self.row_ratios[1]) / row_total,
            1.0,
        ]
        for t in row_bounds[1:3]:
            left_x, right_x, y = edge_at(t)
            cv2.line(image, (int(left_x), int(y)), (int(right_x), int(y)), (255, 255, 0), 2)

        for row in range(3):
            for col in range(3):
                idx = row * 3 + col
                t_center = (row_bounds[row] + row_bounds[row + 1]) / 2.0
                left_x, right_x, y = edge_at(t_center)
                cx = int(left_x + (col + 0.5) * (right_x - left_x) / 3.0)
                cy = int(y)
                cv2.putText(
                    image,
                    str(idx),
                    (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

    def get_current_board_state(self) -> Optional[BoardState]:
        """Return the latest board state from SAM segmentation."""
        return self.latest_board_state


def main():
    """Initialize ROS and run the SAM tic-tac-toe main controller."""
    args = parse_args()
    rospy.init_node("tic_tac_toe_sam_control", anonymous=False)
    controller = TicTacToeMainController(
        args.config,
        dry_run=args.dry_run,
        vision_factory=TicTacToeSAMVisionState,
        vision_name="SAM",
    )
    controller.run(once=args.once)


if __name__ == "__main__":
    main()
