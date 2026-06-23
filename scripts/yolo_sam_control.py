#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""YOLO + SAM ROS controller for the tic-tac-toe robot flow.

YOLO provides the board box, piece boxes, and piece classes.  SAM/FastSAM
refines piece masks so board-cell assignment can use a mask centroid instead of
only the YOLO box center.  The game engine and robot arm control stay unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

try:
    from ultralytics import FastSAM, SAM, YOLO
except ImportError:  # pragma: no cover - gives a clear runtime error on robots.
    FastSAM = None
    SAM = None
    YOLO = None

from scripts.main_control import TicTacToeMainController
from utils.config_loader import load_config, resolve_project_path


BoardState = List[int]
BoxXYXY = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the YOLO+SAM controller."""
    parser = argparse.ArgumentParser(description="Run YOLO+SAM tic-tac-toe robot control.")
    parser.add_argument("--config", default="config/config.yaml", help="Config path.")
    parser.add_argument("--once", action="store_true", help="Execute at most one robot move.")
    parser.add_argument("--dry-run", action="store_true", help="Plan moves without moving the arm.")
    return parser.parse_args()


def _resolve_model_path(model_path: str) -> str:
    """Resolve project-local model paths while preserving Ultralytics model names."""
    candidate = resolve_project_path(model_path)
    if candidate.exists():
        return str(candidate)
    return str(model_path)


def _as_int_triplet(values: Sequence[int], name: str) -> Tuple[int, int, int]:
    """Convert a config value to a validated HSV triplet."""
    out = tuple(int(v) for v in values)
    if len(out) != 3:
        raise ValueError(f"{name} must contain three HSV values, got {values}")
    return out


def _hsv_between(hsv: np.ndarray, lower: Sequence[int], upper: Sequence[int]) -> np.ndarray:
    """Return a boolean mask for HSV pixels inside an inclusive range."""
    lower_arr = np.array(lower, dtype=np.uint8)
    upper_arr = np.array(upper, dtype=np.uint8)
    return cv2.inRange(hsv, lower_arr, upper_arr) > 0


def _box_area(box: BoxXYXY) -> float:
    """Return the area of an ``xyxy`` box."""
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1))


class TicTacToeYoloSAMVisionState:
    """Subscribe to camera images and keep the latest YOLO+SAM board state."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize YOLO, SAM/FastSAM, and the camera subscriber."""
        self.cfg = load_config(config_path)
        self.yolo_cfg = self.cfg["yolo"]
        self.sam_cfg = self.cfg.get("sam", {})
        self.hybrid_cfg = self.cfg.get("yolo_sam", {})
        self.vision_cfg = self.cfg.get("vision", {})
        self.game_cfg = self.cfg.get("game", {})

        self.camera_topic = self.cfg["camera"]["topic"]

        self.yolo_model_path = _resolve_model_path(str(self.yolo_cfg["model_path"]))
        self.yolo_conf = float(self.yolo_cfg.get("conf", 0.25))
        self.yolo_device = self.yolo_cfg.get("device", 0)

        raw_class_names = self.yolo_cfg["class_names"]
        self.class_names = {int(k): str(v) for k, v in raw_class_names.items()}
        self.board_class = self.class_names.get(0, "board")
        self.yellow_class = self.class_names.get(1, "yellow_piece")
        self.blue_class = self.class_names.get(2, "blue_piece")

        self.min_board_conf = float(self.vision_cfg.get("min_board_conf", self.yolo_conf))
        self.min_piece_conf = float(self.vision_cfg.get("min_piece_conf", self.yolo_conf))

        self.sam_model_type = str(
            self.hybrid_cfg.get("sam_model_type", self.sam_cfg.get("model_type", "fastsam"))
        ).lower()
        default_sam_model = (
            "FastSAM-s.pt"
            if self.sam_model_type in ("fastsam", "fast_sam", "fast")
            else "sam_b.pt"
        )
        self.sam_model_path = _resolve_model_path(
            str(
                self.hybrid_cfg.get(
                    "sam_model_path",
                    self.sam_cfg.get("model_path", default_sam_model),
                )
            )
        )
        self.sam_conf = float(self.hybrid_cfg.get("sam_conf", self.sam_cfg.get("conf", 0.25)))
        self.sam_iou = float(self.hybrid_cfg.get("sam_iou", self.sam_cfg.get("iou", 0.7)))
        self.sam_imgsz = self.hybrid_cfg.get("sam_imgsz", self.sam_cfg.get("imgsz", 640))
        self.sam_device = self.hybrid_cfg.get(
            "sam_device",
            self.sam_cfg.get("device", self.yolo_device),
        )
        self.mask_threshold = float(
            self.hybrid_cfg.get("mask_threshold", self.sam_cfg.get("mask_threshold", 0.5))
        )
        self.sam_retina_masks = bool(self.hybrid_cfg.get("sam_retina_masks", True))

        self.show_window = bool(
            self.hybrid_cfg.get("show_window", self.vision_cfg.get("show_window", False))
        )
        self.window_name = str(self.hybrid_cfg.get("window_name", "tic_tac_toe_yolo_sam_vision"))

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
            self.hybrid_cfg.get(
                "yellow_hsv_lower",
                self.sam_cfg.get("yellow_hsv_lower", [15, 45, 50]),
            ),
            "yolo_sam.yellow_hsv_lower",
        )
        self.yellow_hsv_upper = _as_int_triplet(
            self.hybrid_cfg.get(
                "yellow_hsv_upper",
                self.sam_cfg.get("yellow_hsv_upper", [40, 255, 255]),
            ),
            "yolo_sam.yellow_hsv_upper",
        )
        self.blue_hsv_lower = _as_int_triplet(
            self.hybrid_cfg.get(
                "blue_hsv_lower",
                self.sam_cfg.get("blue_hsv_lower", [90, 45, 40]),
            ),
            "yolo_sam.blue_hsv_lower",
        )
        self.blue_hsv_upper = _as_int_triplet(
            self.hybrid_cfg.get(
                "blue_hsv_upper",
                self.sam_cfg.get("blue_hsv_upper", [135, 255, 255]),
            ),
            "yolo_sam.blue_hsv_upper",
        )

        self.fallback_to_yolo_center = bool(
            self.hybrid_cfg.get("fallback_to_yolo_center", True)
        )
        self.fallback_score_scale = float(self.hybrid_cfg.get("fallback_score_scale", 0.65))
        self.mask_match_min_box_coverage = float(
            self.hybrid_cfg.get("mask_match_min_box_coverage", 0.08)
        )
        self.mask_match_min_mask_fraction = float(
            self.hybrid_cfg.get("mask_match_min_mask_fraction", 0.20)
        )
        self.piece_box_expand_ratio = float(self.hybrid_cfg.get("piece_box_expand_ratio", 0.15))
        self.min_piece_area_ratio = float(
            self.hybrid_cfg.get(
                "min_piece_area_ratio",
                self.sam_cfg.get("min_piece_area_ratio", 0.02),
            )
        )
        self.max_piece_area_ratio = float(
            self.hybrid_cfg.get(
                "max_piece_area_ratio",
                self.sam_cfg.get("max_piece_area_ratio", 0.75),
            )
        )
        self.color_check_enabled = bool(self.hybrid_cfg.get("color_check_enabled", True))
        self.reject_color_mismatch = bool(self.hybrid_cfg.get("reject_color_mismatch", False))
        self.min_color_ratio = float(
            self.hybrid_cfg.get("min_color_ratio", self.sam_cfg.get("min_color_ratio", 0.18))
        )
        self.refine_board_box_with_sam = bool(
            self.hybrid_cfg.get("refine_board_box_with_sam", False)
        )
        self.board_box_config = self.hybrid_cfg.get("board_box", self.sam_cfg.get("board_box"))

        self.bridge = CvBridge()
        self.yolo_model = self._load_yolo_model()
        self.sam_model = self._load_sam_model()
        self.latest_board_state: Optional[BoardState] = None

        self.sub = rospy.Subscriber(
            self.camera_topic,
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("[TicTacToeYoloSAMVisionState] initialized")
        rospy.loginfo("[TicTacToeYoloSAMVisionState] camera_topic: %s", self.camera_topic)
        rospy.loginfo("[TicTacToeYoloSAMVisionState] yolo_model_path: %s", self.yolo_model_path)
        rospy.loginfo("[TicTacToeYoloSAMVisionState] sam_model_path: %s", self.sam_model_path)

    def _load_yolo_model(self):
        """Load the configured YOLO detector."""
        if YOLO is None:
            raise RuntimeError("ultralytics YOLO is not available. Install/upgrade ultralytics.")
        return YOLO(str(self.yolo_model_path))

    def _load_sam_model(self):
        """Load the configured Ultralytics SAM implementation."""
        if self.sam_model_type in ("fastsam", "fast_sam", "fast"):
            if FastSAM is None:
                raise RuntimeError("ultralytics FastSAM is not available. Install/upgrade ultralytics.")
            return FastSAM(self.sam_model_path)
        if self.sam_model_type in ("sam", "segment_anything"):
            if SAM is None:
                raise RuntimeError("ultralytics SAM is not available. Install/upgrade ultralytics.")
            return SAM(self.sam_model_path)
        raise ValueError("yolo_sam.sam_model_type must be 'fastsam' or 'sam'")

    def _image_callback(self, msg: Image):
        """Handle one ROS image message from the camera topic."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr("[TicTacToeYoloSAMVisionState] cv_bridge error: %s", exc)
            return

        board_state, debug_image, _ = self.detect_from_cv_image(frame)
        self.latest_board_state = board_state

        if self.show_window and debug_image is not None:
            cv2.imshow(self.window_name, debug_image)
            cv2.waitKey(1)

    def detect_from_cv_image(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[BoardState], np.ndarray, Optional[BoxXYXY]]:
        """Run YOLO detection, refine pieces with SAM, and return board state."""
        debug_image = frame.copy()
        board_candidates, piece_candidates = self._predict_yolo(frame)

        if board_candidates:
            board_box = max(board_candidates, key=lambda item: item["conf"])["xyxy"]
            board_box_source = "yolo"
        else:
            board_box = self._get_config_board_box(frame)
            board_box_source = "config"
            if board_box is None:
                return None, debug_image, None

        masks = self._predict_sam_masks(frame) if piece_candidates or self.refine_board_box_with_sam else []

        if self.refine_board_box_with_sam and masks:
            refined_board_box = self._refine_box_with_matching_mask(masks, board_box, frame.shape[:2])
            if refined_board_box is not None:
                board_box = refined_board_box
                board_box_source = "sam"

        board_state: BoardState = [0] * 9
        cell_best_score = [-1.0] * 9
        accepted_by_cell: Dict[int, Dict[str, Any]] = {}

        for piece in piece_candidates:
            candidate = self._build_piece_candidate(frame, piece, board_box, masks)
            if candidate is None:
                continue

            index = int(candidate["index"])
            score = float(candidate["score"])
            if score <= cell_best_score[index]:
                continue

            cell_best_score[index] = score
            board_state[index] = int(candidate["piece_value"])
            accepted_by_cell[index] = candidate

        self._draw_debug_image(
            debug_image=debug_image,
            board_box=board_box,
            board_box_source=board_box_source,
            board_candidates=board_candidates,
            piece_candidates=piece_candidates,
            accepted=list(accepted_by_cell.values()),
        )
        return board_state, debug_image, board_box

    def _predict_yolo(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return YOLO board and piece candidates for one frame."""
        results = self.yolo_model.predict(
            source=frame,
            conf=self.yolo_conf,
            device=self.yolo_device,
            verbose=False,
        )
        if not results:
            return [], []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return [], []

        board_candidates: List[Dict[str, Any]] = []
        piece_candidates: List[Dict[str, Any]] = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = self.class_names.get(cls_id, str(cls_id))
            xyxy = tuple(int(v) for v in box.xyxy[0].cpu().numpy().astype(int))

            if name == self.board_class and conf >= self.min_board_conf:
                board_candidates.append({"name": name, "conf": conf, "xyxy": xyxy})
            elif name in (self.yellow_class, self.blue_class) and conf >= self.min_piece_conf:
                piece_candidates.append({"name": name, "conf": conf, "xyxy": xyxy})

        return board_candidates, piece_candidates

    def _predict_sam_masks(self, frame: np.ndarray) -> List[np.ndarray]:
        """Return binary masks from one SAM/FastSAM prediction."""
        kwargs = {
            "source": frame,
            "verbose": False,
            "conf": self.sam_conf,
            "iou": self.sam_iou,
            "retina_masks": self.sam_retina_masks,
        }
        if self.sam_device is not None:
            kwargs["device"] = self.sam_device
        if self.sam_imgsz:
            kwargs["imgsz"] = self.sam_imgsz

        try:
            results = self.sam_model.predict(**kwargs)
        except TypeError:
            kwargs.pop("conf", None)
            kwargs.pop("iou", None)
            kwargs.pop("imgsz", None)
            kwargs.pop("retina_masks", None)
            results = self.sam_model.predict(**kwargs)

        if not results:
            return []

        masks_obj = getattr(results[0], "masks", None)
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

    def _build_piece_candidate(
        self,
        frame: np.ndarray,
        piece: Dict[str, Any],
        board_box: BoxXYXY,
        masks: List[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        """Build one board-state candidate from YOLO class and SAM-refined mask."""
        piece_box = piece["xyxy"]
        piece_name = str(piece["name"])
        piece_value = self.yellow_value if piece_name == self.yellow_class else self.blue_value

        match = self._find_matching_piece_mask(masks, piece_box, board_box)
        if match is not None:
            piece_mask, match_score = match
            source = "sam"
            centroid = self._mask_centroid(piece_mask)
            color_mask = piece_mask
            score = float(piece["conf"]) * (0.5 + 0.5 * match_score)
        elif self.fallback_to_yolo_center:
            source = "yolo"
            x1, y1, x2, y2 = piece_box
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            color_mask = self._box_region_mask(frame.shape[:2], piece_box)
            score = float(piece["conf"]) * self.fallback_score_scale
        else:
            return None

        index = self._point_to_cell(centroid[0], centroid[1], board_box)
        if index is None:
            return None

        color_ratio = self._piece_color_ratio(frame, color_mask, piece_name)
        if self.color_check_enabled:
            if self.reject_color_mismatch and color_ratio < self.min_color_ratio:
                return None
            score *= 0.5 + 0.5 * min(1.0, color_ratio / max(self.min_color_ratio, 1e-6))

        return {
            "index": index,
            "piece_value": piece_value,
            "name": piece_name,
            "score": score,
            "centroid": (int(centroid[0]), int(centroid[1])),
            "xyxy": piece_box,
            "mask": piece_mask if match is not None else None,
            "source": source,
            "color_ratio": color_ratio,
            "yolo_conf": float(piece["conf"]),
        }

    def _find_matching_piece_mask(
        self,
        masks: List[np.ndarray],
        piece_box: BoxXYXY,
        board_box: BoxXYXY,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Find the SAM mask that best overlaps a YOLO piece box."""
        if not masks:
            return None

        expanded_box = self._expand_box(piece_box, self.piece_box_expand_ratio, masks[0].shape)
        box_mask = self._box_region_mask(masks[0].shape, piece_box)
        expanded_mask = self._box_region_mask(masks[0].shape, expanded_box)
        board_mask = self._box_region_mask(masks[0].shape, board_box)
        box_area = max(1.0, float(box_mask.sum()))
        cell_area = max(1.0, _box_area(board_box) / 9.0)

        best_mask = None
        best_score = -1.0

        for mask in masks:
            mask_in_board = mask & board_mask
            mask_area_in_board = float(mask_in_board.sum())
            if mask_area_in_board <= 0:
                continue

            mask_in_expanded_box = mask_in_board & expanded_mask
            refined_area = float(mask_in_expanded_box.sum())
            area_ratio = refined_area / cell_area
            if area_ratio < self.min_piece_area_ratio or area_ratio > self.max_piece_area_ratio:
                continue

            intersection = float((mask & box_mask).sum())
            box_coverage = intersection / box_area
            mask_fraction = intersection / max(1.0, mask_area_in_board)

            if (
                box_coverage < self.mask_match_min_box_coverage
                and mask_fraction < self.mask_match_min_mask_fraction
            ):
                continue

            score = 0.7 * box_coverage + 0.3 * mask_fraction
            if score > best_score:
                best_score = score
                best_mask = mask_in_expanded_box

        if best_mask is None:
            return None
        return best_mask, max(0.0, min(1.0, best_score))

    def _refine_box_with_matching_mask(
        self,
        masks: List[np.ndarray],
        box: BoxXYXY,
        frame_shape: Tuple[int, int],
    ) -> Optional[BoxXYXY]:
        """Optionally refine the YOLO board box with a matching SAM mask."""
        box_mask = self._box_region_mask(frame_shape, box)
        box_area = max(1.0, float(box_mask.sum()))
        best_mask = None
        best_score = -1.0

        for mask in masks:
            intersection = float((mask & box_mask).sum())
            if intersection <= 0:
                continue
            score = intersection / box_area
            if score > best_score:
                best_score = score
                best_mask = mask

        if best_mask is None or best_score < 0.20:
            return None

        ys, xs = np.nonzero(best_mask)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _get_config_board_box(self, frame: np.ndarray) -> Optional[BoxXYXY]:
        """Return the configured fallback board box, if present."""
        raw_box = self.board_box_config
        if raw_box is None:
            return None
        if len(raw_box) != 4:
            raise ValueError("yolo_sam.board_box must be [x1, y1, x2, y2] or null")

        frame_h, frame_w = frame.shape[:2]
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
            raise ValueError(f"Invalid yolo_sam.board_box: {raw_box}")
        return x1, y1, x2, y2

    def _piece_color_ratio(self, frame: np.ndarray, mask: np.ndarray, piece_name: str) -> float:
        """Return the HSV color ratio for the class predicted by YOLO."""
        pixels = frame[mask]
        if len(pixels) == 0:
            return 0.0

        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        if piece_name == self.yellow_class:
            return float(_hsv_between(hsv, self.yellow_hsv_lower, self.yellow_hsv_upper).mean())
        return float(_hsv_between(hsv, self.blue_hsv_lower, self.blue_hsv_upper).mean())

    def _point_to_cell(self, cx: float, cy: float, board_box: BoxXYXY) -> Optional[int]:
        """Map a point in the detected board trapezoid to a 0~8 cell index."""
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

    def _draw_debug_image(
        self,
        debug_image: np.ndarray,
        board_box: BoxXYXY,
        board_box_source: str,
        board_candidates: List[Dict[str, Any]],
        piece_candidates: List[Dict[str, Any]],
        accepted: List[Dict[str, Any]],
    ):
        """Draw YOLO boxes, SAM masks, centroids, and board cell IDs."""
        for board in board_candidates:
            x1, y1, x2, y2 = board["xyxy"]
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 180, 0), 1)

        for piece in piece_candidates:
            x1, y1, x2, y2 = piece["xyxy"]
            color = (0, 255, 255) if piece["name"] == self.yellow_class else (255, 0, 0)
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 1)

        overlay = debug_image.copy()
        for candidate in accepted:
            color = (0, 255, 255) if candidate["name"] == self.yellow_class else (255, 0, 0)
            mask = candidate.get("mask")
            if mask is not None:
                overlay[mask] = color
            cx, cy = candidate["centroid"]
            cv2.circle(debug_image, (cx, cy), 4, color, -1)
            cv2.putText(
                debug_image,
                f"{candidate['name']}:{candidate['index']}:{candidate['source']}",
                (max(0, cx - 55), max(15, cy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
        cv2.addWeighted(overlay, 0.25, debug_image, 0.75, 0, dst=debug_image)

        self._draw_board_grid(debug_image, board_box, board_box_source)

    def _draw_board_grid(self, image: np.ndarray, board_box: BoxXYXY, source: str):
        """Draw the board trapezoid, grid, and cell IDs."""
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
            f"board:{source}",
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

    @staticmethod
    def _box_region_mask(shape: Tuple[int, int], box: BoxXYXY) -> np.ndarray:
        """Return a boolean image mask for an ``xyxy`` box."""
        h, w = shape
        x1, y1, x2, y2 = box
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        mask = np.zeros((h, w), dtype=bool)
        if x2 >= x1 and y2 >= y1:
            mask[y1:y2 + 1, x1:x2 + 1] = True
        return mask

    @staticmethod
    def _expand_box(box: BoxXYXY, ratio: float, shape: Tuple[int, int]) -> BoxXYXY:
        """Expand an ``xyxy`` box by a width/height ratio and clamp to frame size."""
        h, w = shape
        x1, y1, x2, y2 = box
        expand_x = int((x2 - x1 + 1) * ratio)
        expand_y = int((y2 - y1 + 1) * ratio)
        return (
            max(0, x1 - expand_x),
            max(0, y1 - expand_y),
            min(w - 1, x2 + expand_x),
            min(h - 1, y2 + expand_y),
        )

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
        """Return the centroid of a binary mask as ``(x, y)``."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return 0.0, 0.0
        return float(xs.mean()), float(ys.mean())

    def get_current_board_state(self) -> Optional[BoardState]:
        """Return the latest YOLO+SAM board state."""
        return self.latest_board_state


def main():
    """Initialize ROS and run the YOLO+SAM tic-tac-toe main controller."""
    args = parse_args()
    rospy.init_node("tic_tac_toe_yolo_sam_control", anonymous=False)
    controller = TicTacToeMainController(
        args.config,
        dry_run=args.dry_run,
        vision_factory=TicTacToeYoloSAMVisionState,
        vision_name="YOLO+SAM",
    )
    controller.run(once=args.once)


if __name__ == "__main__":
    main()
