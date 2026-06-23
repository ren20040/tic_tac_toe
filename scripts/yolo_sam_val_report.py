#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a validation report for YOLO+SAM+Qwen-VLM tic-tac-toe perception.

The script processes every image in the validation split, compares predictions
with YOLO-format validation labels, and writes paper-friendly visual artifacts:

    - YOLO+SAM annotated image and all-mask preview from the frontend
    - ground-truth overlay
    - prediction-vs-ground-truth overlay
    - per-image 3x3 state comparison image for YOLO, YOLO+SAM, VLM,
      and conservative fusion
    - dataset-level confusion matrices and metric summary image
    - summary.csv and summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from scripts.yolo_sam_perception_frontend import (
    BoxXYXY,
    IMAGE_SUFFIXES,
    QwenBoardStateVLM,
    YoloSAMPerceptionFrontend,
    box_area,
    json_safe,
    normalize_vlm_quantization,
)
from utils.config_loader import load_config, resolve_project_path


BoardState = List[int]
STATE_NAMES = {0: "empty", 1: "yellow", 2: "blue"}
STATE_SHORT = {0: ".", 1: "Y", 2: "B"}
STATE_COLORS = {
    0: (235, 235, 235),
    1: (0, 220, 255),
    2: (255, 80, 0),
}
DEFAULT_VLM_SETTINGS = {
    "enabled": True,
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "device_map": "auto",
    "max_new_tokens": 512,
    "quantization": "none",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate YOLO-only vs YOLO+SAM vs YOLO+SAM+Qwen-VLM validation report."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Config path.")
    parser.add_argument("--images", default="dataset/images/val", help="Validation image directory.")
    parser.add_argument("--labels", default="dataset/labels/val", help="Validation label directory.")
    parser.add_argument(
        "--out",
        default="runs/yolo_sam_frontend/val_report",
        help="Output directory for report artifacts.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0 means process all images.")
    parser.add_argument("--no-save-masks", action="store_true", help="Do not write per-piece masks.")
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable Qwen VLM arbitration. By default VLM is enabled for this report.",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Hugging Face model id or local path for Qwen2.5-VL. Overrides config.vlm.model.",
    )
    parser.add_argument(
        "--vlm-device-map",
        default=None,
        help="device_map passed to transformers from_pretrained. Overrides config.vlm.device_map.",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens for VLM JSON output. Overrides config.vlm.max_new_tokens.",
    )
    parser.add_argument(
        "--vlm-quant",
        choices=["none", "8bit", "4bit"],
        default=None,
        help="Optional Qwen VLM quantization mode. Overrides config.vlm.quantization.",
    )
    parser.add_argument(
        "--low-conf-threshold",
        type=float,
        default=0.75,
        help="Confidence threshold used by paper low-confidence sample metric.",
    )
    parser.add_argument(
        "--assignment-iou-threshold",
        type=float,
        default=0.10,
        help="IoU threshold for matching predicted pieces to GT pieces when computing grid-assignment accuracy.",
    )
    return parser.parse_args()


def load_vlm_settings(config_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Load VLM experiment settings from config and apply CLI overrides."""
    config = load_config(config_path) or {}
    config_vlm = dict(DEFAULT_VLM_SETTINGS)
    file_vlm = config.get("vlm") or {}
    if isinstance(file_vlm, dict):
        config_vlm.update(file_vlm)

    report_vlm = (config.get("val_report") or {}).get("vlm") if isinstance(config.get("val_report"), dict) else None
    if isinstance(report_vlm, dict):
        config_vlm.update(report_vlm)

    enabled = bool(config_vlm.get("enabled", True)) and not args.no_vlm
    model = args.vlm_model or config_vlm.get("model") or config_vlm.get("model_id") or DEFAULT_VLM_SETTINGS["model"]
    device_map = args.vlm_device_map or config_vlm.get("device_map") or DEFAULT_VLM_SETTINGS["device_map"]
    max_new_tokens = (
        args.vlm_max_new_tokens
        if args.vlm_max_new_tokens is not None
        else config_vlm.get("max_new_tokens", DEFAULT_VLM_SETTINGS["max_new_tokens"])
    )
    quantization = (
        args.vlm_quant
        if args.vlm_quant is not None
        else config_vlm.get("quantization", config_vlm.get("quant", DEFAULT_VLM_SETTINGS["quantization"]))
    )

    return {
        "enabled": enabled,
        "model": str(model),
        "device_map": str(device_map),
        "max_new_tokens": int(max_new_tokens),
        "quantization": normalize_vlm_quantization(quantization),
    }


def iter_images(image_dir: Path, max_images: int = 0) -> Iterable[Path]:
    """Yield validation images in stable order."""
    count = 0
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        yield path
        count += 1
        if max_images > 0 and count >= max_images:
            break


def yolo_norm_to_xyxy(values: Sequence[float], shape: Tuple[int, int]) -> BoxXYXY:
    """Convert normalized YOLO cx/cy/w/h to clamped xyxy pixels."""
    h, w = shape
    cx, cy, bw, bh = values
    x1 = int(round((cx - bw / 2.0) * w))
    y1 = int(round((cy - bh / 2.0) * h))
    x2 = int(round((cx + bw / 2.0) * w))
    y2 = int(round((cy + bh / 2.0) * h))
    return (
        max(0, min(w - 1, x1)),
        max(0, min(h - 1, y1)),
        max(0, min(w - 1, x2)),
        max(0, min(h - 1, y2)),
    )


def load_yolo_labels(
    label_path: Path,
    image_shape: Tuple[int, int],
    class_names: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Load one YOLO label file as pixel boxes."""
    if not label_path.exists():
        return []

    labels: List[Dict[str, Any]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line {label_path}:{line_no}: {line!r}")
            class_id = int(float(parts[0]))
            xywh = [float(v) for v in parts[1:]]
            box = yolo_norm_to_xyxy(xywh, image_shape)
            labels.append(
                {
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, str(class_id)),
                    "bbox_xyxy": list(box),
                    "center_xy": [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0],
                    "area": box_area(box),
                }
            )
    return labels


def labels_to_board_state(
    frontend: YoloSAMPerceptionFrontend,
    labels: List[Dict[str, Any]],
) -> Tuple[Optional[BoardState], Optional[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Convert ground-truth labels to board_state using the configured grid geometry."""
    messages: List[str] = []
    board_labels = [item for item in labels if item["class_name"] == frontend.board_class]
    if not board_labels:
        return None, None, [], ["missing_gt_board"]

    board = max(board_labels, key=lambda item: float(item["area"]))
    board_box = tuple(int(v) for v in board["bbox_xyxy"])
    state: BoardState = [frontend.empty_value] * 9
    cell_has_label = [False] * 9
    pieces: List[Dict[str, Any]] = []

    for item in labels:
        class_name = item["class_name"]
        if class_name not in (frontend.yellow_class, frontend.blue_class):
            continue
        cx, cy = item["center_xy"]
        cell_index = frontend._point_to_cell(cx, cy, board_box)
        if cell_index is None:
            messages.append(f"gt_piece_outside_board:{class_name}")
            continue
        value = frontend.yellow_value if class_name == frontend.yellow_class else frontend.blue_value
        if cell_has_label[cell_index]:
            messages.append(f"duplicate_gt_cell:{cell_index}")
        state[cell_index] = value
        cell_has_label[cell_index] = True
        piece = dict(item)
        piece["cell_index"] = int(cell_index)
        piece["piece_value"] = int(value)
        pieces.append(piece)

    return state, board, pieces, messages


def detections_to_yolo_only_state(
    frontend: YoloSAMPerceptionFrontend,
    result: Dict[str, Any],
) -> Tuple[Optional[BoardState], List[Dict[str, Any]]]:
    """Build a YOLO-only board state using raw piece box centers."""
    board = result.get("board")
    if not board:
        return None, []
    board_box = tuple(int(v) for v in board["bbox_xyxy"])
    state: BoardState = [frontend.empty_value] * 9
    best_conf = [-1.0] * 9
    pieces: List[Dict[str, Any]] = []

    for item in result.get("detections", {}).get("pieces", []):
        box = tuple(int(v) for v in item["bbox_xyxy"])
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        cell_index = frontend._point_to_cell(cx, cy, board_box)
        if cell_index is None:
            continue
        conf = float(item.get("confidence", 0.0))
        class_name = str(item["class_name"])
        value = frontend.yellow_value if class_name == frontend.yellow_class else frontend.blue_value
        if conf >= best_conf[cell_index]:
            best_conf[cell_index] = conf
            state[cell_index] = int(value)
        piece = dict(item)
        piece["cell_index"] = int(cell_index)
        piece["piece_value"] = int(value)
        piece["center_xy"] = [cx, cy]
        pieces.append(piece)

    return state, pieces


def build_conservative_fusion_state(
    frontend: YoloSAMPerceptionFrontend,
    yolo_state: Optional[BoardState],
    fusion_state: Optional[BoardState],
    vlm_state: Optional[BoardState],
    yolo_pieces: List[Dict[str, Any]],
    fusion_pieces: List[Dict[str, Any]],
    vlm_result: Dict[str, Any],
) -> Tuple[Optional[BoardState], List[Dict[str, Any]]]:
    """Build a YOLO-first conservative fusion state.

    The policy is intentionally conservative:
    - YOLO bbox-center assignment is the default state.
    - SAM and VLM are treated as evidence and abnormality detectors.
    - A non-YOLO value is accepted only when YOLO is empty and both SAM and VLM
      agree on the same non-empty value with reliable SAM evidence.
    - Disagreements are recorded, but they do not overwrite YOLO.

    This prevents the report from unfairly lowering the YOLO baseline by letting
    a bad SAM mask or over-trusting VLM rewrite an already valid YOLO result.
    """
    if yolo_state is None:
        if vlm_state is not None:
            return list(vlm_state), [{"cell_index": i, "decision": "fallback_vlm_no_yolo"} for i in range(9)]
        if fusion_state is not None:
            return list(fusion_state), [{"cell_index": i, "decision": "fallback_sam_no_yolo"} for i in range(9)]
        return None, []

    state = list(yolo_state)
    decisions: List[Dict[str, Any]] = []
    yolo_conf_by_cell = {
        int(piece["cell_index"]): float(piece.get("confidence", piece.get("yolo_confidence", 0.0)))
        for piece in yolo_pieces
    }
    sam_piece_by_cell = {int(piece["cell_index"]): piece for piece in fusion_pieces}
    vlm_parsed = vlm_result.get("parsed") if isinstance(vlm_result, dict) else None
    vlm_abnormal_cells = set()
    vlm_uncertain_cells = set()
    if isinstance(vlm_parsed, dict):
        vlm_uncertain_cells = {int(v) for v in vlm_parsed.get("uncertain_cells", []) if isinstance(v, int)}
        for item in vlm_parsed.get("abnormal_cells", []):
            if isinstance(item, dict) and "cell_index" in item:
                try:
                    vlm_abnormal_cells.add(int(item["cell_index"]))
                except (TypeError, ValueError):
                    pass

    for idx in range(9):
        yolo_value = yolo_state[idx]
        sam_value = fusion_state[idx] if fusion_state is not None else None
        vlm_value = vlm_state[idx] if vlm_state is not None else None
        yolo_confidence = yolo_conf_by_cell.get(idx)
        yolo_low_confidence = yolo_confidence is None or yolo_confidence < 0.75
        sam_piece = sam_piece_by_cell.get(idx)
        sam_reliable = False
        if sam_piece is not None:
            color_ok = bool(sam_piece.get("color", {}).get("verified", False))
            mask_area = int(sam_piece.get("mask_area", 0))
            mask_source = str(sam_piece.get("mask_source", ""))
            sam_reliable = mask_source == "sam" and color_ok and mask_area >= 1200

        if yolo_value != frontend.empty_value:
            decision = "keep_yolo_non_empty"
            if (
                yolo_low_confidence
                and sam_value in (frontend.yellow_value, frontend.blue_value)
                and vlm_value == sam_value
                and sam_value != yolo_value
                and sam_reliable
                and idx not in vlm_abnormal_cells
                and idx not in vlm_uncertain_cells
            ):
                state[idx] = int(sam_value)
                decision = "correct_low_conf_yolo_with_sam_vlm"
            elif (
                yolo_low_confidence
                and sam_value in (None, frontend.empty_value)
                and vlm_value == frontend.empty_value
                and idx not in vlm_abnormal_cells
                and idx not in vlm_uncertain_cells
            ):
                state[idx] = frontend.empty_value
                decision = "clear_low_conf_yolo_with_vlm"
            if sam_value is not None and sam_value != yolo_value:
                decision = decision if decision.startswith(("correct_", "clear_")) else "keep_yolo_sam_disagree"
            if vlm_value is not None and vlm_value != yolo_value:
                decision = decision if decision.startswith(("correct_", "clear_")) else "keep_yolo_vlm_disagree"
            if idx in vlm_abnormal_cells or idx in vlm_uncertain_cells:
                decision += "_vlm_flagged"
            decisions.append(
                {
                    "cell_index": idx,
                    "decision": decision,
                    "final": state[idx],
                    "yolo": yolo_value,
                    "yolo_confidence": yolo_confidence,
                    "sam": sam_value,
                    "vlm": vlm_value,
                    "sam_reliable": sam_reliable,
                }
            )
            continue

        accepted = False
        if (
            sam_value in (frontend.yellow_value, frontend.blue_value)
            and vlm_value == sam_value
            and sam_reliable
            and idx not in vlm_abnormal_cells
            and idx not in vlm_uncertain_cells
        ):
            state[idx] = int(sam_value)
            accepted = True

        decisions.append(
            {
                "cell_index": idx,
                "decision": "accept_sam_vlm_empty_yolo" if accepted else "keep_yolo_empty",
                "final": state[idx],
                "yolo": yolo_value,
                "yolo_confidence": yolo_conf_by_cell.get(idx),
                "sam": sam_value,
                "vlm": vlm_value,
                "sam_reliable": sam_reliable,
            }
        )

    return state, decisions


def build_oracle_upper_bound_state(
    gt_state: Optional[BoardState],
    yolo_state: Optional[BoardState],
    fusion_state: Optional[BoardState],
    vlm_state: Optional[BoardState],
) -> Optional[BoardState]:
    """Return a GT-assisted upper bound over available candidates.

    This is not an executable perception method because it uses validation labels.
    It is useful only to show whether SAM/VLM candidate evidence contains
    recoverable information that a better non-oracle fusion policy could exploit.
    """
    if gt_state is None or yolo_state is None:
        return None
    state = list(yolo_state)
    for idx in range(9):
        candidates = [yolo_state[idx]]
        if fusion_state is not None:
            candidates.append(fusion_state[idx])
        if vlm_state is not None:
            candidates.append(vlm_state[idx])
        if gt_state[idx] in candidates:
            state[idx] = gt_state[idx]
    return state


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def piece_grid_assignment_metrics(
    gt_pieces: List[Dict[str, Any]],
    pred_pieces: List[Dict[str, Any]],
    iou_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Evaluate whether detected piece instances are assigned to correct cells.

    Each GT piece is matched to one predicted piece of the same class by bbox IoU.
    Unmatched GT pieces count as incorrect. The metric corresponds to the paper's
    grid-assignment accuracy for instance-level YOLO/SAM outputs.
    """
    if not gt_pieces:
        false_positive = len(pred_pieces) > 0
        return {
            "valid": True,
            "grid_assignment_accuracy": 0.0 if false_positive else 1.0,
            "correct_assignments": 0,
            "matched_pieces": 0,
            "gt_piece_count": 0,
            "unmatched_gt_count": 0,
            "false_positive_piece_count": len(pred_pieces),
        }

    used_pred: set[int] = set()
    correct = 0
    matched = 0
    unmatched = 0

    for gt_piece in gt_pieces:
        best_idx = None
        best_iou = 0.0
        for pred_idx, pred_piece in enumerate(pred_pieces):
            if pred_idx in used_pred:
                continue
            if pred_piece.get("class_name") != gt_piece.get("class_name"):
                continue
            if "bbox_xyxy" not in pred_piece or "bbox_xyxy" not in gt_piece:
                continue
            iou = box_iou(pred_piece["bbox_xyxy"], gt_piece["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_idx = pred_idx
        if best_idx is None or best_iou < iou_threshold:
            unmatched += 1
            continue
        used_pred.add(best_idx)
        matched += 1
        pred_cell = pred_pieces[best_idx].get("cell_index")
        if pred_cell is not None and int(pred_cell) == int(gt_piece["cell_index"]):
            correct += 1

    return {
        "valid": True,
        "grid_assignment_accuracy": correct / len(gt_pieces),
        "correct_assignments": correct,
        "matched_pieces": matched,
        "gt_piece_count": len(gt_pieces),
        "unmatched_gt_count": unmatched,
        "false_positive_piece_count": max(0, len(pred_pieces) - len(used_pred)),
    }


def state_grid_assignment_metrics(
    gt_pieces: List[Dict[str, Any]],
    pred_state: Optional[BoardState],
) -> Dict[str, Any]:
    """Evaluate occupied GT cells against a predicted board state.

    This makes final/fused board-state outputs comparable with instance-level
    piece assignment: every GT piece is correct when its GT cell has the right
    predicted piece value.
    """
    if pred_state is None:
        return {
            "valid": False,
            "grid_assignment_accuracy": 0.0,
            "correct_assignments": 0,
            "gt_piece_count": len(gt_pieces),
        }
    if not gt_pieces:
        return {
            "valid": True,
            "grid_assignment_accuracy": 1.0 if all(v == 0 for v in pred_state) else 0.0,
            "correct_assignments": 0,
            "gt_piece_count": 0,
        }
    correct = 0
    for piece in gt_pieces:
        cell_index = int(piece["cell_index"])
        if pred_state[cell_index] == int(piece["piece_value"]):
            correct += 1
    return {
        "valid": True,
        "grid_assignment_accuracy": correct / len(gt_pieces),
        "correct_assignments": correct,
        "gt_piece_count": len(gt_pieces),
    }


def winning_values(state: Optional[BoardState]) -> List[int]:
    """Return all piece values with at least one winning line."""
    if state is None or len(state) != 9:
        return []
    win_lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]
    winners: List[int] = []
    for a, b, c in win_lines:
        value = int(state[a])
        if value != 0 and value == int(state[b]) == int(state[c]) and value not in winners:
            winners.append(value)
    return winners


def is_rule_legal_state(state: Optional[BoardState]) -> bool:
    """Check basic tic-tac-toe legality without assuming who moved first."""
    if state is None or len(state) != 9:
        return False
    if any(v not in (0, 1, 2) for v in state):
        return False
    yellow_count = state.count(1)
    blue_count = state.count(2)
    if abs(yellow_count - blue_count) > 1:
        return False
    winners = winning_values(state)
    if len(winners) > 1:
        return False
    if winners:
        winner = winners[0]
        loser = 2 if winner == 1 else 1
        if state.count(winner) < state.count(loser):
            return False
    return True


def low_confidence_sample_info(
    result: Dict[str, Any],
    yolo_pieces: List[Dict[str, Any]],
    fusion_pieces: List[Dict[str, Any]],
    final_decisions: List[Dict[str, Any]],
    low_conf_threshold: float,
) -> Dict[str, Any]:
    """Return image-level low-confidence/conflict flags used by paper metrics."""
    reasons: List[str] = []
    board_conf = result.get("board", {}).get("confidence") if isinstance(result.get("board"), dict) else None
    if board_conf is None:
        reasons.append("board_confidence_missing_or_fallback")
    elif float(board_conf) < low_conf_threshold:
        reasons.append("board_low_confidence")

    low_yolo_cells = [
        int(piece["cell_index"])
        for piece in yolo_pieces
        if float(piece.get("confidence", piece.get("yolo_confidence", 1.0))) < low_conf_threshold
    ]
    if low_yolo_cells:
        reasons.append("piece_low_confidence")

    fallback_cells = [
        int(piece["cell_index"])
        for piece in fusion_pieces
        if str(piece.get("mask_source", "")) != "sam"
    ]
    if fallback_cells:
        reasons.append("mask_fallback")

    color_mismatch_cells = [
        int(piece["cell_index"])
        for piece in fusion_pieces
        if not piece.get("color", {}).get("verified", False)
    ]
    if color_mismatch_cells:
        reasons.append("color_mismatch")

    parsed = result.get("vlm", {}).get("parsed")
    uncertain_cells: List[int] = []
    abnormal_cells: List[Any] = []
    if isinstance(parsed, dict):
        uncertain_cells = [int(v) for v in parsed.get("uncertain_cells", []) if isinstance(v, int)]
        abnormal_cells = parsed.get("abnormal_cells", []) if isinstance(parsed.get("abnormal_cells", []), list) else []
    if uncertain_cells:
        reasons.append("vlm_uncertain")
    if abnormal_cells:
        reasons.append("vlm_abnormal")

    decision_flags = [
        item for item in final_decisions
        if any(key in str(item.get("decision", "")) for key in ("disagree", "correct", "clear", "flagged"))
    ]
    if decision_flags:
        reasons.append("fusion_decision_conflict")

    return {
        "detected": bool(reasons),
        "reasons": sorted(set(reasons)),
        "low_yolo_cells": low_yolo_cells,
        "fallback_cells": fallback_cells,
        "color_mismatch_cells": color_mismatch_cells,
        "vlm_uncertain_cells": uncertain_cells,
        "vlm_abnormal_cells": abnormal_cells,
    }


def compare_states(gt: Optional[BoardState], pred: Optional[BoardState]) -> Dict[str, Any]:
    """Compute state comparison metrics for one image."""
    if gt is None or pred is None:
        return {
            "valid": False,
            "cell_accuracy": 0.0,
            "exact_match": False,
            "correct_cells": 0,
            "total_cells": 9,
            "mismatch_cells": list(range(9)),
            "occupied_accuracy": 0.0,
        }

    correct = [i for i in range(9) if gt[i] == pred[i]]
    mismatch = [i for i in range(9) if gt[i] != pred[i]]
    occupied = [i for i in range(9) if gt[i] != 0]
    occupied_correct = [i for i in occupied if gt[i] == pred[i]]
    occupied_accuracy = len(occupied_correct) / len(occupied) if occupied else 1.0
    return {
        "valid": True,
        "cell_accuracy": len(correct) / 9.0,
        "exact_match": len(mismatch) == 0,
        "correct_cells": len(correct),
        "total_cells": 9,
        "mismatch_cells": mismatch,
        "occupied_accuracy": occupied_accuracy,
    }


def update_confusion(matrix: np.ndarray, gt: Optional[BoardState], pred: Optional[BoardState]):
    """Accumulate a 3x3 confusion matrix of gt value vs predicted value."""
    if gt is None or pred is None:
        return
    for gt_value, pred_value in zip(gt, pred):
        if gt_value in (0, 1, 2) and pred_value in (0, 1, 2):
            matrix[int(gt_value), int(pred_value)] += 1


def draw_gt_overlay(
    frontend: YoloSAMPerceptionFrontend,
    image: np.ndarray,
    gt_board: Optional[Dict[str, Any]],
    gt_pieces: List[Dict[str, Any]],
    path: Path,
):
    """Save ground-truth boxes and grid overlay."""
    canvas = image.copy()
    if gt_board is not None:
        board_box = tuple(int(v) for v in gt_board["bbox_xyxy"])
        frontend._draw_board_grid(canvas, board_box, "gt")
    for piece in gt_pieces:
        x1, y1, x2, y2 = [int(v) for v in piece["bbox_xyxy"]]
        color = (0, 255, 255) if piece["class_name"] == frontend.yellow_class else (255, 0, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cx, cy = piece["center_xy"]
        label = f"GT {piece['cell_index']}:{piece['class_name']}"
        cv2.circle(canvas, (int(cx), int(cy)), 4, color, -1)
        cv2.putText(canvas, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.imwrite(str(path), canvas)


def draw_compare_overlay(
    frontend: YoloSAMPerceptionFrontend,
    image: np.ndarray,
    gt_board: Optional[Dict[str, Any]],
    gt_pieces: List[Dict[str, Any]],
    yolo_pieces: List[Dict[str, Any]],
    sam_pieces: List[Dict[str, Any]],
    mismatch_cells: List[int],
    path: Path,
):
    """Save one image comparing GT, YOLO-only centers, and YOLO+SAM centroids."""
    canvas = image.copy()
    if gt_board is not None:
        board_box = tuple(int(v) for v in gt_board["bbox_xyxy"])
        frontend._draw_board_grid(canvas, board_box, "gt")

    for piece in gt_pieces:
        x1, y1, x2, y2 = [int(v) for v in piece["bbox_xyxy"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 255, 80), 2)

    for piece in yolo_pieces:
        cx, cy = piece["center_xy"]
        cv2.circle(canvas, (int(cx), int(cy)), 5, (255, 255, 255), -1)
        cv2.putText(
            canvas,
            f"Y{piece['cell_index']}",
            (int(cx) + 5, int(cy) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
        )

    for piece in sam_pieces:
        cx, cy = piece["centroid_xy"]
        color = (0, 255, 255) if piece["class_name"] == frontend.yellow_class else (255, 0, 0)
        cv2.circle(canvas, (int(cx), int(cy)), 5, color, -1)
        cv2.putText(
            canvas,
            f"S{piece['cell_index']}",
            (int(cx) - 12, int(cy) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )

    if mismatch_cells:
        text = "mismatch: " + ",".join(str(v) for v in mismatch_cells)
        cv2.putText(canvas, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    else:
        cv2.putText(canvas, "all cells matched", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 0), 2)
    cv2.imwrite(str(path), canvas)


def draw_state_compare(
    gt: Optional[BoardState],
    yolo: Optional[BoardState],
    fusion: Optional[BoardState],
    vlm: Optional[BoardState],
    conservative: Optional[BoardState],
    path: Path,
):
    """Save a compact 3x3 state comparison panel."""
    cell = 170
    margin = 24
    title_h = 52
    width = cell * 3 + margin * 2
    height = title_h + cell * 3 + margin
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "GT / YOLO / YOLO+SAM / VLM / Conservative", (margin, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            x1 = margin + col * cell
            y1 = title_h + row * cell
            x2 = x1 + cell
            y2 = y1 + cell
            gt_v = gt[idx] if gt is not None else -1
            yolo_v = yolo[idx] if yolo is not None else -1
            fusion_v = fusion[idx] if fusion is not None else -1
            vlm_v = vlm[idx] if vlm is not None else -1
            conservative_v = conservative[idx] if conservative is not None else -1
            ok = gt_v == conservative_v
            bg = (225, 255, 225) if ok else (225, 225, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 80, 80), 1)
            lines = [
                f"cell {idx}",
                f"GT: {STATE_SHORT.get(gt_v, '?')}",
                f"Y:  {STATE_SHORT.get(yolo_v, '?')}",
                f"YS: {STATE_SHORT.get(fusion_v, '?')}",
                f"V:  {STATE_SHORT.get(vlm_v, '?')}",
                f"C:  {STATE_SHORT.get(conservative_v, '?')}",
            ]
            for line_i, text in enumerate(lines):
                cv2.putText(
                    canvas,
                    text,
                    (x1 + 15, y1 + 24 + line_i * 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2 if line_i == 0 else 1,
                )
    cv2.imwrite(str(path), canvas)


def draw_confusion_matrix(matrix: np.ndarray, title: str, path: Path):
    """Save a 3x3 confusion matrix image."""
    cell = 120
    margin_left = 115
    margin_top = 90
    width = margin_left + cell * 3 + 40
    height = margin_top + cell * 3 + 70
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, title, (30, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    cv2.putText(canvas, "Pred", (margin_left + 100, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(canvas, "GT", (25, margin_top + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    max_value = max(1, int(matrix.max()))

    for i in range(3):
        cv2.putText(canvas, STATE_NAMES[i], (margin_left + i * cell + 18, margin_top - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, STATE_NAMES[i], (20, margin_top + i * cell + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for j in range(3):
            value = int(matrix[i, j])
            intensity = int(255 - 170 * (value / max_value))
            color = (255, intensity, intensity) if i != j else (intensity, 255, intensity)
            x1 = margin_left + j * cell
            y1 = margin_top + i * cell
            cv2.rectangle(canvas, (x1, y1), (x1 + cell, y1 + cell), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x1 + cell, y1 + cell), (80, 80, 80), 1)
            cv2.putText(canvas, str(value), (x1 + 42, y1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.imwrite(str(path), canvas)


def draw_metrics_summary(summary: Dict[str, Any], path: Path):
    """Save a bar-style metric summary image."""
    paper = summary.get("paper_metrics", {})
    metrics = [
        ("Paper board-state acc", paper.get("board_state_accuracy", summary["final_exact_match_rate"])),
        ("Paper cell classification acc", paper.get("cell_classification_accuracy", summary["final_cell_accuracy"])),
        ("Paper grid-assignment acc", paper.get("grid_assignment_accuracy", 0.0)),
        ("Paper illegal-state intercept", paper.get("illegal_state_interception_rate", 0.0) or 0.0),
        ("Paper low-conf sample rate", paper.get("low_confidence_sample_recognition_rate", 0.0)),
        ("Paper mask fallback rate", paper.get("mask_fallback_rate", 0.0)),
        ("YOLO cell acc", summary["yolo_cell_accuracy"]),
        ("YOLO+SAM raw cell acc", summary["fusion_cell_accuracy"]),
        ("VLM raw cell acc", summary["vlm_cell_accuracy"]),
        ("YOLO+SAM+Qwen-VLM final cell acc", summary["conservative_cell_accuracy"]),
        ("YOLO exact", summary["yolo_exact_match_rate"]),
        ("YOLO+SAM+Qwen-VLM final exact", summary["conservative_exact_match_rate"]),
    ]
    width = 860
    height = 95 + len(metrics) * 70
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Validation Metrics Summary", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    x0 = 260
    bar_w = 500
    for idx, (name, value) in enumerate(metrics):
        y = 90 + idx * 70
        cv2.putText(canvas, name, (30, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + 30), (230, 230, 230), -1)
        if "Paper" in name:
            color = (30, 120, 220)
        elif "Oracle" in name:
            color = (180, 0, 255)
        elif "final" in name:
            color = (0, 120, 255)
        elif "VLM" in name:
            color = (0, 120, 255)
        elif "SAM" in name:
            color = (0, 180, 0)
        else:
            color = (120, 120, 120)
        cv2.rectangle(canvas, (x0, y), (x0 + int(bar_w * value), y + 30), color, -1)
        cv2.putText(canvas, f"{value:.3f}", (x0 + bar_w + 20, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.imwrite(str(path), canvas)


def mean(values: List[float]) -> float:
    """Return arithmetic mean with empty-list fallback."""
    return float(sum(values) / len(values)) if values else 0.0


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    """Write per-image rows as CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    """Run the report builder."""
    args = parse_args()
    image_dir = resolve_project_path(args.images)
    label_dir = resolve_project_path(args.labels)
    out_dir = resolve_project_path(args.out)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    vlm_settings = load_vlm_settings(args.config, args)

    frontend = YoloSAMPerceptionFrontend(args.config)
    save_masks = not args.no_save_masks
    vlm = None
    if vlm_settings["enabled"]:
        print(
            "[info] Loading VLM: "
            f"{vlm_settings['model']} "
            f"(quantization={vlm_settings['quantization']}, "
            f"max_new_tokens={vlm_settings['max_new_tokens']})"
        )
        vlm = QwenBoardStateVLM(
            model_id=vlm_settings["model"],
            device_map=vlm_settings["device_map"],
            max_new_tokens=vlm_settings["max_new_tokens"],
            quantization=vlm_settings["quantization"],
        )

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    yolo_confusion = np.zeros((3, 3), dtype=np.int64)
    fusion_confusion = np.zeros((3, 3), dtype=np.int64)
    vlm_raw_confusion = np.zeros((3, 3), dtype=np.int64)
    final_confusion = np.zeros((3, 3), dtype=np.int64)
    oracle_confusion = np.zeros((3, 3), dtype=np.int64)

    for image_path in iter_images(image_dir, args.max_images):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[warn] failed to read image: {image_path}")
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        labels = load_yolo_labels(label_path, frame.shape[:2], frontend.class_names)
        gt_state, gt_board, gt_pieces, gt_messages = labels_to_board_state(frontend, labels)

        inference_start = time.perf_counter()
        result = frontend.process_image(
            frame=frame,
            image_name=image_path.name,
            output_dir=images_out,
            save_masks=save_masks,
            vlm=vlm,
            use_vlm_board_state=vlm is not None,
        )
        inference_time_sec = time.perf_counter() - inference_start

        yolo_state, yolo_pieces = detections_to_yolo_only_state(frontend, result)
        fusion_state = result.get("board_state_yolo_sam", result.get("board_state"))
        vlm_raw_state = result.get("board_state") if result.get("board_state_source") == "vlm" else None
        fusion_pieces = result.get("pieces", [])
        final_state, final_decisions = build_conservative_fusion_state(
            frontend=frontend,
            yolo_state=yolo_state,
            fusion_state=fusion_state,
            vlm_state=vlm_raw_state,
            yolo_pieces=yolo_pieces,
            fusion_pieces=fusion_pieces,
            vlm_result=result.get("vlm", {}),
        )
        oracle_state = build_oracle_upper_bound_state(
            gt_state=gt_state,
            yolo_state=yolo_state,
            fusion_state=fusion_state,
            vlm_state=vlm_raw_state,
        )

        yolo_metrics = compare_states(gt_state, yolo_state)
        fusion_metrics = compare_states(gt_state, fusion_state)
        vlm_raw_metrics = compare_states(gt_state, vlm_raw_state)
        final_metrics = compare_states(gt_state, final_state)
        oracle_metrics = compare_states(gt_state, oracle_state)
        yolo_grid_metrics = piece_grid_assignment_metrics(
            gt_pieces, yolo_pieces, iou_threshold=args.assignment_iou_threshold
        )
        fusion_grid_metrics = piece_grid_assignment_metrics(
            gt_pieces, fusion_pieces, iou_threshold=args.assignment_iou_threshold
        )
        final_grid_metrics = state_grid_assignment_metrics(gt_pieces, final_state)
        yolo_legal = is_rule_legal_state(yolo_state)
        fusion_legal = is_rule_legal_state(fusion_state)
        vlm_raw_legal = is_rule_legal_state(vlm_raw_state)
        final_legal = is_rule_legal_state(final_state)
        raw_state_candidates = [
            ("yolo", yolo_state, yolo_legal),
            ("yolo_sam_raw", fusion_state, fusion_legal),
        ]
        if vlm_raw_state is not None:
            raw_state_candidates.append(("vlm_raw", vlm_raw_state, vlm_raw_legal))
        raw_illegal_sources = [
            name for name, state, legal in raw_state_candidates
            if state is not None and not legal
        ]
        update_confusion(yolo_confusion, gt_state, yolo_state)
        update_confusion(fusion_confusion, gt_state, fusion_state)
        update_confusion(vlm_raw_confusion, gt_state, vlm_raw_state)
        update_confusion(final_confusion, gt_state, final_state)
        update_confusion(oracle_confusion, gt_state, oracle_state)

        gt_overlay_path = images_out / f"{image_path.stem}_gt_overlay.jpg"
        compare_overlay_path = images_out / f"{image_path.stem}_compare_overlay.jpg"
        state_compare_path = images_out / f"{image_path.stem}_state_compare.jpg"
        draw_gt_overlay(frontend, frame, gt_board, gt_pieces, gt_overlay_path)
        draw_compare_overlay(
            frontend,
            frame,
            gt_board,
            gt_pieces,
            yolo_pieces,
            fusion_pieces,
            final_metrics["mismatch_cells"],
            compare_overlay_path,
        )
        draw_state_compare(gt_state, yolo_state, fusion_state, vlm_raw_state, final_state, state_compare_path)

        color_mismatch_cells = [
            int(piece["cell_index"])
            for piece in fusion_pieces
            if not piece.get("color", {}).get("verified", False)
        ]
        sam_mask_count = sum(1 for piece in fusion_pieces if piece.get("mask_source") == "sam")
        fallback_mask_count = sum(1 for piece in fusion_pieces if piece.get("mask_source") != "sam")
        total_mask_count = sam_mask_count + fallback_mask_count
        mask_fallback_rate = fallback_mask_count / total_mask_count if total_mask_count else 0.0
        low_confidence_info = low_confidence_sample_info(
            result=result,
            yolo_pieces=yolo_pieces,
            fusion_pieces=fusion_pieces,
            final_decisions=final_decisions,
            low_conf_threshold=args.low_conf_threshold,
        )
        illegal_state_intercepted = bool(raw_illegal_sources) and final_legal
        row = {
            "image": image_path.name,
            "gt_state": json.dumps(gt_state, ensure_ascii=False),
            "yolo_state": json.dumps(yolo_state, ensure_ascii=False),
            "yolo_sam_raw_state": json.dumps(fusion_state, ensure_ascii=False),
            "vlm_raw_state": json.dumps(vlm_raw_state, ensure_ascii=False),
            "yolo_sam_vlm_final_state": json.dumps(final_state, ensure_ascii=False),
            "oracle_upper_bound_state": json.dumps(oracle_state, ensure_ascii=False),
            "yolo_cell_accuracy": yolo_metrics["cell_accuracy"],
            "fusion_cell_accuracy": fusion_metrics["cell_accuracy"],
            "vlm_cell_accuracy": vlm_raw_metrics["cell_accuracy"],
            "final_cell_accuracy": final_metrics["cell_accuracy"],
            "oracle_cell_accuracy": oracle_metrics["cell_accuracy"],
            "yolo_exact_match": yolo_metrics["exact_match"],
            "fusion_exact_match": fusion_metrics["exact_match"],
            "vlm_exact_match": vlm_raw_metrics["exact_match"],
            "final_exact_match": final_metrics["exact_match"],
            "oracle_exact_match": oracle_metrics["exact_match"],
            "yolo_occupied_accuracy": yolo_metrics["occupied_accuracy"],
            "fusion_occupied_accuracy": fusion_metrics["occupied_accuracy"],
            "vlm_occupied_accuracy": vlm_raw_metrics["occupied_accuracy"],
            "final_occupied_accuracy": final_metrics["occupied_accuracy"],
            "oracle_occupied_accuracy": oracle_metrics["occupied_accuracy"],
            "yolo_grid_assignment_accuracy": yolo_grid_metrics["grid_assignment_accuracy"],
            "fusion_grid_assignment_accuracy": fusion_grid_metrics["grid_assignment_accuracy"],
            "final_grid_assignment_accuracy": final_grid_metrics["grid_assignment_accuracy"],
            "paper_board_state_accuracy": 1.0 if final_metrics["exact_match"] else 0.0,
            "paper_cell_classification_accuracy": final_metrics["cell_accuracy"],
            "paper_grid_assignment_accuracy": final_grid_metrics["grid_assignment_accuracy"],
            "paper_illegal_state_intercepted": illegal_state_intercepted,
            "paper_illegal_raw_candidate_count": len(raw_illegal_sources),
            "paper_final_state_legal": final_legal,
            "paper_low_confidence_sample_detected": bool(low_confidence_info["detected"]),
            "paper_low_confidence_reasons": json.dumps(low_confidence_info["reasons"], ensure_ascii=False),
            "paper_inference_time_sec": inference_time_sec,
            "paper_mask_fallback_rate": mask_fallback_rate,
            "fusion_mismatch_cells": json.dumps(fusion_metrics["mismatch_cells"]),
            "vlm_mismatch_cells": json.dumps(vlm_raw_metrics["mismatch_cells"]),
            "final_mismatch_cells": json.dumps(final_metrics["mismatch_cells"]),
            "oracle_mismatch_cells": json.dumps(oracle_metrics["mismatch_cells"]),
            "final_decisions": json.dumps(final_decisions, ensure_ascii=False),
            "color_mismatch_cells": json.dumps(color_mismatch_cells),
            "gt_piece_count": len(gt_pieces),
            "fusion_piece_count": len(fusion_pieces),
            "sam_mask_count": sam_mask_count,
            "fallback_mask_count": fallback_mask_count,
            "total_mask_count": total_mask_count,
            "vlm_enabled": bool(result.get("vlm", {}).get("enabled", False)),
            "vlm_valid": bool(result.get("vlm", {}).get("valid", False)),
            "vlm_confidence": (
                result.get("vlm", {}).get("parsed", {}).get("confidence")
                if isinstance(result.get("vlm", {}).get("parsed"), dict)
                else None
            ),
            "vlm_uncertain_cells": json.dumps(
                result.get("vlm", {}).get("parsed", {}).get("uncertain_cells", [])
                if isinstance(result.get("vlm", {}).get("parsed"), dict)
                else []
            ),
            "vlm_abnormal_cells": json.dumps(
                result.get("vlm", {}).get("parsed", {}).get("abnormal_cells", [])
                if isinstance(result.get("vlm", {}).get("parsed"), dict)
                else []
            ),
            "gt_messages": ";".join(gt_messages),
            "yolo_legal_state": yolo_legal,
            "fusion_legal_state": fusion_legal,
            "vlm_raw_legal_state": vlm_raw_legal,
            "final_legal_state": final_legal,
            "raw_illegal_sources": json.dumps(raw_illegal_sources, ensure_ascii=False),
            "annotated_image_path": result.get("annotated_image_path"),
            "all_masks_image_path": result.get("all_masks_image_path"),
            "gt_overlay_path": str(gt_overlay_path),
            "compare_overlay_path": str(compare_overlay_path),
            "state_compare_path": str(state_compare_path),
        }
        rows.append(row)
        details.append(
            {
                "image": image_path.name,
                "gt": {
                    "state": gt_state,
                    "board": gt_board,
                    "pieces": gt_pieces,
                    "messages": gt_messages,
                },
                "yolo_only": {
                    "state": yolo_state,
                    "pieces": yolo_pieces,
                    "metrics": yolo_metrics,
                    "grid_assignment_metrics": yolo_grid_metrics,
                    "legal_state": yolo_legal,
                },
                "yolo_sam": {
                    "state": fusion_state,
                    "pieces": fusion_pieces,
                    "metrics": fusion_metrics,
                    "grid_assignment_metrics": fusion_grid_metrics,
                    "legal_state": fusion_legal,
                    "frontend_json": result.get("json_path"),
                },
                "qwen_vlm_raw": {
                    "state": vlm_raw_state,
                    "metrics": vlm_raw_metrics,
                    "legal_state": vlm_raw_legal,
                    "vlm": result.get("vlm"),
                    "frontend_json": result.get("json_path"),
                },
                "yolo_sam_qwen_vlm_final": {
                    "state": final_state,
                    "metrics": final_metrics,
                    "grid_assignment_metrics": final_grid_metrics,
                    "legal_state": final_legal,
                    "decisions": final_decisions,
                    "frontend_json": result.get("json_path"),
                },
                "paper_metrics": {
                    "board_state_accuracy": row["paper_board_state_accuracy"],
                    "cell_classification_accuracy": row["paper_cell_classification_accuracy"],
                    "grid_assignment_accuracy": row["paper_grid_assignment_accuracy"],
                    "illegal_state_intercepted": row["paper_illegal_state_intercepted"],
                    "raw_illegal_sources": raw_illegal_sources,
                    "low_confidence_sample": low_confidence_info,
                    "inference_time_sec": inference_time_sec,
                    "mask_fallback_rate": mask_fallback_rate,
                },
                "oracle_upper_bound": {
                    "state": oracle_state,
                    "metrics": oracle_metrics,
                    "note": "GT-assisted upper bound, not an executable method.",
                },
                "artifacts": {
                    "annotated_image": result.get("annotated_image_path"),
                    "all_masks_image": result.get("all_masks_image_path"),
                    "gt_overlay": str(gt_overlay_path),
                    "compare_overlay": str(compare_overlay_path),
                    "state_compare": str(state_compare_path),
                },
            }
        )
        print(
            f"{image_path.name}: "
            f"YOLO acc={yolo_metrics['cell_accuracy']:.3f} "
            f"YOLO+SAM raw acc={fusion_metrics['cell_accuracy']:.3f} "
            f"YOLO+SAM+VLM final acc={final_metrics['cell_accuracy']:.3f} "
            f"grid={final_grid_metrics['grid_assignment_accuracy']:.3f} "
            f"fallback={mask_fallback_rate:.3f} "
            f"time={inference_time_sec:.2f}s "
            f"final mismatch={final_metrics['mismatch_cells']}"
        )

    summary = {
        "image_count": len(rows),
        "yolo_cell_accuracy": mean([float(row["yolo_cell_accuracy"]) for row in rows]),
        "fusion_cell_accuracy": mean([float(row["fusion_cell_accuracy"]) for row in rows]),
        "vlm_cell_accuracy": mean([float(row["vlm_cell_accuracy"]) for row in rows]),
        "conservative_cell_accuracy": mean([float(row["final_cell_accuracy"]) for row in rows]),
        "final_cell_accuracy": mean([float(row["final_cell_accuracy"]) for row in rows]),
        "oracle_cell_accuracy": mean([float(row["oracle_cell_accuracy"]) for row in rows]),
        "yolo_exact_match_rate": mean([1.0 if row["yolo_exact_match"] else 0.0 for row in rows]),
        "fusion_exact_match_rate": mean([1.0 if row["fusion_exact_match"] else 0.0 for row in rows]),
        "vlm_exact_match_rate": mean([1.0 if row["vlm_exact_match"] else 0.0 for row in rows]),
        "conservative_exact_match_rate": mean([1.0 if row["final_exact_match"] else 0.0 for row in rows]),
        "final_exact_match_rate": mean([1.0 if row["final_exact_match"] else 0.0 for row in rows]),
        "oracle_exact_match_rate": mean([1.0 if row["oracle_exact_match"] else 0.0 for row in rows]),
        "yolo_occupied_accuracy": mean([float(row["yolo_occupied_accuracy"]) for row in rows]),
        "fusion_occupied_accuracy": mean([float(row["fusion_occupied_accuracy"]) for row in rows]),
        "vlm_occupied_accuracy": mean([float(row["vlm_occupied_accuracy"]) for row in rows]),
        "conservative_occupied_accuracy": mean([float(row["final_occupied_accuracy"]) for row in rows]),
        "final_occupied_accuracy": mean([float(row["final_occupied_accuracy"]) for row in rows]),
        "oracle_occupied_accuracy": mean([float(row["oracle_occupied_accuracy"]) for row in rows]),
        "yolo_grid_assignment_accuracy": mean([float(row["yolo_grid_assignment_accuracy"]) for row in rows]),
        "fusion_grid_assignment_accuracy": mean([float(row["fusion_grid_assignment_accuracy"]) for row in rows]),
        "final_grid_assignment_accuracy": mean([float(row["final_grid_assignment_accuracy"]) for row in rows]),
        "total_sam_masks": int(sum(int(row["sam_mask_count"]) for row in rows)),
        "total_fallback_masks": int(sum(int(row["fallback_mask_count"]) for row in rows)),
        "total_masks": int(sum(int(row["total_mask_count"]) for row in rows)),
        "vlm_enabled": vlm is not None,
        "vlm_model": vlm_settings["model"],
        "vlm_device_map": vlm_settings["device_map"],
        "vlm_max_new_tokens": vlm_settings["max_new_tokens"],
        "vlm_quantization": vlm_settings["quantization"],
        "vlm_config": vlm_settings,
    }
    total_masks = int(summary["total_masks"])
    illegal_candidate_images = [
        row for row in rows
        if int(row["paper_illegal_raw_candidate_count"]) > 0
    ]
    illegal_intercepted_images = [
        row for row in illegal_candidate_images
        if bool(row["paper_illegal_state_intercepted"])
    ]
    summary["paper_metrics"] = {
        "board_state_accuracy": summary["final_exact_match_rate"],
        "cell_classification_accuracy": summary["final_cell_accuracy"],
        "grid_assignment_accuracy": summary["final_grid_assignment_accuracy"],
        "illegal_state_interception_rate": (
            len(illegal_intercepted_images) / len(illegal_candidate_images)
            if illegal_candidate_images
            else None
        ),
        "illegal_candidate_image_count": len(illegal_candidate_images),
        "illegal_intercepted_image_count": len(illegal_intercepted_images),
        "final_legal_state_rate": mean([1.0 if row["paper_final_state_legal"] else 0.0 for row in rows]),
        "low_confidence_sample_recognition_rate": mean(
            [1.0 if row["paper_low_confidence_sample_detected"] else 0.0 for row in rows]
        ),
        "low_confidence_sample_count": int(
            sum(1 for row in rows if row["paper_low_confidence_sample_detected"])
        ),
        "average_inference_time_sec": mean([float(row["paper_inference_time_sec"]) for row in rows]),
        "mask_fallback_rate": (
            int(summary["total_fallback_masks"]) / total_masks
            if total_masks
            else 0.0
        ),
        "total_sam_masks": int(summary["total_sam_masks"]),
        "total_fallback_masks": int(summary["total_fallback_masks"]),
        "total_masks": total_masks,
    }
    for key, value in summary["paper_metrics"].items():
        summary[f"paper_{key}"] = value
    summary["final_cell_accuracy_gain_vs_yolo"] = (
        summary["final_cell_accuracy"] - summary["yolo_cell_accuracy"]
    )
    summary["final_exact_match_gain_vs_yolo"] = (
        summary["final_exact_match_rate"] - summary["yolo_exact_match_rate"]
    )
    summary["final_occupied_accuracy_gain_vs_yolo"] = (
        summary["final_occupied_accuracy"] - summary["yolo_occupied_accuracy"]
    )

    write_csv(out_dir / "summary.csv", rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({"summary": summary, "details": details}), f, ensure_ascii=False, indent=2)

    draw_confusion_matrix(yolo_confusion, "YOLO-only Confusion Matrix", out_dir / "confusion_yolo_only.jpg")
    draw_confusion_matrix(fusion_confusion, "YOLO+SAM Raw Confusion Matrix", out_dir / "confusion_yolo_sam_raw.jpg")
    draw_confusion_matrix(vlm_raw_confusion, "Qwen-VLM Raw Confusion Matrix", out_dir / "confusion_vlm_raw.jpg")
    draw_confusion_matrix(final_confusion, "YOLO+SAM+Qwen-VLM Final Confusion Matrix", out_dir / "confusion_yolo_sam_vlm_final.jpg")
    draw_confusion_matrix(oracle_confusion, "Oracle Upper Bound Confusion Matrix", out_dir / "confusion_oracle_upper_bound.jpg")
    draw_metrics_summary(summary, out_dir / "metrics_summary.jpg")

    print(f"report: {out_dir}")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
