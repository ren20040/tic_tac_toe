#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone YOLO + SAM perception frontend for tic-tac-toe.

This script is intended for fast first-stage validation without ROS:

    image(s) -> YOLO board/piece boxes -> SAM piece masks -> board_state JSON

Outputs include the board box, piece boxes, mask files, centroids, HSV color
ratios, board-cell indexes, an annotated image, and a length-9 board state.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

try:
    from ultralytics import FastSAM, SAM, YOLO
except ImportError:  # pragma: no cover - gives a clear runtime error.
    FastSAM = None
    SAM = None
    YOLO = None

from utils.config_loader import load_config, resolve_project_path


BoardState = List[int]
BoxXYXY = Tuple[int, int, int, int]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_VLM_QUANTIZATIONS = {"none", "8bit", "4bit"}


def normalize_vlm_quantization(value: Any = "none") -> str:
    """Normalize user/config quantization names for Qwen VLM loading."""
    quantization = str(value or "none").strip().lower()
    aliases = {
        "": "none",
        "false": "none",
        "off": "none",
        "no": "none",
        "0": "none",
        "int8": "8bit",
        "8": "8bit",
        "int4": "4bit",
        "4": "4bit",
        "nf4": "4bit",
    }
    quantization = aliases.get(quantization, quantization)
    if quantization not in VALID_VLM_QUANTIZATIONS:
        raise ValueError(
            f"Unsupported VLM quantization '{value}'. "
            "Choose one of: none, 8bit, 4bit."
        )
    return quantization


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run standalone YOLO+SAM tic-tac-toe perception."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, image directory, or camera index such as 0.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Config path.")
    parser.add_argument(
        "--out",
        default="runs/yolo_sam_frontend",
        help="Output directory for JSON, masks, and annotated images.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit images when --source is a directory. 0 means no limit.",
    )
    parser.add_argument(
        "--camera-frames",
        type=int,
        default=1,
        help="Number of frames to process when --source is a camera index.",
    )
    parser.add_argument("--show", action="store_true", help="Show annotated preview window.")
    parser.add_argument(
        "--no-save-masks",
        action="store_true",
        help="Do not write per-piece mask PNG files.",
    )
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help="Run Qwen2.5-VL after YOLO+SAM and append VLM board-state JSON.",
    )
    parser.add_argument(
        "--vlm-model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model id or local path for Qwen2.5-VL.",
    )
    parser.add_argument(
        "--vlm-device-map",
        default="auto",
        help="device_map passed to transformers from_pretrained, usually 'auto'.",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens for VLM JSON output.",
    )
    parser.add_argument(
        "--vlm-quant",
        choices=sorted(VALID_VLM_QUANTIZATIONS),
        default="none",
        help="Optional Qwen VLM quantization mode.",
    )
    parser.add_argument(
        "--use-vlm-board-state",
        action="store_true",
        help="Replace top-level board_state with VLM board_state when VLM output is valid.",
    )
    return parser.parse_args()


def resolve_model_path(model_path: str) -> str:
    """Resolve project-local model paths while preserving Ultralytics model names."""
    candidate = resolve_project_path(model_path)
    if candidate.exists():
        return str(candidate)
    return str(model_path)


def as_int_triplet(values: Sequence[int], name: str) -> Tuple[int, int, int]:
    """Convert a config value to a validated HSV triplet."""
    out = tuple(int(v) for v in values)
    if len(out) != 3:
        raise ValueError(f"{name} must contain three HSV values, got {values}")
    return out


def hsv_between(hsv: np.ndarray, lower: Sequence[int], upper: Sequence[int]) -> np.ndarray:
    """Return a boolean mask for HSV pixels inside an inclusive range."""
    hsv_arr = np.asarray(hsv)
    lower_arr = np.array(lower, dtype=hsv_arr.dtype)
    upper_arr = np.array(upper, dtype=hsv_arr.dtype)
    return np.all((hsv_arr >= lower_arr) & (hsv_arr <= upper_arr), axis=-1)


def box_area(box: BoxXYXY) -> float:
    """Return the area of an xyxy box."""
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1))


def resolve_runtime_device(device: Any) -> Any:
    """Use CPU automatically when config requests CUDA but CUDA is unavailable."""
    if device is None:
        return None

    device_text = str(device).strip().lower()
    if device_text in ("cpu", "mps"):
        return device_text

    # Ultralytics accepts 0 / "0" for CUDA. In CPU-only environments that raises
    # before prediction starts, so downgrade locally for this standalone tool.
    looks_like_cuda_index = device_text.isdigit() or "," in device_text
    if not looks_like_cuda_index:
        return device

    try:
        import torch
    except ImportError:
        print(f"[warn] torch is unavailable; using CPU instead of device={device}")
        return "cpu"

    if not torch.cuda.is_available():
        print(f"[warn] CUDA is unavailable; using CPU instead of device={device}")
        return "cpu"

    return device


def json_safe(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-safe Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first JSON object from model text."""
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def validate_vlm_board_json(parsed: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Validate the VLM JSON schema used by this script."""
    if parsed is None:
        return False, "No JSON object was parsed from VLM output."

    board_state = parsed.get("board_state")
    if not isinstance(board_state, list) or len(board_state) != 9:
        return False, "board_state must be a length-9 list."
    if any(value not in (0, 1, 2) for value in board_state):
        return False, "board_state values must be 0, 1, or 2."

    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        return False, "confidence must be a number between 0 and 1."

    for key in ("uncertain_cells", "abnormal_cells"):
        if key in parsed and not isinstance(parsed[key], list):
            return False, f"{key} must be a list."

    return True, None


class QwenBoardStateVLM:
    """Qwen2.5-VL wrapper for board-state arbitration."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        quantization: str = "none",
    ):
        """Load Qwen2.5-VL with optional dependencies."""
        quantization = normalize_vlm_quantization(quantization)
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-VL dependencies are missing. Install them with: "
                "pip install -r requirements-vlm.txt"
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.quantization = quantization
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_id)
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": device_map,
        }
        if quantization != "none":
            try:
                import torch
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen VLM quantization requires bitsandbytes. Install it with: "
                    "pip install bitsandbytes"
                ) from exc
            if quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs,
        )

    def infer(
        self,
        image_path: Path,
        perception_result: Dict[str, Any],
        use_vlm_board_state: bool = False,
    ) -> Dict[str, Any]:
        """Run VLM inference and return parsed JSON plus raw text."""
        prompt = build_vlm_prompt(perception_result)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = inputs.to(model_device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        parsed = extract_first_json_object(raw_text)
        valid, error = validate_vlm_board_json(parsed)
        output = {
            "enabled": True,
            "model": self.model_id,
            "prompt": prompt,
            "raw_text": raw_text,
            "parsed": parsed,
            "valid": valid,
            "error": error,
            "used_as_top_level_board_state": False,
        }
        if valid and use_vlm_board_state:
            output["used_as_top_level_board_state"] = True
        return output


def build_vlm_prompt(perception_result: Dict[str, Any]) -> str:
    """Build a strict JSON prompt for Qwen2.5-VL board-state arbitration."""
    prompt_payload = {
        "index_layout": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        "cell_values": {
            "empty": 0,
            "yellow_piece": 1,
            "blue_piece": 2,
        },
        "yolo_sam_preliminary_board_state": perception_result.get("board_state"),
        "board": perception_result.get("board"),
        "pieces": [
            {
                "cell_index": piece.get("cell_index"),
                "class_name": piece.get("class_name"),
                "piece_value": piece.get("piece_value"),
                "yolo_confidence": piece.get("yolo_confidence"),
                "bbox_xyxy": piece.get("bbox_xyxy"),
                "mask_source": piece.get("mask_source"),
                "mask_bbox_xyxy": piece.get("mask_bbox_xyxy"),
                "mask_area": piece.get("mask_area"),
                "centroid_xy": piece.get("centroid_xy"),
                "color": piece.get("color"),
            }
            for piece in perception_result.get("pieces", [])
        ],
        "raw_yolo_detections": perception_result.get("detections", {}),
    }

    return (
        "You are a physical tic-tac-toe board-state arbitration model.\n"
        "Use the image and the YOLO+SAM structured evidence to output ONLY one "
        "valid JSON object. Do not output markdown or extra text.\n\n"
        "Task:\n"
        "1. Determine the 3x3 board_state using values 0=empty, "
        "1=yellow_piece, 2=blue_piece.\n"
        "2. Use the cell index layout [[0,1,2],[3,4,5],[6,7,8]].\n"
        "3. Use YOLO/SAM evidence as the primary source. Use the image to catch "
        "obvious errors, ambiguity, color mismatch, duplicate detections, or pieces "
        "assigned to the wrong cell.\n"
        "4. If a cell is ambiguous, keep the most likely value and include the cell "
        "in uncertain_cells.\n"
        "5. Report abnormal_cells for suspicious cells, conflicts, low-confidence "
        "detections, color mismatch, missing board, duplicate pieces, or impossible "
        "geometry.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "board_state": [0,0,0,0,0,0,0,0,0],\n'
        '  "confidence": 0.0,\n'
        '  "uncertain_cells": [0],\n'
        '  "abnormal_cells": [\n'
        '    {"cell_index": 0, "type": "low_confidence", "reason": "short reason"}\n'
        "  ],\n"
        '  "corrections": [\n'
        '    {"cell_index": 0, "from": 0, "to": 1, "reason": "short reason"}\n'
        "  ],\n"
        '  "summary": "short explanation"\n'
        "}\n\n"
        "YOLO+SAM structured evidence:\n"
        f"{json.dumps(json_safe(prompt_payload), ensure_ascii=False, indent=2)}"
    )


class YoloSAMPerceptionFrontend:
    """YOLO + SAM perception pipeline for board-state extraction."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Load config and model weights."""
        self.cfg = load_config(config_path)
        self.yolo_cfg = self.cfg["yolo"]
        self.sam_cfg = self.cfg.get("sam", {})
        self.hybrid_cfg = self.cfg.get("yolo_sam", {})
        self.vision_cfg = self.cfg.get("vision", {})
        self.game_cfg = self.cfg.get("game", {})

        self.yolo_model_path = resolve_model_path(str(self.yolo_cfg["model_path"]))
        self.yolo_conf = float(self.yolo_cfg.get("conf", 0.25))
        self.yolo_device = resolve_runtime_device(self.yolo_cfg.get("device", 0))

        raw_class_names = self.yolo_cfg["class_names"]
        self.class_names = {int(k): str(v) for k, v in raw_class_names.items()}
        self.board_class = self.class_names.get(0, "board")
        self.yellow_class = self.class_names.get(1, "yellow_piece")
        self.blue_class = self.class_names.get(2, "blue_piece")

        self.yellow_value = int(self.game_cfg.get("yellow_value", 1))
        self.blue_value = int(self.game_cfg.get("blue_value", 2))
        self.empty_value = int(self.game_cfg.get("empty_value", 0))

        self.min_board_conf = float(self.vision_cfg.get("min_board_conf", self.yolo_conf))
        self.min_piece_conf = float(self.vision_cfg.get("min_piece_conf", self.yolo_conf))
        self.row_ratios = [float(v) for v in self.vision_cfg.get("row_ratios", [1.0, 1.5, 2.0])]
        if len(self.row_ratios) != 3 or sum(self.row_ratios) <= 0:
            raise ValueError("vision.row_ratios must contain three positive values")
        self.trapezoid_top_width_ratio = float(
            self.vision_cfg.get("trapezoid_top_width_ratio", 2.0 / 3.0)
        )

        self.sam_model_type = str(
            self.hybrid_cfg.get("sam_model_type", self.sam_cfg.get("model_type", "fastsam"))
        ).lower()
        default_sam_model = (
            "FastSAM-s.pt"
            if self.sam_model_type in ("fastsam", "fast_sam", "fast")
            else "sam_b.pt"
        )
        self.sam_model_path = resolve_model_path(
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
        self.sam_device = resolve_runtime_device(
            self.hybrid_cfg.get("sam_device", self.sam_cfg.get("device", self.yolo_device))
        )
        self.sam_retina_masks = bool(self.hybrid_cfg.get("sam_retina_masks", True))
        self.mask_threshold = float(
            self.hybrid_cfg.get("mask_threshold", self.sam_cfg.get("mask_threshold", 0.5))
        )

        self.board_box_config = self.hybrid_cfg.get("board_box", self.sam_cfg.get("board_box"))
        self.mask_match_min_box_coverage = float(
            self.hybrid_cfg.get("mask_match_min_box_coverage", 0.08)
        )
        self.mask_match_min_mask_fraction = float(
            self.hybrid_cfg.get("mask_match_min_mask_fraction", 0.20)
        )
        self.piece_box_expand_ratio = float(self.hybrid_cfg.get("piece_box_expand_ratio", 0.15))
        self.min_piece_area_ratio = float(
            self.hybrid_cfg.get("min_piece_area_ratio", self.sam_cfg.get("min_piece_area_ratio", 0.02))
        )
        self.max_piece_area_ratio = float(
            self.hybrid_cfg.get("max_piece_area_ratio", self.sam_cfg.get("max_piece_area_ratio", 0.75))
        )
        self.fallback_to_yolo_center = bool(self.hybrid_cfg.get("fallback_to_yolo_center", True))

        self.yellow_hsv_lower = as_int_triplet(
            self.hybrid_cfg.get(
                "yellow_hsv_lower",
                self.sam_cfg.get("yellow_hsv_lower", [15, 45, 50]),
            ),
            "yolo_sam.yellow_hsv_lower",
        )
        self.yellow_hsv_upper = as_int_triplet(
            self.hybrid_cfg.get(
                "yellow_hsv_upper",
                self.sam_cfg.get("yellow_hsv_upper", [40, 255, 255]),
            ),
            "yolo_sam.yellow_hsv_upper",
        )
        self.blue_hsv_lower = as_int_triplet(
            self.hybrid_cfg.get(
                "blue_hsv_lower",
                self.sam_cfg.get("blue_hsv_lower", [90, 45, 40]),
            ),
            "yolo_sam.blue_hsv_lower",
        )
        self.blue_hsv_upper = as_int_triplet(
            self.hybrid_cfg.get(
                "blue_hsv_upper",
                self.sam_cfg.get("blue_hsv_upper", [135, 255, 255]),
            ),
            "yolo_sam.blue_hsv_upper",
        )
        self.min_color_ratio = float(
            self.hybrid_cfg.get("min_color_ratio", self.sam_cfg.get("min_color_ratio", 0.18))
        )

        self.yolo_model = self._load_yolo_model()
        self.sam_model = self._load_sam_model()

    def _load_yolo_model(self):
        """Load YOLO detector."""
        if YOLO is None:
            raise RuntimeError("ultralytics YOLO is not available. Install ultralytics.")
        return YOLO(str(self.yolo_model_path))

    def _load_sam_model(self):
        """Load SAM/FastSAM model."""
        if self.sam_model_type in ("fastsam", "fast_sam", "fast"):
            if FastSAM is None:
                raise RuntimeError("ultralytics FastSAM is not available. Install ultralytics.")
            return FastSAM(self.sam_model_path)
        if self.sam_model_type in ("sam", "segment_anything"):
            if SAM is None:
                raise RuntimeError("ultralytics SAM is not available. Install ultralytics.")
            return SAM(self.sam_model_path)
        raise ValueError("yolo_sam.sam_model_type must be 'fastsam' or 'sam'")

    def process_image(
        self,
        frame: np.ndarray,
        image_name: str,
        output_dir: Path,
        save_masks: bool = True,
        vlm: Optional[QwenBoardStateVLM] = None,
        use_vlm_board_state: bool = False,
    ) -> Dict[str, Any]:
        """Run perception and save outputs for one BGR image."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_name).stem
        original_path = output_dir / f"{stem}_original.jpg"
        cv2.imwrite(str(original_path), frame)

        debug_image = frame.copy()
        board_candidates, piece_candidates = self._predict_yolo(frame)
        board_info = self._select_board(frame, board_candidates)

        if board_info is None:
            result = {
                "image": image_name,
                "board": None,
                "board_state": None,
                "pieces": [],
                "message": "No board detected and no fallback board_box configured.",
                "detections": {
                    "boards": self._strip_masks(board_candidates),
                    "pieces": self._strip_masks(piece_candidates),
                },
                "original_image_path": str(original_path),
                "board_state_source": "none",
            }
            self._append_vlm_result(result, original_path, vlm, use_vlm_board_state)
            self._write_outputs(output_dir, image_name, debug_image, result)
            return result

        board_box = tuple(board_info["bbox_xyxy"])
        sam_masks = self._predict_sam_masks(frame) if piece_candidates else []
        board_state: BoardState = [self.empty_value] * 9
        cell_best_score = [-1.0] * 9
        accepted: List[Dict[str, Any]] = []

        mask_dir = output_dir / "masks"
        if save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)

        for piece_id, piece in enumerate(piece_candidates):
            candidate = self._build_piece_candidate(frame, piece, board_box, sam_masks)
            if candidate is None:
                continue
            cell_index = int(candidate["cell_index"])
            score = float(candidate["score"])
            if score <= cell_best_score[cell_index]:
                continue
            cell_best_score[cell_index] = score
            board_state[cell_index] = int(candidate["piece_value"])
            candidate["piece_id"] = piece_id
            accepted = [item for item in accepted if int(item["cell_index"]) != cell_index]
            accepted.append(candidate)

        accepted.sort(key=lambda item: int(item["cell_index"]))
        if save_masks:
            for candidate in accepted:
                mask = candidate["_mask"]
                mask_name = (
                    f"{Path(image_name).stem}_cell_{candidate['cell_index']}_"
                    f"{candidate['class_name']}_mask.png"
                )
                mask_path = mask_dir / mask_name
                cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
                candidate["mask_path"] = str(mask_path)
        else:
            for candidate in accepted:
                candidate["mask_path"] = None

        self._draw_debug_image(debug_image, board_info, piece_candidates, accepted)
        all_masks_path = self._write_all_masks_preview(
            output_dir=output_dir,
            image_name=image_name,
            frame=frame,
            accepted=accepted,
        )
        for candidate in accepted:
            candidate.pop("_mask", None)

        result = {
            "image": image_name,
            "board": board_info,
            "board_state": board_state,
            "board_state_source": "yolo_sam",
            "pieces": accepted,
            "detections": {
                "boards": self._strip_masks(board_candidates),
                "pieces": self._strip_masks(piece_candidates),
            },
            "original_image_path": str(original_path),
            "all_masks_image_path": str(all_masks_path),
        }
        self._append_vlm_result(result, original_path, vlm, use_vlm_board_state)
        self._write_outputs(output_dir, image_name, debug_image, result)
        return result

    def _predict_yolo(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return YOLO board and piece candidates."""
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
            item = {
                "class_id": cls_id,
                "class_name": name,
                "confidence": conf,
                "bbox_xyxy": list(xyxy),
            }
            if name == self.board_class and conf >= self.min_board_conf:
                board_candidates.append(item)
            elif name in (self.yellow_class, self.blue_class) and conf >= self.min_piece_conf:
                piece_candidates.append(item)
        return board_candidates, piece_candidates

    def _select_board(
        self,
        frame: np.ndarray,
        board_candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Select the highest confidence board or configured fallback box."""
        if board_candidates:
            board = max(board_candidates, key=lambda item: float(item["confidence"]))
            return {
                "bbox_xyxy": list(board["bbox_xyxy"]),
                "confidence": float(board["confidence"]),
                "source": "yolo",
            }

        fallback = self._get_config_board_box(frame)
        if fallback is None:
            return None
        return {
            "bbox_xyxy": list(fallback),
            "confidence": None,
            "source": "config",
        }

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
        """Build one output piece from YOLO class and SAM-refined mask."""
        piece_box = tuple(int(v) for v in piece["bbox_xyxy"])
        piece_name = str(piece["class_name"])
        piece_value = self.yellow_value if piece_name == self.yellow_class else self.blue_value

        match = self._find_matching_piece_mask(masks, piece_box, board_box)
        if match is not None:
            piece_mask, match_score = match
            mask_source = "sam"
            centroid = self._mask_centroid(piece_mask)
            score = float(piece["confidence"]) * (0.5 + 0.5 * match_score)
        elif self.fallback_to_yolo_center:
            piece_mask = self._box_region_mask(frame.shape[:2], piece_box)
            mask_source = "yolo_box_fallback"
            x1, y1, x2, y2 = piece_box
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            score = float(piece["confidence"]) * 0.65
        else:
            return None

        cell_index = self._point_to_cell(centroid[0], centroid[1], board_box)
        if cell_index is None:
            return None

        color_info = self._color_info(frame, piece_mask, piece_name)
        mask_bbox = self._mask_bbox(piece_mask)

        return {
            "class_id": int(piece["class_id"]),
            "class_name": piece_name,
            "piece_value": piece_value,
            "yolo_confidence": float(piece["confidence"]),
            "bbox_xyxy": list(piece_box),
            "mask_source": mask_source,
            "mask_bbox_xyxy": list(mask_bbox) if mask_bbox is not None else None,
            "mask_area": int(piece_mask.sum()),
            "centroid_xy": [float(centroid[0]), float(centroid[1])],
            "cell_index": int(cell_index),
            "score": float(score),
            "color": color_info,
            "_mask": piece_mask,
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
        box_area_value = max(1.0, float(box_mask.sum()))
        cell_area = max(1.0, box_area(board_box) / 9.0)

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
            box_coverage = intersection / box_area_value
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

    def _get_config_board_box(self, frame: np.ndarray) -> Optional[BoxXYXY]:
        """Return configured fallback board box."""
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

    def _color_info(self, frame: np.ndarray, mask: np.ndarray, yolo_class: str) -> Dict[str, Any]:
        """Return HSV color ratios and selected color label for a piece mask."""
        pixels = frame[mask]
        if len(pixels) == 0:
            return {
                "yolo_class": yolo_class,
                "hsv_label": "unknown",
                "yellow_ratio": 0.0,
                "blue_ratio": 0.0,
                "verified": False,
            }

        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        yellow_ratio = float(hsv_between(hsv, self.yellow_hsv_lower, self.yellow_hsv_upper).mean())
        blue_ratio = float(hsv_between(hsv, self.blue_hsv_lower, self.blue_hsv_upper).mean())
        if yellow_ratio >= blue_ratio and yellow_ratio >= self.min_color_ratio:
            hsv_label = self.yellow_class
        elif blue_ratio > yellow_ratio and blue_ratio >= self.min_color_ratio:
            hsv_label = self.blue_class
        else:
            hsv_label = "unknown"
        return {
            "yolo_class": yolo_class,
            "hsv_label": hsv_label,
            "yellow_ratio": yellow_ratio,
            "blue_ratio": blue_ratio,
            "verified": hsv_label == yolo_class,
        }

    def _point_to_cell(self, cx: float, cy: float, board_box: BoxXYXY) -> Optional[int]:
        """Map a point in the detected board trapezoid to a 0-8 cell index."""
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
        board_info: Dict[str, Any],
        piece_candidates: List[Dict[str, Any]],
        accepted: List[Dict[str, Any]],
    ):
        """Draw board grid, YOLO boxes, mask overlays, centroids, and cell labels."""
        board_box = tuple(int(v) for v in board_info["bbox_xyxy"])
        for piece in piece_candidates:
            x1, y1, x2, y2 = piece["bbox_xyxy"]
            color = (0, 255, 255) if piece["class_name"] == self.yellow_class else (255, 0, 0)
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 1)

        overlay = debug_image.copy()
        for piece in accepted:
            color = (0, 255, 255) if piece["class_name"] == self.yellow_class else (255, 0, 0)
            mask_path = piece.get("_mask")
            if isinstance(mask_path, np.ndarray):
                overlay[mask_path] = color
            cx, cy = piece["centroid_xy"]
            cv2.circle(debug_image, (int(cx), int(cy)), 4, color, -1)
            cv2.putText(
                debug_image,
                f"{piece['class_name']}:{piece['cell_index']}:{piece['mask_source']}",
                (max(0, int(cx) - 55), max(15, int(cy) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
        cv2.addWeighted(overlay, 0.25, debug_image, 0.75, 0, dst=debug_image)
        self._draw_board_grid(debug_image, board_box, str(board_info["source"]))

    def _draw_board_grid(self, image: np.ndarray, board_box: BoxXYXY, source: str):
        """Draw board trapezoid, grid, and 0-8 cell IDs."""
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

    def _append_vlm_result(
        self,
        result: Dict[str, Any],
        image_path: Path,
        vlm: Optional[QwenBoardStateVLM],
        use_vlm_board_state: bool,
    ):
        """Append optional VLM arbitration output to a perception result."""
        if vlm is None:
            result["vlm"] = {"enabled": False}
            return

        try:
            vlm_result = vlm.infer(
                image_path=image_path,
                perception_result=result,
                use_vlm_board_state=use_vlm_board_state,
            )
        except Exception as exc:  # pragma: no cover - depends on large model runtime.
            result["vlm"] = {
                "enabled": True,
                "valid": False,
                "error": str(exc),
            }
            return

        result["vlm"] = vlm_result
        parsed = vlm_result.get("parsed") if vlm_result.get("valid") else None
        if use_vlm_board_state and isinstance(parsed, dict):
            result["board_state_yolo_sam"] = result.get("board_state")
            result["board_state"] = list(parsed["board_state"])
            result["board_state_source"] = "vlm"

    def _write_all_masks_preview(
        self,
        output_dir: Path,
        image_name: str,
        frame: np.ndarray,
        accepted: List[Dict[str, Any]],
    ) -> Path:
        """Save one preview image with all accepted masks overlaid together."""
        preview = frame.copy()
        overlay = frame.copy()
        palette = [
            (0, 255, 255),
            (255, 0, 0),
            (0, 180, 255),
            (255, 120, 0),
            (0, 255, 0),
            (180, 0, 255),
            (255, 255, 0),
            (0, 120, 255),
            (180, 255, 0),
        ]

        for piece in accepted:
            mask = piece.get("_mask")
            if not isinstance(mask, np.ndarray):
                continue

            cell_index = int(piece["cell_index"])
            color = palette[cell_index % len(palette)]
            overlay[mask] = color

            cx, cy = piece["centroid_xy"]
            label = f"{cell_index}:{piece['class_name']}:{piece['mask_source']}"
            cv2.circle(preview, (int(cx), int(cy)), 5, color, -1)
            cv2.putText(
                preview,
                label,
                (max(0, int(cx) - 70), max(18, int(cy) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
            )

        cv2.addWeighted(overlay, 0.45, preview, 0.55, 0, dst=preview)
        path = output_dir / f"{Path(image_name).stem}_all_masks.jpg"
        cv2.imwrite(str(path), preview)
        return path

    def _write_outputs(
        self,
        output_dir: Path,
        image_name: str,
        debug_image: np.ndarray,
        result: Dict[str, Any],
    ):
        """Write annotated image and JSON sidecar."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_name).stem
        annotated_path = output_dir / f"{stem}_annotated.jpg"
        json_path = output_dir / f"{stem}.json"
        cv2.imwrite(str(annotated_path), debug_image)
        result["annotated_image_path"] = str(annotated_path)
        result["json_path"] = str(json_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(result), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _box_region_mask(shape: Tuple[int, int], box: BoxXYXY) -> np.ndarray:
        """Return a boolean image mask for an xyxy box."""
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
        """Expand an xyxy box by a width/height ratio and clamp to frame size."""
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
        """Return mask centroid as (x, y)."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return 0.0, 0.0
        return float(xs.mean()), float(ys.mean())

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Optional[BoxXYXY]:
        """Return mask bbox or None for empty masks."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _strip_masks(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return items without private mask arrays."""
        clean = []
        for item in items:
            clean.append({k: v for k, v in item.items() if not k.startswith("_")})
        return clean


def iter_image_paths(source: Path, max_images: int = 0) -> Iterable[Path]:
    """Yield image paths from a file or directory source."""
    if source.is_file():
        yield source
        return
    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")

    count = 0
    for path in sorted(source.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        yield path
        count += 1
        if max_images > 0 and count >= max_images:
            break


def is_camera_source(source: str) -> bool:
    """Return True when source looks like an integer camera index."""
    try:
        int(source)
    except ValueError:
        return False
    return True


def process_camera(
    frontend: YoloSAMPerceptionFrontend,
    source: str,
    output_dir: Path,
    frames: int,
    save_masks: bool,
    show: bool,
    vlm: Optional[QwenBoardStateVLM] = None,
    use_vlm_board_state: bool = False,
) -> List[Dict[str, Any]]:
    """Capture and process frames from a local camera index."""
    camera_index = int(source)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}")

    results = []
    try:
        for idx in range(max(1, frames)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read camera frame {idx}")
            image_name = f"camera_{camera_index}_{idx:04d}.jpg"
            result = frontend.process_image(
                frame,
                image_name,
                output_dir,
                save_masks,
                vlm=vlm,
                use_vlm_board_state=use_vlm_board_state,
            )
            results.append(result)
            print(json.dumps(json_safe(result), ensure_ascii=False))
            if show:
                annotated = cv2.imread(result["annotated_image_path"])
                cv2.imshow("yolo_sam_frontend", annotated)
                cv2.waitKey(1)
    finally:
        cap.release()
    return results


def main():
    """Run the standalone frontend."""
    args = parse_args()
    output_dir = resolve_project_path(args.out)
    frontend = YoloSAMPerceptionFrontend(args.config)
    save_masks = not args.no_save_masks
    vlm = None
    if args.enable_vlm:
        print(f"[info] Loading VLM: {args.vlm_model}")
        vlm = QwenBoardStateVLM(
            model_id=args.vlm_model,
            device_map=args.vlm_device_map,
            max_new_tokens=args.vlm_max_new_tokens,
            quantization=args.vlm_quant,
        )

    if is_camera_source(args.source):
        process_camera(
            frontend,
            args.source,
            output_dir,
            args.camera_frames,
            save_masks,
            args.show,
            vlm=vlm,
            use_vlm_board_state=args.use_vlm_board_state,
        )
        return

    source_path = resolve_project_path(args.source)
    results = []
    for image_path in iter_image_paths(source_path, args.max_images):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        result = frontend.process_image(
            frame,
            image_path.name,
            output_dir,
            save_masks,
            vlm=vlm,
            use_vlm_board_state=args.use_vlm_board_state,
        )
        results.append(result)
        print(
            f"{image_path.name}: board_state={result.get('board_state')} "
            f"pieces={len(result.get('pieces', []))} json={result.get('json_path')}"
        )
        if args.show:
            annotated = cv2.imread(result["annotated_image_path"])
            cv2.imshow("yolo_sam_frontend", annotated)
            cv2.waitKey(1)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(results), f, ensure_ascii=False, indent=2)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
