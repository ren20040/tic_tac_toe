#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert camera images into tic-tac-toe board states with YOLO."""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO

from utils.config_loader import load_config, resolve_project_path


BoardState = List[int]
BoxXYXY = Tuple[int, int, int, int]


class TicTacToeVisionState:
    """Subscribe to camera images and keep the latest detected board state.

    Notes:
        This class does not call ``rospy.init_node()``. The main controller owns
        ROS node initialization.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the YOLO detector and camera subscriber.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            KeyError: If required config keys such as ``camera.topic``,
                ``yolo.model_path``, or ``yolo.class_names`` are missing.
        """
        self.cfg = load_config(config_path)

        self.camera_topic = self.cfg["camera"]["topic"]

        self.model_path = resolve_project_path(self.cfg["yolo"]["model_path"])
        self.conf = float(self.cfg["yolo"].get("conf", 0.25))
        self.device = self.cfg["yolo"].get("device", 0)

        raw_class_names = self.cfg["yolo"]["class_names"]
        self.class_names = {int(k): str(v) for k, v in raw_class_names.items()}

        self.board_class = self.class_names.get(0, "board")
        self.yellow_class = self.class_names.get(1, "yellow_piece")
        self.blue_class = self.class_names.get(2, "blue_piece")

        vision_cfg = self.cfg.get("vision", {})
        self.min_board_conf = float(vision_cfg.get("min_board_conf", self.conf))
        self.min_piece_conf = float(vision_cfg.get("min_piece_conf", self.conf))
        self.show_window = bool(vision_cfg.get("show_window", False))
        self.window_name = vision_cfg.get("window_name", "tic_tac_toe_vision")
        self.row_ratios = [
            float(value)
            for value in vision_cfg.get("row_ratios", [1.0, 1.5, 2.0])
        ]
        if len(self.row_ratios) != 3 or sum(self.row_ratios) <= 0:
            raise ValueError("vision.row_ratios must contain three positive values")
        self.trapezoid_top_width_ratio = float(
            vision_cfg.get("trapezoid_top_width_ratio", 2.0 / 3.0)
        )

        self.bridge = CvBridge()
        self.model = YOLO(str(self.model_path))

        self.latest_board_state: Optional[BoardState] = None

        self.sub = rospy.Subscriber(
            self.camera_topic,
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("[TicTacToeVisionState] initialized")
        rospy.loginfo(f"[TicTacToeVisionState] camera_topic: {self.camera_topic}")
        rospy.loginfo(f"[TicTacToeVisionState] model_path: {self.model_path}")
        rospy.loginfo(f"[TicTacToeVisionState] class_names: {self.class_names}")

    def _image_callback(self, msg: Image):
        """Handle one ROS image message from the camera topic.

        Args:
            msg: ``sensor_msgs/Image`` message.

        Returns:
            None. The latest board state is stored on the instance.
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"[TicTacToeVisionState] cv_bridge error: {e}")
            return

        board_state, debug_image, _ = self.detect_from_cv_image(frame)

        if board_state is not None:
            self.latest_board_state = board_state
        else:
            # Clear stale data when the board is not detected in the current frame.
            self.latest_board_state = None

        if self.show_window and debug_image is not None:
            cv2.imshow(self.window_name, debug_image)
            cv2.waitKey(1)

    def detect_from_cv_image(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[BoardState], np.ndarray, Optional[BoxXYXY]]:
        """Run YOLO detection on one OpenCV image.

        Args:
            frame: OpenCV BGR image.

        Returns:
            tuple[Optional[BoardState], np.ndarray, Optional[BoxXYXY]]:
            ``board_state`` is a length-9 board state or ``None`` when no board
            is detected. ``debug_image`` contains drawn detections. ``board_box``
            is the board ``(x1, y1, x2, y2)`` box or ``None``.
        """
        debug_image = frame.copy()

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return None, debug_image, None

        class_id_to_name = self.class_names

        board_candidates: List[Dict[str, Any]] = []
        piece_candidates: List[Dict[str, Any]] = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = class_id_to_name.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            xyxy = (x1, y1, x2, y2)

            if name == self.board_class and conf >= self.min_board_conf:
                board_candidates.append(
                    {
                        "name": name,
                        "conf": conf,
                        "xyxy": xyxy,
                    }
                )

            elif (
                name in (self.yellow_class, self.blue_class)
                and conf >= self.min_piece_conf
            ):
                piece_candidates.append(
                    {
                        "name": name,
                        "conf": conf,
                        "xyxy": xyxy,
                    }
                )

        if not board_candidates:
            return None, debug_image, None

        # If multiple boards are detected, use the highest-confidence board box.
        board = max(board_candidates, key=lambda item: item["conf"])
        board_box = board["xyxy"]

        board_state = self._build_board_state_from_detections(
            board_box=board_box,
            pieces=piece_candidates,
            debug_image=debug_image,
        )

        self._draw_board_grid(debug_image, board_box)

        return board_state, debug_image, board_box

    def _build_board_state_from_detections(
        self,
        board_box: BoxXYXY,
        pieces: List[Dict[str, Any]],
        debug_image: Optional[np.ndarray] = None,
    ) -> BoardState:
        """Build a 3x3 board state from the board box and piece detections.

        Args:
            board_box: Board detection box ``(x1, y1, x2, y2)``.
            pieces: Piece detections. Each item contains ``name``, ``conf``, and
                ``xyxy``.
            debug_image: Optional image to draw piece boxes and cell IDs on.

        Returns:
            BoardState: Length-9 board list.

        Notes:
            The board index layout is:

                0 | 1 | 2
                3 | 4 | 5
                6 | 7 | 8
        """
        bx1, by1, bx2, by2 = board_box
        board_w = bx2 - bx1
        board_h = by2 - by1

        board_state = [0] * 9

        if board_w <= 0 or board_h <= 0:
            return board_state

        # Keep only the highest-confidence piece when multiple detections fall
        # into the same board cell.
        cell_best_conf = [-1.0] * 9

        for piece in pieces:
            px1, py1, px2, py2 = piece["xyxy"]
            name = piece["name"]
            conf = piece["conf"]

            cx = int((px1 + px2) / 2)
            cy = int((py1 + py2) / 2)

            # Ignore pieces whose center is outside the detected board box.
            if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
                continue

            relative_y = cy - by1
            t = relative_y / float(board_h)
            t = max(0.0, min(1.0, t))

            top_width_ratio = self.trapezoid_top_width_ratio
            bottom_width = float(board_w)
            top_width = bottom_width * top_width_ratio

            top_left_x = bx1 + (bottom_width - top_width) / 2.0
            top_right_x = top_left_x + top_width
            bottom_left_x = float(bx1)
            bottom_right_x = float(bx2)

            left_x = top_left_x + (bottom_left_x - top_left_x) * t
            right_x = top_right_x + (bottom_right_x - top_right_x) * t
            current_width = right_x - left_x

            if current_width <= 0:
                continue

            relative_x = (cx - left_x) / current_width
            relative_x = max(0.0, min(1.0, relative_x))

            col = int(relative_x * 3.0)
            col = max(0, min(2, col))

            row_total = sum(self.row_ratios)
            row_bound_1 = self.row_ratios[0] / row_total
            row_bound_2 = (self.row_ratios[0] + self.row_ratios[1]) / row_total

            if t < row_bound_1:
                row = 0
            elif t < row_bound_2:
                row = 1
            else:
                row = 2

            index = row * 3 + col

            if conf < cell_best_conf[index]:
                continue

            if name == self.yellow_class:
                board_state[index] = 1
                draw_color = (0, 255, 255)
            elif name == self.blue_class:
                board_state[index] = 2
                draw_color = (255, 0, 0)
            else:
                continue

            cell_best_conf[index] = conf

            if debug_image is not None:
                cv2.rectangle(debug_image, (px1, py1), (px2, py2), draw_color, 2)
                cv2.circle(debug_image, (cx, cy), 4, draw_color, -1)
                cv2.putText(
                    debug_image,
                    f"{name}:{index}",
                    (px1, max(0, py1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    draw_color,
                    2,
                )

        return board_state

    def _draw_board_grid(self, image: np.ndarray, board_box: BoxXYXY):
        """Draw the detected board box, 3x3 grid, and cell IDs on an image.

        Args:
            image: OpenCV BGR image modified in place.
            board_box: Board detection box ``(x1, y1, x2, y2)``.

        Returns:
            None.
        """
        bx1, by1, bx2, by2 = board_box
        board_w = bx2 - bx1
        board_h = by2 - by1

        if board_w <= 0 or board_h <= 0:
            return

        top_width_ratio = self.trapezoid_top_width_ratio
        bottom_width = float(board_w)
        top_width = bottom_width * top_width_ratio

        top_left_x = bx1 + (bottom_width - top_width) / 2.0
        top_right_x = top_left_x + top_width
        bottom_left_x = float(bx1)
        bottom_right_x = float(bx2)

        def edge_at(t: float):
            """Return the left edge, right edge, and y coordinate at row ratio t."""
            left_x = top_left_x + (bottom_left_x - top_left_x) * t
            right_x = top_right_x + (bottom_right_x - top_right_x) * t
            y = by1 + board_h * t
            return left_x, right_x, y

        outline = np.array(
            [
                [int(top_left_x), int(by1)],
                [int(top_right_x), int(by1)],
                [int(bottom_right_x), int(by2)],
                [int(bottom_left_x), int(by2)],
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [outline], True, (0, 255, 0), 2)
        cv2.putText(
            image,
            self.board_class,
            (bx1, max(0, by1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        for i in range(1, 3):
            fraction = i / 3.0
            top_x = top_left_x + top_width * fraction
            bottom_x = bottom_left_x + bottom_width * fraction
            cv2.line(
                image,
                (int(top_x), int(by1)),
                (int(bottom_x), int(by2)),
                (255, 255, 0),
                2,
            )

        row_total = sum(self.row_ratios)
        row_bounds = [
            0.0,
            self.row_ratios[0] / row_total,
            (self.row_ratios[0] + self.row_ratios[1]) / row_total,
            1.0,
        ]
        for t in row_bounds[1:3]:
            left_x, right_x, y = edge_at(t)
            cv2.line(
                image,
                (int(left_x), int(y)),
                (int(right_x), int(y)),
                (255, 255, 0),
                2,
            )

        # Draw 0~8 cell indexes for visual debugging.
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
        """Return the latest valid board state.

        Returns:
            Optional[BoardState]: Length-9 board state. Returns ``None`` if the
            board is not currently detected or no image has arrived.
        """
        return self.latest_board_state
