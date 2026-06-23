#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Joint-keyframe pick/place helper for the tic-tac-toe robot flow.

This module keeps the controller entry point used by ``main_control.py``:

    get_pick_place(config_path).place_piece_to_cell(vision_index)

The implementation intentionally bypasses Cartesian IK.  Every arm motion is a
pre-tested joint keyframe, and board-cell placement is selected by rotating the
waist plus publishing the corresponding right-arm joint target.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rospy
from kuavo_msgs.msg import (
    armTargetPoses,
    robotHandPosition,
    robotHeadMotionData,
    robotWaistControl,
)
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

from utils.config_loader import load_config


def _as_float_list(values: Sequence[float], length: Optional[int] = None) -> List[float]:
    """Convert values to floats and optionally validate list length."""
    out = [float(v) for v in values]
    if length is not None and len(out) != length:
        raise ValueError(f"Expected {length} values, got {len(out)}: {out}")
    return out


def _as_int_list(values: Sequence[int], length: Optional[int] = None) -> List[int]:
    """Convert values to integers and optionally validate list length."""
    out = [int(v) for v in values]
    if length is not None and len(out) != length:
        raise ValueError(f"Expected {length} values, got {len(out)}: {out}")
    return out


def _vision_index_to_row_col(vision_index: int, board_cols: int = 3) -> Tuple[int, int]:
    """Convert a 0-based board index to 1-based row and column."""
    index = int(vision_index)
    max_index = board_cols * board_cols - 1
    if not 0 <= index <= max_index:
        raise ValueError(f"vision_index must be 0~{max_index}, got {vision_index}")
    return index // board_cols + 1, index % board_cols + 1


class TicTacToePickPlace:
    """Execute tic-tac-toe pick/place moves with fixed joint keyframes."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Load config, connect ROS publishers/services, and move to ready pose."""
        self.cfg = load_config(config_path)
        self.pp_cfg = self.cfg["arm_pick_place"]
        self.arm_cfg = self.cfg["arm"]

        self.arm_joint_count = int(self.pp_cfg.get("arm_joint_count", 8))
        self.single_arm_joint_count = int(
            self.pp_cfg.get("single_arm_joint_count", self.arm_joint_count // 2)
        )
        if self.arm_joint_count != self.single_arm_joint_count * 2:
            raise ValueError(
                "arm_joint_count must equal single_arm_joint_count * 2, "
                f"got {self.arm_joint_count} and {self.single_arm_joint_count}"
            )

        self.right_side = str(self.pp_cfg.get("right_arm_name", "right"))
        self.board_cols = int(self.pp_cfg.get("board_cols", 3))

        keyframes = self.pp_cfg["joint_keyframes"]
        self.pos1 = _as_float_list(keyframes["pos1"], self.arm_joint_count)
        self.pos2a = _as_float_list(keyframes["pos2a"], self.arm_joint_count)
        self.pos2b = _as_float_list(keyframes["pos2b"], self.arm_joint_count)
        self.pos3 = _as_float_list(keyframes["pos3"], self.arm_joint_count)
        self.pos4 = _as_float_list(keyframes["pos4"], self.arm_joint_count)
        self.put_piece_targets = self.pp_cfg["put_piece_targets"]

        self.open_hand_position = _as_int_list(
            self.pp_cfg.get("right_hand_open_position", [0, 0, 0, 0, 0, 0]),
            6,
        )
        self.close_hand_position = _as_int_list(
            self.pp_cfg.get("right_hand_close_position", [80, 20, 80, 80, 80, 80]),
            6,
        )

        self.target_topic = self.arm_cfg["target_topic"]
        self.hand_topic = self.pp_cfg.get("hand_topic", "/control_robot_hand_position")
        self.waist_topic = self.pp_cfg.get("waist_topic", "/robot_waist_motion_data")
        self.head_topic = self.pp_cfg.get("head_topic", "/robot_head_motion_data")
        self.mode_service_name = self.arm_cfg["mode_service"]

        self.target_frame = int(self.pp_cfg.get("target_frame", 0))
        self.connect_timeout = float(self.pp_cfg.get("connect_timeout", 10.0))
        self.require_subscribers = bool(self.pp_cfg.get("require_subscribers", True))
        self.publisher_warmup_time = float(self.pp_cfg.get("publisher_warmup_time", 0.5))
        self.motion_settle_time = float(self.pp_cfg.get("motion_settle_time", 0.3))
        self.hand_settle_time = float(self.pp_cfg.get("hand_settle_time", 0.4))
        self.waist_settle_time = float(self.pp_cfg.get("waist_settle_time", 0.5))

        self.startup_step_duration = float(self.pp_cfg.get("startup_step_duration", 2.0))
        self.move_duration = float(self.pp_cfg.get("move_duration", 2.0))
        self.put_duration = float(self.pp_cfg.get("put_duration", self.move_duration))
        self.return_step_duration = float(self.pp_cfg.get("return_step_duration", 2.0))
        self.finish_step_duration = float(self.pp_cfg.get("finish_step_duration", 2.0))

        self.move_head_on_init = bool(self.pp_cfg.get("move_head_on_init", False))
        self.head_target_on_init = _as_float_list(
            self.pp_cfg.get("head_target_on_init", [0.0, 25.0]),
            2,
        )
        self.set_mode_on_init = bool(self.pp_cfg.get("set_mode_on_init", True))
        self.control_mode = int(self.pp_cfg.get("control_mode", 2))
        self.move_to_ready_on_init = bool(self.pp_cfg.get("move_to_ready_on_init", True))
        self.open_hand_on_init = bool(self.pp_cfg.get("open_hand_on_init", True))
        self.finish_pose_enabled = bool(self.pp_cfg.get("finish_pose_enabled", True))

        self.mode_srv = None
        self.arm_pub = None
        self.hand_pub = None
        self.waist_pub = None
        self.head_pub = None
        self._connect_ros()

        if self.set_mode_on_init:
            self.set_arm_control_mode(self.control_mode)
        if self.move_head_on_init:
            self.publish_head_target(self.head_target_on_init)
        if self.open_hand_on_init:
            self.publish_right_hand(self.open_hand_position)
        if self.move_to_ready_on_init:
            self.move_to_ready_pose()

    def _connect_ros(self):
        """Create ROS service proxies and publishers used by the keyframe flow."""
        rospy.wait_for_service(self.mode_service_name, timeout=self.connect_timeout)
        self.mode_srv = rospy.ServiceProxy(self.mode_service_name, changeArmCtrlMode)

        self.arm_pub = rospy.Publisher(self.target_topic, armTargetPoses, queue_size=10)
        self.hand_pub = rospy.Publisher(self.hand_topic, robotHandPosition, queue_size=10)
        self.waist_pub = rospy.Publisher(self.waist_topic, robotWaistControl, queue_size=10)
        self.head_pub = rospy.Publisher(self.head_topic, robotHeadMotionData, queue_size=10)

        self._wait_for_subscribers(self.arm_pub, self.target_topic)
        self._wait_for_subscribers(self.hand_pub, self.hand_topic)
        self._wait_for_subscribers(self.waist_pub, self.waist_topic)
        if bool(self.pp_cfg.get("head_require_subscribers", False)):
            self._wait_for_subscribers(self.head_pub, self.head_topic)
        else:
            time.sleep(float(self.pp_cfg.get("head_publisher_warmup_time", 0.5)))

    def _wait_for_subscribers(self, publisher, topic_name: str):
        """Wait for a publisher subscriber when subscriber waiting is enabled."""
        if not self.require_subscribers:
            time.sleep(self.publisher_warmup_time)
            return

        deadline = time.monotonic() + self.connect_timeout
        rate = rospy.Rate(10)
        while publisher.get_num_connections() == 0 and not rospy.is_shutdown():
            if time.monotonic() > deadline:
                raise TimeoutError(f"No subscriber connected to {topic_name}")
            rospy.loginfo_throttle(2.0, f"[TicTacToePickPlace] waiting for {topic_name}")
            rate.sleep()

    def set_arm_control_mode(self, mode: int):
        """Set the robot arm control mode through ``/arm_traj_change_mode``."""
        request = changeArmCtrlModeRequest()
        request.control_mode = int(mode)
        response = self.mode_srv(request)
        if not response.result:
            raise RuntimeError(f"Failed to set arm mode {mode}: {response.message}")

    def publish_arm_target_poses(self, times: Sequence[float], values: Sequence[float]):
        """Publish one or more full two-arm joint targets in degrees."""
        expected_values = len(times) * self.arm_joint_count
        if len(values) != expected_values:
            raise ValueError(
                f"arm target values must contain {expected_values} values "
                f"({self.arm_joint_count} joints x {len(times)} time point(s)), "
                f"got {len(values)}"
            )

        msg = armTargetPoses()
        msg.times = [float(t) for t in times]
        msg.values = [float(v) for v in values]
        if hasattr(msg, "frame"):
            msg.frame = self.target_frame
        self.arm_pub.publish(msg)

    def publish_joint_keyframe(
        self,
        name: str,
        joints_deg: Sequence[float],
        duration: Optional[float] = None,
    ):
        """Publish one named joint keyframe and wait for motion completion."""
        joints = _as_float_list(joints_deg, self.arm_joint_count)
        move_time = self.move_duration if duration is None else float(duration)
        rospy.loginfo("[TicTacToePickPlace] arm keyframe=%s joints=%s", name, joints)
        self.publish_arm_target_poses([move_time], joints)
        time.sleep(move_time + self.motion_settle_time)

    def publish_head_target(self, joint_data: Sequence[float]):
        """Publish a head yaw/pitch target."""
        msg = robotHeadMotionData()
        msg.joint_data = _as_float_list(joint_data, 2)
        self.head_pub.publish(msg)
        rospy.loginfo("[TicTacToePickPlace] head target published: %s", msg.joint_data)

    def publish_right_hand(self, positions: Sequence[int]):
        """Publish qiangnao right-hand finger positions."""
        right_hand = _as_int_list(positions, 6)
        msg = robotHandPosition()
        if hasattr(msg, "header"):
            msg.header.stamp = rospy.Time.now()
        msg.left_hand_position = [0, 0, 0, 0, 0, 0]
        msg.right_hand_position = right_hand
        self.hand_pub.publish(msg)
        rospy.loginfo("[TicTacToePickPlace] right hand=%s", right_hand)
        time.sleep(self.hand_settle_time)

    def publish_waist_angle(self, angle_deg: float):
        """Publish a waist yaw angle to ``/robot_waist_motion_data``."""
        msg = robotWaistControl()
        if hasattr(msg, "header"):
            msg.header.stamp = rospy.Time.now()
        msg.data.data = [float(angle_deg)]
        self.waist_pub.publish(msg)
        rospy.loginfo("[TicTacToePickPlace] waist angle=%s", msg.data.data)
        time.sleep(self.waist_settle_time)

    def move_to_ready_pose(self):
        """Run startup path: pos2a then pos2b."""
        self.publish_joint_keyframe("pos2a_startup", self.pos2a, self.startup_step_duration)
        self.publish_joint_keyframe("pos2b_ready", self.pos2b, self.startup_step_duration)

    def move_to_finish_pose(self):
        """Run final path after game over: pos2a then pos1."""
        if not self.finish_pose_enabled:
            return
        self.publish_joint_keyframe("pos2a_finish", self.pos2a, self.finish_step_duration)
        self.publish_joint_keyframe("pos1_finish", self.pos1, self.finish_step_duration)

    def _get_put_piece_target(self, vision_index: int) -> Dict[str, object]:
        """Return waist and arm keyframe data for one board cell."""
        index = int(vision_index)
        target = self.put_piece_targets.get(index)
        if target is None:
            target = self.put_piece_targets.get(str(index))
        if target is None:
            raise ValueError(f"No put_piece target configured for vision_index={vision_index}")
        return target

    def place_piece_to_cell(self, vision_index: int) -> Dict[str, object]:
        """Pick one right-side spare piece and place it on a board cell.

        Args:
            vision_index: Board cell index from 0 to 8.

        Returns:
            dict[str, object]: Debug information for this executed move.
        """
        index = int(vision_index)
        board_row, board_col = _vision_index_to_row_col(index, self.board_cols)
        target = self._get_put_piece_target(index)
        waist_angle = float(target["waist_deg"])
        put_arm_joints = _as_float_list(target["arm_joint_deg"], self.arm_joint_count)

        rospy.loginfo(
            "[TicTacToePickPlace] target=%s row=%s col=%s arm=right waist=%s",
            index,
            board_row,
            board_col,
            waist_angle,
        )

        self.publish_joint_keyframe("pos3_scratch_piece", self.pos3, self.move_duration)
        self.publish_right_hand(self.close_hand_position)
        self.publish_joint_keyframe("pos4_up_hand", self.pos4, self.move_duration)
        self.publish_waist_angle(waist_angle)
        self.publish_joint_keyframe(f"put_piece_{index}", put_arm_joints, self.put_duration)
        self.publish_right_hand(self.open_hand_position)
        self.publish_waist_angle(0.0)
        self.publish_joint_keyframe("pos2b_ready", self.pos2b, self.return_step_duration)

        return {
            "vision_index": index,
            "board_row": board_row,
            "board_col": board_col,
            "arm_side": self.right_side,
            "waist_deg": waist_angle,
            "pos3": self.pos3,
            "pos4": self.pos4,
            "put_arm_joint_deg": put_arm_joints,
            "return_pose": "pos2b",
        }


_DEFAULT_PICK_PLACE: Optional[TicTacToePickPlace] = None


def get_pick_place(config_path: str = "config/config.yaml") -> TicTacToePickPlace:
    """Return the default keyframe controller and keep it alive for the game."""
    global _DEFAULT_PICK_PLACE
    if _DEFAULT_PICK_PLACE is None:
        _DEFAULT_PICK_PLACE = TicTacToePickPlace(config_path=config_path)
    return _DEFAULT_PICK_PLACE
