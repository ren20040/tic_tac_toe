#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main ROS controller for the tic-tac-toe robot flow.

This module only orchestrates existing functional modules:
    - TicTacToeVisionState: YOLO camera inference -> 0~8 board state
    - TicTacToeEngine: board state -> next move
    - TicTacToePickPlace: next move -> robot arm pick/place action
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rospy
from std_msgs.msg import String

from scripts.tictactoe_engine import BoardState, TicTacToeEngine
from scripts.tictactoe_pick_place import TicTacToePickPlace, get_pick_place
from scripts.vision_state import TicTacToeVisionState
from utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the main ROS controller.

    Returns:
        argparse.Namespace: Parsed config path, single-step flag, and dry-run flag.
    """
    parser = argparse.ArgumentParser(description="Run YOLO tic-tac-toe robot control.")
    parser.add_argument("--config", default="config/config.yaml", help="Config path.")
    parser.add_argument("--once", action="store_true", help="Execute at most one robot move.")
    parser.add_argument("--dry-run", action="store_true", help="Plan moves without moving the arm.")
    return parser.parse_args()


def same_board(a: Optional[BoardState], b: Optional[BoardState]) -> bool:
    """Compare two optional board states by value.

    Args:
        a: First board state, or ``None``.
        b: Second board state, or ``None``.

    Returns:
        bool: ``True`` only when both states exist and contain the same values.
    """
    return a is not None and b is not None and list(a) == list(b)


class TicTacToeMainController:
    """Orchestrates recognition, decision making, and arm execution."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        dry_run: bool = False,
        vision_factory: Optional[Callable[[str], object]] = None,
        vision_name: str = "YOLO",
    ):
        """Create all modules needed for one tic-tac-toe game loop.

        Args:
            config_path: Path to the YAML configuration file.
            dry_run: When ``True``, publish decisions and skip arm motion.
            vision_factory: Optional factory used to create the vision module.
            vision_name: Human-readable vision backend name for runtime logs.
        """
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.game_cfg = self.cfg.get("game", {})
        self.ros_cfg = self.cfg.get("ros", {})
        self.arm_cfg = self.cfg.get("arm", {})
        self.vision_name = str(vision_name)

        self.dry_run = bool(dry_run or self.arm_cfg.get("dry_run", False))

        self.engine = TicTacToeEngine(config_path)
        self.pick_place: Optional[TicTacToePickPlace] = None
        if not self.dry_run:
            self.pick_place = get_pick_place(config_path)
        if vision_factory is None:
            vision_factory = TicTacToeVisionState
        self.vision = vision_factory(config_path)

        self.loop_hz = max(0.1, float(self.game_cfg.get("main_loop_hz", 2.0)))
        self.stable_frames = max(1, int(self.game_cfg.get("stable_frames", 3)))
        self.stable_timeout = float(self.game_cfg.get("stable_timeout", 10.0))
        self.verify_timeout = float(self.game_cfg.get("verify_after_move_timeout", 8.0))
        self.robot_starts = bool(self.game_cfg.get("robot_starts", True))
        self.stop_on_draw = bool(self.game_cfg.get("stop_on_draw", True))
        self.set_mode_on_winner = bool(
            self.game_cfg.get(
                "set_arm_mode_on_winner",
                self.game_cfg.get("set_arm_mode_on_game_over", True),
            )
        )
        self.set_mode_on_draw = bool(self.game_cfg.get("set_arm_mode_on_draw", False))
        self.game_over_arm_mode = int(self.game_cfg.get("game_over_arm_mode", 1))
        self.pending_expected_state: Optional[BoardState] = None

        self.board_state_pub = rospy.Publisher(
            self.ros_cfg.get("board_state_topic", "/tic_tac_toe/board_state"),
            String,
            queue_size=10,
        )
        self.next_move_pub = rospy.Publisher(
            self.ros_cfg.get("next_move_topic", "/tic_tac_toe/next_move"),
            String,
            queue_size=10,
        )

        rospy.loginfo(
            "[main] initialized: my_piece=%s robot_starts=%s dry_run=%s vision=%s",
            self.engine.my_piece_name,
            self.robot_starts,
            self.dry_run,
            self.vision_name,
        )

    def run(self, once: bool = False):
        """Run the recognition-decision-execution loop.

        Args:
            once: When ``True``, execute at most one robot move and return.

        Returns:
            None.
        """
        rospy.loginfo("[main] started")
        rate = rospy.Rate(self.loop_hz)

        while not rospy.is_shutdown():
            board_state = self.wait_for_stable_board_state(self.stable_timeout)
            if board_state is None:
                rospy.logwarn("[main] no stable board state; waiting")
                rate.sleep()
                continue

            self.publish_board_state(board_state, event="stable")
            rospy.loginfo("[main] stable board:\n%s", self.engine.format_board_state(board_state))

            if self.pending_expected_state is not None:
                if same_board(board_state, self.pending_expected_state):
                    self.publish_board_state(board_state, event="move_confirmed_late")
                    rospy.loginfo("[main] pending robot move confirmed by later vision")
                    self.pending_expected_state = None
                else:
                    rospy.logwarn_throttle(
                        2.0,
                        "[main] detected board does not match pending expected state; "
                        "continuing %s detection",
                        self.vision_name,
                    )
                    rate.sleep()
                    continue

            if self.is_terminal_state(board_state):
                self.finish_game(board_state)
                return

            if not self.is_reachable_turn_state(board_state):
                rospy.logwarn("[main] board piece counts are not reachable; waiting")
                rate.sleep()
                continue

            if not self.is_robot_turn(board_state):
                rospy.loginfo("[main] waiting for opponent move")
                rate.sleep()
                continue

            move = self.engine.get_next_move(board_state)
            if move is None:
                rospy.logwarn(
                    "[main] no legal move for detected board; continuing %s detection",
                    self.vision_name,
                )
                rate.sleep()
                continue

            confirmed_state = self.execute_move(board_state, move)
            if confirmed_state is None:
                rospy.logwarn(
                    "[main] move was not confirmed or was skipped; continuing %s detection",
                    self.vision_name,
                )
                rate.sleep()
                continue

            if self.is_terminal_state(confirmed_state):
                self.finish_game(confirmed_state)
                return
            if once or self.dry_run:
                return

            rate.sleep()

    def wait_for_stable_board_state(self, timeout: float) -> Optional[BoardState]:
        """Wait until the same board state is observed for configured frames.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            Optional[BoardState]: Stable board state, or ``None`` on timeout.
        """
        deadline = time.monotonic() + float(timeout)
        rate = rospy.Rate(max(1.0, self.loop_hz * 2.0))
        last_state: Optional[BoardState] = None
        same_count = 0

        while not rospy.is_shutdown():
            state = self.vision.get_current_board_state()
            if state is None:
                last_state = None
                same_count = 0
            elif same_board(state, last_state):
                same_count += 1
            else:
                last_state = list(state)
                same_count = 1

            if last_state is not None and same_count >= self.stable_frames:
                return list(last_state)

            if time.monotonic() > deadline:
                return None

            rate.sleep()

        return None

    def execute_move(self, board_state: BoardState, move: Dict[str, int]) -> Optional[BoardState]:
        """Execute one planned robot move and optionally confirm it by vision.

        Args:
            board_state: Stable board state before the move.
            move: Move payload containing ``vision_index`` and ``piece_value``.

        Returns:
            Optional[BoardState]: Expected board state after the move, or ``None``
            when the move is skipped or visual confirmation fails.

        Raises:
            RuntimeError: If the target cell is occupied or pick/place is missing.
        """
        target_index = int(move["vision_index"])
        piece_value = int(move["piece_value"])

        if board_state[target_index] != self.engine.empty_value:
            rospy.logwarn(
                "[main] target cell %s is not empty in detected board; continuing %s detection",
                target_index,
                self.vision_name,
            )
            return None

        expected = list(board_state)
        expected[target_index] = piece_value

        payload = {
            "event": "next_move",
            "vision_index": target_index,
            "piece_value": piece_value,
            "board_before": list(board_state),
            "board_expected": expected,
            "dry_run": self.dry_run,
        }
        self.publish_next_move(payload)
        rospy.loginfo("[main] next move: %s", json.dumps(payload, ensure_ascii=False))

        if self.dry_run:
            rospy.logwarn("[main] dry-run: skipped arm motion")
            return expected

        if self.pick_place is None:
            raise RuntimeError("pick/place controller is not initialized")

        motion_result = self.pick_place.place_piece_to_cell(target_index)
        self.pending_expected_state = list(expected)
        self.publish_next_move(
            {
                "event": "move_executed",
                "vision_index": target_index,
                "motion_result": motion_result,
            }
        )

        if self.verify_timeout <= 0:
            self.pending_expected_state = None
            return expected

        confirmed = self.wait_for_expected_state(expected, self.verify_timeout)
        if confirmed:
            rospy.loginfo("[main] robot move confirmed by vision")
            self.pending_expected_state = None
            return expected

        rospy.logwarn("[main] vision did not confirm the robot move before timeout")
        self.publish_next_move(
            {
                "event": "move_not_confirmed",
                "board_expected": expected,
            }
        )
        return None

    def wait_for_expected_state(self, expected: BoardState, timeout: float) -> bool:
        """Wait for vision to match an expected post-move board state.

        Args:
            expected: Board state expected after the robot move.
            timeout: Maximum wait time in seconds.

        Returns:
            bool: ``True`` if the expected state is observed before timeout.
        """
        deadline = time.monotonic() + float(timeout)
        rate = rospy.Rate(max(1.0, self.loop_hz * 2.0))

        while not rospy.is_shutdown():
            state = self.vision.get_current_board_state()
            if same_board(state, expected):
                self.publish_board_state(list(state), event="move_confirmed")
                return True

            if time.monotonic() > deadline:
                return False

            rate.sleep()

        return False

    def is_robot_turn(self, board_state: BoardState) -> bool:
        """Check whether the current piece counts mean it is the robot's turn.

        Args:
            board_state: Current board state.

        Returns:
            bool: ``True`` when the robot should move next.
        """
        my_count = board_state.count(self.engine.my_value)
        opponent_count = board_state.count(self.engine.opponent_value)

        if self.robot_starts:
            return my_count == opponent_count
        return opponent_count == my_count + 1

    def is_reachable_turn_state(self, board_state: BoardState) -> bool:
        """Check whether board piece counts are reachable for the turn order.

        Args:
            board_state: Current board state.

        Returns:
            bool: ``True`` when the counts match a legal game progression.
        """
        my_count = board_state.count(self.engine.my_value)
        opponent_count = board_state.count(self.engine.opponent_value)

        if self.robot_starts:
            return my_count == opponent_count or my_count == opponent_count + 1
        return opponent_count == my_count or opponent_count == my_count + 1

    def is_terminal_state(self, board_state: BoardState) -> bool:
        """Check whether the main loop should stop on this board state.

        Args:
            board_state: Current board state.

        Returns:
            bool: ``True`` when someone won, or when draw-stop is enabled and
            the board is full.
        """
        winner = self.engine.get_winner(board_state)
        return winner is not None or (self.stop_on_draw and self.engine.is_full(board_state))

    def finish_game(self, board_state: BoardState):
        """Publish final game state and apply configured end-of-game arm mode.

        Args:
            board_state: Terminal board state.

        Returns:
            None.
        """
        winner = self.engine.get_winner(board_state)
        if winner == self.engine.my_value:
            result = "robot_win"
        elif winner == self.engine.opponent_value:
            result = "opponent_win"
        elif self.engine.is_full(board_state):
            result = "draw"
        else:
            result = "stopped"

        self.publish_board_state(board_state, event="game_over", result=result)
        rospy.loginfo(
            "[main] game over: %s\n%s",
            result,
            self.engine.format_board_state(board_state),
        )

        if self.pick_place is not None:
            try:
                self.pick_place.move_to_finish_pose()
            except Exception as exc:
                rospy.logerr("[main] failed to move arm to finish pose: %s", exc)

        should_set_mode = (
            (winner is not None and self.set_mode_on_winner)
            or (result == "draw" and self.set_mode_on_draw)
        )
        if should_set_mode:
            self.set_arm_mode_after_game()

    def set_arm_mode_after_game(self):
        """Switch the arm control mode after a winner or configured draw.

        Returns:
            None.
        """
        if self.dry_run:
            rospy.loginfo("[dry-run] set arm control mode -> %s", self.game_over_arm_mode)
            return

        if self.pick_place is None:
            rospy.logwarn("[main] no pick/place controller; cannot set arm mode")
            return

        try:
            self.pick_place.set_arm_control_mode(self.game_over_arm_mode)
            rospy.loginfo("[main] arm control mode set to %s", self.game_over_arm_mode)
        except Exception as exc:
            rospy.logerr("[main] failed to set arm mode after game: %s", exc)

    def publish_board_state(
        self,
        board_state: BoardState,
        event: str,
        result: Optional[str] = None,
    ):
        """Publish board-state JSON to the configured ROS topic.

        Args:
            board_state: Board values to publish.
            event: Event name such as ``stable`` or ``game_over``.
            result: Optional final result string.

        Returns:
            None.
        """
        payload = {
            "event": event,
            "board_state": list(board_state),
            "result": result,
            "winner": self.engine.get_winner(board_state),
            "my_value": self.engine.my_value,
            "opponent_value": self.engine.opponent_value,
        }
        self.board_state_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def publish_next_move(self, payload: Dict[str, object]):
        """Publish a planned or executed move payload as JSON.

        Args:
            payload: JSON-serializable move event payload.

        Returns:
            None.
        """
        self.next_move_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def main():
    """Initialize ROS and run the tic-tac-toe main controller."""
    args = parse_args()
    rospy.init_node("tic_tac_toe_main_control", anonymous=False)
    controller = TicTacToeMainController(args.config, dry_run=args.dry_run)
    controller.run(once=args.once)


if __name__ == "__main__":
    main()
