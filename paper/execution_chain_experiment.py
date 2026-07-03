#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect execution-chain experiment metrics for the tic-tac-toe robot.

This script is designed for the paper's execution-link evaluation. It records
per-trial data and computes:

    - placement success rate
    - target-cell hit rate
    - post-execution state consistency rate
    - failure recovery success rate
    - single-step closed-loop time

By default the script is safe and does not move the robot. Add ``--execute`` to
call the existing ROS pick/place template backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.tictactoe_engine import BoardState, TicTacToeEngine
from utils.config_loader import resolve_project_path


FAILURE_TYPES = [
    "none",
    "grasp_failed",
    "place_offset",
    "wrong_cell",
    "vision_mismatch",
    "illegal_state",
    "timeout",
    "skipped",
    "other",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect physical execution-chain experiment data for the tic-tac-toe robot."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Config path.")
    parser.add_argument(
        "--out",
        default="runs/execution_chain_experiment",
        help="Output root directory. A timestamped run directory is created inside it unless --run-name is set.",
    )
    parser.add_argument("--run-name", default="", help="Optional run directory name.")
    parser.add_argument(
        "--strategy",
        default="template",
        choices=["template", "template_param", "template_residual", "template_residual_rl"],
        help="Execution strategy label recorded in the report.",
    )
    parser.add_argument(
        "--target-cells",
        nargs="+",
        default=["0,1,2,3,4,5,6,7,8"],
        help="Target cells to test, e.g. --target-cells 0 1 2 or --target-cells \"0,1,2\". Ignored when --use-minimax is set.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Repeat count for each target cell.")
    parser.add_argument(
        "--board-before",
        default="[0,0,0,0,0,0,0,0,0]",
        help="Default board state before every trial as JSON list of 9 integers.",
    )
    parser.add_argument(
        "--before-state-per-trial",
        action="store_true",
        help="Prompt for board_before before every trial.",
    )
    parser.add_argument(
        "--use-minimax",
        action="store_true",
        help="Use minimax to choose the target cell from board_before instead of --target-cells.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call TicTacToePickPlace.place_piece_to_cell(). Without this flag, no robot motion is sent.",
    )
    parser.add_argument(
        "--assume-success",
        action="store_true",
        help="Set board_after to expected state and failure_type=none without prompting.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt for board_after/failure metadata. Missing fields are left null unless --assume-success is used.",
    )
    parser.add_argument(
        "--failure-type",
        default="",
        choices=[""] + FAILURE_TYPES,
        help="Fixed failure type for all trials. Empty means prompt unless --non-interactive or --assume-success.",
    )
    parser.add_argument(
        "--recovery-attempted",
        default="",
        choices=["", "yes", "no"],
        help="Fixed recovery-attempted flag. Empty means prompt for failed trials.",
    )
    parser.add_argument(
        "--recovery-success",
        default="",
        choices=["", "yes", "no"],
        help="Fixed recovery-success flag. Empty means prompt when recovery is attempted.",
    )
    parser.add_argument("--notes", default="", help="Notes saved to every trial row.")
    parser.add_argument(
        "--strategy-params",
        default="{}",
        help="JSON object describing template parameters/residual settings used in this run.",
    )
    return parser.parse_args()


def parse_board_state(text: str, *, field_name: str = "board_state") -> BoardState:
    """Parse a length-9 board state JSON list."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be JSON list, got {text!r}") from exc
    if not isinstance(value, list) or len(value) != 9:
        raise ValueError(f"{field_name} must be a JSON list with length 9.")
    out = [int(v) for v in value]
    if any(v not in (0, 1, 2) for v in out):
        raise ValueError(f"{field_name} values must be 0, 1, or 2: {out}")
    return out


def parse_target_cells(values: Sequence[str] | str) -> List[int]:
    """Parse target cell indexes from comma-separated or space-separated values."""
    if isinstance(values, str):
        raw_parts = values.split(",")
    else:
        raw_parts = []
        for value in values:
            raw_parts.extend(str(value).split(","))
    cells = [int(part.strip()) for part in raw_parts if part.strip()]
    if not cells:
        raise ValueError("--target-cells must contain at least one cell index.")
    bad = [idx for idx in cells if idx < 0 or idx > 8]
    if bad:
        raise ValueError(f"target cell indexes must be 0~8, got {bad}")
    return cells


def parse_json_object(text: str, field_name: str) -> Dict[str, Any]:
    """Parse a JSON object argument."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON object.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return value


def format_board(state: Optional[BoardState]) -> str:
    """Return a compact 3x3 board string."""
    if state is None:
        return "<missing>"
    symbols = {0: ".", 1: "Y", 2: "B"}
    return "\n".join(
        " ".join(symbols.get(int(state[row * 3 + col]), "?") for col in range(3))
        for row in range(3)
    )


def prompt_board_state(prompt: str, default: Optional[BoardState] = None) -> Optional[BoardState]:
    """Prompt the user for a board state.

    Empty input uses the default. The strings "skip" and "none" return None.
    """
    default_text = json.dumps(default, ensure_ascii=False) if default is not None else "none"
    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip()
        if not raw and default is not None:
            return list(default)
        if raw.lower() in {"skip", "none", "null"}:
            return None
        try:
            return parse_board_state(raw, field_name=prompt)
        except ValueError as exc:
            print(f"[warn] {exc}")


def prompt_choice(prompt: str, choices: Sequence[str], default: str) -> str:
    """Prompt for one value from choices."""
    choices_text = "/".join(choices)
    while True:
        raw = input(f"{prompt} ({choices_text}) [{default}]: ").strip()
        value = raw or default
        if value in choices:
            return value
        print(f"[warn] expected one of: {choices_text}")


def prompt_bool(prompt: str, default: bool = False) -> bool:
    """Prompt for yes/no."""
    default_text = "yes" if default else "no"
    value = prompt_choice(prompt, ["yes", "no"], default_text)
    return value == "yes"


def expected_after_state(board_before: BoardState, target_cell: int, piece_value: int) -> BoardState:
    """Return expected state after placing one piece."""
    expected = list(board_before)
    expected[int(target_cell)] = int(piece_value)
    return expected


def evaluate_trial(
    board_before: Optional[BoardState],
    board_expected: Optional[BoardState],
    board_after: Optional[BoardState],
    target_cell: Optional[int],
    piece_value: int,
    failure_type: str,
    attempted: bool,
) -> Dict[str, Any]:
    """Compute per-trial execution metrics."""
    target_cell_empty_before = None
    target_cell_hit = None
    non_target_unchanged = None
    state_consistent = None
    placement_success = None
    target_cell_valid = target_cell is not None and 0 <= int(target_cell) <= 8

    if board_before is not None and target_cell_valid:
        target_cell_empty_before = board_before[int(target_cell)] == 0

    if board_after is not None and target_cell_valid:
        target_cell_hit = board_after[int(target_cell)] == int(piece_value)
        if board_before is not None:
            non_target_unchanged = all(
                board_after[idx] == board_before[idx]
                for idx in range(9)
                if idx != int(target_cell)
            )
        if board_expected is not None:
            state_consistent = list(board_after) == list(board_expected)
        elif non_target_unchanged is not None:
            state_consistent = bool(target_cell_hit and non_target_unchanged)

    if attempted:
        if board_after is not None and target_cell_hit is not None:
            placement_success = bool(target_cell_hit and failure_type == "none")
        else:
            placement_success = failure_type == "none"

    return {
        "target_cell_empty_before": target_cell_empty_before,
        "target_cell_hit": target_cell_hit,
        "non_target_unchanged": non_target_unchanged,
        "state_consistent": state_consistent,
        "placement_success": placement_success,
    }


def bool_rate(rows: Iterable[Dict[str, Any]], key: str, eligible_key: Optional[str] = None) -> Optional[float]:
    """Return mean of a boolean field, excluding None."""
    values: List[float] = []
    for row in rows:
        if eligible_key is not None and not row.get(eligible_key):
            continue
        value = row.get(key)
        if value is None:
            continue
        values.append(1.0 if bool(value) else 0.0)
    return sum(values) / len(values) if values else None


def mean(values: Sequence[float]) -> Optional[float]:
    """Mean with None on empty input."""
    return float(sum(values) / len(values)) if values else None


def stdev(values: Sequence[float]) -> Optional[float]:
    """Sample standard deviation with None on fewer than two samples."""
    return float(statistics.stdev(values)) if len(values) >= 2 else None


def json_dumps(value: Any) -> str:
    """JSON dump helper for CSV cells."""
    return json.dumps(value, ensure_ascii=False)


def make_run_dir(out_root: Path, run_name: str) -> Path:
    """Create one run directory."""
    if run_name:
        run_dir = out_root / run_name
    else:
        run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def init_pick_place(config_path: str):
    """Initialize ROS and return the existing pick/place backend."""
    import rospy

    from utils.tictactoe_pick_place import get_pick_place

    if not rospy.core.is_initialized():
        rospy.init_node("tic_tac_toe_execution_experiment", anonymous=True)
    return get_pick_place(config_path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def group_summary(rows: List[Dict[str, Any]], group_key: str) -> Dict[str, Dict[str, Any]]:
    """Compute summary grouped by target cell or strategy."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key)), []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for key, group_rows in groups.items():
        times = [
            float(row["closed_loop_time_sec"])
            for row in group_rows
            if row.get("attempted") and row.get("closed_loop_time_sec") is not None
        ]
        failures = [row for row in group_rows if row.get("attempted") and row.get("failure_type") != "none"]
        out[key] = {
            "trial_count": len(group_rows),
            "attempted_count": sum(1 for row in group_rows if row.get("attempted")),
            "placement_success_rate": bool_rate(group_rows, "placement_success", "attempted"),
            "target_cell_hit_rate": bool_rate(group_rows, "target_cell_hit", "attempted"),
            "state_consistency_rate": bool_rate(group_rows, "state_consistent", "attempted"),
            "failure_count": len(failures),
            "failure_recovery_success_rate": (
                sum(1 for row in failures if row.get("recovery_success")) / len(failures)
                if failures
                else None
            ),
            "average_closed_loop_time_sec": mean(times),
            "std_closed_loop_time_sec": stdev(times),
        }
    return out


def build_summary(rows: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Compute paper metrics from all trials."""
    attempted_rows = [row for row in rows if row.get("attempted")]
    closed_loop_times = [
        float(row["closed_loop_time_sec"])
        for row in attempted_rows
        if row.get("closed_loop_time_sec") is not None
    ]
    failures = [row for row in attempted_rows if row.get("failure_type") != "none"]
    recovery_attempted = [row for row in failures if row.get("recovery_attempted")]

    failure_type_counts: Dict[str, int] = {}
    for row in attempted_rows:
        failure_type = str(row.get("failure_type", "none"))
        failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1

    summary = {
        "metadata": metadata,
        "trial_count": len(rows),
        "attempted_count": len(attempted_rows),
        "skipped_count": sum(1 for row in rows if not row.get("attempted")),
        "placement_success_rate": bool_rate(rows, "placement_success", "attempted"),
        "target_cell_hit_rate": bool_rate(rows, "target_cell_hit", "attempted"),
        "post_state_consistency_rate": bool_rate(rows, "state_consistent", "attempted"),
        "failure_count": len(failures),
        "failure_recovery_success_rate": (
            sum(1 for row in failures if row.get("recovery_success")) / len(failures)
            if failures
            else None
        ),
        "recovery_attempt_success_rate": (
            sum(1 for row in recovery_attempted if row.get("recovery_success")) / len(recovery_attempted)
            if recovery_attempted
            else None
        ),
        "average_closed_loop_time_sec": mean(closed_loop_times),
        "std_closed_loop_time_sec": stdev(closed_loop_times),
        "failure_type_counts": failure_type_counts,
        "by_target_cell": group_summary(rows, "target_cell"),
        "by_strategy": group_summary(rows, "strategy"),
    }
    return summary


def run_trial(
    args: argparse.Namespace,
    engine: TicTacToeEngine,
    pick_place: Any,
    trial_id: int,
    target_cell: Optional[int],
    repeat_index: int,
    default_board_before: BoardState,
) -> Dict[str, Any]:
    """Run or record one execution trial."""
    if args.before_state_per_trial and not args.non_interactive:
        board_before = prompt_board_state("board_before", default_board_before)
        if board_before is None:
            board_before = list(default_board_before)
    else:
        board_before = list(default_board_before)

    planning_start = time.perf_counter()
    move = None
    planned_target = target_cell
    if args.use_minimax:
        move = engine.get_next_move(board_before)
        planned_target = int(move["vision_index"]) if move is not None else None
        piece_value = int(move["piece_value"]) if move is not None else engine.my_value
    else:
        piece_value = engine.my_value
        move = {"vision_index": planned_target, "piece_value": piece_value} if planned_target is not None else None
    planning_time = time.perf_counter() - planning_start

    target_valid = planned_target is not None and 0 <= int(planned_target) <= 8
    target_empty = target_valid and board_before[int(planned_target)] == engine.empty_value
    attempted = bool(move is not None and target_valid and target_empty)
    board_expected = (
        expected_after_state(board_before, int(planned_target), piece_value)
        if attempted
        else None
    )

    print("\n" + "=" * 72)
    print(f"trial={trial_id} strategy={args.strategy} repeat={repeat_index}")
    print(f"board_before:\n{format_board(board_before)}")
    print(f"target_cell={planned_target} piece_value={piece_value} attempted={attempted}")
    if board_expected is not None:
        print(f"board_expected:\n{format_board(board_expected)}")

    motion_result: Optional[Dict[str, Any]] = None
    motion_error: Optional[str] = None
    motion_start = time.perf_counter()
    if attempted and args.execute:
        try:
            if args.strategy != "template":
                print(
                    "[warn] Only the template backend is currently implemented; "
                    f"recording strategy={args.strategy} but executing base template."
                )
            motion_result = pick_place.place_piece_to_cell(int(planned_target))
        except Exception as exc:  # pragma: no cover - depends on robot runtime.
            motion_error = str(exc)
    elif attempted:
        print("[info] dry collection mode: robot motion skipped. Add --execute to move the arm.")
    motion_time = time.perf_counter() - motion_start if attempted else 0.0

    verification_start = time.perf_counter()
    if args.assume_success and board_expected is not None:
        board_after = list(board_expected)
        failure_type = "none"
        recovery_attempted = False
        recovery_success = False
    elif args.non_interactive:
        board_after = None
        failure_type = args.failure_type or ("other" if motion_error else "none")
        recovery_attempted = args.recovery_attempted == "yes"
        recovery_success = args.recovery_success == "yes"
    else:
        board_after = prompt_board_state("board_after, or type none if not available", board_expected)
        if args.failure_type:
            failure_type = args.failure_type
        else:
            failure_type = prompt_choice("failure_type", FAILURE_TYPES, "none" if motion_error is None else "other")
        if failure_type == "none":
            recovery_attempted = False if args.recovery_attempted == "" else args.recovery_attempted == "yes"
            recovery_success = False if args.recovery_success == "" else args.recovery_success == "yes"
        else:
            recovery_attempted = (
                args.recovery_attempted == "yes"
                if args.recovery_attempted
                else prompt_bool("recovery_attempted", False)
            )
            recovery_success = (
                args.recovery_success == "yes"
                if args.recovery_success
                else (prompt_bool("recovery_success", False) if recovery_attempted else False)
            )
    verification_time = time.perf_counter() - verification_start
    closed_loop_time = planning_time + motion_time + verification_time

    if motion_error and failure_type == "none":
        failure_type = "other"

    metrics = evaluate_trial(
        board_before=board_before,
        board_expected=board_expected,
        board_after=board_after,
        target_cell=planned_target,
        piece_value=piece_value,
        failure_type=failure_type,
        attempted=attempted,
    )

    row: Dict[str, Any] = {
        "trial_id": trial_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy": args.strategy,
        "repeat_index": repeat_index,
        "target_cell": planned_target,
        "piece_value": piece_value,
        "attempted": attempted,
        "skipped_reason": "" if attempted else ("target_not_empty_or_no_move"),
        "board_before": json_dumps(board_before),
        "board_expected": json_dumps(board_expected),
        "board_after": json_dumps(board_after),
        "failure_type": failure_type,
        "recovery_attempted": recovery_attempted,
        "recovery_success": recovery_success,
        "target_cell_empty_before": metrics["target_cell_empty_before"],
        "target_cell_hit": metrics["target_cell_hit"],
        "non_target_unchanged": metrics["non_target_unchanged"],
        "state_consistent": metrics["state_consistent"],
        "placement_success": metrics["placement_success"],
        "planning_time_sec": planning_time,
        "motion_time_sec": motion_time,
        "verification_time_sec": verification_time,
        "closed_loop_time_sec": closed_loop_time if attempted else None,
        "motion_result": json_dumps(motion_result),
        "motion_error": motion_error,
        "strategy_params": args.strategy_params,
        "notes": args.notes,
    }
    print(
        "[trial] "
        f"placement_success={row['placement_success']} "
        f"target_hit={row['target_cell_hit']} "
        f"state_consistent={row['state_consistent']} "
        f"failure={failure_type} "
        f"time={closed_loop_time:.2f}s"
    )
    return row


def main() -> None:
    """Collect execution experiment trials and write reports."""
    args = parse_args()
    default_board_before = parse_board_state(args.board_before, field_name="--board-before")
    parse_json_object(args.strategy_params, "--strategy-params")

    engine = TicTacToeEngine(args.config)
    targets = [None] if args.use_minimax else parse_target_cells(args.target_cells)
    run_dir = make_run_dir(resolve_project_path(args.out), args.run_name)

    pick_place = init_pick_place(args.config) if args.execute else None

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "strategy": args.strategy,
        "execute": bool(args.execute),
        "use_minimax": bool(args.use_minimax),
        "target_cells": targets,
        "repeats": int(args.repeats),
        "board_before_default": default_board_before,
        "robot_piece_value": engine.my_value,
        "robot_piece_name": engine.my_piece_name,
        "strategy_params": json.loads(args.strategy_params),
        "notes": args.notes,
    }

    rows: List[Dict[str, Any]] = []
    trial_id = 0
    for target in targets:
        for repeat in range(1, int(args.repeats) + 1):
            trial_id += 1
            row = run_trial(
                args=args,
                engine=engine,
                pick_place=pick_place,
                trial_id=trial_id,
                target_cell=target,
                repeat_index=repeat,
                default_board_before=default_board_before,
            )
            rows.append(row)
            write_csv(run_dir / "trials.csv", rows)
            summary = build_summary(rows, metadata)
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    summary = build_summary(rows, metadata)
    write_csv(run_dir / "trials.csv", rows)
    write_csv(run_dir / "summary_flat.csv", [flatten_summary(summary)])
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nreport:", run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def flatten_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the top-level paper metrics for a one-row CSV."""
    return {
        "trial_count": summary.get("trial_count"),
        "attempted_count": summary.get("attempted_count"),
        "skipped_count": summary.get("skipped_count"),
        "placement_success_rate": summary.get("placement_success_rate"),
        "target_cell_hit_rate": summary.get("target_cell_hit_rate"),
        "post_state_consistency_rate": summary.get("post_state_consistency_rate"),
        "failure_count": summary.get("failure_count"),
        "failure_recovery_success_rate": summary.get("failure_recovery_success_rate"),
        "recovery_attempt_success_rate": summary.get("recovery_attempt_success_rate"),
        "average_closed_loop_time_sec": summary.get("average_closed_loop_time_sec"),
        "std_closed_loop_time_sec": summary.get("std_closed_loop_time_sec"),
        "failure_type_counts": json_dumps(summary.get("failure_type_counts", {})),
    }


if __name__ == "__main__":
    main()
