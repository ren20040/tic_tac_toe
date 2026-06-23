#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Provide tic-tac-toe board evaluation and minimax move selection."""

from typing import Dict, List, Optional

from utils.config_loader import load_config

BoardState = List[int]


class TicTacToeEngine:
    """Choose the robot's next move from the current tic-tac-toe board state."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the tic-tac-toe decision engine from YAML config.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            ValueError: If ``game.my_piece`` is not ``yellow_piece`` or
                ``blue_piece``.
        """
        self.cfg = load_config(config_path)

        game_cfg = self.cfg.get("game", {})

        self.empty_value = int(game_cfg.get("empty_value", 0))
        self.yellow_value = int(game_cfg.get("yellow_value", 1))
        self.blue_value = int(game_cfg.get("blue_value", 2))

        self.my_piece_name = str(game_cfg.get("my_piece", "blue_piece"))

        if self.my_piece_name == "yellow_piece":
            self.my_value = self.yellow_value
            self.opponent_value = self.blue_value
        elif self.my_piece_name == "blue_piece":
            self.my_value = self.blue_value
            self.opponent_value = self.yellow_value
        else:
            raise ValueError(
                f"Unsupported game.my_piece: {self.my_piece_name}. "
                "Expected yellow_piece or blue_piece."
            )

        self.prefer_center = bool(game_cfg.get("prefer_center", True))
        self.prefer_corners = bool(game_cfg.get("prefer_corners", True))

    def get_next_move(self, board_state: BoardState) -> Optional[Dict[str, int]]:
        """Return the robot move that should be executed for ``board_state``.

        Args:
            board_state: Length-9 board list using the same index layout as vision.

        Returns:
            Optional[dict[str, int]]: ``None`` when the game is over or no legal
            move exists. Otherwise returns ``vision_index`` and ``piece_value``.

        Raises:
            TypeError: If ``board_state`` is not a list.
            ValueError: If the board length or cell values are invalid.
        """
        self._validate_board_state(board_state)

        if self.is_game_over(board_state):
            return None

        best_index = self._find_best_move(board_state)
        if best_index is None:
            return None

        return {"vision_index": best_index, "piece_value": self.my_value}

    def _find_best_move(self, board_state: BoardState) -> Optional[int]:
        """Search the best legal move for the current board.

        Args:
            board_state: Current validated board state.

        Returns:
            Optional[int]: Best ``vision_index``. Returns ``None`` when no empty
            cell exists.
        """
        best_score = -10_000
        best_index = None

        for index in self._ordered_empty_indices(board_state):
            next_state = board_state.copy()
            next_state[index] = self.my_value
            score = self._minimax(next_state, is_my_turn=False, depth=0)
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _minimax(self, board_state: BoardState, is_my_turn: bool, depth: int) -> int:
        """Recursively score a board assuming both players make optimal moves.

        Args:
            board_state: Simulated board state.
            is_my_turn: ``True`` when simulating the robot's move; ``False`` for
                the opponent's move.
            depth: Current recursion depth. Faster wins and later losses are scored
                better.

        Returns:
            int: Positive scores favor the robot, negative scores favor the
            opponent, and zero means a draw.
        """
        winner = self.get_winner(board_state)
        if winner == self.my_value:
            return 10 - depth
        if winner == self.opponent_value:
            return depth - 10
        if self.is_full(board_state):
            return 0

        if is_my_turn:
            best_score = -10_000
            for index in self._ordered_empty_indices(board_state):
                next_state = board_state.copy()
                next_state[index] = self.my_value
                score = self._minimax(next_state, False, depth + 1)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = 10_000
            for index in self._ordered_empty_indices(board_state):
                next_state = board_state.copy()
                next_state[index] = self.opponent_value
                score = self._minimax(next_state, True, depth + 1)
                best_score = min(best_score, score)
            return best_score

    def _ordered_empty_indices(self, board_state: BoardState) -> List[int]:
        """Return empty cell indexes in the configured search preference order.

        Args:
            board_state: Current validated board state.

        Returns:
            list[int]: Ordered empty ``vision_index`` values.
        """
        empty = set(self.get_empty_indices(board_state))
        order = []

        if self.prefer_center:
            order.append(4)
        if self.prefer_corners:
            order.extend([0, 2, 6, 8])
        order.extend([1, 3, 5, 7])

        ordered = [idx for idx in order if idx in empty]
        # Keep any legal empty index even if a custom preference skipped it.
        for idx in sorted(empty):
            if idx not in ordered:
                ordered.append(idx)
        return ordered

    def get_winner(self, board_state: BoardState) -> Optional[int]:
        """Return the winner for the current board.

        Args:
            board_state: Length-9 board state list.

        Returns:
            Optional[int]: Winning piece value. Returns ``None`` when no winner
            exists.

        Raises:
            TypeError: If ``board_state`` is not a list.
            ValueError: If the board length or cell values are invalid.
        """
        self._validate_board_state(board_state)
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
        for a, b, c in win_lines:
            if (
                board_state[a] != self.empty_value
                and board_state[a] == board_state[b] == board_state[c]
            ):
                return board_state[a]
        return None

    def is_full(self, board_state: BoardState) -> bool:
        """Check whether the board has no empty cells.

        Args:
            board_state: Length-9 board state list.

        Returns:
            bool: ``True`` when every cell is occupied.
        """
        self._validate_board_state(board_state)
        return all(v != self.empty_value for v in board_state)

    def is_game_over(self, board_state: BoardState) -> bool:
        """Check whether the current board has reached a terminal state.

        Args:
            board_state: Length-9 board state list.

        Returns:
            bool: ``True`` when a player has won or the board is full.
        """
        return self.get_winner(board_state) is not None or self.is_full(board_state)

    def get_empty_indices(self, board_state: BoardState) -> List[int]:
        """Return all empty board indexes.

        Args:
            board_state: Length-9 board state list.

        Returns:
            list[int]: Indexes whose value equals ``empty_value``.
        """
        self._validate_board_state(board_state)
        return [i for i, v in enumerate(board_state) if v == self.empty_value]

    def _validate_board_state(self, board_state: BoardState):
        """Validate board type, length, and legal cell values.

        Args:
            board_state: Board state to validate.

        Raises:
            TypeError: If ``board_state`` is not a list.
            ValueError: If the board length is not 9 or a cell value is illegal.
        """
        if not isinstance(board_state, list):
            raise TypeError(f"board_state must be list, got {type(board_state)}")
        if len(board_state) != 9:
            raise ValueError(f"board_state length must be 9, got {len(board_state)}")
        allowed = {self.empty_value, self.yellow_value, self.blue_value}
        for idx, val in enumerate(board_state):
            if val not in allowed:
                raise ValueError(f"board_state[{idx}]={val} not in {allowed}")

    def format_board_state(self, board_state: BoardState) -> str:
        """Format a board state as three printable rows.

        Args:
            board_state: Length-9 board state list.

        Returns:
            str: Three-line text using ``.``, ``Y``, and ``B`` for empty, yellow,
            and blue cells.
        """
        symbol = {self.empty_value: ".", self.yellow_value: "Y", self.blue_value: "B"}
        rows = []
        for r in range(3):
            row_vals = board_state[r * 3:(r + 1) * 3]
            rows.append(" ".join(symbol.get(v, "?") for v in row_vals))
        return "\n".join(rows)
