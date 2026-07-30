"""Canonical deterministic 2048 environment state transitions."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Sequence

from llm2048.policy_contracts import Action


Board = list[list[int]]


@dataclass(frozen=True)
class SpawnedTile:
    row: int
    column: int
    value: int

    def resolved(self) -> dict[str, int]:
        return {
            "column": self.column,
            "row": self.row,
            "value": self.value,
        }


@dataclass(frozen=True)
class MoveOutcome:
    board_after_move: Board
    next_board: Board
    score_delta: int
    spawned_tile: SpawnedTile


class Game2048:
    """A seeded 4x4 2048 game with canonical movement and spawning."""

    def __init__(self, seed: int) -> None:
        self.board: Board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.moves = 0
        self._random = random.Random(seed)
        self._spawn_tile()
        self._spawn_tile()

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "Game2048":
        game = cls.__new__(cls)
        board = snapshot.get("board")
        score = snapshot.get("score")
        moves = snapshot.get("moves")
        rng_state = snapshot.get("rng_state")
        if (
            not isinstance(board, list)
            or len(board) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in board)
            or any(
                not isinstance(tile, int)
                or isinstance(tile, bool)
                or tile < 0
                or (tile != 0 and (tile < 2 or tile & (tile - 1) != 0))
                for row in board
                for tile in row
            )
        ):
            raise ValueError("snapshot board is not a canonical 4x4 board")
        if (
            not isinstance(score, int)
            or isinstance(score, bool)
            or score < 0
            or score % 4 != 0
        ):
            raise ValueError("snapshot score must be a non-negative multiple of four")
        if (
            not isinstance(moves, int)
            or isinstance(moves, bool)
            or moves < 0
        ):
            raise ValueError("snapshot moves must be a non-negative integer")
        if not isinstance(rng_state, list):
            raise ValueError("snapshot RNG state must be a JSON array")
        game.board = [row[:] for row in board]
        game.score = score
        game.moves = moves
        game._random = random.Random()
        try:
            game._random.setstate(_nested_tuple(rng_state))
        except (TypeError, ValueError) as error:
            raise ValueError("snapshot RNG state is invalid") from error
        return game

    def snapshot(self) -> dict[str, Any]:
        return {
            "board": [row[:] for row in self.board],
            "moves": self.moves,
            "rng_state": self._random.getstate(),
            "score": self.score,
        }

    def move(self, action: Action) -> MoveOutcome:
        board_after_move, score_delta = moved_board(self.board, action)
        if board_after_move == self.board:
            raise ValueError("action does not change the board")
        self.board = board_after_move
        self.score += score_delta
        self.moves += 1
        pre_spawn_board = [row[:] for row in board_after_move]
        spawned_tile = self._spawn_tile()
        return MoveOutcome(
            board_after_move=pre_spawn_board,
            next_board=[row[:] for row in self.board],
            score_delta=score_delta,
            spawned_tile=spawned_tile,
        )

    def _spawn_tile(self) -> SpawnedTile:
        empty_cells = [
            (row, column)
            for row in range(4)
            for column in range(4)
            if self.board[row][column] == 0
        ]
        if not empty_cells:
            raise RuntimeError("cannot spawn a tile on a full board")
        row, column = empty_cells[self._random.randrange(len(empty_cells))]
        value = 2 if self._random.random() < 0.9 else 4
        self.board[row][column] = value
        return SpawnedTile(row=row, column=column, value=value)


def moved_board(board: Sequence[Sequence[int]], action: Action) -> tuple[Board, int]:
    """Move without spawning and return the resulting board and merge score."""
    if action in ("LEFT", "RIGHT"):
        lines = [list(row) for row in board]
        if action == "RIGHT":
            lines = [list(reversed(line)) for line in lines]
        moved_lines, score_delta = _move_lines(lines)
        if action == "RIGHT":
            moved_lines = [list(reversed(line)) for line in moved_lines]
        return moved_lines, score_delta

    lines = [[board[row][column] for row in range(4)] for column in range(4)]
    if action == "DOWN":
        lines = [list(reversed(line)) for line in lines]
    moved_lines, score_delta = _move_lines(lines)
    if action == "DOWN":
        moved_lines = [list(reversed(line)) for line in moved_lines]
    return (
        [
            [moved_lines[column][row] for column in range(4)]
            for row in range(4)
        ],
        score_delta,
    )


def _move_lines(lines: Sequence[Sequence[int]]) -> tuple[Board, int]:
    moved_lines: Board = []
    total_score = 0
    for line in lines:
        moved_line, line_score = _slide_and_merge(line)
        moved_lines.append(moved_line)
        total_score += line_score
    return moved_lines, total_score


def _slide_and_merge(line: Sequence[int]) -> tuple[list[int], int]:
    tiles = [tile for tile in line if tile != 0]
    merged: list[int] = []
    score = 0
    index = 0
    while index < len(tiles):
        if index + 1 < len(tiles) and tiles[index] == tiles[index + 1]:
            value = tiles[index] * 2
            merged.append(value)
            score += value
            index += 2
        else:
            merged.append(tiles[index])
            index += 1
    return merged + [0] * (4 - len(merged)), score


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value
