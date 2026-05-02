#!/usr/bin/env python3
"""Validate TD teacher JSONL datasets for SFT conversion."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ACTIONS = ["up", "down", "left", "right"]
ACTION_SET = set(ACTIONS)
REQUIRED_FIELDS = {
    "board",
    "valid_moves",
    "action_scores",
    "action_ranking",
    "teacher_action",
    "max_tile",
    "empty_cells",
    "legal_move_count",
    "top1_score",
    "top2_score",
    "score_margin",
    "search_depth",
    "source",
}
EPS = 1e-4


class ValidationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input teacher JSONL path. Can be repeated.")
    parser.add_argument("--max-errors", type=int, default=20, help="Maximum errors to print.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any row is invalid.")
    return parser.parse_args()


def is_power_of_two_or_zero(value: int) -> bool:
    return value == 0 or (value > 0 and (value & (value - 1)) == 0)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def nearly_equal(a: Any, b: Any, eps: float = EPS) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if not is_number(a) or not is_number(b):
        return False
    scale = max(1.0, abs(float(a)), abs(float(b)))
    return abs(float(a) - float(b)) <= eps * scale


def margin_bucket(value: Any) -> str:
    if value is None:
        return "null"
    value = float(value)
    if value == 0:
        return "0"
    if value < 0.001:
        return "(0,0.001)"
    if value < 0.01:
        return "[0.001,0.01)"
    if value < 0.1:
        return "[0.01,0.1)"
    if value < 1:
        return "[0.1,1)"
    if value < 10:
        return "[1,10)"
    if value < 100:
        return "[10,100)"
    if value < 1000:
        return "[100,1000)"
    return ">=1000"


def validate_board(row: dict[str, Any]) -> list[list[int]]:
    board = row["board"]
    if not isinstance(board, list) or len(board) != 4:
        raise ValidationError("board must be a 4x4 list")
    flat: list[int] = []
    for r, line in enumerate(board):
        if not isinstance(line, list) or len(line) != 4:
            raise ValidationError(f"board row {r} must have length 4")
        for c, cell in enumerate(line):
            if not isinstance(cell, int) or isinstance(cell, bool):
                raise ValidationError(f"board[{r}][{c}] must be an int")
            if not is_power_of_two_or_zero(cell):
                raise ValidationError(f"board[{r}][{c}] must be 0 or a power of two")
            flat.append(cell)
    if row["max_tile"] != max(flat):
        raise ValidationError(f"max_tile {row['max_tile']} does not match board max {max(flat)}")
    empty = sum(1 for value in flat if value == 0)
    if row["empty_cells"] != empty:
        raise ValidationError(f"empty_cells {row['empty_cells']} does not match board empty count {empty}")
    return board


def validate_moves(row: dict[str, Any]) -> list[str]:
    valid_moves = row["valid_moves"]
    if not isinstance(valid_moves, list):
        raise ValidationError("valid_moves must be a list")
    if not valid_moves:
        raise ValidationError("valid_moves must contain at least one legal action")
    if any(not isinstance(move, str) for move in valid_moves):
        raise ValidationError("valid_moves entries must be strings")
    if any(move not in ACTION_SET for move in valid_moves):
        raise ValidationError("valid_moves may only contain up/down/left/right")
    if len(set(valid_moves)) != len(valid_moves):
        raise ValidationError("valid_moves must not contain duplicates")
    if row["legal_move_count"] != len(valid_moves):
        raise ValidationError("legal_move_count must equal len(valid_moves)")
    return valid_moves


def validate_scores(row: dict[str, Any], valid_moves: list[str]) -> dict[str, Any]:
    scores = row["action_scores"]
    if not isinstance(scores, dict):
        raise ValidationError("action_scores must be an object")
    if set(scores) != ACTION_SET:
        raise ValidationError("action_scores must contain exactly up/down/left/right")
    valid_set = set(valid_moves)
    for action in ACTIONS:
        score = scores[action]
        if action in valid_set:
            if not is_number(score):
                raise ValidationError(f"legal action {action} score must be a finite number")
        elif score is not None:
            raise ValidationError(f"illegal action {action} score must be null")
    return scores


def validate_ranking(row: dict[str, Any], valid_moves: list[str], scores: dict[str, Any]) -> list[str]:
    ranking = row["action_ranking"]
    if not isinstance(ranking, list) or any(not isinstance(action, str) for action in ranking):
        raise ValidationError("action_ranking must be a list of strings")
    if set(ranking) != ACTION_SET or len(ranking) != len(ACTIONS):
        raise ValidationError("action_ranking must cover up/down/left/right exactly once")
    valid_set = set(valid_moves)
    legal_prefix = ranking[: len(valid_moves)]
    illegal_suffix = ranking[len(valid_moves) :]
    if set(legal_prefix) != valid_set:
        raise ValidationError("legal actions must appear before illegal actions in action_ranking")
    if any(action in valid_set for action in illegal_suffix):
        raise ValidationError("illegal suffix contains a legal action")
    for prev, cur in zip(legal_prefix, legal_prefix[1:]):
        if float(scores[prev]) + EPS < float(scores[cur]):
            raise ValidationError("action_ranking legal prefix is not sorted by descending action_scores")
    max_score = max(float(scores[action]) for action in valid_moves)
    if row["teacher_action"] not in valid_set or not nearly_equal(scores[row["teacher_action"]], max_score):
        raise ValidationError("teacher_action must equal highest-scoring legal action")
    return ranking


def validate_top_scores(row: dict[str, Any], valid_moves: list[str], scores: dict[str, Any]) -> None:
    sorted_scores = sorted((float(scores[action]) for action in valid_moves), reverse=True)
    top1 = sorted_scores[0]
    if not nearly_equal(row["top1_score"], top1):
        raise ValidationError("top1_score must equal highest legal action score")
    if len(sorted_scores) == 1:
        # Current exporter convention uses top2_score = top1_score and score_margin = 0
        # for a single legal action; null/null is also accepted for hand-written fixtures.
        if row["top2_score"] is None and row["score_margin"] is None:
            return
        if not nearly_equal(row["top2_score"], top1):
            raise ValidationError("single-action top2_score must be null or equal top1_score")
        if not nearly_equal(row["score_margin"], 0.0):
            raise ValidationError("single-action score_margin must be null or 0")
        return
    top2 = sorted_scores[1]
    if not nearly_equal(row["top2_score"], top2):
        raise ValidationError("top2_score must equal second-highest legal action score")
    if not nearly_equal(row["score_margin"], top1 - top2):
        raise ValidationError("score_margin must equal top1_score - top2_score")


def validate_row(row: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        raise ValidationError(f"missing required fields: {', '.join(sorted(missing))}")
    validate_board(row)
    valid_moves = validate_moves(row)
    scores = validate_scores(row, valid_moves)
    validate_ranking(row, valid_moves, scores)
    validate_top_scores(row, valid_moves, scores)


def update_summary(summary: dict[str, Counter], row: dict[str, Any]) -> None:
    summary["action_distribution"][row["teacher_action"]] += 1
    summary["max_tile_histogram"][str(row["max_tile"])] += 1
    summary["empty_cells_histogram"][str(row["empty_cells"])] += 1
    summary["legal_move_count_histogram"][str(row["legal_move_count"])] += 1
    summary["search_depth_histogram"][str(row["search_depth"])] += 1
    summary["source_histogram"][str(row["source"])] += 1
    summary["score_margin_histogram"][margin_bucket(row["score_margin"])] += 1


def print_counter(title: str, counter: Counter) -> None:
    print(f"{title}:")
    for key, value in sorted(counter.items(), key=lambda item: (str(item[0]))):
        print(f"  {key}: {value}")


def main() -> int:
    args = parse_args()
    summary = {
        "action_distribution": Counter(),
        "max_tile_histogram": Counter(),
        "empty_cells_histogram": Counter(),
        "legal_move_count_histogram": Counter(),
        "search_depth_histogram": Counter(),
        "source_histogram": Counter(),
        "score_margin_histogram": Counter(),
    }
    total = valid = invalid = 0
    printed_errors = 0

    for input_path in args.input:
        path = Path(input_path)
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                total += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValidationError("row must be a JSON object")
                    validate_row(row)
                except (json.JSONDecodeError, ValidationError) as exc:
                    invalid += 1
                    if printed_errors < args.max_errors:
                        print(f"ERROR {path}:{line_no}: {exc}")
                        printed_errors += 1
                    continue
                valid += 1
                update_summary(summary, row)

    print(f"total rows: {total}")
    print(f"valid rows: {valid}")
    print(f"invalid rows: {invalid}")
    for title, counter in summary.items():
        print_counter(title, counter)

    if args.strict and invalid:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
