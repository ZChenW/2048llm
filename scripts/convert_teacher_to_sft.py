#!/usr/bin/env python3
"""Convert TD teacher JSONL records into chat-style SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "data" / "teacher_1ply_100k.jsonl",
    ROOT / "data" / "teacher_2ply_50k.jsonl",
]
DEFAULT_OUT_DIR = ROOT / "data" / "sft"
ACTIONS = ["up", "down", "left", "right"]
ACTION_SET = set(ACTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", help="Input teacher JSONL path. Can be repeated.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--format", choices=["raw_move", "answer_tag"], required=True)
    parser.add_argument("--task", choices=["best_action", "action_ranking"], required=True)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-max-tile", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--include-depth", action="store_true")
    parser.add_argument("--include-features", action="store_true")
    return parser.parse_args()


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


def normalize_scores(row: dict[str, Any]) -> dict[str, Any]:
    scores = dict(row["action_scores"])
    valid = set(row["valid_moves"])
    for action in ACTIONS:
        if action not in scores:
            scores[action] = None
        if action not in valid:
            scores[action] = None
    return scores


def normalize_ranking(row: dict[str, Any]) -> list[str]:
    scores = normalize_scores(row)
    valid_moves = list(row["valid_moves"])
    legal = sorted(valid_moves, key=lambda action: (-float(scores[action]), ACTIONS.index(action)))
    illegal = [action for action in ACTIONS if action not in set(valid_moves)]
    return legal + illegal


def board_text(board: list[list[int]]) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in board)


def user_prompt(row: dict[str, Any], task: str, include_depth: bool, include_features: bool) -> str:
    parts = [f"Board:\n{board_text(row['board'])}", f"Valid moves: {', '.join(row['valid_moves'])}"]
    if include_depth:
        parts.append(f"Teacher search depth: {row['search_depth']}")
    if include_features:
        parts.append(
            "Features: "
            f"max_tile={row['max_tile']}, "
            f"empty_cells={row['empty_cells']}, "
            f"legal_move_count={row['legal_move_count']}"
        )
    if task == "best_action":
        parts.append("Choose the best move. Answer only one of: up, down, left, right.")
    else:
        parts.append("Rank all moves from best to worst. Use exactly this format:\nbest > second > third > fourth")
    return "\n\n".join(parts)


def assistant_content(row: dict[str, Any], task: str, fmt: str) -> str:
    if task == "best_action":
        answer = row["teacher_action"]
    else:
        answer = " > ".join(normalize_ranking(row))
    if fmt == "answer_tag":
        return f"<answer>{answer}</answer>"
    return answer


def make_record(row: dict[str, Any], source_file: str, source_line: int, task: str, fmt: str, include_depth: bool, include_features: bool) -> dict[str, Any]:
    metadata = {
        "source_file": source_file,
        "source_line": source_line,
        "search_depth": row["search_depth"],
        "source": row["source"],
        "max_tile": row["max_tile"],
        "empty_cells": row["empty_cells"],
        "legal_move_count": row["legal_move_count"],
        "teacher_action": row["teacher_action"],
        "score_margin": row["score_margin"],
        "valid_moves": row["valid_moves"],
        "action_ranking": normalize_ranking(row),
    }
    return {
        "messages": [
            {"role": "user", "content": user_prompt(row, task, include_depth, include_features)},
            {"role": "assistant", "content": assistant_content(row, task, fmt)},
        ],
        "metadata": metadata,
    }


def update_summary(summary: dict[str, Any], row: dict[str, Any]) -> None:
    summary["action_distribution"][row["teacher_action"]] += 1
    summary["max_tile_histogram"][str(row["max_tile"])] += 1
    summary["empty_cells_histogram"][str(row["empty_cells"])] += 1
    summary["legal_move_count_histogram"][str(row["legal_move_count"])] += 1
    summary["search_depth_histogram"][str(row["search_depth"])] += 1
    summary["score_margin_histogram"][margin_bucket(row["score_margin"])] += 1


def load_records(args: argparse.Namespace, input_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    records: list[dict[str, Any]] = []
    total_input = 0
    summary = {
        "action_distribution": Counter(),
        "max_tile_histogram": Counter(),
        "empty_cells_histogram": Counter(),
        "legal_move_count_histogram": Counter(),
        "search_depth_histogram": Counter(),
        "score_margin_histogram": Counter(),
    }
    for path in input_paths:
        with path.open(encoding="utf-8") as f:
            for source_line, line in enumerate(f, 1):
                total_input += 1
                row = json.loads(line)
                if args.min_max_tile is not None and row["max_tile"] < args.min_max_tile:
                    continue
                records.append(
                    make_record(
                        row,
                        str(path),
                        source_line,
                        args.task,
                        args.format,
                        args.include_depth,
                        args.include_features,
                    )
                )
                update_summary(summary, row)
                if args.max_rows is not None and len(records) >= args.max_rows:
                    return records, summary, total_input
    return records, summary, total_input


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def main() -> int:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be in [0, 1)")
    input_paths = [Path(p) for p in (args.input if args.input else DEFAULT_INPUTS)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, summary_counters, total_input = load_records(args, input_paths)
    if args.shuffle:
        random.Random(args.seed).shuffle(records)
    val_count = int(math.floor(len(records) * args.val_ratio))
    val_records = records[:val_count]
    train_records = records[val_count:]

    stem = f"{args.task}_{args.format}"
    train_path = out_dir / f"{stem}_train.jsonl"
    val_path = out_dir / f"{stem}_val.jsonl"
    summary_path = out_dir / f"{stem}_summary.json"
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    summary = {
        "total_input_rows": total_input,
        "used_rows": len(records),
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "task": args.task,
        "format": args.format,
        "input_files": [str(path) for path in input_paths],
        "action_distribution": counter_to_dict(summary_counters["action_distribution"]),
        "max_tile_histogram": counter_to_dict(summary_counters["max_tile_histogram"]),
        "empty_cells_histogram": counter_to_dict(summary_counters["empty_cells_histogram"]),
        "legal_move_count_histogram": counter_to_dict(summary_counters["legal_move_count_histogram"]),
        "search_depth_histogram": counter_to_dict(summary_counters["search_depth_histogram"]),
        "score_margin_histogram": counter_to_dict(summary_counters["score_margin_histogram"]),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"wrote train: {train_path} ({len(train_records)} rows)")
    print(f"wrote val: {val_path} ({len(val_records)} rows)")
    print(f"wrote summary: {summary_path}")
    print("first 3 samples:")
    for record in records[:3]:
        print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
