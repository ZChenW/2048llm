#!/usr/bin/env python3
"""Create a focused-small SFT train split by oversampling weak slices."""

from __future__ import annotations

import argparse
import heapq
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "sft" / "best_action_raw_move_train.jsonl"
DEFAULT_VAL = ROOT / "data" / "sft" / "best_action_raw_move_val.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "sft_focused_small"
OUTPUT_TRAIN_NAME = "best_action_raw_move_focused_small_train.jsonl"
OUTPUT_VAL_NAME = "best_action_raw_move_val.jsonl"
OUTPUT_SUMMARY_NAME = "best_action_raw_move_focused_small_summary.json"

SCORE_BUCKET_ORDER = [
    "0",
    "(0,0.001)",
    "[0.001,0.01)",
    "[0.01,0.1)",
    "[0.1,1)",
    "[1,10)",
    "[10,100)",
    "[100,1000)",
    ">=1000",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--extra-rows", type=int, default=78000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assistant_action(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("teacher_action"):
        return str(metadata["teacher_action"])
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def score_margin(row: dict[str, Any]) -> float:
    return float((row.get("metadata") or {}).get("score_margin", 0.0) or 0.0)


def score_margin_bucket(value: Any) -> str:
    margin = float(value or 0.0)
    if margin == 0:
        return "0"
    if margin < 0.001:
        return "(0,0.001)"
    if margin < 0.01:
        return "[0.001,0.01)"
    if margin < 0.1:
        return "[0.01,0.1)"
    if margin < 1:
        return "[0.1,1)"
    if margin < 10:
        return "[1,10)"
    if margin < 100:
        return "[10,100)"
    if margin < 1000:
        return "[100,1000)"
    return ">=1000"


def sort_key(value: str) -> tuple[int, Any]:
    if value in SCORE_BUCKET_ORDER:
        return (0, SCORE_BUCKET_ORDER.index(value))
    try:
        return (1, int(value))
    except ValueError:
        return (2, value)


def counter_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(counter[k]) for k in sorted(counter, key=lambda x: sort_key(str(x)))}


def distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    action = Counter()
    depth = Counter()
    legal = Counter()
    margin = Counter()
    max_tile = Counter()
    empty = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        action[assistant_action(row)] += 1
        depth[str(metadata.get("search_depth", ""))] += 1
        legal[str(metadata.get("legal_move_count", ""))] += 1
        margin[score_margin_bucket(metadata.get("score_margin", 0.0))] += 1
        max_tile[str(metadata.get("max_tile", ""))] += 1
        empty[str(metadata.get("empty_cells", ""))] += 1
    return {
        "action_distribution": counter_dict(action),
        "search_depth_histogram": counter_dict(depth),
        "legal_move_count_histogram": counter_dict(legal),
        "score_margin_histogram": counter_dict(margin),
        "max_tile_histogram": counter_dict(max_tile),
        "empty_cells_histogram": counter_dict(empty),
    }


def priority_rules(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") or {}
    margin = score_margin(row)
    rules: list[str] = []
    if assistant_action(row) in {"left", "right"}:
        rules.append("gold_left_right")
    if int(metadata.get("legal_move_count", 0) or 0) == 4:
        rules.append("legal_move_count_4")
    if int(metadata.get("search_depth", 0) or 0) == 2:
        rules.append("search_depth_2")
    if 1 <= margin < 100:
        rules.append("score_margin_[1,100)")
    if 0.1 <= margin < 1:
        rules.append("score_margin_[0.1,1)")
    return rules


def sample_without_replacement(
    candidates: list[tuple[int, dict[str, Any]]],
    k: int,
    seed: int,
    weight_fn,
) -> list[tuple[int, dict[str, Any]]]:
    if k <= 0:
        return []
    if k >= len(candidates):
        return list(candidates)
    rng = random.Random(seed)
    heap: list[tuple[float, int, dict[str, Any]]] = []
    for index, row in candidates:
        weight = max(float(weight_fn(row)), 1e-6)
        # Efraimidis-Spirakis weighted sample without replacement.
        key = rng.random() ** (1.0 / weight)
        item = (key, index, row)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif key > heap[0][0]:
            heapq.heapreplace(heap, item)
    return [(index, row) for _, index, row in sorted(heap, key=lambda item: item[1])]


def focused_weight(row: dict[str, Any]) -> float:
    rules = set(priority_rules(row))
    weight = 0.0
    if "gold_left_right" in rules:
        weight += 4.0
    if "legal_move_count_4" in rules:
        weight += 3.0
    if "search_depth_2" in rules:
        weight += 3.0
    if "score_margin_[1,100)" in rules:
        weight += 2.0
    if "score_margin_[0.1,1)" in rules:
        weight += 1.0
    return weight


def build_focused_small_train(
    train_rows: list[dict[str, Any]],
    extra_rows: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, Any]]], Counter]:
    if extra_rows < 0:
        raise ValueError("--extra-rows must be non-negative")
    candidates = [(idx, row) for idx, row in enumerate(train_rows) if priority_rules(row)]
    selected = sample_without_replacement(candidates, extra_rows, seed, focused_weight)
    rule_hits = Counter()
    for _, row in selected:
        for rule in priority_rules(row):
            rule_hits[rule] += 1
    return list(train_rows) + [row for _, row in selected], selected, rule_hits


def run(train_file: Path, val_file: Path, out_dir: Path, extra_rows: int, seed: int) -> dict[str, Any]:
    train_rows = read_jsonl(train_file)
    val_rows = read_jsonl(val_file)
    focused_train, selected_extra, rule_hits = build_focused_small_train(train_rows, extra_rows, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / OUTPUT_TRAIN_NAME
    val_out = out_dir / OUTPUT_VAL_NAME
    summary_out = out_dir / OUTPUT_SUMMARY_NAME
    write_jsonl(train_out, focused_train)
    write_jsonl(val_out, val_rows)

    extra_rows_data = [row for _, row in selected_extra]
    summary = {
        "inputs": {
            "train_file": str(train_file),
            "val_file": str(val_file),
        },
        "outputs": {
            "focused_train_file": str(train_out),
            "val_file": str(val_out),
            "summary_file": str(summary_out),
        },
        "sampling_policy": {
            "base": "copy every original train row exactly once",
            "extra_rows_requested": extra_rows,
            "extra_sampling": "weighted sample without replacement from rows matching any priority rule",
            "seed": seed,
            "priority_rules": [
                "teacher_action in {left,right}",
                "legal_move_count == 4",
                "search_depth == 2",
                "1 <= score_margin < 100",
                "small inclusion: 0.1 <= score_margin < 1",
            ],
            "weights": {
                "teacher_action in {left,right}": 4.0,
                "legal_move_count == 4": 3.0,
                "search_depth == 2": 3.0,
                "1 <= score_margin < 100": 2.0,
                "0.1 <= score_margin < 1": 1.0,
            },
            "val_policy": "copied unchanged",
        },
        "train_input_rows": len(train_rows),
        "val_rows": len(val_rows),
        "extra_rows_selected": len(selected_extra),
        "focused_train_rows": len(focused_train),
        "focused_to_original_train_ratio": len(focused_train) / len(train_rows) if train_rows else 0.0,
        "extra_rule_hits": counter_dict(rule_hits),
        "train_input_distribution": distribution(train_rows),
        "extra_distribution": distribution(extra_rows_data),
        "focused_train_distribution": distribution(focused_train),
        "val_distribution": distribution(val_rows),
    }
    write_json(summary_out, summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run(args.train_file, args.val_file, args.out_dir, args.extra_rows, args.seed)
    print(f"Wrote focused-small train: {summary['outputs']['focused_train_file']}")
    print(f"Wrote val copy: {summary['outputs']['val_file']}")
    print(f"Wrote summary: {summary['outputs']['summary_file']}")
    print(
        "rows: "
        f"train_input={summary['train_input_rows']} "
        f"extra={summary['extra_rows_selected']} "
        f"focused_train={summary['focused_train_rows']} "
        f"val={summary['val_rows']} "
        f"ratio={summary['focused_to_original_train_ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
