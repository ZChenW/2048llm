#!/usr/bin/env python3
"""Smoke tests for make_focused_sft_dataset.py."""

import json
import tempfile
from pathlib import Path

import make_focused_sft_dataset


def row(action: str, depth: int, legal_count: int, margin: float) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Board:\n0 0 0 0\n\nChoose the best move."},
            {"role": "assistant", "content": action},
        ],
        "metadata": {
            "teacher_action": action,
            "search_depth": depth,
            "legal_move_count": legal_count,
            "score_margin": margin,
            "max_tile": 2048,
            "empty_cells": 4,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_focused_small_keeps_original_train_and_adds_capped_priority_sample():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train = root / "train.jsonl"
        val = root / "val.jsonl"
        out = root / "focused_small"

        write_jsonl(
            train,
            [
                row("left", 2, 4, 10.0),
                row("right", 1, 4, 0.5),
                row("up", 2, 3, 50.0),
                row("down", 1, 2, 1000.0),
            ],
        )
        write_jsonl(val, [row("down", 2, 4, 10.0)])

        make_focused_sft_dataset.main(
            [
                "--train-file",
                str(train),
                "--val-file",
                str(val),
                "--out-dir",
                str(out),
                "--extra-rows",
                "2",
                "--seed",
                "7",
            ]
        )

        focused_train = read_jsonl(out / "best_action_raw_move_focused_small_train.jsonl")
        focused_val = read_jsonl(out / "best_action_raw_move_val.jsonl")
        summary = json.loads((out / "best_action_raw_move_focused_small_summary.json").read_text())

        assert len(focused_train) == 6
        assert focused_train[:4] == read_jsonl(train)
        assert len(focused_val) == 1
        assert summary["train_input_rows"] == 4
        assert summary["extra_rows_selected"] == 2
        assert summary["focused_train_rows"] == 6
        assert summary["focused_train_rows"] < 8
        assert summary["sampling_policy"]["val_policy"] == "copied unchanged"
        assert summary["train_input_distribution"]["action_distribution"] == {
            "down": 1,
            "left": 1,
            "right": 1,
            "up": 1,
        }
        assert summary["extra_distribution"]["score_margin_histogram"]


if __name__ == "__main__":
    test_focused_small_keeps_original_train_and_adds_capped_priority_sample()
    print("make_focused_sft_dataset smoke test passed")
