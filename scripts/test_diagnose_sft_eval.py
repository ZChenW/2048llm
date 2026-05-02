#!/usr/bin/env python3
"""Smoke tests for diagnose_sft_eval.py."""

import json
import tempfile
from pathlib import Path

import diagnose_sft_eval


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def sft_row(action, depth, margin, max_tile, empty_cells, legal_count):
    return {
        "messages": [
            {"role": "user", "content": "Board:\n0 0 0 0\n\nChoose the best move."},
            {"role": "assistant", "content": action},
        ],
        "metadata": {
            "teacher_action": action,
            "search_depth": depth,
            "score_margin": margin,
            "max_tile": max_tile,
            "empty_cells": empty_cells,
            "legal_move_count": legal_count,
        },
    }


def test_diagnose_sft_eval_smoke_outputs_expected_tables_and_samples():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        eval_json = root / "eval.json"
        bad_cases = root / "bad.jsonl"
        val_file = root / "val.jsonl"
        out_dir = root / "diagnostics"

        eval_json.write_text(
            json.dumps(
                {
                    "total": 4,
                    "exact_match": 2,
                    "exact_match_rate": 0.5,
                    "parse_fail_count": 0,
                    "parse_fail_rate": 0.0,
                    "invalid_action_count": 0,
                    "invalid_action_rate": 0.0,
                    "confusion_matrix": {
                        "left": {"left": 1, "down": 1},
                        "right": {"right": 1, "up": 1},
                    },
                    "accuracy_by_search_depth": {
                        "1": {"correct": 1, "total": 2, "accuracy": 0.5},
                        "2": {"correct": 1, "total": 2, "accuracy": 0.5},
                    },
                    "accuracy_by_score_margin_bucket": {},
                    "accuracy_by_max_tile": {},
                    "accuracy_by_empty_cells": {},
                }
            ),
            encoding="utf-8",
        )
        write_jsonl(
            val_file,
            [
                sft_row("left", 1, 150.0, 2048, 3, 2),
                sft_row("left", 2, 0.5, 4096, 5, 4),
                sft_row("right", 2, 1200.0, 4096, 4, 2),
                sft_row("right", 1, 5.0, 1024, 6, 3),
            ],
        )
        write_jsonl(
            bad_cases,
            [
                {
                    "index": 1,
                    "expected": "left",
                    "prediction": "down",
                    "metadata": {
                        "search_depth": 1,
                        "score_margin": 150.0,
                        "max_tile": 2048,
                        "empty_cells": 3,
                        "legal_move_count": 2,
                    },
                },
                {
                    "index": 3,
                    "expected": "right",
                    "prediction": "up",
                    "metadata": {
                        "search_depth": 2,
                        "score_margin": 1200.0,
                        "max_tile": 4096,
                        "empty_cells": 4,
                        "legal_move_count": 2,
                    },
                },
            ],
        )

        diagnose_sft_eval.main(
            [
                "--eval-json",
                str(eval_json),
                "--bad-cases",
                str(bad_cases),
                "--val-file",
                str(val_file),
                "--out-dir",
                str(out_dir),
                "--sample-limit",
                "10",
            ]
        )

        summary = (out_dir / "diagnosis_summary.md").read_text(encoding="utf-8")
        assert "exact_match_rate" in summary
        assert "Accuracy by legal_move_count" in summary

        legal_stats = json.loads((out_dir / "diagnostics_summary.json").read_text())
        assert legal_stats["accuracy_by_legal_move_count"]["2"]["total"] == 2
        assert legal_stats["accuracy_by_legal_move_count"]["2"]["correct"] == 0
        assert legal_stats["cross_gold_action_by_search_depth"]["left"]["1"]["total"] == 1
        assert legal_stats["cross_search_depth_by_score_margin_bucket"]["2"][">=1000"]["errors"] == 1

        assert sum(1 for _ in (out_dir / "samples" / "very_high_margin_errors.jsonl").open()) == 1
        assert sum(1 for _ in (out_dir / "samples" / "left_right_to_up_down_errors.jsonl").open()) == 2


if __name__ == "__main__":
    test_diagnose_sft_eval_smoke_outputs_expected_tables_and_samples()
    print("diagnose_sft_eval smoke test passed")
