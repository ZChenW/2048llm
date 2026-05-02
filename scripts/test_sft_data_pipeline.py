#!/usr/bin/env python3
"""Regression test for teacher validation and SFT conversion scripts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATE = ROOT / "scripts" / "validate_teacher_dataset.py"
CONVERT = ROOT / "scripts" / "convert_teacher_to_sft.py"
ACTIONS = {"up", "down", "left", "right"}


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def write_teacher(path: Path) -> None:
    rows = [
        {
            "board": [[2, 4, 8, 16], [0, 0, 2, 32], [0, 4, 8, 64], [2, 8, 16, 128]],
            "valid_moves": ["left", "up", "down"],
            "action_scores": {"up": 7.0, "down": 5.0, "left": 9.0, "right": None},
            "action_ranking": ["left", "up", "down", "right"],
            "teacher_action": "left",
            "max_tile": 128,
            "empty_cells": 3,
            "legal_move_count": 3,
            "top1_score": 9.0,
            "top2_score": 7.0,
            "score_margin": 2.0,
            "search_depth": 1,
            "source": "best_highwin_td",
        },
        {
            "board": [[4, 8, 16, 32], [2, 4, 8, 64], [0, 2, 4, 128], [0, 0, 2, 256]],
            "valid_moves": ["right", "up"],
            "action_scores": {"up": 8.5, "down": None, "left": None, "right": 10.25},
            "action_ranking": ["right", "up", "down", "left"],
            "teacher_action": "right",
            "max_tile": 256,
            "empty_cells": 3,
            "legal_move_count": 2,
            "top1_score": 10.25,
            "top2_score": 8.5,
            "score_margin": 1.75,
            "search_depth": 2,
            "source": "best_highwin_td",
        },
        {
            "board": [[8, 16, 32, 64], [4, 8, 16, 128], [2, 4, 8, 256], [0, 2, 4, 512]],
            "valid_moves": ["up"],
            "action_scores": {"up": 11.0, "down": None, "left": None, "right": None},
            "action_ranking": ["up", "down", "left", "right"],
            "teacher_action": "up",
            "max_tile": 512,
            "empty_cells": 1,
            "legal_move_count": 1,
            "top1_score": 11.0,
            "top2_score": None,
            "score_margin": None,
            "search_depth": 2,
            "source": "best_highwin_td",
        },
        {
            "board": [[16, 32, 64, 128], [8, 16, 32, 256], [4, 8, 16, 512], [2, 4, 8, 1024]],
            "valid_moves": ["down", "right"],
            "action_scores": {"up": None, "down": 12.0, "left": None, "right": 13.0},
            "action_ranking": ["right", "down", "up", "left"],
            "teacher_action": "right",
            "max_tile": 1024,
            "empty_cells": 0,
            "legal_move_count": 2,
            "top1_score": 13.0,
            "top2_score": 12.0,
            "score_margin": 1.0,
            "search_depth": 1,
            "source": "best_highwin_td",
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def unwrap_answer(content: str, answer_tag: bool) -> str:
    if not answer_tag:
        return content
    assert content.startswith("<answer>") and content.endswith("</answer>"), content
    return content[len("<answer>") : -len("</answer>")]


def check_outputs(out_dir: Path, task: str, fmt: str, expected_total: int) -> None:
    stem = f"{task}_{fmt}"
    train_path = out_dir / f"{stem}_train.jsonl"
    val_path = out_dir / f"{stem}_val.jsonl"
    summary_path = out_dir / f"{stem}_summary.json"
    assert train_path.exists(), train_path
    assert val_path.exists(), val_path
    assert summary_path.exists(), summary_path

    rows = read_jsonl(train_path) + read_jsonl(val_path)
    assert len(rows) == expected_total, (task, fmt, len(rows))
    answer_tag = fmt == "answer_tag"
    for row in rows:
        messages = row["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        answer = unwrap_answer(messages[1]["content"], answer_tag)
        meta = row["metadata"]
        if task == "best_action":
            assert answer in meta["valid_moves"], (answer, meta["valid_moves"])
        else:
            parts = [part.strip() for part in answer.split(">")]
            assert set(parts) == ACTIONS, answer
            assert len(parts) == 4, answer


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        teacher_path = tmp_path / "teacher.jsonl"
        out_dir = tmp_path / "sft"
        write_teacher(teacher_path)

        run_cmd([sys.executable, str(VALIDATE), "--input", str(teacher_path), "--strict"])

        for task in ["best_action", "action_ranking"]:
            for fmt in ["raw_move", "answer_tag"]:
                run_cmd(
                    [
                        sys.executable,
                        str(CONVERT),
                        "--input",
                        str(teacher_path),
                        "--out-dir",
                        str(out_dir),
                        "--task",
                        task,
                        "--format",
                        fmt,
                        "--val-ratio",
                        "0.25",
                        "--seed",
                        "7",
                    ]
                )
                check_outputs(out_dir, task, fmt, expected_total=4)

    print("sft data pipeline regression test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
