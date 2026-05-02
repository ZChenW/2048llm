#!/usr/bin/env python3
"""Smoke tests for SFT train/eval script CLI and parsing helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    train = ROOT / "scripts" / "train_sft.py"
    eval_script = ROOT / "scripts" / "eval_sft_offline.py"
    run_cmd([sys.executable, str(train), "--help"])
    run_cmd([sys.executable, str(eval_script), "--help"])

    evaluator = load_module(eval_script)
    assert evaluator.parse_prediction("left", "raw_move") == "left"
    assert evaluator.parse_prediction(" left\n", "raw_move") == "left"
    assert evaluator.parse_prediction("<answer>up</answer>", "answer_tag") == "up"
    assert evaluator.parse_prediction("move: right", "raw_move") == "right"
    assert evaluator.parse_prediction("nonsense", "raw_move") is None
    assert evaluator.margin_bucket(None) == "null"
    assert evaluator.margin_bucket(0) == "0"
    assert evaluator.margin_bucket(0.5) == "[0.1,1)"
    print("sft train/eval script smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
