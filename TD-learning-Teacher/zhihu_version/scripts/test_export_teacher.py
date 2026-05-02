#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (
    Path("/home/chakew/Projects/llm_rl_2048")
    / "artifacts"
    / "best_td_teacher"
    / "best_highwin_td.bin"
)


def run(cmd, *, timeout=60):
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def main():
    if not WEIGHTS.exists():
        raise AssertionError(f"missing frozen teacher weights: {WEIGHTS}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bin_path = tmp_path / "export_teacher"
        out_path = tmp_path / "teacher.jsonl"

        build = run(["g++", "-O3", "-std=c++17", "export_teacher.cpp", "-o", str(bin_path)], timeout=120)
        if build.returncode != 0:
            raise AssertionError(build.stdout)

        export = run(
            [
                str(bin_path),
                "--weights",
                str(WEIGHTS),
                "--samples",
                "8",
                "--depth",
                "1",
                "--seed",
                "7",
                "--out",
                str(out_path),
                "--max-games",
                "100",
                "--min-max-tile",
                "0",
                "--hard-state-ratio",
                "0.25",
                "--report-every",
                "0",
            ],
            timeout=60,
        )
        if export.returncode != 0:
            raise AssertionError(export.stdout)

        rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        if len(rows) != 8:
            raise AssertionError(f"expected 8 rows, found {len(rows)}")

        required = {
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
        for row in rows:
            missing = required - set(row)
            if missing:
                raise AssertionError(f"missing keys {sorted(missing)} in {row}")
            if row["source"] != "best_highwin_td":
                raise AssertionError(row)
            if row["teacher_action"] not in row["valid_moves"]:
                raise AssertionError(row)
            if len(row["board"]) != 4 or any(len(line) != 4 for line in row["board"]):
                raise AssertionError(row["board"])
            if row["search_depth"] != 1:
                raise AssertionError(row)


if __name__ == "__main__":
    sys.exit(main())
