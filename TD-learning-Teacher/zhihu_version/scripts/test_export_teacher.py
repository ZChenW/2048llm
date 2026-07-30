#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(
    os.environ.get(
        "TEACHER_CHECKPOINT",
        str(
            ROOT.parents[1]
            / "runs"
            / "zhihu_repro_20260729"
            / "best"
            / "ckpt_100000000.bin"
        ),
    )
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
        out_dir = tmp_path / "corpus"
        out_dir.mkdir()

        build = run(["g++", "-O3", "-std=c++17", "export_teacher.cpp", "-o", str(bin_path)], timeout=120)
        if build.returncode != 0:
            raise AssertionError(build.stdout)

        export = run(
            [
                str(bin_path),
                "--weights",
                str(WEIGHTS),
                "--depth",
                "2",
                "--seed",
                "7",
                "--out-dir",
                str(out_dir),
                "--max-trajectories",
                "100",
                "--late-min-max-tile",
                "512",
                "--teacher-core-count",
                "10",
                "--split-target",
                "train:5:3:2",
                "--split-target",
                "validation:5:3:2",
                "--split-target",
                "test:5:3:2",
            ],
            timeout=60,
        )
        if export.returncode != 0:
            raise AssertionError(export.stdout)

        required = {
            "schema_version",
            "record_id",
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
            "split",
            "stratum",
            "teacher_core",
            "lineage",
        }
        seen_trajectories = {}
        seen_orbits = {}
        for split in ("train", "validation", "test"):
            rows = [
                json.loads(line)
                for line in (out_dir / f"{split}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if len(rows) != 10:
                raise AssertionError(f"expected 10 {split} rows, found {len(rows)}")
            strata = {"natural": 0, "hard": 0, "late": 0}
            for row in rows:
                missing = required - set(row)
                if missing:
                    raise AssertionError(f"missing keys {sorted(missing)} in {row}")
                if row["source"] != "retained_100m_teacher_policy":
                    raise AssertionError(row)
                if row["teacher_action"] not in row["valid_moves"]:
                    raise AssertionError(row)
                if set(row["action_scores"]) != {"up", "down", "left", "right"}:
                    raise AssertionError(row)
                if set(row["action_ranking"]) != {"up", "down", "left", "right"}:
                    raise AssertionError(row)
                if len(row["board"]) != 4 or any(len(line) != 4 for line in row["board"]):
                    raise AssertionError(row["board"])
                if row["search_depth"] != 2 or row["split"] != split:
                    raise AssertionError(row)
                if row["stratum"] == "late" and row["max_tile"] < 512:
                    raise AssertionError(row)
                strata[row["stratum"]] += 1
                for key, owners in (
                    ("trajectory_id", seen_trajectories),
                    ("orbit_id", seen_orbits),
                ):
                    identifier = row["lineage"][key]
                    owner = owners.setdefault(identifier, split)
                    if owner != split:
                        raise AssertionError(f"{key} crosses {owner} and {split}")
            if strata != {"natural": 5, "hard": 3, "late": 2}:
                raise AssertionError(strata)


if __name__ == "__main__":
    sys.exit(main())
