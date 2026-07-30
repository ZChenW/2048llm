from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


FAKE_EXPORTER = r"""#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

ACTIONS = ("up", "down", "left", "right")
SCORES = {"up": 4.0, "down": 3.0, "left": 2.0, "right": 1.0}

parser = argparse.ArgumentParser()
parser.add_argument("--weights")
parser.add_argument("--depth")
parser.add_argument("--seed")
parser.add_argument("--out-dir", type=Path)
parser.add_argument("--max-trajectories")
parser.add_argument("--late-min-max-tile", type=int)
parser.add_argument("--teacher-core-count", type=int)
parser.add_argument("--split-target", action="append")
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

targets = {}
for target in args.split_target:
    split, natural, hard, late = target.split(":")
    targets[split] = {
        "natural": int(natural),
        "hard": int(hard),
        "late": int(late),
    }

boards = {
    "natural": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 0, 0]],
    "hard": [[2, 2, 4, 8], [16, 32, 64, 128], [2, 4, 8, 16], [4, 8, 16, 32]],
    "late": [[2, 2, 4, 8], [16, 32, 64, 512], [2, 4, 8, 16], [4, 8, 16, 32]],
}

def slide(line):
    values = [value for value in line if value]
    result = []
    index = 0
    while index < len(values):
        if index + 1 < len(values) and values[index] == values[index + 1]:
            result.append(values[index] * 2)
            index += 2
        else:
            result.append(values[index])
            index += 1
    return result + [0] * (4 - len(result))

def moved(board, action):
    rows = [row[:] for row in board]
    if action == "left":
        return [slide(row) for row in rows]
    if action == "right":
        return [list(reversed(slide(list(reversed(row))))) for row in rows]
    columns = [[rows[row][column] for row in range(4)] for column in range(4)]
    if action == "down":
        columns = [list(reversed(slide(list(reversed(column))))) for column in columns]
    else:
        columns = [slide(column) for column in columns]
    return [[columns[column][row] for column in range(4)] for row in range(4)]

def orbit_id(board):
    def rotate(value):
        return [[value[3 - column][row] for column in range(4)] for row in range(4)]
    reflected = [list(reversed(row)) for row in board]
    variants = []
    for start in (board, reflected):
        value = start
        for _ in range(4):
            variants.append(value)
            value = rotate(value)
    def packed(value):
        result = 0
        for index, tile in enumerate(cell for row in value for cell in row):
            exponent = 0 if tile == 0 else tile.bit_length() - 1
            result |= exponent << (index * 4)
        return result
    return f"{min(packed(value) for value in variants):016x}"

core_remaining = args.teacher_core_count
record_number = 0
for split in ("train", "validation", "test"):
    with (args.out_dir / f"{split}.jsonl").open("w") as output:
        for stratum in ("natural", "hard", "late"):
            for stratum_index in range(targets[split][stratum]):
                board = [row[:] for row in boards[stratum]]
                # Give every record a distinct orbit without changing its stratum.
                board[0][2] = 2 ** (record_number % 7 + 1)
                board[0][3] = 2 ** (record_number // 7 + 1)
                valid_moves = [
                    action for action in ACTIONS if moved(board, action) != board
                ]
                action_scores = {
                    action: SCORES[action] if action in valid_moves else None
                    for action in ACTIONS
                }
                ranking = sorted(
                    valid_moves, key=lambda action: (-SCORES[action], ACTIONS.index(action))
                )
                ranking += [action for action in ACTIONS if action not in valid_moves]
                top_scores = sorted(
                    (SCORES[action] for action in valid_moves), reverse=True
                )
                top1 = top_scores[0]
                top2 = top_scores[1] if len(top_scores) > 1 else top1
                teacher_core = split == "train" and core_remaining > 0
                if teacher_core:
                    core_remaining -= 1
                row = {
                    "schema_version": 1,
                    "record_id": f"{split}-{record_number:08d}",
                    "board": board,
                    "valid_moves": valid_moves,
                    "action_scores": action_scores,
                    "action_ranking": ranking,
                    "teacher_action": ranking[0],
                    "max_tile": max(max(line) for line in board),
                    "empty_cells": sum(cell == 0 for line in board for cell in line),
                    "legal_move_count": len(valid_moves),
                    "top1_score": top1,
                    "top2_score": top2,
                    "score_margin": top1 - top2,
                    "search_depth": int(args.depth),
                    "source": "retained_100m_teacher_policy",
                    "split": split,
                    "stratum": stratum,
                    "teacher_core": teacher_core,
                    "lineage": {
                        "trajectory_id": f"{split}-trajectory-{record_number}",
                        "trajectory_step": stratum_index,
                        "orbit_id": orbit_id(board),
                        "symmetry_id": 0,
                    },
                }
                output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                record_number += 1
"""


class TeacherCorpusCliTests(unittest.TestCase):
    def run_runner(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        source_path = str(REPO_ROOT / "src")
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path
            if not existing_pythonpath
            else os.pathsep.join((source_path, existing_pythonpath))
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, "-m", "llm2048.experiment_runner", *arguments],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def write_fixture(self, root: Path, exporter_source: str = FAKE_EXPORTER) -> Path:
        checkpoint = root / "ckpt_100000000.bin"
        checkpoint.write_bytes(b"retained Teacher Policy checkpoint")
        metadata = root / "ckpt_100000000.json"
        metadata.write_text(
            json.dumps(
                {
                    "episodes": 100_000_000,
                    "lr": 0.100000001,
                    "seed": 100,
                    "mode": "highwin8192",
                    "tuple_count": 8,
                    "tuple_len": 6,
                    "alphabet_size": 16,
                    "weight_file_size": checkpoint.stat().st_size,
                }
            )
        )
        exporter = root / "fake_exporter.py"
        exporter.write_text(textwrap.dedent(exporter_source))
        exporter.chmod(0o755)
        config = root / "corpus.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "export_name": "depth2-smoke",
                    "seed": 2048,
                    "checkpoint": {
                        "path": str(checkpoint),
                        "sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                        "metadata_path": str(metadata),
                        "metadata_sha256": sha256(metadata.read_bytes()).hexdigest(),
                    },
                    "teacher_policy": {
                        "search_depth": 2,
                        "source": "retained_100m_teacher_policy",
                    },
                    "exporter": {"path": str(exporter)},
                    "corpus": {
                        "splits": {
                            "train": {"natural": 5, "hard": 3, "late": 2},
                            "validation": {"natural": 1, "hard": 1, "late": 0},
                            "test": {"natural": 1, "hard": 1, "late": 0},
                        },
                        "teacher_core_count": 10,
                        "late_min_max_tile": 512,
                        "max_trajectories": 100,
                    },
                },
                sort_keys=True,
            )
        )
        return config

    def test_teacher_corpus_export_is_validated_and_manifested_deterministically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = self.write_fixture(root)
            checkpoint = root / "ckpt_100000000.bin"
            first_output = root / "first"
            second_output = root / "second"

            for output_directory in (first_output, second_output):
                completed = self.run_runner(
                    "--teacher-corpus-config",
                    str(config),
                    "--output-dir",
                    str(output_directory),
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            first_manifest = json.loads((first_output / "manifest.json").read_text())
            self.assertEqual(first_manifest["schema_version"], 1)
            self.assertEqual(first_manifest["export"]["name"], "depth2-smoke")
            self.assertEqual(first_manifest["teacher_policy"]["search_depth"], 2)
            self.assertEqual(
                first_manifest["teacher_policy"]["checkpoint"]["sha256"],
                sha256(checkpoint.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                first_manifest["teacher_policy"]["checkpoint"]["episodes"],
                100_000_000,
            )
            self.assertEqual(
                first_manifest["corpus"]["counts"],
                {"test": 2, "train": 10, "validation": 2},
            )
            self.assertEqual(
                first_manifest["corpus"]["strata"]["train"],
                {"hard": 3, "late": 2, "natural": 5},
            )
            self.assertEqual(first_manifest["teacher_core"]["count"], 10)
            self.assertEqual(
                first_manifest["teacher_core"]["strata"],
                {"hard": 3, "late": 2, "natural": 5},
            )
            self.assertEqual(
                first_manifest["corpus"]["symmetry_augmentation"],
                "none_identity_lineage_only",
            )
            self.assertEqual(first_manifest["calibration"]["method"], "median_positive_margin")
            self.assertEqual(first_manifest["calibration"]["scope"], "train")
            self.assertGreater(first_manifest["calibration"]["tau"], 0)

            artifacts = {
                artifact["name"]: artifact for artifact in first_manifest["artifacts"]
            }
            self.assertEqual(set(artifacts), {"test", "train", "validation"})
            for split in ("train", "validation", "test"):
                path = first_output / artifacts[split]["path"]
                self.assertEqual(
                    artifacts[split]["sha256"], sha256(path.read_bytes()).hexdigest()
                )
                self.assertEqual(
                    path.read_bytes(), (second_output / f"{split}.jsonl").read_bytes()
                )
            self.assertEqual(
                first_manifest,
                json.loads((second_output / "manifest.json").read_text()),
            )

    def test_teacher_corpus_rejects_a_symmetry_orbit_crossing_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            colliding_exporter = FAKE_EXPORTER.replace(
                "2 ** (record_number % 7 + 1)",
                "2 ** (stratum_index % 7 + 1)",
            ).replace(
                "2 ** (record_number // 7 + 1)",
                "2 ** (stratum_index // 7 + 1)",
            )
            config = self.write_fixture(root, colliding_exporter)
            output_directory = root / "output"

            completed = self.run_runner(
                "--teacher-corpus-config",
                str(config),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("symmetry orbit", completed.stderr)
            self.assertIn("crosses train and validation", completed.stderr)
            self.assertFalse((output_directory / "manifest.json").exists())

    def test_teacher_corpus_rejects_moves_that_are_not_legal_on_the_board(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            illegal_move_exporter = FAKE_EXPORTER.replace(
                "                action_scores = {\n",
                '                if split == "validation":\n'
                '                    valid_moves = ["up"]\n'
                "                action_scores = {\n",
            )
            config = self.write_fixture(root, illegal_move_exporter)
            output_directory = root / "output"

            completed = self.run_runner(
                "--teacher-corpus-config",
                str(config),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "valid_moves does not match board legality", completed.stderr
            )
            self.assertFalse((output_directory / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
