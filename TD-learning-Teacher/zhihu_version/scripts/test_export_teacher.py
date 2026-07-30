#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
CONFIG = Path(
    os.environ.get(
        "TEACHER_CORPUS_CONFIG",
        str(REPO_ROOT / "configs" / "teacher_corpus_depth2_smoke.json"),
    )
)


def run(cmd, *, cwd=ROOT, env=None, timeout=60):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def main():
    config = json.loads(CONFIG.read_text())
    checkpoint = (CONFIG.resolve().parent / config["checkpoint"]["path"]).resolve()
    if not checkpoint.exists():
        raise AssertionError(f"missing frozen teacher weights: {checkpoint}")
    metadata = (
        CONFIG.resolve().parent / config["checkpoint"]["metadata_path"]
    ).resolve()
    if not metadata.exists():
        raise AssertionError(f"missing frozen teacher metadata: {metadata}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_dir = tmp_path / "corpus"
        runner_env = os.environ.copy()
        source_root = str(REPO_ROOT / "src")
        runner_env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (source_root, runner_env.get("PYTHONPATH")))
        )
        export = run(
            [
                sys.executable,
                "-m",
                "llm2048.experiment_runner",
                "--teacher-corpus-config",
                str(CONFIG.resolve()),
                "--output-dir",
                str(out_dir),
            ],
            cwd=REPO_ROOT,
            env=runner_env,
            timeout=120,
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

        # Corpus mode needs a strict four-action schema. The pre-existing
        # single-file mode remains a legal-action-only compatibility format.
        legacy_path = tmp_path / "legacy.jsonl"
        legacy_export = run(
            [
                str(out_dir / "export_teacher"),
                "--weights",
                str(checkpoint),
                "--samples",
                "8",
                "--depth",
                "1",
                "--seed",
                "7",
                "--out",
                str(legacy_path),
                "--max-games",
                "100",
                "--min-max-tile",
                "0",
                "--hard-state-ratio",
                "0.25",
                "--report-every",
                "0",
            ]
        )
        if legacy_export.returncode != 0:
            raise AssertionError(legacy_export.stdout)
        legacy_rows = [
            json.loads(line)
            for line in legacy_path.read_text().splitlines()
            if line.strip()
        ]
        if len(legacy_rows) != 8:
            raise AssertionError(
                f"expected 8 legacy rows, found {len(legacy_rows)}"
            )
        for row in legacy_rows:
            valid_moves = set(row["valid_moves"])
            if set(row["action_scores"]) != valid_moves:
                raise AssertionError(row)
            if set(row["action_ranking"]) != valid_moves:
                raise AssertionError(row)
            if any(score is None for score in row["action_scores"].values()):
                raise AssertionError(row)


if __name__ == "__main__":
    sys.exit(main())
