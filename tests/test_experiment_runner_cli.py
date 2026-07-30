from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = REPO_ROOT / "tests" / "fixtures" / "experiment_runner_tracer.json"
FIXTURE_BOARD = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [2, 2, 0, 0],
]


class ExperimentRunnerCliTests(unittest.TestCase):
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
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["WANDB_MODE"] = "offline"
        environment["WANDB_SILENT"] = "true"

        return subprocess.run(
            [
                sys.executable,
                "-m",
                "llm2048.experiment_runner",
                *arguments,
            ],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_deterministic_fixture_run_emits_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "run"

            completed = self.run_runner(
                "--config",
                str(FIXTURE_CONFIG),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)

            result = json.loads((output_directory / "result.json").read_text())
            self.assertEqual(
                result,
                {
                    "completed_steps": 4,
                    "metrics": {
                        "mean_reward": 0.625,
                        "reward_total": 2.5,
                    },
                    "status": "completed",
                },
            )

            manifest = json.loads((output_directory / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["experiment"]["name"], "issue-2-tracer")
            self.assertEqual(manifest["experiment"]["seed"], 2048)
            self.assertEqual(manifest["experiment"]["total_steps"], 4)
            self.assertEqual(
                manifest["experiment"]["id"],
                "6b3e53816d725eaff4455b4ac0f318db68aa3e2683020c76a80ec09495b55e94",
            )
            self.assertEqual(
                manifest["input"]["sha256"],
                "44f82ecfa33f089a6b9eb0937b1c3d9ced7aef87b6f84c515720f02014e1e8fd",
            )
            self.assertEqual(manifest["environment"]["packages"]["wandb"], version("wandb"))
            self.assertEqual(
                manifest["environment"]["packages"]["tensorboard"],
                version("tensorboard"),
            )
            self.assertIsInstance(manifest["environment"]["source_dirty"], bool)
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(manifest["environment"]["source_revision"], revision)

            artifacts = {artifact["name"]: artifact for artifact in manifest["artifacts"]}
            self.assertEqual(
                set(artifacts),
                {
                    "checkpoint",
                    "events",
                    "invocations",
                    "result",
                    "tensorboard",
                    "wandb",
                },
            )
            for artifact_name in ("checkpoint", "events", "invocations", "result"):
                artifact_path = output_directory / artifacts[artifact_name]["path"]
                self.assertEqual(
                    artifacts[artifact_name]["sha256"],
                    sha256(artifact_path.read_bytes()).hexdigest(),
                )

            checkpoint = output_directory / artifacts["checkpoint"]["path"]
            events = output_directory / artifacts["events"]["path"]
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(
                [json.loads(line) for line in events.read_text().splitlines()],
                [
                    {
                        "action": "LEFT",
                        "board": FIXTURE_BOARD,
                        "reward": 1.0,
                        "reward_total": 1.0,
                        "step": 1,
                    },
                    {
                        "action": "UP",
                        "board": FIXTURE_BOARD,
                        "reward": 0.5,
                        "reward_total": 1.5,
                        "step": 2,
                    },
                    {
                        "action": "RIGHT",
                        "board": FIXTURE_BOARD,
                        "reward": -0.25,
                        "reward_total": 1.25,
                        "step": 3,
                    },
                    {
                        "action": "DOWN",
                        "board": FIXTURE_BOARD,
                        "reward": 1.25,
                        "reward_total": 2.5,
                        "step": 4,
                    },
                ],
            )

            wandb_directory = output_directory / artifacts["wandb"]["path"]
            tensorboard_directory = output_directory / artifacts["tensorboard"]["path"]
            self.assertTrue(any(wandb_directory.rglob("offline-run-*")))
            self.assertTrue(any(tensorboard_directory.glob("events.out.tfevents.*")))
            for telemetry_name in ("wandb", "tensorboard"):
                self.assertTrue(artifacts[telemetry_name]["members"])
                for member in artifacts[telemetry_name]["members"]:
                    member_path = output_directory / member["path"]
                    self.assertTrue(member_path.is_file())
                    self.assertEqual(
                        member["sha256"],
                        sha256(member_path.read_bytes()).hexdigest(),
                    )
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )

            tensorboard_events = EventAccumulator(str(tensorboard_directory))
            tensorboard_events.Reload()
            self.assertEqual(
                [
                    (event.step, event.value)
                    for event in tensorboard_events.Scalars("train/reward")
                ],
                [(1, 1.0), (2, 0.5), (3, -0.25), (4, 1.25)],
            )
            self.assertEqual(
                [
                    (event.step, event.value)
                    for event in tensorboard_events.Scalars("train/reward_total")
                ],
                [(1, 1.0), (2, 1.5), (3, 1.25), (4, 2.5)],
            )
            self.assertEqual(
                [
                    (event.step, event.value)
                    for event in tensorboard_events.Scalars("eval/mean_reward")
                ],
                [(4, 0.625)],
            )

    def test_resume_continues_without_repeating_completed_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            uninterrupted_directory = root / "uninterrupted"
            resumed_directory = root / "resumed"

            uninterrupted = self.run_runner(
                "--config",
                str(FIXTURE_CONFIG),
                "--output-dir",
                str(uninterrupted_directory),
            )
            self.assertEqual(uninterrupted.returncode, 0, uninterrupted.stderr)

            paused = self.run_runner(
                "--config",
                str(FIXTURE_CONFIG),
                "--output-dir",
                str(resumed_directory),
                "--stop-after-step",
                "2",
            )
            self.assertEqual(paused.returncode, 0, paused.stderr)
            self.assertEqual(
                json.loads((resumed_directory / "result.json").read_text()),
                {
                    "completed_steps": 2,
                    "metrics": {
                        "mean_reward": 0.75,
                        "reward_total": 1.5,
                    },
                    "status": "paused",
                },
            )

            checkpoint_path = resumed_directory / "checkpoints" / "latest.json"
            paused_checkpoint = json.loads(checkpoint_path.read_text())
            self.assertEqual(paused_checkpoint["schema_version"], 1)
            self.assertEqual(paused_checkpoint["completed_steps"], 2)

            resumed = self.run_runner(
                "--config",
                str(FIXTURE_CONFIG),
                "--output-dir",
                str(resumed_directory),
                "--resume",
                str(checkpoint_path),
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)

            for relative_path in (
                Path("events.jsonl"),
                Path("result.json"),
                Path("checkpoints/latest.json"),
            ):
                self.assertEqual(
                    (resumed_directory / relative_path).read_bytes(),
                    (uninterrupted_directory / relative_path).read_bytes(),
                    relative_path,
                )

            invocations = [
                json.loads(line)
                for line in (resumed_directory / "invocations.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                invocations,
                [
                    {"end_step": 2, "start_step": 0, "steps_executed": 2},
                    {"end_step": 4, "start_step": 2, "steps_executed": 2},
                ],
            )
            manifest = json.loads((resumed_directory / "manifest.json").read_text())
            artifact_names = {artifact["name"] for artifact in manifest["artifacts"]}
            self.assertIn("invocations", artifact_names)
            uninterrupted_manifest = json.loads(
                (uninterrupted_directory / "manifest.json").read_text()
            )
            resumed_artifacts = {
                artifact["name"]: artifact for artifact in manifest["artifacts"]
            }
            uninterrupted_artifacts = {
                artifact["name"]: artifact
                for artifact in uninterrupted_manifest["artifacts"]
            }
            self.assertEqual(
                resumed_artifacts["events"]["equivalence"],
                "content",
            )
            self.assertEqual(
                resumed_artifacts["invocations"]["equivalence"],
                "attempt_audit",
            )
            self.assertEqual(
                resumed_artifacts["tensorboard"]["equivalence"],
                "semantic_metrics",
            )

            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )

            def scalars(
                run_directory: Path, artifact: dict[str, object], tag: str
            ) -> list[tuple[int, float]]:
                accumulator = EventAccumulator(
                    str(run_directory / str(artifact["path"]))
                )
                accumulator.Reload()
                return [
                    (event.step, event.value)
                    for event in accumulator.Scalars(tag)
                ]

            for tag in ("train/reward", "train/reward_total"):
                self.assertEqual(
                    scalars(
                        resumed_directory,
                        resumed_artifacts["tensorboard"],
                        tag,
                    ),
                    scalars(
                        uninterrupted_directory,
                        uninterrupted_artifacts["tensorboard"],
                        tag,
                    ),
                )
            self.assertEqual(
                scalars(
                    resumed_directory,
                    resumed_artifacts["tensorboard"],
                    "eval/mean_reward",
                )[-1],
                scalars(
                    uninterrupted_directory,
                    uninterrupted_artifacts["tensorboard"],
                    "eval/mean_reward",
                )[-1],
            )

    def test_invalid_configuration_exits_without_creating_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invalid_config = root / "invalid.json"
            output_directory = root / "run"
            configuration = json.loads(FIXTURE_CONFIG.read_text())
            configuration["fixture"]["policy_actions"][0] = "JUMP"
            invalid_config.write_text(json.dumps(configuration))

            completed = self.run_runner(
                "--config",
                str(invalid_config),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertEqual(
                completed.stderr,
                "experiment runner error: fixture.policy_actions contains "
                "unsupported action 'JUMP'\n",
            )
            self.assertFalse(output_directory.exists())


if __name__ == "__main__":
    unittest.main()
