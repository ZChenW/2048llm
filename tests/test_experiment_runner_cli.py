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
POLICY_CONTRACT_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "policy_response_contracts.json"
)
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
                ),
                scalars(
                    uninterrupted_directory,
                    uninterrupted_artifacts["tensorboard"],
                    "eval/mean_reward",
                ),
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

    def test_policy_response_contracts_are_enforced_through_the_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "run"

            completed = self.run_runner(
                "--config",
                str(POLICY_CONTRACT_FIXTURE),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((output_directory / "result.json").read_text())
            expected_rates = {
                "illegal_action_rate": 1 / 7,
                "parse_rate": 2 / 7,
                "policy_failure_rate": 6 / 7,
                "truncation_rate": 1 / 7,
                "valid_action_rate": 1 / 7,
            }
            self.assertEqual(
                result["policy_metrics"]["direct_action"],
                {
                    **expected_rates,
                    "mean_response_length_tokens": 24 / 7,
                    "responses": 7,
                },
            )
            self.assertEqual(
                result["policy_metrics"]["reasoning"],
                {
                    **expected_rates,
                    "mean_response_length_tokens": 164 / 7,
                    "responses": 7,
                },
            )

            events = [
                json.loads(line)
                for line in (output_directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(events), 14)
            direct_events = [
                event for event in events if event["variant"] == "direct_action"
            ]
            reasoning_events = [
                event for event in events if event["variant"] == "reasoning"
            ]

            for event in events:
                compact_board = json.dumps(event["board"], separators=(",", ":"))
                self.assertEqual(event["prompt"].count(compact_board), 1)
                self.assertIn("4x4 JSON array; 0 means empty", event["prompt"])
                for prohibited in ("legal", "hint", "history", "score", "teacher"):
                    self.assertNotIn(prohibited, event["prompt"].lower())

            self.assertEqual(direct_events[0]["action"], "LEFT")
            self.assertTrue(direct_events[0]["parsed"])
            self.assertTrue(direct_events[0]["valid_action"])
            self.assertFalse(direct_events[0]["policy_failure"])
            self.assertNotIn("<think>", direct_events[0]["prompt"])
            self.assertEqual(
                direct_events[0]["response"],
                "<action>LEFT</action>",
            )

            self.assertIn(
                "<think>POLICY_REASONING_TRACE</think><action>ACTION</action>",
                reasoning_events[0]["prompt"],
            )
            self.assertEqual(reasoning_events[0]["max_generation_tokens"], 96)
            self.assertEqual(
                reasoning_events[0]["policy_reasoning_trace"],
                "Keep the largest tile in a corner.",
            )
            self.assertEqual(reasoning_events[0]["action"], "UP")
            self.assertTrue(reasoning_events[0]["valid_action"])

            expected_failure_reasons = [
                "malformed_response",
                "truncated_response",
                "missing_action",
                "multiple_actions",
                "out_of_vocabulary_action",
                "illegal_action",
            ]
            self.assertEqual(
                [event["policy_failure_reason"] for event in direct_events[1:]],
                expected_failure_reasons,
            )
            self.assertEqual(
                [event["policy_failure_reason"] for event in reasoning_events[1:]],
                expected_failure_reasons,
            )
            self.assertTrue(
                all(event["action"] is None for event in direct_events[1:-1])
            )
            self.assertTrue(
                all(event["action"] is None for event in reasoning_events[1:-1])
            )
            self.assertEqual(direct_events[-1]["action"], "DOWN")
            self.assertTrue(direct_events[-1]["parsed"])
            self.assertFalse(direct_events[-1]["valid_action"])
            self.assertEqual(reasoning_events[-1]["action"], "RIGHT")
            self.assertTrue(reasoning_events[-1]["parsed"])

            manifest = json.loads((output_directory / "manifest.json").read_text())
            self.assertEqual(
                manifest["configuration"]["policy_contracts"],
                {
                    "action_envelope": "<action>ACTION</action>",
                    "reasoning_envelope": (
                        "<think>POLICY_REASONING_TRACE</think>"
                        "<action>ACTION</action>"
                    ),
                    "reasoning_max_generation_tokens": 96,
                },
            )

            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )

            tensorboard_artifact = next(
                artifact
                for artifact in manifest["artifacts"]
                if artifact["name"] == "tensorboard"
            )
            tensorboard_events = EventAccumulator(
                str(output_directory / tensorboard_artifact["path"])
            )
            tensorboard_events.Reload()
            self.assertEqual(
                [
                    (event.step, event.value)
                    for event in tensorboard_events.Scalars(
                        "policy/direct_action/parsed"
                    )
                ],
                [
                    (1, 1.0),
                    (2, 0.0),
                    (3, 0.0),
                    (4, 0.0),
                    (5, 0.0),
                    (6, 0.0),
                    (7, 1.0),
                ],
            )
            self.assertEqual(
                [
                    (event.step, event.value)
                    for event in tensorboard_events.Scalars(
                        "policy/reasoning/response_length_tokens"
                    )
                ],
                [
                    (8, 12.0),
                    (9, 14.0),
                    (10, 96.0),
                    (11, 6.0),
                    (12, 15.0),
                    (13, 10.0),
                    (14, 11.0),
                ],
            )

    def test_reasoning_generation_budget_is_fixed_at_96_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invalid_config = root / "invalid.json"
            output_directory = root / "run"
            configuration = json.loads(POLICY_CONTRACT_FIXTURE.read_text())
            configuration["total_steps"] = 1
            configuration["fixture"]["policy_cases"] = [
                configuration["fixture"]["policy_cases"][7]
            ]
            configuration["fixture"]["policy_cases"][0][
                "response_length_tokens"
            ] = 97
            invalid_config.write_text(json.dumps(configuration))

            completed = self.run_runner(
                "--config",
                str(invalid_config),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "response_length_tokens must not exceed 96 for a Reasoning Policy",
                completed.stderr,
            )
            self.assertFalse(output_directory.exists())

    def test_policy_prompt_rejects_non_tile_board_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invalid_config = root / "invalid.json"
            output_directory = root / "run"
            configuration = json.loads(POLICY_CONTRACT_FIXTURE.read_text())
            configuration["total_steps"] = 1
            configuration["fixture"]["policy_cases"] = [
                configuration["fixture"]["policy_cases"][0]
            ]
            configuration["fixture"]["policy_cases"][0]["board"][0][0] = 1
            invalid_config.write_text(json.dumps(configuration))

            completed = self.run_runner(
                "--config",
                str(invalid_config),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "board tiles must be 0 or powers of two",
                completed.stderr,
            )
            self.assertFalse(output_directory.exists())


if __name__ == "__main__":
    unittest.main()
