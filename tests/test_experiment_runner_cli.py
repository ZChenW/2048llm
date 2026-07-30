from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = REPO_ROOT / "tests" / "fixtures" / "experiment_runner_tracer.json"
POLICY_CONTRACT_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "policy_response_contracts.json"
)
ENVIRONMENT_GAME_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "environment_game_seeded.json"
)
COMPLETE_ENVIRONMENT_GAME_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "environment_game_complete.json"
)
TEACHER_GUIDED_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "teacher_guided_rollout_group.json"
)
TEACHER_REWARD_MANIFEST = (
    REPO_ROOT / "tests" / "fixtures" / "teacher_corpus_reward_manifest.json"
)
GRPO_SMOKE_FIXTURE = (
    REPO_ROOT / "configs" / "qwen35_4b_grpo_smoke.json"
)
ZERO_SHOT_SFT_CONFIG = (
    REPO_ROOT / "configs" / "qwen35_4b_zero_shot_sft_gate.json"
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

    def test_grpo_smoke_dry_run_resolves_the_safe_real_training_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "must-not-exist"

            completed = self.run_runner(
                "--grpo-smoke-config",
                str(GRPO_SMOKE_FIXTURE),
                "--output-dir",
                str(output_directory),
                "--dry-run",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = json.loads(completed.stdout)
            self.assertEqual(plan["status"], "validated")
            self.assertEqual(
                plan["model"],
                {
                    "fast_inference": False,
                    "id": "Qwen/Qwen3.5-4B",
                    "load_in_4bit": False,
                    "max_sequence_length": 256,
                    "precision": "bf16",
                    "revision": "c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97",
                    "text_only": True,
                },
            )
            self.assertEqual(
                plan["lora"],
                {
                    "alpha": 64,
                    "dropout": 0.0,
                    "finetune_attention_modules": True,
                    "finetune_language_layers": True,
                    "finetune_mlp_modules": True,
                    "finetune_vision_layers": False,
                    "gradient_checkpointing": "unsloth",
                    "rank": 64,
                },
            )
            self.assertEqual(
                plan["grpo"],
                {
                    "generation_batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    "learning_rate": 5e-6,
                    "loss_type": "grpo",
                    "mask_truncated_completions": False,
                    "max_completion_length": 96,
                    "max_prompt_length": 160,
                    "max_steps": 1,
                    "num_generations": 4,
                    "optimizer": "adamw_8bit",
                    "per_device_train_batch_size": 1,
                    "safe_group8_reserved_headroom_gib": 2.0,
                    "temperature": 1.0,
                    "beta": 0.0,
                    "trainer": "trl.GRPOTrainer",
                    "use_vllm": False,
                },
            )
            self.assertEqual(
                plan["telemetry"],
                {
                    "tensorboard": True,
                    "upload_model_checkpoints": False,
                    "wandb_entity": "auto",
                    "wandb_mode": "online",
                    "wandb_project": "2048llm-feasibility",
                    "wandb_project_visibility": "private",
                },
            )
            self.assertFalse(output_directory.exists())

    def test_real_grpo_smoke_fails_closed_when_wandb_credential_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "must-not-exist"

            completed = self.run_runner(
                "--grpo-smoke-config",
                str(GRPO_SMOKE_FIXTURE),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertEqual(
                completed.stderr,
                "experiment runner error: WANDB_API_KEY is required for the "
                "private online W&B smoke run\n",
            )
            self.assertNotIn("api_key", completed.stdout.lower())
            self.assertFalse(output_directory.exists())

    def test_planned_finalization_dispatches_without_a_failure_artifact(
        self,
    ) -> None:
        from llm2048.experiment_runner import main

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "successful-phase-one"
            source.mkdir()
            lock = source / "finalization-lock.json"
            lock.write_text("{}\n", encoding="utf-8")
            output = root / "final"
            expected = {
                "status": "completed",
                "fresh_training_performed": False,
            }
            stdout = StringIO()
            with patch(
                "llm2048.zero_shot_sft.run_fresh_process_finalization",
                return_value=expected,
            ) as finalize:
                with redirect_stdout(stdout):
                    return_code = main(
                        [
                            "--zero-shot-sft-config",
                            str(ZERO_SHOT_SFT_CONFIG),
                            "--finalize-from",
                            str(source),
                            "--finalization-lock",
                            str(lock),
                            "--output-dir",
                            str(output),
                        ]
                    )

            self.assertEqual(return_code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), expected)
            self.assertFalse((source / "failure.json").exists())
            finalize.assert_called_once()
            call = finalize.call_args.kwargs
            self.assertEqual(call["source_run_directory"], source)
            self.assertEqual(call["finalization_lock_path"], lock)
            self.assertEqual(call["output_directory"], output)

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
                    "illegal_action_rate": 1 / 9,
                    "mean_response_length_tokens": 186 / 9,
                    "parse_rate": 2 / 9,
                    "policy_failure_rate": 8 / 9,
                    "responses": 9,
                    "truncation_rate": 1 / 9,
                    "valid_action_rate": 1 / 9,
                },
            )

            events = [
                json.loads(line)
                for line in (output_directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(events), 16)
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
            self.assertNotIn(
                "<action>ACTION</action>",
                direct_events[0]["prompt"],
            )
            self.assertIn(
                "<action>LEFT</action>, <action>RIGHT</action>",
                direct_events[0]["prompt"],
            )
            self.assertEqual(
                direct_events[0]["response"],
                "<action>LEFT</action>",
            )

            self.assertNotIn(
                "<action>ACTION</action>",
                reasoning_events[0]["prompt"],
            )
            self.assertIn(
                "<think>...</think>",
                reasoning_events[0]["prompt"],
            )
            self.assertEqual(reasoning_events[0]["max_generation_tokens"], 96)
            self.assertEqual(
                reasoning_events[0]["policy_reasoning_trace"],
                "Keep the largest tile in a corner.",
            )
            self.assertEqual(reasoning_events[0]["action"], "UP")
            self.assertTrue(reasoning_events[0]["valid_action"])

            direct_failure_reasons = [
                "malformed_response",
                "truncated_response",
                "missing_action",
                "multiple_actions",
                "out_of_vocabulary_action",
                "illegal_action",
            ]
            self.assertEqual(
                [event["policy_failure_reason"] for event in direct_events[1:]],
                direct_failure_reasons,
            )
            self.assertEqual(
                [event["policy_failure_reason"] for event in reasoning_events[1:]],
                [
                    *direct_failure_reasons[:-1],
                    "non_english_reasoning_trace",
                    "non_english_reasoning_trace",
                    "illegal_action",
                ],
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
                    (14, 10.0),
                    (15, 12.0),
                    (16, 11.0),
                ],
            )

    def test_teacher_guided_reward_callback_scores_exact_best_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "run"

            completed = self.run_runner(
                "--config",
                str(TEACHER_GUIDED_FIXTURE),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            event = json.loads(
                (output_directory / "events.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(event["action"], "UP")
            self.assertEqual(event["teacher_action"], "UP")
            self.assertEqual(event["selected_action_score"], 10.0)
            self.assertEqual(event["teacher_top1_score"], 10.0)
            self.assertEqual(event["regret"], 0.0)
            self.assertEqual(
                event["reward_components"],
                {
                    "action_quality": 1.0,
                    "best_action_bonus": 0.1,
                    "illegal_action_penalty": 0.0,
                    "policy_failure_penalty": 0.0,
                },
            )
            self.assertEqual(event["reward"], 1.1)

    def test_teacher_guided_group_preserves_soft_legal_action_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "run"

            completed = self.run_runner(
                "--config",
                str(TEACHER_GUIDED_FIXTURE),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [
                json.loads(line)
                for line in (output_directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            legal_events = events[:3]
            self.assertEqual(
                [event["action"] for event in legal_events],
                ["UP", "LEFT", "RIGHT"],
            )
            self.assertEqual(
                [event["regret"] for event in legal_events],
                [0.0, 0.1999999999999993, 3.0],
            )
            self.assertEqual(
                [
                    event["reward_components"]["action_quality"]
                    for event in legal_events
                ],
                [1.0, 0.9048374180359599, 0.22313016014842982],
            )
            self.assertEqual(
                [event["reward"] for event in legal_events],
                [1.1, 0.9048374180359599, 0.22313016014842982],
            )
            self.assertGreater(events[1]["reward"], events[2]["reward"])
            self.assertLess(events[1]["reward"], events[0]["reward"])

    def test_teacher_guided_group_applies_distinct_failure_penalties(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "run"

            completed = self.run_runner(
                "--config",
                str(TEACHER_GUIDED_FIXTURE),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [
                json.loads(line)
                for line in (output_directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [event["policy_failure_reason"] for event in events[3:]],
                [
                    "illegal_action",
                    "malformed_response",
                    "truncated_response",
                ],
            )
            self.assertEqual(
                [event["reward"] for event in events[3:]],
                [-1.0, -1.25, -1.25],
            )
            self.assertEqual(
                [
                    event["reward_components"]
                    for event in events[3:]
                ],
                [
                    {
                        "action_quality": 0.0,
                        "best_action_bonus": 0.0,
                        "illegal_action_penalty": -1.0,
                        "policy_failure_penalty": 0.0,
                    },
                    {
                        "action_quality": 0.0,
                        "best_action_bonus": 0.0,
                        "illegal_action_penalty": 0.0,
                        "policy_failure_penalty": -1.25,
                    },
                    {
                        "action_quality": 0.0,
                        "best_action_bonus": 0.0,
                        "illegal_action_penalty": 0.0,
                        "policy_failure_penalty": -1.25,
                    },
                ],
            )

    def test_teacher_guided_group_results_and_offline_metrics_are_reviewable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "run"

            completed = self.run_runner(
                "--config",
                str(TEACHER_GUIDED_FIXTURE),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((output_directory / "result.json").read_text())
            self.assertEqual(
                result["teacher_guided_rollout_group"],
                {
                    "group_size": 6,
                    "reward_component_means": {
                        "action_quality": 0.3546612630307316,
                        "best_action_bonus": 0.016666666666666666,
                        "illegal_action_penalty": -0.16666666666666666,
                        "policy_failure_penalty": -0.4166666666666667,
                    },
                    "teacher_action_agreement_rate": 0.16666666666666666,
                    "teacher_margin_scale": 2.0,
                    "valid_legal_mean_regret": 1.0666666666666664,
                },
            )

            events = [
                json.loads(line)
                for line in (output_directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(events), 6)
            self.assertTrue(
                all(event["board"] == events[0]["board"] for event in events)
            )

            manifest = json.loads((output_directory / "manifest.json").read_text())
            resolved_group = manifest["configuration"]["fixture"][
                "teacher_guided_rollout_group"
            ]
            self.assertEqual(resolved_group["group_size"], 6)
            self.assertEqual(resolved_group["board"], events[0]["board"])
            self.assertEqual(
                resolved_group["corpus_manifest"]["calibration"],
                {
                    "method": "median_positive_margin",
                    "scope": "train",
                    "tau": 2.0,
                },
            )
            wandb_artifact = next(
                artifact
                for artifact in manifest["artifacts"]
                if artifact["name"] == "wandb"
            )
            self.assertTrue(wandb_artifact["members"])
            self.assertTrue(
                all(
                    member["path"].endswith(".wandb")
                    for member in wandb_artifact["members"]
                )
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
            expected_tags = {
                "teacher_guided/action_quality",
                "teacher_guided/best_action_bonus",
                "teacher_guided/illegal_action_penalty",
                "teacher_guided/policy_failure_penalty",
                "teacher_guided/regret",
                "teacher_guided/group/action_quality_mean",
                "teacher_guided/group/best_action_bonus_mean",
                "teacher_guided/group/illegal_action_penalty_mean",
                "teacher_guided/group/mean_reward",
                "teacher_guided/group/policy_failure_penalty_mean",
                "teacher_guided/group/teacher_action_agreement_rate",
                "teacher_guided/group/valid_legal_mean_regret",
            }
            self.assertTrue(
                expected_tags.issubset(
                    set(tensorboard_events.Tags()["scalars"])
                )
            )
            component_events = tensorboard_events.Scalars(
                "teacher_guided/illegal_action_penalty"
            )
            self.assertEqual(
                [(event.step, event.value) for event in component_events],
                [
                    (1, 0.0),
                    (2, 0.0),
                    (3, 0.0),
                    (4, -1.0),
                    (5, 0.0),
                    (6, 0.0),
                ],
            )
            group_mean = tensorboard_events.Scalars(
                "teacher_guided/group/mean_reward"
            )
            self.assertEqual(len(group_mean), 1)
            self.assertEqual(group_mean[0].step, 6)
            self.assertAlmostEqual(group_mean[0].value, -0.2120054)

    def test_teacher_guided_group_rejects_per_candidate_boards_and_wrong_size(
        self,
    ) -> None:
        invalid_cases = {
            "candidate board": (
                lambda configuration: configuration["fixture"][
                    "teacher_guided_rollout_group"
                ]["candidates"][0].update(
                    {
                        "board": [
                            [2, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ]
                    }
                ),
                "candidates[0] must contain exactly",
            ),
            "wrong group size": (
                lambda configuration: configuration["fixture"][
                    "teacher_guided_rollout_group"
                ].update({"group_size": 5}),
                "group_size must equal total_steps",
            ),
        }
        for label, (mutate, expected_error) in invalid_cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                config_path = root / "invalid.json"
                output_directory = root / "run"
                configuration = json.loads(TEACHER_GUIDED_FIXTURE.read_text())
                configuration["fixture"]["teacher_guided_rollout_group"][
                    "corpus_manifest"
                ] = str(TEACHER_REWARD_MANIFEST)
                mutate(configuration)
                config_path.write_text(json.dumps(configuration))

                completed = self.run_runner(
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_directory),
                )

                self.assertEqual(completed.returncode, 2)
                self.assertIn(expected_error, completed.stderr)
                self.assertFalse(output_directory.exists())

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

    def test_policy_legality_is_derived_from_board_changes(self) -> None:
        movement_boards = {
            "LEFT": [
                [0, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            "RIGHT": [
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            "UP": [
                [0, 0, 0, 0],
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            "DOWN": [
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        }
        merge_boards = {
            "LEFT": [
                [2, 2, 4, 8],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            "RIGHT": [
                [8, 4, 2, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            "UP": [
                [2, 0, 0, 0],
                [2, 0, 0, 0],
                [4, 0, 0, 0],
                [8, 0, 0, 0],
            ],
            "DOWN": [
                [8, 0, 0, 0],
                [4, 0, 0, 0],
                [2, 0, 0, 0],
                [2, 0, 0, 0],
            ],
        }
        locked_board = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 4],
            [8, 16, 32, 64],
        ]
        policy_cases = []
        for boards in (movement_boards, merge_boards):
            for action, board in boards.items():
                policy_cases.append(
                    {
                        "variant": "direct_action",
                        "board": board,
                        "response": f"<action>{action}</action>",
                        "response_length_tokens": 3,
                        "truncated": False,
                    }
                )
        for action in ("LEFT", "RIGHT", "UP", "DOWN"):
            policy_cases.append(
                {
                    "variant": "direct_action",
                    "board": locked_board,
                    "response": f"<action>{action}</action>",
                    "response_length_tokens": 3,
                    "truncated": False,
                }
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "config.json"
            output_directory = root / "run"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_name": "canonical-policy-legality",
                        "seed": 2048,
                        "total_steps": len(policy_cases),
                        "fixture": {"policy_cases": policy_cases},
                        "telemetry": {"wandb_project": "2048llm-fixture"},
                    }
                )
            )

            completed = self.run_runner(
                "--config",
                str(config_path),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [
                json.loads(line)
                for line in (output_directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [event["valid_action"] for event in events],
                [True] * 8 + [False] * 4,
            )
            self.assertEqual(
                [event["policy_failure_reason"] for event in events],
                [None] * 8 + ["illegal_action"] * 4,
            )
            self.assertEqual(
                events[-1]["change_making_actions"],
                [],
            )

    def test_seeded_environment_game_moves_merges_scores_and_spawns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "run"

            completed = self.run_runner(
                "--config",
                str(ENVIRONMENT_GAME_FIXTURE),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [
                json.loads(line)
                for line in (output_directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [
                    {
                        "action": event["action"],
                        "board": event["board"],
                        "board_after_move": event["board_after_move"],
                        "next_board": event["next_board"],
                        "score_delta": event["score_delta"],
                        "spawned_tile": event["spawned_tile"],
                    }
                    for event in events
                ],
                [
                    {
                        "action": "LEFT",
                        "board": [
                            [0, 0, 0, 0],
                            [0, 0, 2, 0],
                            [0, 0, 4, 0],
                            [0, 0, 0, 0],
                        ],
                        "board_after_move": [
                            [0, 0, 0, 0],
                            [2, 0, 0, 0],
                            [4, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "next_board": [
                            [0, 2, 0, 0],
                            [2, 0, 0, 0],
                            [4, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "score_delta": 0,
                        "spawned_tile": {"column": 1, "row": 0, "value": 2},
                    },
                    {
                        "action": "UP",
                        "board": [
                            [0, 2, 0, 0],
                            [2, 0, 0, 0],
                            [4, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "board_after_move": [
                            [2, 2, 0, 0],
                            [4, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "next_board": [
                            [2, 2, 0, 2],
                            [4, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "score_delta": 0,
                        "spawned_tile": {"column": 3, "row": 0, "value": 2},
                    },
                    {
                        "action": "LEFT",
                        "board": [
                            [2, 2, 0, 2],
                            [4, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "board_after_move": [
                            [4, 2, 0, 0],
                            [4, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "next_board": [
                            [4, 2, 4, 0],
                            [4, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "score_delta": 4,
                        "spawned_tile": {"column": 2, "row": 0, "value": 4},
                    },
                    {
                        "action": "UP",
                        "board": [
                            [4, 2, 4, 0],
                            [4, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "board_after_move": [
                            [8, 2, 4, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "next_board": [
                            [8, 2, 4, 0],
                            [0, 0, 2, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                        ],
                        "score_delta": 8,
                        "spawned_tile": {"column": 2, "row": 1, "value": 2},
                    },
                ],
            )
            for event in events:
                compact_board = json.dumps(event["board"], separators=(",", ":"))
                self.assertEqual(event["prompt"].count(compact_board), 1)
                for prohibited in (
                    "legal",
                    "rng",
                    "score",
                    "history",
                    "prior",
                    "teacher",
                ):
                    self.assertNotIn(prohibited, event["prompt"].lower())

            result = json.loads((output_directory / "result.json").read_text())
            self.assertEqual(
                result["game"],
                {
                    "2048_success": False,
                    "empty_cells": 12,
                    "maximum_tile": 8,
                    "moves": 4,
                    "policy_failure": False,
                    "score": 12,
                    "termination_reason": "script_exhausted",
                    "tile_histogram": {"2": 2, "4": 1, "8": 1},
                },
            )

    def test_environment_snapshot_resumes_the_identical_rng_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            uninterrupted_directory = root / "uninterrupted"
            resumed_directory = root / "resumed"

            uninterrupted = self.run_runner(
                "--config",
                str(ENVIRONMENT_GAME_FIXTURE),
                "--output-dir",
                str(uninterrupted_directory),
            )
            self.assertEqual(uninterrupted.returncode, 0, uninterrupted.stderr)

            paused = self.run_runner(
                "--config",
                str(ENVIRONMENT_GAME_FIXTURE),
                "--output-dir",
                str(resumed_directory),
                "--stop-after-step",
                "2",
            )
            self.assertEqual(paused.returncode, 0, paused.stderr)
            checkpoint_path = resumed_directory / "checkpoints" / "latest.json"
            snapshot = json.loads(checkpoint_path.read_text())["game_snapshot"]
            self.assertEqual(
                snapshot["board"],
                [
                    [2, 2, 0, 2],
                    [4, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            )
            self.assertEqual(snapshot["score"], 0)
            self.assertEqual(snapshot["moves"], 2)
            self.assertIsInstance(snapshot["rng_state"], list)
            self.assertEqual(len(snapshot["rng_state"]), 3)

            resumed = self.run_runner(
                "--config",
                str(ENVIRONMENT_GAME_FIXTURE),
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

    def test_environment_resume_rejects_a_snapshot_not_reached_by_the_script(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "run"
            paused = self.run_runner(
                "--config",
                str(ENVIRONMENT_GAME_FIXTURE),
                "--output-dir",
                str(output_directory),
                "--stop-after-step",
                "2",
            )
            self.assertEqual(paused.returncode, 0, paused.stderr)
            checkpoint_path = output_directory / "checkpoints" / "latest.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            checkpoint["game_snapshot"]["board"][0][0] = 4
            checkpoint_path.write_text(json.dumps(checkpoint))

            resumed = self.run_runner(
                "--config",
                str(ENVIRONMENT_GAME_FIXTURE),
                "--output-dir",
                str(output_directory),
                "--resume",
                str(checkpoint_path),
            )

            self.assertEqual(resumed.returncode, 2)
            self.assertIn(
                "game_snapshot does not match the seeded response history",
                resumed.stderr,
            )

    def test_environment_policy_failures_terminate_without_retry_or_replacement(
        self,
    ) -> None:
        cases: dict[str, dict[str, Any]] = {
            "malformed": {
                "responses": [
                    {
                        "response": "<action>LEFT</action> extra",
                        "response_length_tokens": 4,
                        "truncated": False,
                    },
                    {
                        "response": "<action>LEFT</action>",
                        "response_length_tokens": 3,
                        "truncated": False,
                    },
                ],
                "expected_events": 1,
                "expected_moves": 0,
                "expected_reason": "malformed_response",
            },
            "truncated": {
                "responses": [
                    {
                        "response": "<action>LEFT",
                        "response_length_tokens": 2,
                        "truncated": True,
                    },
                    {
                        "response": "<action>LEFT</action>",
                        "response_length_tokens": 3,
                        "truncated": False,
                    },
                ],
                "expected_events": 1,
                "expected_moves": 0,
                "expected_reason": "truncated_response",
            },
            "illegal": {
                "responses": [
                    {
                        "response": f"<action>{action}</action>",
                        "response_length_tokens": 3,
                        "truncated": False,
                    }
                    for action in ("LEFT", "UP", "LEFT", "LEFT", "DOWN")
                ],
                "expected_events": 4,
                "expected_moves": 3,
                "expected_reason": "illegal_action",
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name, case in cases.items():
                with self.subTest(name=name):
                    config_path = root / f"{name}.json"
                    output_directory = root / name
                    responses = case["responses"]
                    config_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "experiment_name": f"policy-failure-{name}",
                                "seed": 7,
                                "total_steps": len(responses),
                                "fixture": {
                                    "environment_game": {
                                        "variant": "direct_action",
                                        "responses": responses,
                                    }
                                },
                                "telemetry": {
                                    "wandb_project": "2048llm-fixture"
                                },
                            }
                        )
                    )

                    completed = self.run_runner(
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(output_directory),
                    )

                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    events = [
                        json.loads(line)
                        for line in (output_directory / "events.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    self.assertEqual(len(events), case["expected_events"])
                    self.assertEqual(
                        events[-1]["policy_failure_reason"],
                        case["expected_reason"],
                    )
                    self.assertTrue(events[-1]["policy_failure"])
                    result = json.loads(
                        (output_directory / "result.json").read_text()
                    )
                    self.assertEqual(
                        result["completed_steps"],
                        case["expected_events"],
                    )
                    self.assertEqual(
                        result["game"]["moves"],
                        case["expected_moves"],
                    )
                    self.assertTrue(result["game"]["policy_failure"])
                    self.assertEqual(
                        result["game"]["termination_reason"],
                        "policy_failure",
                    )

    def test_complete_strict_markov_policy_game_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_directory = root / "first"
            second_directory = root / "second"

            for output_directory in (first_directory, second_directory):
                completed = self.run_runner(
                    "--config",
                    str(COMPLETE_ENVIRONMENT_GAME_FIXTURE),
                    "--output-dir",
                    str(output_directory),
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            for relative_path in (
                Path("events.jsonl"),
                Path("result.json"),
                Path("checkpoints/latest.json"),
            ):
                self.assertEqual(
                    (first_directory / relative_path).read_bytes(),
                    (second_directory / relative_path).read_bytes(),
                    relative_path,
                )

            events = [
                json.loads(line)
                for line in (first_directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(events), 193)
            self.assertEqual(
                {
                    "action": events[1]["action"],
                    "board": events[1]["board"],
                    "board_after_move": events[1]["board_after_move"],
                    "score_delta": events[1]["score_delta"],
                },
                {
                    "action": "DOWN",
                    "board": [
                        [0, 0, 0, 0],
                        [2, 0, 0, 0],
                        [2, 0, 0, 0],
                        [2, 0, 0, 0],
                    ],
                    "board_after_move": [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [2, 0, 0, 0],
                        [4, 0, 0, 0],
                    ],
                    "score_delta": 4,
                },
            )
            self.assertEqual(
                {
                    "action": events[4]["action"],
                    "board": events[4]["board"],
                    "board_after_move": events[4]["board_after_move"],
                    "score_delta": events[4]["score_delta"],
                },
                {
                    "action": "RIGHT",
                    "board": [
                        [2, 0, 0, 0],
                        [4, 0, 0, 0],
                        [2, 0, 0, 0],
                        [4, 0, 0, 0],
                    ],
                    "board_after_move": [
                        [0, 0, 0, 2],
                        [0, 0, 0, 4],
                        [0, 0, 0, 2],
                        [0, 0, 0, 4],
                    ],
                    "score_delta": 0,
                },
            )
            self.assertEqual(
                {
                    "action": events[147]["action"],
                    "board": events[147]["board"],
                    "board_after_move": events[147]["board_after_move"],
                    "score_delta": events[147]["score_delta"],
                },
                {
                    "action": "LEFT",
                    "board": [
                        [2, 2, 4, 2],
                        [0, 32, 16, 4],
                        [8, 32, 8, 2],
                        [128, 64, 16, 8],
                    ],
                    "board_after_move": [
                        [4, 4, 2, 0],
                        [32, 16, 4, 0],
                        [8, 32, 8, 2],
                        [128, 64, 16, 8],
                    ],
                    "score_delta": 4,
                },
            )

            for index, event in enumerate(events):
                self.assertFalse(event["policy_failure"])
                self.assertTrue(event["valid_action"])
                self.assertEqual(
                    sum(tile for row in event["next_board"] for tile in row),
                    sum(tile for row in event["board"] for tile in row)
                    + event["spawned_tile"]["value"],
                )
                self.assertGreaterEqual(event["score_delta"], 0)
                self.assertEqual(event["score_delta"] % 4, 0)
                if index + 1 < len(events):
                    self.assertEqual(
                        event["next_board"],
                        events[index + 1]["board"],
                    )
                for board_field in (
                    "board",
                    "board_after_move",
                    "next_board",
                ):
                    board = event[board_field]
                    self.assertEqual(len(board), 4)
                    self.assertTrue(all(len(row) == 4 for row in board))
                    self.assertTrue(
                        all(
                            tile == 0
                            or (tile >= 2 and tile & (tile - 1) == 0)
                            for row in board
                            for tile in row
                        )
                    )

            result = json.loads((first_directory / "result.json").read_text())
            self.assertEqual(result["completed_steps"], 193)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                result["game"],
                {
                    "2048_success": False,
                    "empty_cells": 0,
                    "maximum_tile": 256,
                    "moves": 193,
                    "policy_failure": False,
                    "score": 2304,
                    "termination_reason": "game_over",
                    "tile_histogram": {
                        "2": 3,
                        "4": 4,
                        "8": 5,
                        "16": 1,
                        "32": 1,
                        "64": 1,
                        "256": 1,
                    },
                },
            )

    def test_reachable_four_equal_tiles_merge_in_pairs_without_chaining(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "merge-edge.json"
            output_directory = root / "run"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_name": "reachable-merge-edge",
                        "seed": 35,
                        "total_steps": 42,
                        "fixture": {
                            "environment_game": {
                                "variant": "direct_action",
                                "action_preferences": [
                                    "LEFT",
                                    "DOWN",
                                    "RIGHT",
                                    "UP",
                                ],
                            }
                        },
                        "telemetry": {
                            "wandb_project": "2048llm-fixture"
                        },
                    }
                )
            )

            completed = self.run_runner(
                "--config",
                str(config_path),
                "--output-dir",
                str(output_directory),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [
                json.loads(line)
                for line in (output_directory / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(events), 42)
            self.assertEqual(
                {
                    "action": events[-1]["action"],
                    "board": events[-1]["board"],
                    "board_after_move": events[-1]["board_after_move"],
                    "score_delta": events[-1]["score_delta"],
                },
                {
                    "action": "LEFT",
                    "board": [
                        [2, 0, 0, 0],
                        [8, 0, 0, 0],
                        [2, 2, 2, 2],
                        [32, 32, 8, 4],
                    ],
                    "board_after_move": [
                        [2, 0, 0, 0],
                        [8, 0, 0, 0],
                        [4, 4, 0, 0],
                        [64, 8, 4, 0],
                    ],
                    "score_delta": 72,
                },
            )


if __name__ == "__main__":
    unittest.main()
