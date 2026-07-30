from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping
import unittest
from unittest.mock import patch

from llm2048.teacher_guided_block import (
    TeacherGuidedBlockConfig,
    _training_summary,
    build_grpo_arguments,
    build_reward_function,
    checkpoint_contract,
    inspect_upstream_evidence,
    paired_reward_comparison,
    run_teacher_guided_block,
    summarize_evaluation_events,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY
    / "tests"
    / "fixtures"
    / "teacher_guided_block_blocked.json"
)


class TeacherGuidedBlockCliTests(unittest.TestCase):
    def run_runner(
        self, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment.pop("WANDB_API_KEY", None)
        environment.pop("WANDB_MODE", None)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "llm2048.experiment_runner",
                *arguments,
            ],
            cwd=REPOSITORY,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_real_grpo_fails_preflight_before_artifacts_without_wandb_key(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"

            completed = self.run_runner(
                "--teacher-guided-block-config",
                str(CONFIG),
                "--policy-variant",
                "direct_action",
                "--output-dir",
                str(output),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "WANDB_API_KEY is required for the private online "
                "Teacher-guided GRPO run",
                completed.stderr,
            )
            self.assertFalse(output.exists())
            self.assertEqual(completed.stdout, "")

    def test_upstream_inspection_reports_candidates_and_failed_gate_risk(
        self,
    ) -> None:
        result = inspect_upstream_evidence(
            TeacherGuidedBlockConfig.load(CONFIG),
            environ={},
        )

        self.assertTrue(result["training_authorized_by_issue_9"])
        self.assertFalse(result["upstream_gate_passed"])
        self.assertEqual(
            result["risks"],
            [
                {
                    "code": "upstream_gate_not_passed",
                    "message": (
                        "#8 selected non-passing candidates; #9 "
                        "authorizes the bounded comparison but this "
                        "does not make the gate pass"
                    ),
                }
            ],
        )
        self.assertEqual(
            result["selected_starting_points"]["direct_action"]["kind"],
            "unchanged_base_model",
        )
        self.assertEqual(
            result["selected_starting_points"]["reasoning"]["kind"],
            "lora_adapter",
        )
        self.assertEqual(result["selected_group_size"], 4)


class TeacherGuidedBlockContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TeacherGuidedBlockConfig.load(CONFIG)

    def test_grpo_arguments_lock_sampling_checkpoint_and_resume_contract(
        self,
    ) -> None:
        arguments = build_grpo_arguments(
            config=self.config,
            output_directory=Path("/tmp/teacher-guided-contract"),
            variant="reasoning",
            target_step=250,
        )

        self.assertEqual(arguments["num_generations"], 4)
        self.assertEqual(arguments["generation_batch_size"], 4)
        self.assertEqual(arguments["gradient_accumulation_steps"], 4)
        self.assertEqual(arguments["max_steps"], 250)
        self.assertEqual(arguments["max_completion_length"], 96)
        self.assertEqual(arguments["temperature"], 1.0)
        self.assertEqual(arguments["top_p"], 0.95)
        self.assertEqual(arguments["top_k"], 20)
        self.assertEqual(arguments["save_steps"], 125)
        self.assertFalse(arguments["save_only_model"])
        self.assertEqual(arguments["report_to"], ["wandb", "tensorboard"])

    def test_reward_function_scores_each_complete_rollout_group(
        self,
    ) -> None:
        events: list[dict[str, object]] = []
        reward = build_reward_function(
            variant="direct_action",
            max_completion_length=96,
            eos_token_id=0,
            pad_token_id=0,
            event_sink=events.append,
        )
        metadata = {
            "board_json": [
                "[[0,2,0,0],[0,0,0,0],[0,0,2,0],[0,0,0,0]]"
            ]
            * 4,
            "teacher_action_scores_json": [
                '{"LEFT":10.0,"RIGHT":9.0,"UP":8.0,"DOWN":7.0}'
            ]
            * 4,
            "teacher_action": ["LEFT"] * 4,
            "teacher_margin_scale": [2.0] * 4,
            "record_id": ["train-1"] * 4,
        }

        scores = reward(
            completions=[
                "<action>LEFT</action>",
                "<action>RIGHT</action>",
                "<action>UP</action>",
                "<action>DOWN</action>",
            ],
            completion_ids=[[1], [1], [1], [1]],
            **metadata,
        )

        self.assertEqual(len(scores), 4)
        self.assertAlmostEqual(scores[0], 1.1)
        self.assertGreater(scores[1], scores[2])
        self.assertGreater(scores[2], scores[3])
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0]["record_id"], "train-1")

    def test_reward_truncation_uses_eos_at_the_exact_token_budget(
        self,
    ) -> None:
        events: list[dict[str, object]] = []
        reward = build_reward_function(
            variant="direct_action",
            max_completion_length=2,
            eos_token_id=0,
            pad_token_id=0,
            event_sink=events.append,
        )
        metadata = {
            "board_json": [
                "[[0,2,0,0],[0,0,0,0],[0,0,2,0],[0,0,0,0]]"
            ]
            * 4,
            "teacher_action_scores_json": [
                '{"LEFT":10.0,"RIGHT":9.0,"UP":8.0,"DOWN":7.0}'
            ]
            * 4,
            "teacher_action": ["LEFT"] * 4,
            "teacher_margin_scale": [2.0] * 4,
            "record_id": ["train-1"] * 4,
        }

        reward(
            completions=["<action>LEFT</action>"] * 4,
            completion_ids=[[1, 0], [1, 0], [1, 0], [1, 0]],
            **metadata,
        )

        self.assertEqual(
            [event["response_length_tokens"] for event in events],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            [event["policy_failure"] for event in events],
            [False, False, False, False],
        )

        events.clear()
        reward(
            completions=["<action>LEFT</action>"] * 4,
            completion_ids=[[1, 2], [1, 2], [1, 2], [1, 2]],
            **metadata,
        )
        self.assertEqual(
            [event["policy_failure_reason"] for event in events],
            [
                "truncated_response",
                "truncated_response",
                "truncated_response",
                "truncated_response",
            ],
        )

    def test_paired_comparison_requires_positive_lower_confidence_bound(
        self,
    ) -> None:
        learned = paired_reward_comparison(
            initial=[0.0, 0.1, 0.2, 0.3],
            final=[0.5, 0.6, 0.7, 0.8],
            bootstrap_samples=2000,
            confidence_level=0.95,
            seed=2048,
        )
        flat = paired_reward_comparison(
            initial=[0.2, 0.2, 0.2, 0.2],
            final=[0.2, 0.2, 0.2, 0.2],
            bootstrap_samples=2000,
            confidence_level=0.95,
            seed=2048,
        )

        self.assertTrue(learned["measurable_learning"])
        self.assertGreater(learned["reward_delta_ci"]["lower"], 0.0)
        self.assertFalse(flat["measurable_learning"])
        self.assertEqual(flat["reward_delta_ci"]["lower"], 0.0)

    def test_evaluation_summary_keeps_reward_failures_and_sampling_metrics(
        self,
    ) -> None:
        events = [
            {
                "reward": 1.1,
                "reward_components": {
                    "action_quality": 1.0,
                    "best_action_bonus": 0.1,
                    "illegal_action_penalty": 0.0,
                    "policy_failure_penalty": 0.0,
                },
                "action": "LEFT",
                "teacher_action_agreement": True,
                "policy_failure": False,
                "policy_failure_reason": None,
                "response_length_tokens": 3,
            },
            {
                "reward": -1.0,
                "reward_components": {
                    "action_quality": 0.0,
                    "best_action_bonus": 0.0,
                    "illegal_action_penalty": -1.0,
                    "policy_failure_penalty": 0.0,
                },
                "action": "UP",
                "teacher_action_agreement": False,
                "policy_failure": True,
                "policy_failure_reason": "illegal_action",
                "response_length_tokens": 5,
            },
        ]

        summary = summarize_evaluation_events(
            events,
            wall_seconds=2.0,
            generated_tokens=8,
        )

        self.assertEqual(summary["boards"], 2)
        self.assertAlmostEqual(summary["reward"]["mean"], 0.05)
        self.assertEqual(summary["teacher_action_agreement"]["rate"], 0.5)
        self.assertEqual(
            summary["policy_failures"]["classes"], {"illegal_action": 1}
        )
        self.assertEqual(
            summary["action_distribution"]["counts"]["LEFT"], 1
        )
        self.assertEqual(summary["sampling"]["kl"], None)
        self.assertEqual(summary["response_length_tokens"]["p95"], 5)
        self.assertEqual(summary["latency"]["tokens_per_second"], 4.0)

    def test_checkpoint_contract_requires_adapter_optimizer_scheduler_and_rng(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-250"
            checkpoint.mkdir()
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
            ):
                (checkpoint / name).write_text(name, encoding="utf-8")

            contract = checkpoint_contract(checkpoint, expected_step=250)

            self.assertEqual(contract["optimizer_step"], 250)
            self.assertEqual(contract["artifact_kind"], "lora_adapter")
            self.assertEqual(len(contract["required_members"]), 6)

            (checkpoint / "scheduler.pt").unlink()
            with self.assertRaisesRegex(
                RuntimeError, "scheduler state"
            ):
                checkpoint_contract(checkpoint, expected_step=250)

    def test_training_summary_preserves_pre_resume_reward_events(self) -> None:
        before_resume: list[Mapping[str, Any]] = [
            {
                "reward": 0.25,
                "policy_failure": False,
                "policy_failure_reason": None,
            }
        ]
        after_resume: list[Mapping[str, Any]] = [
            {
                "reward": -1.0,
                "policy_failure": True,
                "policy_failure_reason": "illegal_action",
            }
        ]

        summary = _training_summary(
            reward_events=after_resume,
            all_reward_events=before_resume + after_resume,
            log_history=[],
            trainer_metrics={},
            wall_seconds=1.0,
            global_step=250,
        )

        self.assertEqual(summary["reward_events_this_invocation"], 1)
        self.assertEqual(summary["reward_events_all_invocations"], 2)
        self.assertAlmostEqual(
            summary["reward_mean_all_invocations"],
            -0.375,
        )
        self.assertEqual(
            summary["policy_failure_classes_all_invocations"],
            {"illegal_action": 1},
        )

    def test_public_runner_completes_controlled_interrupt_and_resume_via_trl(
        self,
    ) -> None:
        class FakeDataset:
            @staticmethod
            def from_list(rows: list[dict[str, object]]) -> list[dict[str, object]]:
                return rows

        class FakeProcessor:
            eos_token_id = 0
            pad_token_id = 0
            eos_token = "<eos>"
            padding_side = "left"

            def apply_chat_template(self, *_: object, **__: object) -> str:
                return "prompt"

            def __call__(
                self,
                *,
                text: object,
                **__: object,
            ) -> dict[str, list[int]]:
                del text
                return {"input_ids": [1]}

        class FakeTrainer:
            calls: list[str | None] = []

            def __init__(self, **arguments: object) -> None:
                raw_args = arguments["args"]
                if not isinstance(raw_args, dict):
                    raise TypeError("fake GRPO args must be a dictionary")
                self.args: dict[str, object] = raw_args
                self.state = SimpleNamespace(global_step=0, log_history=[])

            def train(
                self, resume_from_checkpoint: str | None = None
            ) -> SimpleNamespace:
                type(self).calls.append(resume_from_checkpoint)
                raw_step = self.args["max_steps"]
                if not isinstance(raw_step, int):
                    raise TypeError("fake max_steps must be an integer")
                step = raw_step
                self.state.global_step = step
                self.state.log_history = [
                    {"step": step, "entropy": 0.25, "reward": 0.5}
                ]
                checkpoint = Path(
                    str(self.args["output_dir"])
                ) / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                members = {
                    "adapter_config.json": "{}",
                    "adapter_model.safetensors": "adapter",
                    "optimizer.pt": "optimizer",
                    "scheduler.pt": "scheduler",
                    "rng_state.pth": "rng",
                    "trainer_state.json": json.dumps(
                        {"global_step": step}
                    ),
                }
                for name, content in members.items():
                    (checkpoint / name).write_text(
                        content, encoding="utf-8"
                    )
                reward_path = (
                    Path(str(self.args["output_dir"])).parent
                    / "training"
                    / "reward-events.jsonl"
                )
                reward_path.parent.mkdir(parents=True, exist_ok=True)
                prior_step = 125 if resume_from_checkpoint is not None else 0
                event = {
                    "reward": 0.5,
                    "reward_components": {
                        "action_quality": 0.5,
                        "best_action_bonus": 0.0,
                        "illegal_action_penalty": 0.0,
                        "policy_failure_penalty": 0.0,
                    },
                    "policy_failure": False,
                    "policy_failure_reason": None,
                }
                with reward_path.open("a", encoding="utf-8") as destination:
                    for _ in range((step - prior_step) * 4):
                        destination.write(json.dumps(event) + "\n")
                tensorboard_event = (
                    Path(str(self.args["output_dir"]))
                    / "runs"
                    / "fake"
                    / f"events.out.tfevents.fake-{step}"
                )
                tensorboard_event.parent.mkdir(parents=True, exist_ok=True)
                tensorboard_event.write_bytes(b"event")
                return SimpleNamespace(metrics={"train_loss": 0.5})

        class FakeRun:
            entity = "private-entity"
            project = "2048llm-feasibility"
            id = "fake-run"
            url = "https://wandb.invalid/fake-run"

            def log(self, *_: object, **__: object) -> None:
                return None

            def finish(self, *_: object, **__: object) -> None:
                return None

        fake_cuda = SimpleNamespace(
            empty_cache=lambda: None,
            get_device_properties=lambda _: SimpleNamespace(
                name="fake-gpu", total_memory=12 * 1024**3
            ),
            get_device_capability=lambda _: (9, 0),
        )
        fake_torch = SimpleNamespace(
            cuda=fake_cuda,
            version=SimpleNamespace(cuda="fake"),
            backends=SimpleNamespace(
                cudnn=SimpleNamespace(version=lambda: 0)
            ),
        )
        fake_stack = SimpleNamespace(
            torch=fake_torch,
            wandb=SimpleNamespace(
                util=SimpleNamespace(generate_id=lambda: "fake-run")
            ),
            dataset_class=FakeDataset,
            fast_vision_model=SimpleNamespace(
                for_training=lambda _: None
            ),
            grpo_config_class=lambda **arguments: arguments,
            grpo_trainer_class=FakeTrainer,
            versions={"trl": "fake"},
            api_signatures={"GRPOTrainer": "fake"},
        )
        record = {
            "record_id": "validation-1",
            "split": "validation",
            "stratum": "natural",
            "board": [[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
            "action_scores": {
                "left": 10.0,
                "right": 9.0,
                "up": 8.0,
                "down": 7.0,
            },
            "teacher_action": "left",
            "lineage": {
                "trajectory_id": "trajectory-1",
                "orbit_id": "orbit-1",
            },
        }
        data_evidence = {
            "dynamic_board_pool": {
                "identity_sha256": "pool",
                "teacher_core": {"member_ids_sha256": "members"},
            },
            "validation_snapshot": {
                "member_ids_sha256": "validation"
            },
        }
        data_plan = {
            "pool_records": [record],
            "validation_records": [record],
            "tau": 2.0,
            "evidence": data_evidence,
        }

        def fake_evaluate(**arguments: object) -> dict[str, object]:
            output = Path(str(arguments["output_directory"]))
            phase = str(arguments["phase"])
            reward = 0.0 if phase == "initialization" else 0.5
            event = {
                "record_id": "validation-1",
                "reward": reward,
            }
            event_path = output / "evaluation" / f"{phase}.jsonl"
            event_path.parent.mkdir(parents=True, exist_ok=True)
            event_path.write_text(
                json.dumps(event) + "\n", encoding="utf-8"
            )
            return {
                "status": "completed",
                "phase": phase,
                "events_path": str(event_path.relative_to(output)),
                "events_sha256": "",
                "metrics": {
                    "reward": {"mean": reward},
                    "teacher_action_agreement": {"rate": 1.0},
                    "policy_failures": {"rate": 0.0},
                    "sampling": {"policy_action_entropy_nats": 0.0},
                    "response_length_tokens": {"mean": 1.0},
                    "latency": {"seconds_per_board": 0.01},
                },
            }

        config = TeacherGuidedBlockConfig.load(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with (
                patch.dict(
                    os.environ,
                    {
                        "WANDB_API_KEY": "test-key",
                        "WANDB_LOG_MODEL": "false",
                    },
                    clear=False,
                ),
                patch(
                    "llm2048.teacher_guided_block.prepare_block_data_plan",
                    return_value=data_plan,
                ),
                patch(
                    "llm2048.teacher_guided_block._load_runtime_stack",
                    return_value=fake_stack,
                ),
                patch("llm2048.teacher_guided_block._preflight_cuda"),
                patch(
                    "llm2048.teacher_guided_block._validate_maintained_stack"
                ),
                patch(
                    "llm2048.teacher_guided_block._load_starting_model",
                    return_value=(
                        object(),
                        FakeProcessor(),
                        {"lora_trainable_parameters": 1},
                    ),
                ),
                patch(
                    "llm2048.teacher_guided_block._start_wandb_run",
                    return_value=(FakeRun(), "PRIVATE"),
                ),
                patch(
                    "llm2048.teacher_guided_block._evaluate_policy",
                    side_effect=fake_evaluate,
                ),
            ):
                paused = run_teacher_guided_block(
                    config=config,
                    variant="direct_action",
                    output_directory=output,
                    stop_after_step=125,
                )
                completed = run_teacher_guided_block(
                    config=config,
                    variant="direct_action",
                    output_directory=output,
                    resume_path=output / "trainer" / "checkpoint-125",
                )

        self.assertEqual(paused["status"], "paused_for_controlled_resume")
        self.assertEqual(completed["status"], "completed")
        self.assertTrue(
            completed["comparison_to_initialization"][
                "measurable_learning"
            ]
        )
        self.assertEqual(FakeTrainer.calls[0], None)
        self.assertTrue(
            str(FakeTrainer.calls[1]).endswith("checkpoint-125")
        )


if __name__ == "__main__":
    unittest.main()
