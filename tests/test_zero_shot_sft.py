from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from llm2048.zero_shot_sft import (
    IGNORE_INDEX,
    CorpusSelection,
    GatePreflightError,
    SelectionConfig,
    ZeroShotSftConfig,
    _load_adapter,
    _assert_planned_finalization_has_no_failure,
    _complete_native_policy_response,
    _render_prompt,
    _select_starting_point,
    _source_gate_result,
    _trim_completion,
    _verify_recovery_artifact_hashes,
    _run_fresh_process_adapter_evaluation,
    assert_split_isolation,
    build_sft_record,
    encode_masked_sft_record,
    gate_metrics,
    preflight_real_gate,
    run_adapter_reload_recovery,
    run_zero_shot_sft_gate,
    select_corpus_records,
)


class _CharacterProcessor:
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        **kwargs: object,
    ) -> str:
        if kwargs != {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }:
            raise AssertionError("unexpected Qwen chat template arguments")
        return "rendered-prompt:<think>\n\n</think>\n\n"

    def __call__(
        self, text: str, *, add_special_tokens: bool
    ) -> dict[str, list[int]]:
        if add_special_tokens:
            raise AssertionError("segments must not add special tokens")
        return {"input_ids": [ord(character) for character in text]}


class _PromptRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        **kwargs: object,
    ) -> str:
        self.calls.append(kwargs)
        return "rendered"


def _row(
    *,
    record_id: str,
    split: str,
    stratum: str,
    trajectory: str,
    orbit: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_id": record_id,
        "board": [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        "valid_moves": ["down", "right"],
        "teacher_action": "down",
        "split": split,
        "stratum": stratum,
        "lineage": {
            "trajectory_id": trajectory,
            "trajectory_step": 0,
            "orbit_id": orbit,
            "symmetry_id": 0,
        },
    }


def _selection(
    name: str,
    split: str,
    rows: list[dict[str, object]],
) -> CorpusSelection:
    return CorpusSelection(
        records=rows,
        split=split,
        source_path=Path(f"/{name}.jsonl"),
        source_sha256="0" * 64,
        source_records=len(rows),
        member_ids_sha256="1" * 64,
    )


class ConfigurationTests(unittest.TestCase):
    def test_adapter_reload_gate_uses_the_12gb_headroom_batch_plan(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, _ = ZeroShotSftConfig.load(
            repository / "configs" / "qwen35_4b_zero_shot_sft_gate.json"
        )

        self.assertEqual(config.gate.evaluation_batch_size, 8)
        self.assertEqual(config.gate.adapter_evaluation_batch_size, 4)
        self.assertEqual(config.gate.recovery_evaluation_batch_size, 2)
        self.assertEqual(config.gate.batch_size("zero_shot"), 8)
        self.assertEqual(config.gate.batch_size("post_sft"), 4)
        self.assertEqual(config.gate.batch_size("post_sft_recovery"), 2)

    def test_repository_config_pins_the_fixed_gate_and_approximately_1000_sft_rows(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, digest = ZeroShotSftConfig.load(
            repository / "configs" / "qwen35_4b_zero_shot_sft_gate.json"
        )

        self.assertEqual(len(digest), 64)
        self.assertEqual(config.gate.selection.count, 500)
        self.assertEqual(
            config.gate.selection.strata,
            {"natural": 250, "hard": 150, "late": 100},
        )
        self.assertEqual(config.sft.train.count, 900)
        self.assertEqual(config.sft.validation.count, 100)
        self.assertEqual(config.sft.total_examples, 1000)
        self.assertEqual(config.gate.parse_rate_minimum, 0.95)
        self.assertEqual(config.gate.illegal_action_rate_maximum, 0.02)
        self.assertEqual(config.sft.trainer, "transformers.Trainer")

    def test_real_preflight_requires_private_online_telemetry_credential(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, _ = ZeroShotSftConfig.load(
            repository / "configs" / "qwen35_4b_zero_shot_sft_gate.json"
        )

        with self.assertRaisesRegex(GatePreflightError, "WANDB_API_KEY"):
            preflight_real_gate(config, environ={})


class CorpusSelectionTests(unittest.TestCase):
    def test_hash_selection_is_deterministic_and_stratified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "test.jsonl"
            rows = [
                _row(
                    record_id=f"test-{stratum}-{index}",
                    split="test",
                    stratum=stratum,
                    trajectory=f"trajectory-{stratum}-{index}",
                    orbit=f"orbit-{stratum}-{index}",
                )
                for stratum in ("natural", "hard", "late")
                for index in range(3)
            ]
            raw = b"".join(
                (
                    json.dumps(row, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                for row in rows
            )
            source.write_bytes(raw)
            manifest = {
                "artifacts": [
                    {
                        "name": "test",
                        "path": "test.jsonl",
                        "records": len(rows),
                        "sha256": sha256(raw).hexdigest(),
                    }
                ]
            }
            selection_config = SelectionConfig(
                split="test",
                count=3,
                strata={"natural": 1, "hard": 1, "late": 1},
            )

            first = select_corpus_records(
                manifest_path=root / "manifest.json",
                manifest=manifest,
                selection=selection_config,
                seed=2048,
                purpose="gate",
            )
            second = select_corpus_records(
                manifest_path=root / "manifest.json",
                manifest=manifest,
                selection=selection_config,
                seed=2048,
                purpose="gate",
            )

            self.assertEqual(
                [row["record_id"] for row in first.records],
                [row["record_id"] for row in second.records],
            )
            self.assertEqual(
                {row["stratum"] for row in first.records},
                {"natural", "hard", "late"},
            )
            self.assertEqual(
                first.member_ids_sha256, second.member_ids_sha256
            )

    def test_lineage_or_orbit_overlap_fails_closed(self) -> None:
        gate = _selection(
            "gate",
            "test",
            [
                _row(
                    record_id="test-1",
                    split="test",
                    stratum="natural",
                    trajectory="shared-trajectory",
                    orbit="gate-orbit",
                )
            ],
        )
        train = _selection(
            "train",
            "train",
            [
                _row(
                    record_id="train-1",
                    split="train",
                    stratum="natural",
                    trajectory="shared-trajectory",
                    orbit="train-orbit",
                )
            ],
        )
        validation = _selection(
            "validation",
            "validation",
            [
                _row(
                    record_id="validation-1",
                    split="validation",
                    stratum="natural",
                    trajectory="validation-trajectory",
                    orbit="validation-orbit",
                )
            ],
        )

        with self.assertRaisesRegex(
            GatePreflightError, "split isolation failed"
        ):
            assert_split_isolation(gate, train, validation)


class MaskedSftTests(unittest.TestCase):
    def test_reasoning_masks_only_prompt_and_policy_reasoning_trace_body(
        self,
    ) -> None:
        processor = _CharacterProcessor()
        record = build_sft_record(
            row=_row(
                record_id="train-1",
                split="train",
                stratum="natural",
                trajectory="trajectory-1",
                orbit="orbit-1",
            ),
            variant="reasoning",
            reasoning_trace_placeholder="hidden trace",
        )

        encoded = encode_masked_sft_record(
            record, processor, max_sequence_length=256
        )
        input_text = "".join(chr(token) for token in encoded["input_ids"])
        self.assertNotIn("<think>\n\n</think>\n\n<think>", input_text)
        opening_start = input_text.index("<think>")
        trace_start = input_text.index("hidden trace")
        tail_start = input_text.index("</think>")

        self.assertTrue(
            all(
                label == IGNORE_INDEX
                for label in encoded["labels"][:opening_start]
            )
        )
        self.assertEqual(
            encoded["labels"][opening_start:trace_start],
            encoded["input_ids"][opening_start:trace_start],
        )
        self.assertTrue(
            all(
                label == IGNORE_INDEX
                for label in encoded["labels"][trace_start:tail_start]
            )
        )
        self.assertEqual(
            encoded["labels"][tail_start:],
            encoded["input_ids"][tail_start:],
        )
        self.assertIn("<action>DOWN</action>", input_text)

    def test_direct_action_supervises_complete_response_envelope(self) -> None:
        processor = _CharacterProcessor()
        record = build_sft_record(
            row=_row(
                record_id="train-1",
                split="train",
                stratum="hard",
                trajectory="trajectory-1",
                orbit="orbit-1",
            ),
            variant="direct_action",
            reasoning_trace_placeholder="unused",
        )

        encoded = encode_masked_sft_record(
            record, processor, max_sequence_length=256
        )
        prompt_length = len(
            "rendered-prompt:<think>\n\n</think>\n\n"
        )

        self.assertTrue(
            all(
                label == IGNORE_INDEX
                for label in encoded["labels"][:prompt_length]
            )
        )
        self.assertEqual(
            encoded["labels"][prompt_length:],
            encoded["input_ids"][prompt_length:],
        )


class GateMetricTests(unittest.TestCase):
    def test_regressed_adapter_is_not_promoted_as_the_starting_point(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, _ = ZeroShotSftConfig.load(
            repository / "configs" / "qwen35_4b_zero_shot_sft_gate.json"
        )
        zero_shot = {
            "metrics": {
                "passed": False,
                "parse_rate": 1.0,
                "truncation_rate": 0.0,
                "illegal_action_rate": 0.094,
                "policy_failure_rate": 0.094,
            }
        }
        post_sft = {
            "metrics": {
                "passed": False,
                "parse_rate": 1.0,
                "truncation_rate": 0.0,
                "illegal_action_rate": 0.158,
                "policy_failure_rate": 0.158,
            }
        }

        selected = _select_starting_point(
            config=config,
            variant="direct_action",
            zero_shot=zero_shot,
            post_sft=post_sft,
            adapter_path="adapters/direct_action",
        )

        self.assertEqual(selected["kind"], "unchanged_base_model")
        self.assertEqual(selected["decision"], "hold_no_passing_candidate")
        self.assertFalse(selected["gate_passed"])
        self.assertEqual(
            selected["rejected_adapter"]["path"],
            "adapters/direct_action",
        )

    def test_gate_enables_native_qwen_thinking_only_for_reasoning_policy(
        self,
    ) -> None:
        processor = _PromptRecorder()
        board = [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        _render_prompt(processor, "direct_action", board)
        _render_prompt(processor, "reasoning", board)

        self.assertFalse(processor.calls[0]["enable_thinking"])
        self.assertTrue(processor.calls[1]["enable_thinking"])

    def test_reasoning_contract_restores_native_prefix_from_qwen_prompt(
        self,
    ) -> None:
        response = _complete_native_policy_response(
            "reasoning",
            "Keep large tiles together.</think><action>LEFT</action>",
        )

        self.assertEqual(
            response,
            "<think>\nKeep large tiles together.</think>"
            "<action>LEFT</action>",
        )

    def test_thresholds_use_all_boards_as_the_denominator(self) -> None:
        events: list[dict[str, object]] = [
            {
                "parsed": True,
                "truncated": False,
                "policy_failure_reason": None,
                "valid_action": True,
                "policy_failure": False,
                "response_length_tokens": 4,
            }
            for _ in range(95)
        ]
        events.extend(
            [
                {
                    "parsed": True,
                    "truncated": False,
                    "policy_failure_reason": "illegal_action",
                    "valid_action": False,
                    "policy_failure": True,
                    "response_length_tokens": 4,
                }
                for _ in range(2)
            ]
        )
        events.extend(
            [
                {
                    "parsed": False,
                    "truncated": True,
                    "policy_failure_reason": "truncated_response",
                    "valid_action": False,
                    "policy_failure": True,
                    "response_length_tokens": 96,
                }
                for _ in range(3)
            ]
        )

        metrics = gate_metrics(
            events,
            parse_rate_minimum=0.95,
            illegal_action_rate_maximum=0.02,
        )

        self.assertEqual(metrics["parse_rate"], 0.97)
        self.assertEqual(metrics["illegal_action_rate"], 0.02)
        self.assertEqual(metrics["truncation_rate"], 0.03)
        self.assertTrue(metrics["passed"])

    def test_completion_is_truncated_only_without_eos_at_budget(self) -> None:
        completed, completed_truncated = _trim_completion(
            [10, 11, 2, 0],
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=4,
        )
        exhausted, exhausted_truncated = _trim_completion(
            [10, 11, 12, 13],
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=4,
        )

        self.assertEqual(completed, [10, 11])
        self.assertFalse(completed_truncated)
        self.assertEqual(exhausted, [10, 11, 12, 13])
        self.assertTrue(exhausted_truncated)


class AdapterReloadTests(unittest.TestCase):
    def test_default_phase_one_defers_reasoning_reload_without_failure(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, digest = ZeroShotSftConfig.load(
            repository / "configs" / "qwen35_4b_zero_shot_sft_gate.json"
        )
        failed_metrics = {
            "passed": False,
            "parse_rate": 1.0,
            "truncation_rate": 0.0,
            "illegal_action_rate": 0.1,
            "valid_action_rate": 0.9,
            "policy_failure_rate": 0.1,
            "response_length_tokens": {"mean": 7.0},
        }
        gate_selection = SimpleNamespace(
            records=[{"record_id": "gate-1"}],
            member_ids_sha256="1" * 64,
        )
        data_plan = {
            "manifest_path": Path("/corpus/manifest.json"),
            "manifest_sha256": "2" * 64,
            "gate": gate_selection,
            "train": SimpleNamespace(records=[{"record_id": "train-1"}]),
            "validation": SimpleNamespace(
                records=[{"record_id": "validation-1"}]
            ),
        }
        writer = MagicMock()
        wandb_run = SimpleNamespace(
            entity="entity",
            project="project",
            id="phase-one",
            url="https://wandb.invalid/phase-one",
            log=MagicMock(),
            finish=MagicMock(),
        )
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                empty_cache=MagicMock(),
            )
        )
        stack = SimpleNamespace(
            torch=torch,
            summary_writer_class=MagicMock(return_value=writer),
        )
        load_adapter = MagicMock(return_value=(object(), object()))

        def write_lock(**kwargs: object) -> dict[str, object]:
            lock_path = kwargs["lock_path"]
            assert isinstance(lock_path, Path)
            lock: dict[str, object] = {
                "artifacts": {"handoff.json": "0" * 64}
            }
            lock_path.write_text(json.dumps(lock) + "\n", encoding="utf-8")
            return lock

        def write_data_plan(**kwargs: object) -> dict[str, object]:
            output_directory = kwargs["output_directory"]
            assert isinstance(output_directory, Path)
            manifest: dict[str, object] = {}
            (output_directory / "data_manifest.json").write_text(
                json.dumps(manifest) + "\n",
                encoding="utf-8",
            )
            return manifest

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase-one"
            patches = (
                patch(
                    "llm2048.zero_shot_sft.preflight_real_gate"
                ),
                patch(
                    "llm2048.zero_shot_sft.prepare_data_plan",
                    return_value=data_plan,
                ),
                patch(
                    "llm2048.zero_shot_sft._load_runtime_stack",
                    return_value=stack,
                ),
                patch("llm2048.zero_shot_sft._preflight_cuda"),
                patch("llm2048.zero_shot_sft._configure_telemetry"),
                patch(
                    "llm2048.zero_shot_sft._write_data_plan",
                    side_effect=write_data_plan,
                ),
                patch(
                    "llm2048.zero_shot_sft._start_wandb_run",
                    return_value=(wandb_run, "PRIVATE"),
                ),
                patch(
                    "llm2048.zero_shot_sft._load_base_model",
                    return_value=(object(), object()),
                ),
                patch(
                    "llm2048.zero_shot_sft._evaluate_gate",
                    return_value={"metrics": failed_metrics},
                ),
                patch(
                    "llm2048.zero_shot_sft._write_variant_sft_data"
                ),
                patch(
                    "llm2048.zero_shot_sft._train_variant",
                    return_value={
                        "status": "completed",
                        "adapter_reloaded_for_gate": False,
                    },
                ),
                patch(
                    "llm2048.zero_shot_sft._load_adapter",
                    load_adapter,
                ),
                patch(
                    "llm2048.zero_shot_sft.create_finalization_lock",
                    side_effect=write_lock,
                ),
                patch(
                    "llm2048.zero_shot_sft._build_run_manifest",
                    return_value={"schema_version": 1},
                ),
            )
            entered = [context.start() for context in patches]
            try:
                result = run_zero_shot_sft_gate(
                    config=config,
                    input_sha256=digest,
                    output_directory=output,
                )
            finally:
                for context in reversed(patches):
                    context.stop()
                del entered

            self.assertEqual(
                result["status"],
                "awaiting_fresh_process_finalization",
            )
            self.assertTrue((output / "handoff.json").is_file())
            self.assertTrue((output / "finalization-lock.json").is_file())
            self.assertFalse((output / "failure.json").exists())
            self.assertEqual(load_adapter.call_count, 1)
            self.assertEqual(
                load_adapter.call_args.kwargs["adapter_directory"].name,
                "direct_action",
            )

    def test_planned_finalization_rejects_failure_artifact_dependency(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "failure.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                GatePreflightError, "must not depend on failure.json"
            ):
                _assert_planned_finalization_has_no_failure(source)

    def test_recovery_rejects_gate_member_or_order_drift(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, _ = ZeroShotSftConfig.load(
            repository / "configs" / "qwen35_4b_zero_shot_sft_gate.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            path = source / "gate.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "record_id": "different-board",
                        "variant": "reasoning",
                        "phase": "zero_shot",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                GatePreflightError, "fixed gate members"
            ):
                _source_gate_result(
                    path=path,
                    records=[{"record_id": "expected-board"}],
                    variant="reasoning",
                    phase="zero_shot",
                    config=config,
                    source=source,
                )

    def test_recovery_rejects_a_changed_locked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            artifact = source / "adapters" / "reasoning" / "adapter_model.bin"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"locked")
            locked_hash = sha256(b"locked").hexdigest()
            artifact.write_bytes(b"changed")

            with self.assertRaisesRegex(
                GatePreflightError, "source hash mismatch"
            ):
                _verify_recovery_artifact_hashes(
                    source=source,
                    artifacts={
                        "adapters/reasoning/adapter_model.bin": locked_hash
                    },
                )

    def test_recovery_lock_rejects_paths_outside_the_source_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                GatePreflightError, "escapes the source run"
            ):
                _verify_recovery_artifact_hashes(
                    source=Path(temporary),
                    artifacts={"../adapter.bin": "0" * 64},
                )

    def test_recovery_entrypoint_cannot_invoke_training(self) -> None:
        source = (
            inspect.getsource(run_adapter_reload_recovery)
            + inspect.getsource(_run_fresh_process_adapter_evaluation)
        )

        self.assertNotIn("_train_variant(", source)
        self.assertIn('"fresh_training_performed": False', source)

    def test_adapter_reload_uses_the_exact_pinned_base_revision(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, _ = ZeroShotSftConfig.load(
            repository / "configs" / "qwen35_4b_zero_shot_sft_gate.json"
        )
        base_model = object()
        processor = object()

        class Peft:
            called: dict[str, object] | None = None

            @classmethod
            def from_pretrained(
                cls, *, model: object, model_id: str, is_trainable: bool
            ) -> object:
                cls.called = {
                    "model": model,
                    "model_id": model_id,
                    "is_trainable": is_trainable,
                }
                return "adapted"

        stack = SimpleNamespace(peft_model_class=Peft)
        with patch(
            "llm2048.zero_shot_sft._load_base_model",
            return_value=(base_model, processor),
        ) as load_base:
            model, loaded_processor = _load_adapter(
                stack=stack,  # type: ignore[arg-type]
                config=config,
                adapter_directory=Path("/tmp/adapter"),
            )

        load_base.assert_called_once_with(stack, config)
        self.assertEqual(model, "adapted")
        self.assertIs(loaded_processor, processor)
        self.assertEqual(
            Peft.called,
            {
                "model": base_model,
                "model_id": "/tmp/adapter",
                "is_trainable": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
