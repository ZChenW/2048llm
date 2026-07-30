"""Zero-shot gate and conditional masked SFT for the two Student Policies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gc
import heapq
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence, cast

from llm2048.grpo_smoke import (
    MODEL_REVISION_PATTERN,
    OFFICIAL_QWEN35_4B,
    _assert_adapter_only_checkpoint,
    _ensure_generation_architecture,
    _require_keywords,
    _trainable_parameter_evidence,
    _verify_private_wandb_project,
)
from llm2048.policy_contracts import (
    Action,
    PolicyVariant,
    build_policy_prompt,
    change_making_actions,
    enforce_policy_response,
)


IGNORE_INDEX = -100
VARIANTS: tuple[PolicyVariant, ...] = ("direct_action", "reasoning")
STRATA = ("natural", "hard", "late")
SPLITS = ("train", "validation", "test")
QWEN_DISABLED_THINKING_SUFFIX = "<think>\n\n</think>\n\n"


class GateConfigurationError(ValueError):
    """Raised when the zero-shot/SFT configuration is unsafe or ambiguous."""


class GatePreflightError(RuntimeError):
    """Raised before a real model operation when prerequisites are absent."""


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str
    precision: str
    load_in_4bit: bool
    fast_inference: bool
    text_only: bool
    max_sequence_length: int

    def resolved(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "revision": self.revision,
            "precision": self.precision,
            "load_in_4bit": self.load_in_4bit,
            "fast_inference": self.fast_inference,
            "text_only": self.text_only,
            "max_sequence_length": self.max_sequence_length,
        }


@dataclass(frozen=True)
class CorpusConfig:
    manifest_path: str
    manifest_environment_override: str

    def resolved(self) -> dict[str, str]:
        return {
            "manifest_path": self.manifest_path,
            "manifest_environment_override": self.manifest_environment_override,
        }


@dataclass(frozen=True)
class SelectionConfig:
    split: str
    count: int
    strata: dict[str, int]

    def resolved(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "count": self.count,
            "strata": self.strata,
        }


@dataclass(frozen=True)
class GateConfig:
    selection: SelectionConfig
    parse_rate_minimum: float
    illegal_action_rate_maximum: float
    evaluation_batch_size: int
    adapter_evaluation_batch_size: int
    recovery_evaluation_batch_size: int
    direct_action_max_new_tokens: int
    reasoning_max_new_tokens: int

    def max_new_tokens(self, variant: PolicyVariant) -> int:
        return (
            self.direct_action_max_new_tokens
            if variant == "direct_action"
            else self.reasoning_max_new_tokens
        )

    def batch_size(self, phase: str) -> int:
        if phase == "post_sft_recovery":
            return self.recovery_evaluation_batch_size
        return (
            self.adapter_evaluation_batch_size
            if phase == "post_sft"
            else self.evaluation_batch_size
        )

    def resolved(self) -> dict[str, Any]:
        return {
            **self.selection.resolved(),
            "parse_rate_minimum": self.parse_rate_minimum,
            "illegal_action_rate_maximum": self.illegal_action_rate_maximum,
            "evaluation_batch_size": self.evaluation_batch_size,
            "adapter_evaluation_batch_size": (
                self.adapter_evaluation_batch_size
            ),
            "recovery_evaluation_batch_size": (
                self.recovery_evaluation_batch_size
            ),
            "direct_action_max_new_tokens": self.direct_action_max_new_tokens,
            "reasoning_max_new_tokens": self.reasoning_max_new_tokens,
        }


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    finetune_vision_layers: bool
    finetune_language_layers: bool
    finetune_attention_modules: bool
    finetune_mlp_modules: bool
    gradient_checkpointing: str

    def resolved(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "finetune_vision_layers": self.finetune_vision_layers,
            "finetune_language_layers": self.finetune_language_layers,
            "finetune_attention_modules": self.finetune_attention_modules,
            "finetune_mlp_modules": self.finetune_mlp_modules,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


@dataclass(frozen=True)
class TrainingConfig:
    num_train_epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    optimizer: str
    logging_steps: int

    def resolved(self) -> dict[str, Any]:
        return {
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "logging_steps": self.logging_steps,
        }


@dataclass(frozen=True)
class SftConfig:
    trainer: str
    total_examples: int
    train: SelectionConfig
    validation: SelectionConfig
    policy_reasoning_trace_placeholder: str
    mask_policy_reasoning_trace_body: bool
    lora: LoraConfig
    training: TrainingConfig

    def resolved(self) -> dict[str, Any]:
        return {
            "trainer": self.trainer,
            "total_examples": self.total_examples,
            "train": self.train.resolved(),
            "validation": self.validation.resolved(),
            "policy_reasoning_trace_placeholder": (
                self.policy_reasoning_trace_placeholder
            ),
            "mask_policy_reasoning_trace_body": (
                self.mask_policy_reasoning_trace_body
            ),
            "lora": self.lora.resolved(),
            "training": self.training.resolved(),
        }


@dataclass(frozen=True)
class TelemetryConfig:
    wandb_mode: str
    wandb_entity: str
    wandb_project: str
    wandb_project_visibility: str
    tensorboard: bool
    upload_model_checkpoints: bool

    def resolved(self) -> dict[str, Any]:
        return {
            "wandb_mode": self.wandb_mode,
            "wandb_entity": self.wandb_entity,
            "wandb_project": self.wandb_project,
            "wandb_project_visibility": self.wandb_project_visibility,
            "tensorboard": self.tensorboard,
            "upload_model_checkpoints": self.upload_model_checkpoints,
        }


@dataclass(frozen=True)
class ZeroShotSftConfig:
    schema_version: int
    experiment_name: str
    seed: int
    model: ModelConfig
    corpus: CorpusConfig
    gate: GateConfig
    sft: SftConfig
    telemetry: TelemetryConfig
    repository_root: Path

    @classmethod
    def load(cls, path: Path) -> tuple["ZeroShotSftConfig", str]:
        try:
            raw_bytes = path.read_bytes()
            raw = json.loads(raw_bytes)
        except OSError as error:
            raise GateConfigurationError(
                f"cannot read zero-shot/SFT configuration: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise GateConfigurationError(
                f"zero-shot/SFT configuration is not valid JSON: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise GateConfigurationError(
                "zero-shot/SFT configuration must be a JSON object"
            )
        _exact_keys(
            raw,
            {
                "schema_version",
                "experiment_name",
                "seed",
                "model",
                "corpus",
                "gate",
                "sft",
                "telemetry",
            },
            "zero-shot/SFT configuration",
        )
        _require(raw["schema_version"], 1, "schema_version")
        experiment_name = _nonempty_string(
            raw["experiment_name"], "experiment_name"
        )
        seed = _integer(raw["seed"], "seed", minimum=0)
        model = _load_model(_object(raw["model"], "model"))
        corpus = _load_corpus(_object(raw["corpus"], "corpus"))
        gate = _load_gate(_object(raw["gate"], "gate"))
        sft = _load_sft(_object(raw["sft"], "sft"))
        telemetry = _load_telemetry(_object(raw["telemetry"], "telemetry"))
        if sft.train.count + sft.validation.count != sft.total_examples:
            raise GateConfigurationError(
                "sft.total_examples must equal train.count + validation.count"
            )
        return (
            cls(
                schema_version=1,
                experiment_name=experiment_name,
                seed=seed,
                model=model,
                corpus=corpus,
                gate=gate,
                sft=sft,
                telemetry=telemetry,
                repository_root=path.resolve().parent.parent,
            ),
            sha256(raw_bytes).hexdigest(),
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "model": self.model.resolved(),
            "corpus": self.corpus.resolved(),
            "gate": self.gate.resolved(),
            "sft": self.sft.resolved(),
            "telemetry": self.telemetry.resolved(),
        }

    def corpus_manifest_path(
        self, environ: Mapping[str, str] | None = None
    ) -> Path:
        environment = os.environ if environ is None else environ
        override = environment.get(self.corpus.manifest_environment_override)
        if override:
            return Path(override).expanduser().resolve()
        return (self.repository_root / self.corpus.manifest_path).resolve()


@dataclass(frozen=True)
class RuntimeStack:
    torch: Any
    wandb: Any
    dataset_class: Any
    trainer_class: Any
    training_arguments_class: Any
    fast_vision_model: Any
    snapshot_download: Any
    peft_model_class: Any
    summary_writer_class: Any
    versions: dict[str, str]
    api_signatures: dict[str, str]


@dataclass(frozen=True)
class CorpusSelection:
    records: list[dict[str, Any]]
    split: str
    source_path: Path
    source_sha256: str
    source_records: int
    member_ids_sha256: str

    def evidence(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "source_records": self.source_records,
            "selected_records": len(self.records),
            "member_ids_sha256": self.member_ids_sha256,
            "strata": dict(Counter(row["stratum"] for row in self.records)),
        }


def preflight_real_gate(
    config: ZeroShotSftConfig,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fail closed before model loading or online telemetry creation."""
    environment = os.environ if environ is None else environ
    if not environment.get("WANDB_API_KEY", "").strip():
        raise GatePreflightError(
            "WANDB_API_KEY is required for the private online gate run"
        )
    if environment.get("WANDB_MODE") not in (None, "online"):
        raise GatePreflightError("WANDB_MODE must be 'online'")
    if environment.get("WANDB_LOG_MODEL", "false").lower() != "false":
        raise GatePreflightError(
            "WANDB_LOG_MODEL must be 'false' so adapters are not uploaded"
        )
    manifest_path = config.corpus_manifest_path(environment)
    if not manifest_path.is_file():
        raise GatePreflightError(
            "Teacher Policy Corpus manifest is missing; set "
            f"{config.corpus.manifest_environment_override} to the retained "
            "production manifest"
        )


def prepare_data_plan(
    config: ZeroShotSftConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate corpus provenance and deterministically select every split."""
    manifest_path = config.corpus_manifest_path(environ)
    manifest, manifest_sha256 = _load_corpus_manifest(manifest_path)
    gate = select_corpus_records(
        manifest_path=manifest_path,
        manifest=manifest,
        selection=config.gate.selection,
        seed=config.seed,
        purpose="gate",
    )
    train = select_corpus_records(
        manifest_path=manifest_path,
        manifest=manifest,
        selection=config.sft.train,
        seed=config.seed,
        purpose="sft-train",
    )
    validation = select_corpus_records(
        manifest_path=manifest_path,
        manifest=manifest,
        selection=config.sft.validation,
        seed=config.seed,
        purpose="sft-validation",
    )
    isolation = assert_split_isolation(gate, train, validation)
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "gate": gate,
        "train": train,
        "validation": validation,
        "split_isolation": isolation,
    }


def select_corpus_records(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    selection: SelectionConfig,
    seed: int,
    purpose: str,
) -> CorpusSelection:
    """Select the lowest seeded record hashes per Corpus Stratum."""
    artifact = _corpus_artifact(manifest, selection.split)
    source_path = manifest_path.parent / artifact["path"]
    expected_sha = artifact["sha256"]
    expected_records = artifact["records"]
    if not source_path.is_file():
        raise GatePreflightError(
            f"Teacher Policy Corpus {selection.split} artifact is missing: "
            f"{source_path}"
        )
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
        stratum: [] for stratum in STRATA
    }
    digest = sha256()
    source_records = 0
    with source_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            digest.update(raw_line)
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise GatePreflightError(
                    f"{source_path}:{line_number} is not valid JSON"
                ) from error
            _validate_corpus_row(
                row,
                expected_split=selection.split,
                source_path=source_path,
                line_number=line_number,
            )
            source_records += 1
            stratum = row["stratum"]
            target = selection.strata[stratum]
            if target == 0:
                continue
            record_id = row["record_id"]
            priority = int.from_bytes(
                sha256(
                    f"{seed}:{purpose}:{record_id}".encode("utf-8")
                ).digest(),
                "big",
            )
            item = (-priority, record_id, row)
            heap = heaps[stratum]
            if len(heap) < target:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha:
        raise GatePreflightError(
            f"{selection.split} corpus SHA-256 differs from its manifest"
        )
    if source_records != expected_records:
        raise GatePreflightError(
            f"{selection.split} corpus record count differs from its manifest"
        )
    records: list[dict[str, Any]] = []
    for stratum in STRATA:
        selected = heaps[stratum]
        target = selection.strata[stratum]
        if len(selected) != target:
            raise GatePreflightError(
                f"{selection.split} lacks {target} {stratum} records"
            )
        records.extend(
            row
            for _, _, row in sorted(
                selected,
                key=lambda item: (-item[0], item[1]),
            )
        )
    if len(records) != selection.count:
        raise RuntimeError("internal selected-record count mismatch")
    ids = "\n".join(row["record_id"] for row in records) + "\n"
    return CorpusSelection(
        records=records,
        split=selection.split,
        source_path=source_path,
        source_sha256=actual_sha,
        source_records=source_records,
        member_ids_sha256=sha256(ids.encode("utf-8")).hexdigest(),
    )


def assert_split_isolation(
    gate: CorpusSelection,
    train: CorpusSelection,
    validation: CorpusSelection,
) -> dict[str, Any]:
    """Prove no Trajectory Lineage or Symmetry Orbit crosses selected splits."""
    selections = {"gate": gate, "train": train, "validation": validation}
    lineage_sets = {
        name: {row["lineage"]["trajectory_id"] for row in value.records}
        for name, value in selections.items()
    }
    orbit_sets = {
        name: {row["lineage"]["orbit_id"] for row in value.records}
        for name, value in selections.items()
    }
    checked_pairs: list[dict[str, Any]] = []
    names = tuple(selections)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared_lineages = lineage_sets[left] & lineage_sets[right]
            shared_orbits = orbit_sets[left] & orbit_sets[right]
            if shared_lineages or shared_orbits:
                raise GatePreflightError(
                    "Teacher Policy Corpus split isolation failed between "
                    f"{left} and {right}"
                )
            checked_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "shared_trajectory_lineages": 0,
                    "shared_symmetry_orbits": 0,
                }
            )
    return {
        "status": "passed",
        "split_unit": "trajectory_and_symmetry_orbit",
        "pairs": checked_pairs,
    }


def gate_metrics(
    events: Sequence[Mapping[str, Any]],
    *,
    parse_rate_minimum: float,
    illegal_action_rate_maximum: float,
) -> dict[str, Any]:
    """Aggregate one variant/phase without hiding Policy Failure cases."""
    if not events:
        raise ValueError("gate events must not be empty")
    total = len(events)
    parsed = sum(bool(event["parsed"]) for event in events)
    truncated = sum(bool(event["truncated"]) for event in events)
    illegal = sum(
        event["policy_failure_reason"] == "illegal_action" for event in events
    )
    valid = sum(bool(event["valid_action"]) for event in events)
    failures = sum(bool(event["policy_failure"]) for event in events)
    lengths = sorted(int(event["response_length_tokens"]) for event in events)
    parse_rate = parsed / total
    illegal_rate = illegal / total
    return {
        "boards": total,
        "parsed": parsed,
        "parse_rate": parse_rate,
        "truncated": truncated,
        "truncation_rate": truncated / total,
        "illegal_actions": illegal,
        "illegal_action_rate": illegal_rate,
        "valid_actions": valid,
        "valid_action_rate": valid / total,
        "policy_failures": failures,
        "policy_failure_rate": failures / total,
        "policy_failure_reasons": dict(
            sorted(
                Counter(
                    str(event["policy_failure_reason"])
                    for event in events
                    if event["policy_failure_reason"] is not None
                ).items()
            )
        ),
        "response_length_tokens": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p95": _nearest_rank(lengths, 0.95),
            "maximum": max(lengths),
        },
        "thresholds": {
            "parse_rate_minimum": parse_rate_minimum,
            "illegal_action_rate_maximum": illegal_action_rate_maximum,
        },
        "passed": (
            parse_rate >= parse_rate_minimum
            and illegal_rate <= illegal_action_rate_maximum
        ),
    }


def _select_starting_point(
    *,
    config: ZeroShotSftConfig,
    variant: PolicyVariant,
    zero_shot: Mapping[str, Any],
    post_sft: Mapping[str, Any],
    adapter_path: str,
) -> dict[str, Any]:
    """Select a passing or best available start without promoting regression."""
    zero_metrics = cast(Mapping[str, Any], zero_shot["metrics"])
    post_metrics = cast(Mapping[str, Any], post_sft["metrics"])
    adapter_is_better = _gate_quality(post_metrics) > _gate_quality(
        zero_metrics
    )
    if bool(post_metrics["passed"]) or adapter_is_better:
        return {
            "kind": "lora_adapter",
            "path": adapter_path,
            "base_model_id": config.model.id,
            "base_model_revision": config.model.revision,
            "gate_phase": "post_sft",
            "gate_passed": bool(post_metrics["passed"]),
            "decision": (
                "selected_passing_candidate"
                if bool(post_metrics["passed"])
                else "hold_no_passing_candidate"
            ),
            "selected_metrics": dict(post_metrics),
            **(
                {}
                if bool(post_metrics["passed"])
                else {
                    "warning": (
                        "adapter is the best available candidate but remains "
                        "below the gate threshold"
                    )
                }
            ),
        }
    return {
        "kind": "unchanged_base_model",
        "model_id": config.model.id,
        "revision": config.model.revision,
        "gate_phase": "zero_shot",
        "gate_passed": bool(zero_metrics["passed"]),
        "decision": "hold_no_passing_candidate",
        "selected_metrics": dict(zero_metrics),
        "rejected_adapter": {
            "path": adapter_path,
            "reason": "post-SFT gate regressed relative to zero-shot",
            "metrics": dict(post_metrics),
        },
    }


def _gate_quality(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(bool(metrics["passed"])),
        -float(metrics["policy_failure_rate"]),
        float(metrics["parse_rate"]),
        -float(metrics["illegal_action_rate"]),
        -float(metrics["truncation_rate"]),
    )


def build_sft_record(
    *,
    row: Mapping[str, Any],
    variant: PolicyVariant,
    reasoning_trace_placeholder: str,
) -> dict[str, Any]:
    """Build Teacher-derived supervision without inventing Teacher reasoning."""
    action = cast(Action, str(row["teacher_action"]).upper())
    assistant_content = (
        f"<action>{action}</action>"
        if variant == "direct_action"
        else (
            f"<think>{reasoning_trace_placeholder}</think>"
            f"<action>{action}</action>"
        )
    )
    return {
        "schema_version": 1,
        "record_id": row["record_id"],
        "source_split": row["split"],
        "source_stratum": row["stratum"],
        "lineage": row["lineage"],
        "variant": variant,
        "messages": [
            {
                "role": "user",
                "content": build_policy_prompt(variant, row["board"]),
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "supervision": {
            "source": "Teacher Policy final action",
            "policy_reasoning_trace_source": "fixed_non_teacher_placeholder",
            "policy_reasoning_trace_body_masked": variant == "reasoning",
            "response_envelope_supervised": True,
            "final_action_supervised": True,
        },
    }


def encode_masked_sft_record(
    record: Mapping[str, Any],
    processor: Any,
    *,
    max_sequence_length: int,
) -> dict[str, list[int]]:
    """Encode one SFT row and mask prompt plus Reasoning Trace body labels."""
    messages = cast(list[dict[str, str]], record["messages"])
    user_message = messages[0]
    assistant_content = messages[1]["content"]
    variant = cast(PolicyVariant, record["variant"])
    prompt_text = _render_sft_prompt(
        processor,
        variant=variant,
        user_content=user_message["content"],
    )
    eos_token = getattr(processor, "eos_token", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise RuntimeError("Qwen processor does not expose an EOS token")

    prompt_ids = _token_ids(processor, prompt_text)
    if variant == "direct_action":
        response_ids = _token_ids(processor, assistant_content + eos_token)
        input_ids = [*prompt_ids, *response_ids]
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    else:
        opening = "<think>"
        closing = "</think>"
        if not assistant_content.startswith(opening):
            raise GateConfigurationError(
                "Reasoning Policy SFT response must begin with <think>"
            )
        closing_index = assistant_content.find(closing, len(opening))
        if closing_index < 0:
            raise GateConfigurationError(
                "Reasoning Policy SFT response must close </think>"
            )
        trace = assistant_content[len(opening) : closing_index]
        tail = assistant_content[closing_index:]
        opening_ids = _token_ids(processor, opening)
        trace_ids = _token_ids(processor, trace)
        tail_ids = _token_ids(processor, tail + eos_token)
        input_ids = [
            *prompt_ids,
            *opening_ids,
            *trace_ids,
            *tail_ids,
        ]
        labels = [
            *([IGNORE_INDEX] * len(prompt_ids)),
            *opening_ids,
            *([IGNORE_INDEX] * len(trace_ids)),
            *tail_ids,
        ]
        if not opening_ids or not trace_ids or not tail_ids:
            raise RuntimeError("Reasoning Policy SFT token segments must be non-empty")
    if len(input_ids) > max_sequence_length:
        raise GatePreflightError(
            "masked SFT row exceeds model.max_sequence_length: "
            f"{len(input_ids)} > {max_sequence_length}"
        )
    if len(input_ids) != len(labels):
        raise RuntimeError("masked SFT input/label length mismatch")
    if not any(label != IGNORE_INDEX for label in labels):
        raise RuntimeError("masked SFT row has no supervised response tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _render_sft_prompt(
    processor: Any,
    *,
    variant: PolicyVariant,
    user_content: str,
) -> str:
    """Render the assistant boundary without duplicating Reasoning tags."""
    rendered = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_content}],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeError("Qwen processor did not render a text SFT prompt")
    if variant == "reasoning":
        if not rendered.endswith(QWEN_DISABLED_THINKING_SUFFIX):
            raise GatePreflightError(
                "pinned Qwen disabled-thinking template no longer ends with "
                "the expected empty thinking scaffold"
            )
        return rendered.removesuffix(QWEN_DISABLED_THINKING_SUFFIX)
    return rendered


class CausalSftCollator:
    """Right-pad causal SFT examples while retaining the ignore mask."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(
        self, features: list[dict[str, list[int]]]
    ) -> dict[str, Any]:
        torch = import_module("torch")
        pad_id = getattr(self.processor, "pad_token_id", None)
        if not isinstance(pad_id, int):
            raise RuntimeError("Qwen processor does not expose a pad token id")
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * padding)
            attention_masks.append(
                feature["attention_mask"] + [0] * padding
            )
            labels.append(feature["labels"] + [IGNORE_INDEX] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def run_zero_shot_sft_gate(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Run both gates and conditionally train only failing policy variants."""
    preflight_real_gate(config)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise GatePreflightError(
            "output directory must be absent or empty for the gate run"
        )
    data_plan = prepare_data_plan(config)
    stack = _load_runtime_stack()
    _preflight_cuda(stack)
    output_directory.mkdir(parents=True, exist_ok=True)
    telemetry_directory = output_directory / "telemetry"
    telemetry_directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_directory / "resolved_config.json",
        {
            **config.resolved(),
            "resolved_corpus_manifest_path": str(
                data_plan["manifest_path"]
            ),
        },
    )
    data_manifest = _write_data_plan(
        config=config,
        input_sha256=input_sha256,
        data_plan=data_plan,
        output_directory=output_directory,
    )
    _configure_telemetry(config, telemetry_directory)
    wandb_run: Any | None = None
    writer: Any | None = None
    started = time.perf_counter()
    try:
        wandb_run, project_access = _start_wandb_run(
            stack=stack,
            config=config,
            telemetry_directory=telemetry_directory,
        )
        writer = stack.summary_writer_class(
            str(telemetry_directory / "tensorboard" / "gate")
        )
        base_model, processor = _load_base_model(stack, config)
        zero_shot: dict[PolicyVariant, dict[str, Any]] = {}
        for variant in VARIANTS:
            zero_shot[variant] = _evaluate_gate(
                stack=stack,
                config=config,
                model=base_model,
                processor=processor,
                records=data_plan["gate"].records,
                variant=variant,
                phase="zero_shot",
                output_directory=output_directory,
            )
            _log_gate_metrics(
                wandb_run=wandb_run,
                writer=writer,
                variant=variant,
                phase="zero_shot",
                metrics=zero_shot[variant]["metrics"],
                step=0,
            )
        del base_model
        del processor
        gc.collect()
        stack.torch.cuda.empty_cache()

        post_sft: dict[PolicyVariant, dict[str, Any] | None] = {
            "direct_action": None,
            "reasoning": None,
        }
        sft_results: dict[PolicyVariant, dict[str, Any]] = {}
        selections: dict[PolicyVariant, dict[str, Any]] = {}
        deferred_variant: PolicyVariant | None = None
        for variant in VARIANTS:
            if zero_shot[variant]["metrics"]["passed"]:
                sft_results[variant] = {
                    "status": "not_performed",
                    "reason": "zero-shot gate passed",
                    "examples": 0,
                }
                selections[variant] = {
                    "kind": "unchanged_base_model",
                    "model_id": config.model.id,
                    "revision": config.model.revision,
                    "gate_phase": "zero_shot",
                    "gate_passed": True,
                }
                continue
            _write_variant_sft_data(
                config=config,
                variant=variant,
                train=data_plan["train"],
                validation=data_plan["validation"],
                output_directory=output_directory,
            )
            sft_result = _train_variant(
                stack=stack,
                config=config,
                variant=variant,
                train_records=data_plan["train"].records,
                validation_records=data_plan["validation"].records,
                data_manifest=data_manifest,
                output_directory=output_directory,
            )
            sft_results[variant] = sft_result
            adapter_directory = (
                output_directory / "adapters" / variant
            )
            if variant == "reasoning":
                deferred_variant = variant
                break
            adapted_model, adapted_processor = _load_adapter(
                stack=stack,
                config=config,
                adapter_directory=adapter_directory,
            )
            post_result = _evaluate_gate(
                stack=stack,
                config=config,
                model=adapted_model,
                processor=adapted_processor,
                records=data_plan["gate"].records,
                variant=variant,
                phase="post_sft",
                output_directory=output_directory,
            )
            post_sft[variant] = post_result
            sft_results[variant]["adapter_reloaded_for_gate"] = True
            _log_gate_metrics(
                wandb_run=wandb_run,
                writer=writer,
                variant=variant,
                phase="post_sft",
                metrics=post_result["metrics"],
                step=1,
            )
            selections[variant] = _select_starting_point(
                config=config,
                variant=variant,
                zero_shot=zero_shot[variant],
                post_sft=post_result,
                adapter_path=str(
                    adapter_directory.relative_to(output_directory)
                ),
            )
            del adapted_model
            del adapted_processor
            gc.collect()
            stack.torch.cuda.empty_cache()

        if deferred_variant is not None:
            selections[deferred_variant] = {
                "kind": "pending_fresh_process_evaluation",
                "gate_phase": "post_sft_fresh_process",
                "gate_passed": False,
                "decision": "finalization_required",
            }
            handoff = {
                "schema_version": 1,
                "status": "ready",
                "mode": "planned_fresh_process_adapter_finalization",
                "deferred_variant": deferred_variant,
                "source_configuration_sha256": input_sha256,
                "data_manifest_sha256": _file_sha256(
                    output_directory / "data_manifest.json"
                ),
                "gate_member_ids_sha256": data_plan[
                    "gate"
                ].member_ids_sha256,
                "training_complete": True,
                "fresh_training_required": False,
                "failure_artifact_required": False,
            }
            _write_json(output_directory / "handoff.json", handoff)
            finalization_lock = create_finalization_lock(
                config=config,
                input_sha256=input_sha256,
                source_run_directory=output_directory,
                lock_path=output_directory / "finalization-lock.json",
            )
            result = {
                "schema_version": 1,
                "status": "awaiting_fresh_process_finalization",
                "zero_shot": zero_shot,
                "sft": sft_results,
                "post_sft": post_sft,
                "selected_starting_points": selections,
                "ready_for_teacher_guided_grpo": False,
                "required_next_step": {
                    "mode": "planned_fresh_process_adapter_finalization",
                    "source_run_directory": str(
                        output_directory.resolve()
                    ),
                    "lock_path": str(
                        (output_directory / "finalization-lock.json").resolve()
                    ),
                    "lock_sha256": _file_sha256(
                        output_directory / "finalization-lock.json"
                    ),
                    "locked_artifacts": len(
                        cast(
                            Mapping[str, Any],
                            finalization_lock["artifacts"],
                        )
                    ),
                },
                "scope": {
                    "teacher_guided_grpo_performed": False,
                    "custom_training_loop": False,
                    "phase_one_completed_without_failure": True,
                },
                "telemetry": {
                    "wandb": {
                        "mode": "online",
                        "entity": wandb_run.entity,
                        "project": wandb_run.project,
                        "project_access": project_access,
                        "run_id": wandb_run.id,
                        "url": wandb_run.url,
                        "checkpoint_upload": False,
                    },
                    "tensorboard": {
                        "enabled": True,
                        "path": "telemetry/tensorboard",
                    },
                    "wall_seconds": time.perf_counter() - started,
                },
            }
            _write_json(output_directory / "result.json", result)
            manifest = _build_run_manifest(
                config=config,
                input_sha256=input_sha256,
                data_plan=data_plan,
                output_directory=output_directory,
                result=result,
                stack=stack,
            )
            _write_json(output_directory / "manifest.json", manifest)
            wandb_run.log(
                {
                    "gate/phase_one_completed": 1.0,
                    "gate/finalization_required": 1.0,
                    "gate/ready_for_teacher_guided_grpo": 0.0,
                }
            )
            writer.flush()
            writer.close()
            writer = None
            wandb_run.finish(exit_code=0)
            wandb_run = None
            return result

        ready_for_teacher_guided_grpo = all(
            bool(selection["gate_passed"])
            for selection in selections.values()
        )
        result = {
            "schema_version": 1,
            "status": "completed",
            "zero_shot": zero_shot,
            "sft": sft_results,
            "post_sft": post_sft,
            "selected_starting_points": selections,
            "ready_for_teacher_guided_grpo": (
                ready_for_teacher_guided_grpo
            ),
            "scope": {
                "teacher_guided_grpo_performed": False,
                "custom_training_loop": False,
            },
            "telemetry": {
                "wandb": {
                    "mode": "online",
                    "entity": wandb_run.entity,
                    "project": wandb_run.project,
                    "project_access": project_access,
                    "run_id": wandb_run.id,
                    "url": wandb_run.url,
                    "checkpoint_upload": False,
                },
                "tensorboard": {
                    "enabled": True,
                    "path": "telemetry/tensorboard",
                },
                "wall_seconds": time.perf_counter() - started,
            },
        }
        _write_json(output_directory / "result.json", result)
        manifest = _build_run_manifest(
            config=config,
            input_sha256=input_sha256,
            data_plan=data_plan,
            output_directory=output_directory,
            result=result,
            stack=stack,
        )
        _write_json(output_directory / "manifest.json", manifest)
        wandb_run.log(
            {
                "gate/ready_for_teacher_guided_grpo": float(
                    ready_for_teacher_guided_grpo
                ),
                "gate/completed": 1.0,
            }
        )
        writer.flush()
        writer.close()
        writer = None
        wandb_run.finish(exit_code=0)
        wandb_run = None
        return result
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": _safe_error(error),
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json(output_directory / "failure.json", failure)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            writer = None
        if wandb_run is not None:
            try:
                wandb_run.log({"gate/completed": 0.0})
                wandb_run.finish(exit_code=1)
            except Exception:
                pass
            wandb_run = None
        raise
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except Exception:
                pass


def create_finalization_lock(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    source_run_directory: Path,
    lock_path: Path,
) -> dict[str, Any]:
    """Freeze a successful phase-one run for fresh-process finalization."""
    return _create_adapter_evaluation_lock(
        config=config,
        input_sha256=input_sha256,
        source_run_directory=source_run_directory,
        lock_path=lock_path,
        mode="adapter_reload_finalization_lock",
        required_files=_finalization_required_files(
            source_run_directory.resolve()
        ),
    )


def create_recovery_lock(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    source_run_directory: Path,
    lock_path: Path,
) -> dict[str, Any]:
    """Freeze every source artifact required by a fresh-process reload."""
    return _create_adapter_evaluation_lock(
        config=config,
        input_sha256=input_sha256,
        source_run_directory=source_run_directory,
        lock_path=lock_path,
        mode="adapter_reload_recovery_lock",
        required_files=_recovery_required_files(
            source_run_directory.resolve()
        ),
    )


def _create_adapter_evaluation_lock(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    source_run_directory: Path,
    lock_path: Path,
    mode: str,
    required_files: Sequence[Path],
) -> dict[str, Any]:
    if lock_path.exists():
        raise GatePreflightError(
            "adapter evaluation lock path must not already exist"
        )
    source = source_run_directory.resolve()
    data_manifest = _read_json_object(
        source / "data_manifest.json", "source data manifest"
    )
    source_configuration_sha256 = data_manifest.get("configuration_sha256")
    if (
        not isinstance(source_configuration_sha256, str)
        or len(source_configuration_sha256) != 64
    ):
        raise GatePreflightError(
            "source data manifest configuration hash is malformed"
        )
    lock = {
        "schema_version": 1,
        "mode": mode,
        "source_run_directory": str(source),
        "source_configuration_sha256": source_configuration_sha256,
        "recovery_configuration_sha256": input_sha256,
        "model": config.model.resolved(),
        "artifacts": {
            str(path.relative_to(source)): _file_sha256(path)
            for path in required_files
        },
    }
    _write_json(lock_path, lock)
    return lock


def run_adapter_reload_recovery(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    source_run_directory: Path,
    recovery_lock_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Recover an OOM-interrupted Reasoning reload without any training."""
    return _run_fresh_process_adapter_evaluation(
        config=config,
        input_sha256=input_sha256,
        source_run_directory=source_run_directory,
        adapter_evaluation_lock_path=recovery_lock_path,
        output_directory=output_directory,
        source_mode="failure_recovery",
    )


def run_fresh_process_finalization(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    source_run_directory: Path,
    finalization_lock_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Complete planned Reasoning adapter evaluation without retraining."""
    return _run_fresh_process_adapter_evaluation(
        config=config,
        input_sha256=input_sha256,
        source_run_directory=source_run_directory,
        adapter_evaluation_lock_path=finalization_lock_path,
        output_directory=output_directory,
        source_mode="planned_finalization",
    )


def _run_fresh_process_adapter_evaluation(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    source_run_directory: Path,
    adapter_evaluation_lock_path: Path,
    output_directory: Path,
    source_mode: str,
) -> dict[str, Any]:
    preflight_real_gate(config)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise GatePreflightError(
            "output directory must be absent or empty for recovery"
        )
    data_plan = prepare_data_plan(config)
    source_evidence = _validate_recovery_source(
        config=config,
        input_sha256=input_sha256,
        source_run_directory=source_run_directory,
        adapter_evaluation_lock_path=adapter_evaluation_lock_path,
        data_plan=data_plan,
        expected_source_mode=source_mode,
    )
    run_mode = (
        "fresh_process_adapter_finalization"
        if source_mode == "planned_finalization"
        else "fresh_process_adapter_reload_recovery"
    )
    resolved_filename = (
        "resolved_finalization.json"
        if source_mode == "planned_finalization"
        else "resolved_recovery.json"
    )
    stack = _load_runtime_stack()
    _preflight_cuda(stack)
    output_directory.mkdir(parents=True, exist_ok=True)
    telemetry_directory = output_directory / "telemetry"
    telemetry_directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_directory / resolved_filename,
        {
            "schema_version": 1,
            "mode": run_mode,
            "configuration": config.resolved(),
            "configuration_sha256": input_sha256,
            "source": source_evidence["lineage"],
            "fresh_training_performed": False,
        },
    )
    _configure_telemetry(config, telemetry_directory)
    wandb_run: Any | None = None
    writer: Any | None = None
    started = time.perf_counter()
    try:
        wandb_run, project_access = _start_wandb_run(
            stack=stack,
            config=config,
            telemetry_directory=telemetry_directory,
            run_name=f"{config.experiment_name}-reload-recovery",
            run_config={
                **config.resolved(),
                "recovery": source_evidence["lineage"],
                "fresh_training_performed": False,
            },
            tags=[
                "issue-8",
                (
                    "adapter-reload-finalization"
                    if source_mode == "planned_finalization"
                    else "adapter-reload-recovery"
                ),
                "no-training",
                "private",
            ],
        )
        writer = stack.summary_writer_class(
            str(telemetry_directory / "tensorboard" / "recovery")
        )
        zero_shot = source_evidence["zero_shot"]
        source_post_direct = source_evidence["post_sft_direct_action"]
        for variant in VARIANTS:
            _log_gate_metrics(
                wandb_run=wandb_run,
                writer=writer,
                variant=variant,
                phase="zero_shot",
                metrics=zero_shot[variant]["metrics"],
                step=0,
            )
        if source_post_direct is not None:
            _log_gate_metrics(
                wandb_run=wandb_run,
                writer=writer,
                variant="direct_action",
                phase="post_sft",
                metrics=source_post_direct["metrics"],
                step=1,
            )

        adapter_directory = (
            source_run_directory.resolve() / "adapters" / "reasoning"
        )
        model, processor = _load_adapter(
            stack=stack,
            config=config,
            adapter_directory=adapter_directory,
        )
        reasoning_recovery = _evaluate_gate(
            stack=stack,
            config=config,
            model=model,
            processor=processor,
            records=data_plan["gate"].records,
            variant="reasoning",
            phase="post_sft_recovery",
            output_directory=output_directory,
        )
        _log_gate_metrics(
            wandb_run=wandb_run,
            writer=writer,
            variant="reasoning",
            phase="post_sft",
            metrics=reasoning_recovery["metrics"],
            step=1,
        )
        del model
        del processor
        gc.collect()
        stack.torch.cuda.empty_cache()

        direct_selection = (
            {
                "kind": "unchanged_base_model",
                "model_id": config.model.id,
                "revision": config.model.revision,
                "gate_phase": "zero_shot",
                "gate_passed": True,
                "decision": "selected_passing_candidate",
                "selected_metrics": zero_shot["direct_action"]["metrics"],
            }
            if source_post_direct is None
            else _select_starting_point(
                config=config,
                variant="direct_action",
                zero_shot=zero_shot["direct_action"],
                post_sft=source_post_direct,
                adapter_path=str(
                    source_run_directory.resolve()
                    / "adapters"
                    / "direct_action"
                ),
            )
        )
        reasoning_selection = _select_starting_point(
            config=config,
            variant="reasoning",
            zero_shot=zero_shot["reasoning"],
            post_sft=reasoning_recovery,
            adapter_path=str(adapter_directory),
        )
        selections = {
            "direct_action": direct_selection,
            "reasoning": reasoning_selection,
        }
        ready = all(
            bool(selection["gate_passed"])
            for selection in selections.values()
        )
        result = {
            "schema_version": 1,
            "status": "completed",
            "mode": run_mode,
            "source_run": source_evidence["lineage"],
            "fresh_training_performed": False,
            "zero_shot": zero_shot,
            "sft": {
                variant: (
                    {
                        "status": "consumed_from_source_run",
                        "fresh_training_performed": False,
                        "adapter": source_evidence["adapters"][variant],
                        "examples": {
                            "train": config.sft.train.count,
                            "validation": config.sft.validation.count,
                            "total": config.sft.total_examples,
                        },
                    }
                    if variant in source_evidence["adapters"]
                    else {
                        "status": "not_performed",
                        "reason": "zero-shot gate passed",
                        "examples": 0,
                        "fresh_training_performed": False,
                    }
                )
                for variant in VARIANTS
            },
            "post_sft": {
                "direct_action": source_post_direct,
                "reasoning": reasoning_recovery,
            },
            "selected_starting_points": selections,
            "ready_for_teacher_guided_grpo": ready,
            "scope": {
                "teacher_guided_grpo_performed": False,
                "custom_training_loop": False,
                "fresh_training_performed": False,
            },
            "telemetry": {
                "wandb": {
                    "mode": "online",
                    "entity": wandb_run.entity,
                    "project": wandb_run.project,
                    "project_access": project_access,
                    "run_id": wandb_run.id,
                    "url": wandb_run.url,
                    "checkpoint_upload": False,
                },
                "tensorboard": {
                    "enabled": True,
                    "path": "telemetry/tensorboard",
                },
                "wall_seconds": time.perf_counter() - started,
            },
        }
        _write_json(output_directory / "result.json", result)
        manifest = _build_recovery_manifest(
            config=config,
            input_sha256=input_sha256,
            source_evidence=source_evidence,
            adapter_evaluation_lock_path=adapter_evaluation_lock_path,
            output_directory=output_directory,
            result=result,
            stack=stack,
            mode=run_mode,
            resolved_filename=resolved_filename,
        )
        _write_json(output_directory / "manifest.json", manifest)
        wandb_run.log(
            {
                "gate/ready_for_teacher_guided_grpo": float(ready),
                "gate/recovery_completed": 1.0,
                "gate/fresh_training_performed": 0.0,
            }
        )
        writer.flush()
        writer.close()
        writer = None
        wandb_run.finish(exit_code=0)
        wandb_run = None
        return result
    except Exception as error:
        _write_json(
            output_directory / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "mode": run_mode,
                "fresh_training_performed": False,
                "source_run": source_evidence["lineage"],
                "error_type": type(error).__name__,
                "error": _safe_error(error),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            writer = None
        if wandb_run is not None:
            try:
                wandb_run.log({"gate/recovery_completed": 0.0})
                wandb_run.finish(exit_code=1)
            except Exception:
                pass
            wandb_run = None
        raise
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except Exception:
                pass


def _validate_recovery_source(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    source_run_directory: Path,
    adapter_evaluation_lock_path: Path,
    data_plan: Mapping[str, Any],
    expected_source_mode: str,
) -> dict[str, Any]:
    source = source_run_directory.resolve()
    lock = _read_json_object(
        adapter_evaluation_lock_path, "adapter evaluation lock"
    )
    _exact_keys(
        lock,
        {
            "schema_version",
            "mode",
            "source_run_directory",
            "source_configuration_sha256",
            "recovery_configuration_sha256",
            "model",
            "artifacts",
        },
        "adapter evaluation lock",
    )
    _require(
        lock["schema_version"],
        1,
        "adapter evaluation lock.schema_version",
    )
    expected_lock_mode = (
        "adapter_reload_finalization_lock"
        if expected_source_mode == "planned_finalization"
        else "adapter_reload_recovery_lock"
    )
    _require(
        lock["mode"],
        expected_lock_mode,
        "adapter evaluation lock.mode",
    )
    _require(
        lock["source_run_directory"],
        str(source),
        "adapter evaluation lock.source_run_directory",
    )
    _require(
        lock["recovery_configuration_sha256"],
        input_sha256,
        "adapter evaluation lock.recovery_configuration_sha256",
    )
    if lock["model"] != config.model.resolved():
        raise GatePreflightError(
            "adapter evaluation lock model does not match configuration"
        )
    artifacts = _object(
        lock["artifacts"], "adapter evaluation lock.artifacts"
    )
    expected_paths = {
        str(path.relative_to(source))
        for path in (
            _finalization_required_files(source)
            if expected_source_mode == "planned_finalization"
            else _recovery_required_files(source)
        )
    }
    if set(artifacts) != expected_paths:
        raise GatePreflightError(
            "adapter evaluation lock artifact members differ from the source "
            "run"
        )
    _verify_recovery_artifact_hashes(source=source, artifacts=artifacts)

    resolved = _read_json_object(
        source / "resolved_config.json", "source resolved configuration"
    )
    expected_configuration = config.resolved()
    expected_gate = dict(
        _object(expected_configuration["gate"], "configuration.gate")
    )
    expected_gate.pop("recovery_evaluation_batch_size")
    expected_configuration = {
        **expected_configuration,
        "gate": expected_gate,
    }
    for field, value in expected_configuration.items():
        if resolved.get(field) != value:
            raise GatePreflightError(
                f"source resolved configuration differs at {field}"
            )
    data_manifest = _read_json_object(
        source / "data_manifest.json", "source data manifest"
    )
    source_configuration_sha256 = data_manifest.get(
        "configuration_sha256"
    )
    if (
        source_configuration_sha256
        != lock["source_configuration_sha256"]
    ):
        raise GatePreflightError(
            "source data manifest configuration hash differs"
        )
    selections = _object(
        data_manifest.get("selections"), "source data manifest.selections"
    )
    for source_name, plan_name in (
        ("gate", "gate"),
        ("sft_train", "train"),
        ("sft_validation", "validation"),
    ):
        evidence = _object(
            selections.get(source_name),
            f"source data manifest.selections.{source_name}",
        )
        if (
            evidence.get("member_ids_sha256")
            != data_plan[plan_name].member_ids_sha256
        ):
            raise GatePreflightError(
                f"source {source_name} member-ID hash differs"
            )
    if data_manifest.get("split_isolation") != data_plan["split_isolation"]:
        raise GatePreflightError("source split-isolation evidence differs")
    data_manifest_sha = _file_sha256(source / "data_manifest.json")

    adapters: dict[PolicyVariant, dict[str, Any]] = {}
    for variant in VARIANTS:
        adapter = source / "adapters" / variant
        if not adapter.is_dir():
            continue
        _assert_adapter_only_checkpoint(adapter)
        initialization = _read_json_object(
            adapter / "llm2048_initialization.json",
            f"{variant} initialization",
        )
        if initialization.get("variant") != variant:
            raise GatePreflightError(
                f"{variant} adapter initialization variant differs"
            )
        if initialization.get("base_model") != config.model.resolved():
            raise GatePreflightError(
                f"{variant} adapter base model differs"
            )
        initialization_data = _object(
            initialization.get("data_manifest"),
            f"{variant} initialization.data_manifest",
        )
        if initialization_data.get("sha256") != data_manifest_sha:
            raise GatePreflightError(
                f"{variant} adapter data-manifest hash differs"
            )
        adapters[variant] = {
            "path": str(adapter),
            "members": _directory_members(adapter, source),
            "initialization_sha256": _file_sha256(
                adapter / "llm2048_initialization.json"
            ),
            "adapter_weights_sha256": _file_sha256(
                next(adapter.glob("adapter_model.*"))
            ),
        }

    gate_records = data_plan["gate"].records
    zero_shot = {
        variant: _source_gate_result(
            path=source / "gate" / f"zero_shot-{variant}.jsonl",
            records=gate_records,
            variant=variant,
            phase="zero_shot",
            config=config,
            source=source,
        )
        for variant in VARIANTS
    }
    for variant in VARIANTS:
        zero_passed = bool(zero_shot[variant]["metrics"]["passed"])
        if zero_passed and variant in adapters:
            raise GatePreflightError(
                f"{variant} zero-shot passed but source contains an adapter"
            )
        if not zero_passed and variant not in adapters:
            raise GatePreflightError(
                f"{variant} zero-shot failed but source adapter is missing"
            )
    post_direct_path = (
        source / "gate" / "post_sft-direct_action.jsonl"
    )
    post_direct = (
        _source_gate_result(
            path=post_direct_path,
            records=gate_records,
            variant="direct_action",
            phase="post_sft",
            config=config,
            source=source,
        )
        if post_direct_path.is_file()
        else None
    )
    if (post_direct is None) != bool(
        zero_shot["direct_action"]["metrics"]["passed"]
    ):
        raise GatePreflightError(
            "Direct-action post-SFT evidence does not match its zero-shot "
            "decision"
        )
    source_outcome: dict[str, Any]
    if expected_source_mode == "planned_finalization":
        handoff = _read_json_object(
            source / "handoff.json", "source finalization handoff"
        )
        if (
            handoff.get("status") != "ready"
            or handoff.get("mode")
            != "planned_fresh_process_adapter_finalization"
            or handoff.get("deferred_variant") != "reasoning"
            or handoff.get("source_configuration_sha256")
            != source_configuration_sha256
            or handoff.get("data_manifest_sha256")
            != data_manifest_sha
            or handoff.get("training_complete") is not True
            or handoff.get("fresh_training_required") is not False
            or handoff.get("failure_artifact_required") is not False
            or handoff.get("gate_member_ids_sha256")
            != data_plan["gate"].member_ids_sha256
        ):
            raise GatePreflightError(
                "planned finalization handoff is incomplete or inconsistent"
            )
        _assert_planned_finalization_has_no_failure(source)
        source_outcome = {
            "source_mode": "planned_finalization",
            "handoff_sha256": _file_sha256(source / "handoff.json"),
            "handoff": handoff,
        }
    else:
        failure = _read_json_object(source / "failure.json", "source failure")
        if (
            failure.get("status") != "failed"
            or failure.get("error_type") != "OutOfMemoryError"
            or "CUDA out of memory" not in str(failure.get("error"))
        ):
            raise GatePreflightError(
                "recovery source must preserve the Reasoning reload CUDA OOM"
            )
        source_outcome = {
            "source_mode": "failure_recovery",
            "failure_sha256": _file_sha256(source / "failure.json"),
            "failure": failure,
        }
    return {
        "lineage": {
            "source_run_directory": str(source),
            "adapter_evaluation_lock_path": str(
                adapter_evaluation_lock_path.resolve()
            ),
            "adapter_evaluation_lock_sha256": _file_sha256(
                adapter_evaluation_lock_path
            ),
            "source_configuration_sha256": source_configuration_sha256,
            "recovery_configuration_sha256": input_sha256,
            "data_manifest_sha256": data_manifest_sha,
            **source_outcome,
        },
        "adapters": adapters,
        "zero_shot": zero_shot,
        "post_sft_direct_action": post_direct,
        "locked_artifacts": dict(artifacts),
    }


def _assert_planned_finalization_has_no_failure(source: Path) -> None:
    if (source / "failure.json").exists():
        raise GatePreflightError(
            "planned finalization source must not depend on failure.json"
        )


def _verify_recovery_artifact_hashes(
    *,
    source: Path,
    artifacts: Mapping[str, Any],
) -> None:
    source = source.resolve()
    for relative_path, expected_sha in artifacts.items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
        ):
            raise GatePreflightError(
                "recovery lock contains an invalid artifact path"
            )
        path = (source / relative_path).resolve()
        try:
            path.relative_to(source)
        except ValueError as error:
            raise GatePreflightError(
                "recovery lock artifact path escapes the source run"
            ) from error
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise GatePreflightError(
                f"recovery lock hash is malformed for {relative_path}"
            )
        if not path.is_file():
            raise GatePreflightError(
                f"recovery source is missing locked artifact {relative_path}"
            )
        if _file_sha256(path) != expected_sha:
            raise GatePreflightError(
                f"recovery source hash mismatch for {relative_path}"
            )


def _finalization_required_files(source: Path) -> list[Path]:
    return _adapter_evaluation_required_files(
        source=source,
        source_outcome_path=source / "handoff.json",
    )


def _recovery_required_files(source: Path) -> list[Path]:
    return _adapter_evaluation_required_files(
        source=source,
        source_outcome_path=source / "failure.json",
    )


def _adapter_evaluation_required_files(
    *, source: Path, source_outcome_path: Path
) -> list[Path]:
    fixed = [
        source / "resolved_config.json",
        source / "data_manifest.json",
        source_outcome_path,
        source / "data" / "gate.jsonl",
        source / "gate" / "zero_shot-direct_action.jsonl",
        source / "gate" / "zero_shot-reasoning.jsonl",
    ]
    post_direct = source / "gate" / "post_sft-direct_action.jsonl"
    if post_direct.is_file():
        fixed.append(post_direct)
    for variant in VARIANTS:
        adapter = source / "adapters" / variant
        if adapter.is_dir():
            fixed.extend(
                [
                    source / "data" / variant / "train.jsonl",
                    source / "data" / variant / "validation.jsonl",
                ]
            )
            fixed.extend(
                path for path in sorted(adapter.rglob("*")) if path.is_file()
            )
    missing = [path for path in fixed if not path.is_file()]
    if missing:
        raise GatePreflightError(
            "recovery source is missing required artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    return sorted(set(fixed))


def _source_gate_result(
    *,
    path: Path,
    records: Sequence[Mapping[str, Any]],
    variant: PolicyVariant,
    phase: str,
    config: ZeroShotSftConfig,
    source: Path,
) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    expected_ids = [row["record_id"] for row in records]
    actual_ids = [
        event.get("record_id") if isinstance(event, dict) else None
        for event in events
    ]
    if actual_ids != expected_ids:
        raise GatePreflightError(
            f"source {phase}/{variant} events differ from fixed gate members"
        )
    if any(
        event.get("variant") != variant or event.get("phase") != phase
        for event in events
    ):
        raise GatePreflightError(
            f"source {phase}/{variant} event metadata differs"
        )
    return {
        "status": "consumed_from_source_run",
        "variant": variant,
        "phase": phase,
        "metrics": gate_metrics(
            events,
            parse_rate_minimum=config.gate.parse_rate_minimum,
            illegal_action_rate_maximum=(
                config.gate.illegal_action_rate_maximum
            ),
        ),
        "events_path": str(path.relative_to(source)),
        "events_sha256": _file_sha256(path),
        "source_run_directory": str(source),
    }


def _build_recovery_manifest(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    source_evidence: Mapping[str, Any],
    adapter_evaluation_lock_path: Path,
    output_directory: Path,
    result: Mapping[str, Any],
    stack: RuntimeStack,
    mode: str,
    resolved_filename: str,
) -> dict[str, Any]:
    properties = stack.torch.cuda.get_device_properties(0)
    artifacts = [
        {
            "path": relative,
            "sha256": _file_sha256(output_directory / relative),
        }
        for relative in (
            resolved_filename,
            "gate/post_sft_recovery-reasoning.jsonl",
            "result.json",
        )
    ]
    return {
        "schema_version": 1,
        "mode": mode,
        "configuration_sha256": input_sha256,
        "result_status": result["status"],
        "fresh_training_performed": False,
        "source": source_evidence["lineage"],
        "source_locked_artifacts": source_evidence["locked_artifacts"],
        "adapter_evaluation_lock": {
            "path": str(adapter_evaluation_lock_path.resolve()),
            "sha256": _file_sha256(adapter_evaluation_lock_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": stack.versions,
            "cuda": stack.torch.version.cuda,
            "cudnn": stack.torch.backends.cudnn.version(),
            "gpu": {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": list(
                    stack.torch.cuda.get_device_capability(0)
                ),
            },
            "api_signatures": stack.api_signatures,
        },
        "artifacts": artifacts,
        "scope": {
            "trainer_invoked": False,
            "teacher_guided_grpo_performed": False,
            "custom_training_loop": False,
            "checkpoint_upload": False,
        },
    }


def _load_runtime_stack() -> RuntimeStack:
    try:
        unsloth = import_module("unsloth")
        torch = import_module("torch")
        wandb = import_module("wandb")
        datasets = import_module("datasets")
        transformers = import_module("transformers")
        huggingface_hub = import_module("huggingface_hub")
        peft = import_module("peft")
        tensorboard = import_module("torch.utils.tensorboard")
    except ImportError as error:
        raise GatePreflightError(
            "the real gate requires torch, Unsloth Core, Transformers Trainer, "
            "datasets, PEFT, W&B, and TensorBoard"
        ) from error
    required = {
        "FastVisionModel.from_pretrained": (
            unsloth.FastVisionModel.from_pretrained,
            {
                "model_name",
                "revision",
                "tokenizer_name",
                "max_seq_length",
                "dtype",
                "load_in_4bit",
                "fast_inference",
                "text_only",
                "use_exact_model_name",
                "full_finetuning",
            },
        ),
        "FastVisionModel.get_peft_model": (
            unsloth.FastVisionModel.get_peft_model,
            {
                "model",
                "finetune_vision_layers",
                "finetune_language_layers",
                "finetune_attention_modules",
                "finetune_mlp_modules",
                "r",
                "lora_alpha",
            },
        ),
        "PeftModel.from_pretrained": (
            peft.PeftModel.from_pretrained,
            {"model", "model_id", "is_trainable"},
        ),
        "transformers.Trainer": (
            transformers.Trainer,
            {
                "model",
                "args",
                "data_collator",
                "train_dataset",
                "eval_dataset",
                "processing_class",
            },
        ),
    }
    for name, (function, keywords) in required.items():
        _require_keywords(function, keywords, name)
    return RuntimeStack(
        torch=torch,
        wandb=wandb,
        dataset_class=datasets.Dataset,
        trainer_class=transformers.Trainer,
        training_arguments_class=transformers.TrainingArguments,
        fast_vision_model=unsloth.FastVisionModel,
        snapshot_download=huggingface_hub.snapshot_download,
        peft_model_class=peft.PeftModel,
        summary_writer_class=tensorboard.SummaryWriter,
        versions={
            package: _installed_version(package)
            for package in (
                "torch",
                "unsloth",
                "unsloth_zoo",
                "transformers",
                "datasets",
                "huggingface_hub",
                "peft",
                "bitsandbytes",
                "wandb",
                "tensorboard",
            )
        },
        api_signatures={
            name: _callable_signature(function)
            for name, (function, _) in required.items()
        },
    )


def _preflight_cuda(stack: RuntimeStack) -> None:
    if not stack.torch.cuda.is_available():
        raise GatePreflightError("a CUDA GPU is required for the BF16 gate")
    if not stack.torch.cuda.is_bf16_supported():
        raise GatePreflightError("the active CUDA GPU does not support BF16")


def _load_base_model(
    stack: RuntimeStack, config: ZeroShotSftConfig
) -> tuple[Any, Any]:
    snapshot_path = _exact_snapshot_path(stack, config)
    arguments = {
        "model_name": config.model.id,
        "revision": config.model.revision,
        "tokenizer_name": snapshot_path,
        "max_seq_length": config.model.max_sequence_length,
        "dtype": stack.torch.bfloat16,
        "load_in_4bit": False,
        "fast_inference": False,
        "text_only": True,
        "use_exact_model_name": True,
        "full_finetuning": False,
    }
    _require_keywords(
        stack.fast_vision_model.from_pretrained,
        set(arguments),
        "FastVisionModel.from_pretrained",
    )
    model, processor = stack.fast_vision_model.from_pretrained(**arguments)
    _ensure_generation_architecture(model)
    if getattr(processor, "pad_token_id", None) is None:
        processor.pad_token = processor.eos_token
    return model, processor


def _load_trainable_model(
    stack: RuntimeStack, config: ZeroShotSftConfig
) -> tuple[Any, Any, dict[str, Any]]:
    model, processor = _load_base_model(stack, config)
    lora = config.sft.lora
    arguments = {
        "model": model,
        "finetune_vision_layers": lora.finetune_vision_layers,
        "finetune_language_layers": lora.finetune_language_layers,
        "finetune_attention_modules": lora.finetune_attention_modules,
        "finetune_mlp_modules": lora.finetune_mlp_modules,
        "r": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "bias": "none",
        "random_state": config.seed,
        "use_rslora": False,
        "loftq_config": None,
        "use_gradient_checkpointing": lora.gradient_checkpointing,
    }
    _require_keywords(
        stack.fast_vision_model.get_peft_model,
        set(arguments),
        "FastVisionModel.get_peft_model",
    )
    model = stack.fast_vision_model.get_peft_model(**arguments)
    evidence = _trainable_parameter_evidence(model)
    if evidence["vision_trainable_parameter_names"]:
        raise RuntimeError("masked SFT unexpectedly exposed trainable vision layers")
    if evidence["lora_trainable_parameters"] <= 0:
        raise RuntimeError("masked SFT did not expose trainable LoRA parameters")
    return model, processor, evidence


def _load_adapter(
    *,
    stack: RuntimeStack,
    config: ZeroShotSftConfig,
    adapter_directory: Path,
) -> tuple[Any, Any]:
    model, processor = _load_base_model(stack, config)
    arguments = {
        "model": model,
        "model_id": str(adapter_directory),
        "is_trainable": False,
    }
    _require_keywords(
        stack.peft_model_class.from_pretrained,
        set(arguments),
        "PeftModel.from_pretrained",
    )
    model = stack.peft_model_class.from_pretrained(**arguments)
    return model, processor


def _evaluate_gate(
    *,
    stack: RuntimeStack,
    config: ZeroShotSftConfig,
    model: Any,
    processor: Any,
    records: Sequence[Mapping[str, Any]],
    variant: PolicyVariant,
    phase: str,
    output_directory: Path,
) -> dict[str, Any]:
    stack.fast_vision_model.for_inference(model)
    processor.padding_side = "left"
    max_new_tokens = config.gate.max_new_tokens(variant)
    event_path = output_directory / "gate" / f"{phase}-{variant}.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    stack.torch.cuda.empty_cache()
    stack.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    total_generated_tokens = 0
    evaluation_batch_size = config.gate.batch_size(phase)
    for batch_start in range(0, len(records), evaluation_batch_size):
        batch = records[
            batch_start : batch_start + evaluation_batch_size
        ]
        prompts = [
            _render_prompt(processor, variant, row["board"]) for row in batch
        ]
        encoded = processor(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = encoded.to("cuda")
        prompt_width = int(encoded["input_ids"].shape[-1])
        with stack.torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.pad_token_id,
                eos_token_id=processor.eos_token_id,
            )
        completion_rows = generated[:, prompt_width:].detach().cpu().tolist()
        for row, completion_ids in zip(batch, completion_rows):
            trimmed_ids, truncated = _trim_completion(
                completion_ids,
                eos_token_id=processor.eos_token_id,
                pad_token_id=processor.pad_token_id,
                max_new_tokens=max_new_tokens,
            )
            response = processor.decode(
                trimmed_ids,
                skip_special_tokens=True,
            )
            response = _complete_native_policy_response(variant, response)
            contract = enforce_policy_response(
                variant=variant,
                response=response,
                truncated=truncated,
                board_change_actions=change_making_actions(row["board"]),
            )
            event = {
                "record_id": row["record_id"],
                "source_split": row["split"],
                "source_stratum": row["stratum"],
                "lineage": row["lineage"],
                "variant": variant,
                "phase": phase,
                "board": row["board"],
                "response": response,
                "response_length_tokens": len(trimmed_ids),
                "truncated": truncated,
                "action": contract.action,
                "parsed": contract.parsed,
                "illegal_action": (
                    contract.policy_failure_reason == "illegal_action"
                ),
                "valid_action": contract.valid_action,
                "policy_failure": contract.policy_failure,
                "policy_failure_reason": contract.policy_failure_reason,
            }
            _append_json_line(event_path, event)
            events.append(event)
            total_generated_tokens += len(trimmed_ids)
        del generated
        del encoded
    wall_seconds = time.perf_counter() - started
    metrics = gate_metrics(
        events,
        parse_rate_minimum=config.gate.parse_rate_minimum,
        illegal_action_rate_maximum=config.gate.illegal_action_rate_maximum,
    )
    return {
        "status": "completed",
        "variant": variant,
        "phase": phase,
        "metrics": metrics,
        "generation": {
            "generated_tokens": total_generated_tokens,
            "wall_seconds": wall_seconds,
            "tokens_per_second": (
                total_generated_tokens / wall_seconds
                if wall_seconds > 0
                else None
            ),
            "batch_size": evaluation_batch_size,
            "max_new_tokens": max_new_tokens,
            "peak_allocated_vram_bytes": (
                stack.torch.cuda.max_memory_allocated()
            ),
            "peak_reserved_vram_bytes": (
                stack.torch.cuda.max_memory_reserved()
            ),
        },
        "events_path": str(event_path.relative_to(output_directory)),
    }


def _train_variant(
    *,
    stack: RuntimeStack,
    config: ZeroShotSftConfig,
    variant: PolicyVariant,
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    data_manifest: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    model, processor, trainable_evidence = _load_trainable_model(stack, config)
    train_rows = [
        build_sft_record(
            row=row,
            variant=variant,
            reasoning_trace_placeholder=(
                config.sft.policy_reasoning_trace_placeholder
            ),
        )
        for row in train_records
    ]
    validation_rows = [
        build_sft_record(
            row=row,
            variant=variant,
            reasoning_trace_placeholder=(
                config.sft.policy_reasoning_trace_placeholder
            ),
        )
        for row in validation_records
    ]
    encoded_train = [
        encode_masked_sft_record(
            row,
            processor,
            max_sequence_length=config.model.max_sequence_length,
        )
        for row in train_rows
    ]
    encoded_validation = [
        encode_masked_sft_record(
            row,
            processor,
            max_sequence_length=config.model.max_sequence_length,
        )
        for row in validation_rows
    ]
    train_dataset = stack.dataset_class.from_list(encoded_train)
    validation_dataset = stack.dataset_class.from_list(encoded_validation)
    trainer_directory = output_directory / "trainers" / variant
    tensorboard_directory = (
        output_directory / "telemetry" / "tensorboard" / f"sft-{variant}"
    )
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tensorboard_directory)
    training = config.sft.training
    arguments = {
        "output_dir": str(trainer_directory),
        "num_train_epochs": training.num_train_epochs,
        "learning_rate": training.learning_rate,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "warmup_ratio": training.warmup_ratio,
        "weight_decay": training.weight_decay,
        "optim": training.optimizer,
        "logging_strategy": "steps",
        "logging_steps": training.logging_steps,
        "logging_first_step": True,
        "eval_strategy": "epoch",
        "save_strategy": "no",
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "report_to": ["wandb", "tensorboard"],
        "run_name": f"{config.experiment_name}-sft-{variant}",
        "remove_unused_columns": False,
        "push_to_hub": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "disable_tqdm": True,
    }
    _require_keywords(
        stack.training_arguments_class,
        set(arguments),
        "transformers.TrainingArguments",
    )
    training_args = stack.training_arguments_class(**arguments)
    trainer = stack.trainer_class(
        model=model,
        args=training_args,
        data_collator=CausalSftCollator(processor),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
    )
    stack.torch.cuda.empty_cache()
    stack.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = trainer.train()
    train_wall_seconds = time.perf_counter() - started
    optimizer_steps = int(trainer.state.global_step)
    if optimizer_steps <= 0:
        raise RuntimeError("Transformers Trainer completed no SFT optimizer step")
    eval_metrics = trainer.evaluate()
    adapter_directory = output_directory / "adapters" / variant
    model.save_pretrained(adapter_directory)
    processor.save_pretrained(adapter_directory)
    data_manifest_path = output_directory / "data_manifest.json"
    _write_json(
        adapter_directory / "llm2048_initialization.json",
        {
            "schema_version": 1,
            "variant": variant,
            "base_model": config.model.resolved(),
            "data_manifest": {
                "path": str(data_manifest_path.relative_to(output_directory)),
                "sha256": _file_sha256(data_manifest_path),
            },
            "masking": {
                "prompt": "masked",
                "policy_reasoning_trace_body": (
                    "masked" if variant == "reasoning" else "not_applicable"
                ),
                "response_envelope": "supervised",
                "final_action": "supervised",
            },
            "trainer": "transformers.Trainer",
            "teacher_guided_grpo_performed": False,
        },
    )
    _assert_adapter_only_checkpoint(adapter_directory)
    result = {
        "status": "completed",
        "variant": variant,
        "examples": {
            "train": len(train_records),
            "validation": len(validation_records),
            "total": len(train_records) + len(validation_records),
        },
        "optimizer_steps": optimizer_steps,
        "train_wall_seconds": train_wall_seconds,
        "peak_allocated_vram_bytes": (
            stack.torch.cuda.max_memory_allocated()
        ),
        "peak_reserved_vram_bytes": (
            stack.torch.cuda.max_memory_reserved()
        ),
        "train_metrics": _json_mapping(dict(output.metrics)),
        "eval_metrics": _json_mapping(eval_metrics),
        "trainable_parameter_evidence": trainable_evidence,
        "adapter_path": str(adapter_directory.relative_to(output_directory)),
        "adapter_reloaded_for_gate": False,
        "data_manifest_member_ids": {
            "train": data_manifest["selections"]["sft_train"][
                "member_ids_sha256"
            ],
            "validation": data_manifest["selections"]["sft_validation"][
                "member_ids_sha256"
            ],
        },
    }
    del trainer
    del model
    del processor
    del train_dataset
    del validation_dataset
    gc.collect()
    stack.torch.cuda.empty_cache()
    return result


def _write_data_plan(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    data_plan: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "configuration_sha256": input_sha256,
        "teacher_policy_corpus": {
            "manifest_path": str(data_plan["manifest_path"]),
            "manifest_sha256": data_plan["manifest_sha256"],
            "teacher_policy": data_plan["manifest"]["teacher_policy"],
            "split_assignment": data_plan["manifest"]["corpus"][
                "split_assignment"
            ],
            "split_unit": data_plan["manifest"]["corpus"]["split_unit"],
        },
        "selections": {
            "gate": data_plan["gate"].evidence(),
            "sft_train": data_plan["train"].evidence(),
            "sft_validation": data_plan["validation"].evidence(),
        },
        "split_isolation": data_plan["split_isolation"],
        "sft_contract": {
            "total_examples": config.sft.total_examples,
            "teacher_supplies": ["final_action"],
            "teacher_does_not_supply": ["policy_reasoning_trace"],
            "policy_reasoning_trace_body_masked": True,
            "response_envelope_supervised": True,
            "final_action_supervised": True,
        },
    }
    _write_json(output_directory / "data_manifest.json", manifest)
    gate_path = output_directory / "data" / "gate.jsonl"
    for row in data_plan["gate"].records:
        _append_json_line(gate_path, row)
    return manifest


def _write_variant_sft_data(
    *,
    config: ZeroShotSftConfig,
    variant: PolicyVariant,
    train: CorpusSelection,
    validation: CorpusSelection,
    output_directory: Path,
) -> None:
    directory = output_directory / "data" / variant
    for name, selection in (("train", train), ("validation", validation)):
        path = directory / f"{name}.jsonl"
        for row in selection.records:
            _append_json_line(
                path,
                build_sft_record(
                    row=row,
                    variant=variant,
                    reasoning_trace_placeholder=(
                        config.sft.policy_reasoning_trace_placeholder
                    ),
                ),
            )


def _start_wandb_run(
    *,
    stack: RuntimeStack,
    config: ZeroShotSftConfig,
    telemetry_directory: Path,
    run_name: str | None = None,
    run_config: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
) -> tuple[Any, str]:
    stack.wandb.login(
        key=os.environ["WANDB_API_KEY"],
        relogin=True,
        verify=True,
    )
    api = stack.wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    if not isinstance(entity, str) or not entity:
        raise RuntimeError("W&B did not resolve an authenticated default entity")
    project_access = _verify_private_wandb_project(
        api=api,
        entity=entity,
        project=config.telemetry.wandb_project,
    )
    run = stack.wandb.init(
        entity=entity,
        project=config.telemetry.wandb_project,
        name=run_name or config.experiment_name,
        config=(
            dict(run_config)
            if run_config is not None
            else config.resolved()
        ),
        dir=str(telemetry_directory / "wandb"),
        mode="online",
        tags=(
            list(tags)
            if tags is not None
            else ["issue-8", "zero-shot-gate", "masked-sft", "private"]
        ),
        save_code=False,
        reinit=True,
    )
    if run is None:
        raise RuntimeError("W&B did not create an online run")
    run_mode = getattr(getattr(run, "settings", None), "mode", None)
    if run_mode != "online":
        run.finish(exit_code=1)
        raise RuntimeError("W&B did not honor online mode")
    if run.entity != entity or run.project != config.telemetry.wandb_project:
        run.finish(exit_code=1)
        raise RuntimeError("W&B run target differs from verified private project")
    return run, project_access


def _configure_telemetry(
    config: ZeroShotSftConfig, telemetry_directory: Path
) -> None:
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_PROJECT"] = config.telemetry.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_DIR"] = str(telemetry_directory / "wandb")


def _log_gate_metrics(
    *,
    wandb_run: Any,
    writer: Any,
    variant: PolicyVariant,
    phase: str,
    metrics: Mapping[str, Any],
    step: int,
) -> None:
    scalar_names = (
        "parse_rate",
        "truncation_rate",
        "illegal_action_rate",
        "valid_action_rate",
        "policy_failure_rate",
    )
    values = {
        f"gate/{phase}/{variant}/{name}": float(metrics[name])
        for name in scalar_names
    }
    values[f"gate/{phase}/{variant}/response_length_mean"] = float(
        metrics["response_length_tokens"]["mean"]
    )
    values[f"gate/{phase}/{variant}/passed"] = float(metrics["passed"])
    wandb_run.log(values)
    for name, value in values.items():
        writer.add_scalar(name, value, step)
    writer.flush()


def _build_run_manifest(
    *,
    config: ZeroShotSftConfig,
    input_sha256: str,
    data_plan: Mapping[str, Any],
    output_directory: Path,
    result: Mapping[str, Any],
    stack: RuntimeStack,
) -> dict[str, Any]:
    properties = stack.torch.cuda.get_device_properties(0)
    artifact_entries: list[dict[str, Any]] = []
    for relative_path in (
        "resolved_config.json",
        "data_manifest.json",
        "result.json",
    ):
        path = output_directory / relative_path
        artifact_entries.append(
            {
                "path": relative_path,
                "sha256": _file_sha256(path),
            }
        )
    for relative_path in ("handoff.json", "finalization-lock.json"):
        path = output_directory / relative_path
        if path.is_file():
            artifact_entries.append(
                {
                    "path": relative_path,
                    "sha256": _file_sha256(path),
                }
            )
    for path in sorted((output_directory / "gate").glob("*.jsonl")):
        artifact_entries.append(
            {
                "path": str(path.relative_to(output_directory)),
                "sha256": _file_sha256(path),
            }
        )
    for variant in VARIANTS:
        adapter = output_directory / "adapters" / variant
        if adapter.exists():
            artifact_entries.append(
                {
                    "path": str(adapter.relative_to(output_directory)),
                    "members": _directory_members(adapter, output_directory),
                    "wandb_uploaded": False,
                }
            )
    return {
        "schema_version": 1,
        "configuration_sha256": input_sha256,
        "result_status": result["status"],
        "corpus_manifest_sha256": data_plan["manifest_sha256"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": stack.versions,
            "cuda": stack.torch.version.cuda,
            "cudnn": stack.torch.backends.cudnn.version(),
            "gpu": {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": list(
                    stack.torch.cuda.get_device_capability(0)
                ),
            },
            "api_signatures": stack.api_signatures,
        },
        "metric_definitions": {
            "parse_rate": "parsed Policy Responses divided by all 500 boards",
            "truncation_rate": (
                "responses reaching max_new_tokens without EOS divided by "
                "all 500 boards"
            ),
            "illegal_action_rate": (
                "parsed actions that do not change the board divided by "
                "all 500 boards"
            ),
            "valid_action_rate": (
                "contract-valid, board-changing actions divided by all boards"
            ),
            "policy_failure_rate": (
                "invalid-format, truncated, or illegal responses divided by "
                "all boards"
            ),
            "response_length_tokens": (
                "completion tokens before EOS, excluding prompt and padding"
            ),
        },
        "artifacts": artifact_entries,
        "scope": {
            "trainer": "transformers.Trainer",
            "teacher_guided_grpo_performed": False,
            "custom_training_loop": False,
            "checkpoint_upload": False,
        },
    }


def _render_prompt(
    processor: Any,
    variant: PolicyVariant,
    board: Sequence[Sequence[int]],
) -> str:
    rendered = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_policy_prompt(variant, board)}
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=variant == "reasoning",
    )
    if not isinstance(rendered, str):
        raise RuntimeError("Qwen processor did not render a text policy prompt")
    return rendered


def _complete_native_policy_response(
    variant: PolicyVariant, generated_completion: str
) -> str:
    """Restore the native Qwen thinking prefix that lives in the prompt."""
    if variant == "reasoning":
        return "<think>\n" + generated_completion
    return generated_completion


def _trim_completion(
    completion_ids: Sequence[int],
    *,
    eos_token_id: int | Sequence[int],
    pad_token_id: int,
    max_new_tokens: int,
) -> tuple[list[int], bool]:
    eos_ids = (
        {eos_token_id}
        if isinstance(eos_token_id, int)
        else set(eos_token_id)
    )
    trimmed: list[int] = []
    terminated = False
    for token_id in completion_ids:
        if token_id in eos_ids:
            terminated = True
            break
        if token_id == pad_token_id:
            continue
        trimmed.append(token_id)
    return trimmed, len(completion_ids) >= max_new_tokens and not terminated


def _load_corpus_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw_bytes = path.read_bytes()
        manifest = json.loads(raw_bytes)
    except OSError as error:
        raise GatePreflightError(
            f"cannot read Teacher Policy Corpus manifest: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise GatePreflightError(
            "Teacher Policy Corpus manifest is not valid JSON"
        ) from error
    if not isinstance(manifest, dict):
        raise GatePreflightError(
            "Teacher Policy Corpus manifest must be an object"
        )
    try:
        if manifest["schema_version"] != 1:
            raise KeyError
        if manifest["corpus"]["split_unit"] != (
            "trajectory_and_symmetry_orbit"
        ):
            raise KeyError
        if manifest["teacher_policy"]["search_depth"] != 2:
            raise KeyError
        if manifest["teacher_policy"]["source"] != (
            "retained_100m_teacher_policy"
        ):
            raise KeyError
    except (KeyError, TypeError) as error:
        raise GatePreflightError(
            "Teacher Policy Corpus manifest does not match the retained "
            "Depth-2 split-isolated contract"
        ) from error
    return manifest, sha256(raw_bytes).hexdigest()


def _corpus_artifact(
    manifest: Mapping[str, Any], split: str
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise GatePreflightError("corpus manifest artifacts must be a list")
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("name") == split
    ]
    if len(matches) != 1:
        raise GatePreflightError(
            f"corpus manifest must contain exactly one {split} artifact"
        )
    artifact = matches[0]
    if (
        not isinstance(artifact.get("path"), str)
        or not isinstance(artifact.get("sha256"), str)
        or not isinstance(artifact.get("records"), int)
    ):
        raise GatePreflightError(f"corpus {split} artifact is malformed")
    return artifact


def _validate_corpus_row(
    row: Any,
    *,
    expected_split: str,
    source_path: Path,
    line_number: int,
) -> None:
    try:
        if not isinstance(row, dict):
            raise KeyError
        if row["split"] != expected_split:
            raise KeyError
        if row["stratum"] not in STRATA:
            raise KeyError
        if not isinstance(row["record_id"], str) or not row["record_id"]:
            raise KeyError
        board = row["board"]
        if (
            not isinstance(board, list)
            or len(board) != 4
            or any(not isinstance(line, list) or len(line) != 4 for line in board)
        ):
            raise KeyError
        teacher_action = str(row["teacher_action"]).upper()
        if teacher_action not in {"LEFT", "RIGHT", "UP", "DOWN"}:
            raise KeyError
        lineage = row["lineage"]
        if (
            not isinstance(lineage, dict)
            or not isinstance(lineage["trajectory_id"], str)
            or not isinstance(lineage["orbit_id"], str)
        ):
            raise KeyError
    except (KeyError, TypeError) as error:
        raise GatePreflightError(
            f"{source_path}:{line_number} violates the Teacher Policy Corpus "
            "row contract"
        ) from error


def _load_model(raw: Mapping[str, Any]) -> ModelConfig:
    _exact_keys(
        raw,
        {
            "id",
            "revision",
            "precision",
            "load_in_4bit",
            "fast_inference",
            "text_only",
            "max_sequence_length",
        },
        "model",
    )
    _require(raw["id"], OFFICIAL_QWEN35_4B, "model.id")
    revision = _nonempty_string(raw["revision"], "model.revision")
    if MODEL_REVISION_PATTERN.fullmatch(revision) is None:
        raise GateConfigurationError(
            "model.revision must pin a 40-character commit SHA"
        )
    _require(raw["precision"], "bf16", "model.precision")
    _require(raw["load_in_4bit"], False, "model.load_in_4bit")
    _require(raw["fast_inference"], False, "model.fast_inference")
    _require(raw["text_only"], True, "model.text_only")
    _require(raw["max_sequence_length"], 256, "model.max_sequence_length")
    return ModelConfig(
        id=OFFICIAL_QWEN35_4B,
        revision=revision,
        precision="bf16",
        load_in_4bit=False,
        fast_inference=False,
        text_only=True,
        max_sequence_length=256,
    )


def _load_corpus(raw: Mapping[str, Any]) -> CorpusConfig:
    _exact_keys(
        raw,
        {"manifest_path", "manifest_environment_override"},
        "corpus",
    )
    manifest_path = _nonempty_string(
        raw["manifest_path"], "corpus.manifest_path"
    )
    if Path(manifest_path).is_absolute():
        raise GateConfigurationError(
            "corpus.manifest_path must be repository-relative"
        )
    override = _nonempty_string(
        raw["manifest_environment_override"],
        "corpus.manifest_environment_override",
    )
    _require(
        override,
        "LLM2048_TEACHER_CORPUS_MANIFEST",
        "corpus.manifest_environment_override",
    )
    return CorpusConfig(manifest_path, override)


def _load_gate(raw: Mapping[str, Any]) -> GateConfig:
    _exact_keys(
        raw,
        {
            "split",
            "count",
            "strata",
            "parse_rate_minimum",
            "illegal_action_rate_maximum",
            "evaluation_batch_size",
            "adapter_evaluation_batch_size",
            "recovery_evaluation_batch_size",
            "direct_action_max_new_tokens",
            "reasoning_max_new_tokens",
        },
        "gate",
    )
    selection = _load_selection(
        raw,
        field="gate",
        expected_split="test",
        expected_count=500,
        expected_strata={"natural": 250, "hard": 150, "late": 100},
        extra_keys={
            "parse_rate_minimum",
            "illegal_action_rate_maximum",
            "evaluation_batch_size",
            "adapter_evaluation_batch_size",
            "recovery_evaluation_batch_size",
            "direct_action_max_new_tokens",
            "reasoning_max_new_tokens",
        },
    )
    parse_minimum = _finite(raw["parse_rate_minimum"], "gate.parse_rate_minimum")
    illegal_maximum = _finite(
        raw["illegal_action_rate_maximum"],
        "gate.illegal_action_rate_maximum",
    )
    _require(parse_minimum, 0.95, "gate.parse_rate_minimum")
    _require(illegal_maximum, 0.02, "gate.illegal_action_rate_maximum")
    batch_size = _integer(
        raw["evaluation_batch_size"],
        "gate.evaluation_batch_size",
        minimum=1,
    )
    if batch_size > 16:
        raise GateConfigurationError(
            "gate.evaluation_batch_size must not exceed 16 on the target GPU"
        )
    adapter_batch_size = _integer(
        raw["adapter_evaluation_batch_size"],
        "gate.adapter_evaluation_batch_size",
        minimum=1,
    )
    _require(
        adapter_batch_size,
        4,
        "gate.adapter_evaluation_batch_size",
    )
    recovery_batch_size = _integer(
        raw["recovery_evaluation_batch_size"],
        "gate.recovery_evaluation_batch_size",
        minimum=1,
    )
    _require(
        recovery_batch_size,
        2,
        "gate.recovery_evaluation_batch_size",
    )
    direct_tokens = _integer(
        raw["direct_action_max_new_tokens"],
        "gate.direct_action_max_new_tokens",
        minimum=8,
    )
    reasoning_tokens = _integer(
        raw["reasoning_max_new_tokens"],
        "gate.reasoning_max_new_tokens",
        minimum=8,
    )
    if reasoning_tokens > 96:
        raise GateConfigurationError(
            "gate.reasoning_max_new_tokens must not exceed 96"
        )
    return GateConfig(
        selection=selection,
        parse_rate_minimum=parse_minimum,
        illegal_action_rate_maximum=illegal_maximum,
        evaluation_batch_size=batch_size,
        adapter_evaluation_batch_size=adapter_batch_size,
        recovery_evaluation_batch_size=recovery_batch_size,
        direct_action_max_new_tokens=direct_tokens,
        reasoning_max_new_tokens=reasoning_tokens,
    )


def _load_sft(raw: Mapping[str, Any]) -> SftConfig:
    _exact_keys(
        raw,
        {
            "trainer",
            "total_examples",
            "train",
            "validation",
            "policy_reasoning_trace_placeholder",
            "mask_policy_reasoning_trace_body",
            "lora",
            "training",
        },
        "sft",
    )
    _require(raw["trainer"], "transformers.Trainer", "sft.trainer")
    total = _integer(raw["total_examples"], "sft.total_examples", minimum=1)
    _require(total, 1000, "sft.total_examples")
    train = _load_selection(
        _object(raw["train"], "sft.train"),
        field="sft.train",
        expected_split="train",
        expected_count=900,
        expected_strata={"natural": 450, "hard": 270, "late": 180},
    )
    validation = _load_selection(
        _object(raw["validation"], "sft.validation"),
        field="sft.validation",
        expected_split="validation",
        expected_count=100,
        expected_strata={"natural": 50, "hard": 30, "late": 20},
    )
    placeholder = _nonempty_string(
        raw["policy_reasoning_trace_placeholder"],
        "sft.policy_reasoning_trace_placeholder",
    )
    if "<" in placeholder or ">" in placeholder:
        raise GateConfigurationError(
            "Policy Reasoning Trace placeholder must not contain tags"
        )
    _require(
        raw["mask_policy_reasoning_trace_body"],
        True,
        "sft.mask_policy_reasoning_trace_body",
    )
    return SftConfig(
        trainer="transformers.Trainer",
        total_examples=total,
        train=train,
        validation=validation,
        policy_reasoning_trace_placeholder=placeholder,
        mask_policy_reasoning_trace_body=True,
        lora=_load_lora(_object(raw["lora"], "sft.lora")),
        training=_load_training(
            _object(raw["training"], "sft.training")
        ),
    )


def _load_selection(
    raw: Mapping[str, Any],
    *,
    field: str,
    expected_split: str,
    expected_count: int,
    expected_strata: Mapping[str, int],
    extra_keys: set[str] | None = None,
) -> SelectionConfig:
    expected_keys = {"split", "count", "strata"} | (extra_keys or set())
    _exact_keys(raw, expected_keys, field)
    _require(raw["split"], expected_split, f"{field}.split")
    count = _integer(raw["count"], f"{field}.count", minimum=1)
    _require(count, expected_count, f"{field}.count")
    strata_raw = _object(raw["strata"], f"{field}.strata")
    _exact_keys(strata_raw, set(STRATA), f"{field}.strata")
    strata = {
        stratum: _integer(
            strata_raw[stratum], f"{field}.strata.{stratum}", minimum=0
        )
        for stratum in STRATA
    }
    if strata != dict(expected_strata):
        raise GateConfigurationError(
            f"{field}.strata must be {dict(expected_strata)!r}"
        )
    if sum(strata.values()) != count:
        raise GateConfigurationError(
            f"{field}.strata must sum to {field}.count"
        )
    return SelectionConfig(expected_split, count, strata)


def _load_lora(raw: Mapping[str, Any]) -> LoraConfig:
    expected = {
        "rank": 64,
        "alpha": 64,
        "dropout": 0.0,
        "finetune_vision_layers": False,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
        "gradient_checkpointing": "unsloth",
    }
    _exact_keys(raw, set(expected), "sft.lora")
    for name, value in expected.items():
        _require(raw[name], value, f"sft.lora.{name}")
    return LoraConfig(
        rank=64,
        alpha=64,
        dropout=0.0,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        gradient_checkpointing="unsloth",
    )


def _load_training(raw: Mapping[str, Any]) -> TrainingConfig:
    expected = {
        "num_train_epochs": 1.0,
        "learning_rate": 0.0002,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "optimizer": "adamw_8bit",
        "logging_steps": 10,
    }
    _exact_keys(raw, set(expected), "sft.training")
    for name, value in expected.items():
        _require(raw[name], value, f"sft.training.{name}")
    return TrainingConfig(
        num_train_epochs=1.0,
        learning_rate=0.0002,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        weight_decay=0.0,
        optimizer="adamw_8bit",
        logging_steps=10,
    )


def _load_telemetry(raw: Mapping[str, Any]) -> TelemetryConfig:
    expected = {
        "wandb_mode": "online",
        "wandb_entity": "auto",
        "wandb_project_visibility": "private",
        "tensorboard": True,
        "upload_model_checkpoints": False,
    }
    _exact_keys(raw, set(expected) | {"wandb_project"}, "telemetry")
    for name, value in expected.items():
        _require(raw[name], value, f"telemetry.{name}")
    project = _nonempty_string(
        raw["wandb_project"], "telemetry.wandb_project"
    )
    return TelemetryConfig(
        wandb_mode="online",
        wandb_entity="auto",
        wandb_project=project,
        wandb_project_visibility="private",
        tensorboard=True,
        upload_model_checkpoints=False,
    )


def _token_ids(processor: Any, text: str) -> list[int]:
    encoded = processor(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _nearest_rank(sorted_values: Sequence[int], quantile: float) -> int:
    rank = max(1, math.ceil(quantile * len(sorted_values)))
    return sorted_values[rank - 1]


def _exact_snapshot_path(
    stack: RuntimeStack, config: ZeroShotSftConfig
) -> str:
    try:
        path = stack.snapshot_download(
            repo_id=config.model.id,
            revision=config.model.revision,
            local_files_only=True,
        )
    except Exception as error:
        raise GatePreflightError(
            "the exact Qwen model revision is not complete in the local "
            "Hugging Face cache"
        ) from error
    return str(path)


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _callable_signature(function: Any) -> str:
    try:
        return str(inspect.signature(function))
    except (TypeError, ValueError):
        return "unavailable"


def _json_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_scalar(value) for key, value in values.items()}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _safe_error(error: Exception) -> str:
    message = str(error)
    credential = os.environ.get("WANDB_API_KEY")
    if credential:
        message = message.replace(credential, "[REDACTED]")
    return message


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_json_line(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GatePreflightError(f"{field} is not readable JSON") from error
    if not isinstance(value, dict):
        raise GatePreflightError(f"{field} must be a JSON object")
    return value


def _directory_members(
    directory: Path, output_directory: Path
) -> list[dict[str, str]]:
    return [
        {
            "path": str(path.relative_to(output_directory)),
            "sha256": _file_sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GateConfigurationError(f"{field} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], field: str
) -> None:
    if set(value) != expected:
        raise GateConfigurationError(
            f"{field} must contain exactly {', '.join(sorted(expected))}"
        )


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateConfigurationError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise GateConfigurationError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _finite(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise GateConfigurationError(f"{field} must be a finite number")
    return float(value)


def _require(value: Any, expected: Any, field: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise GateConfigurationError(f"{field} must be {expected!r}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args(argv)
    config, input_sha256 = ZeroShotSftConfig.load(parsed.config)
    if parsed.dry_run:
        plan = prepare_data_plan(config)
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "configuration_sha256": input_sha256,
                    "corpus_manifest_path": str(plan["manifest_path"]),
                    "corpus_manifest_sha256": plan["manifest_sha256"],
                    "gate": plan["gate"].evidence(),
                    "sft_train": plan["train"].evidence(),
                    "sft_validation": plan["validation"].evidence(),
                    "split_isolation": plan["split_isolation"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if parsed.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is used")
    result = run_zero_shot_sft_gate(
        config=config,
        input_sha256=input_sha256,
        output_directory=parsed.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
