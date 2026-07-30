"""Paired 250-step Teacher-guided GRPO block orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any, Callable, cast, Mapping, Sequence

from llm2048.grpo_smoke import (
    MODEL_REVISION_PATTERN,
    OFFICIAL_QWEN35_4B,
    RuntimeStack,
    _ensure_generation_architecture,
    _load_runtime_stack,
    _preflight_cuda,
    _require_keywords,
    _trainable_parameter_evidence,
    _verify_private_wandb_project,
    _completion_text,
)
from llm2048.policy_contracts import (
    Action,
    PolicyVariant,
    build_policy_prompt,
)
from llm2048.teacher_guided_rewards import (
    TeacherGuidedCompletion,
    teacher_guided_reward_callback,
)
from llm2048.zero_shot_sft import (
    GatePreflightError,
    SelectionConfig,
    select_corpus_records,
)


class TeacherGuidedBlockConfigurationError(ValueError):
    """Raised when the paired block configuration is unsafe or ambiguous."""


class TeacherGuidedBlockPreflightError(RuntimeError):
    """Raised before artifacts, telemetry, or GPU state are created."""


@dataclass(frozen=True)
class EvidenceReference:
    path: str
    environment_override: str
    sha256: str

    def resolve(
        self, *, configuration_path: Path, environ: Mapping[str, str]
    ) -> Path:
        override = environ.get(self.environment_override)
        if override:
            return Path(override).expanduser().resolve()
        return (configuration_path.resolve().parent / self.path).resolve()

    def resolved(self) -> dict[str, str]:
        return {
            "path": self.path,
            "environment_override": self.environment_override,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TeacherGuidedBlockConfig:
    schema_version: int
    experiment_name: str
    seed: int
    configuration_path: Path
    input_sha256: str
    model: Mapping[str, Any]
    gate_requirement: str
    zero_shot_gate: EvidenceReference
    grpo_feasibility: EvidenceReference
    corpus: Mapping[str, Any]
    dynamic_board_pool: Mapping[str, Any]
    validation: Mapping[str, Any]
    lora: Mapping[str, Any]
    grpo: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    comparison: Mapping[str, Any]
    telemetry: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> "TeacherGuidedBlockConfig":
        try:
            raw_bytes = path.read_bytes()
            raw = json.loads(raw_bytes)
        except OSError as error:
            raise TeacherGuidedBlockConfigurationError(
                f"cannot read Teacher-guided block configuration: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise TeacherGuidedBlockConfigurationError(
                "Teacher-guided block configuration is not valid JSON"
            ) from error
        root = _object(raw, "Teacher-guided block configuration")
        _exact_keys(
            root,
            {
                "schema_version",
                "experiment_name",
                "seed",
                "model",
                "upstream",
                "corpus",
                "dynamic_board_pool",
                "validation",
                "lora",
                "grpo",
                "checkpoint",
                "comparison",
                "telemetry",
            },
            "Teacher-guided block configuration",
        )
        _require(root["schema_version"], 1, "schema_version")
        experiment_name = _nonempty(root["experiment_name"], "experiment_name")
        seed = _integer(root["seed"], "seed", minimum=0)
        model = _validate_model(_object(root["model"], "model"))
        upstream = _object(root["upstream"], "upstream")
        _exact_keys(
            upstream,
            {
                "gate_requirement",
                "zero_shot_gate",
                "grpo_feasibility",
            },
            "upstream",
        )
        _require(
            upstream["gate_requirement"],
            "selected_candidates_with_failed_gate_risk",
            "upstream.gate_requirement",
        )
        zero_shot_gate = _load_evidence_reference(
            _object(upstream["zero_shot_gate"], "upstream.zero_shot_gate"),
            "upstream.zero_shot_gate",
        )
        feasibility = _load_evidence_reference(
            _object(
                upstream["grpo_feasibility"],
                "upstream.grpo_feasibility",
            ),
            "upstream.grpo_feasibility",
        )
        corpus = _validate_corpus(_object(root["corpus"], "corpus"))
        pool = _validate_pool(
            _object(root["dynamic_board_pool"], "dynamic_board_pool")
        )
        validation = _validate_validation(
            _object(root["validation"], "validation")
        )
        lora = _validate_lora(_object(root["lora"], "lora"))
        grpo = _validate_grpo(_object(root["grpo"], "grpo"))
        checkpoint = _validate_checkpoint(
            _object(root["checkpoint"], "checkpoint")
        )
        comparison = _validate_comparison(
            _object(root["comparison"], "comparison")
        )
        telemetry = _validate_telemetry(
            _object(root["telemetry"], "telemetry")
        )
        if grpo["max_steps"] != checkpoint["final_step"]:
            raise TeacherGuidedBlockConfigurationError(
                "grpo.max_steps must equal checkpoint.final_step"
            )
        return cls(
            schema_version=1,
            experiment_name=experiment_name,
            seed=seed,
            configuration_path=path.resolve(),
            input_sha256=sha256(raw_bytes).hexdigest(),
            model=model,
            gate_requirement="selected_candidates_with_failed_gate_risk",
            zero_shot_gate=zero_shot_gate,
            grpo_feasibility=feasibility,
            corpus=corpus,
            dynamic_board_pool=pool,
            validation=validation,
            lora=lora,
            grpo=grpo,
            checkpoint=checkpoint,
            comparison=comparison,
            telemetry=telemetry,
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "model": dict(self.model),
            "upstream": {
                "gate_requirement": self.gate_requirement,
                "zero_shot_gate": self.zero_shot_gate.resolved(),
                "grpo_feasibility": self.grpo_feasibility.resolved(),
            },
            "corpus": dict(self.corpus),
            "dynamic_board_pool": dict(self.dynamic_board_pool),
            "validation": dict(self.validation),
            "lora": dict(self.lora),
            "grpo": dict(self.grpo),
            "checkpoint": dict(self.checkpoint),
            "comparison": dict(self.comparison),
            "telemetry": dict(self.telemetry),
        }

    def corpus_manifest_path(
        self, environ: Mapping[str, str] | None = None
    ) -> Path:
        environment = os.environ if environ is None else environ
        override = environment.get(
            str(self.corpus["manifest_environment_override"])
        )
        if override:
            return Path(override).expanduser().resolve()
        return (
            self.configuration_path.parent
            / str(self.corpus["manifest_path"])
        ).resolve()


def build_grpo_arguments(
    *,
    config: TeacherGuidedBlockConfig,
    output_directory: Path,
    variant: PolicyVariant,
    target_step: int,
) -> dict[str, Any]:
    """Resolve the maintained TRL contract used by both policy variants."""
    if target_step not in {
        config.checkpoint["controlled_interruption_step"],
        config.checkpoint["final_step"],
    }:
        raise TeacherGuidedBlockConfigurationError(
            "target_step must be the controlled interruption or final step"
        )
    grpo = config.grpo
    return {
        "output_dir": str(output_directory / "trainer"),
        "learning_rate": grpo["learning_rate"],
        "optim": grpo["optimizer"],
        "lr_scheduler_type": grpo["lr_scheduler_type"],
        "warmup_ratio": grpo["warmup_ratio"],
        "weight_decay": grpo["weight_decay"],
        "per_device_train_batch_size": (
            grpo["per_device_train_batch_size"]
        ),
        "gradient_accumulation_steps": (
            grpo["gradient_accumulation_steps"]
        ),
        "num_generations": grpo["group_size"],
        "generation_batch_size": grpo["generation_batch_size"],
        "max_prompt_length": grpo["max_prompt_length"],
        "max_completion_length": grpo["max_completion_length"],
        "max_steps": target_step,
        "temperature": grpo["temperature"],
        "top_p": grpo["top_p"],
        "top_k": grpo["top_k"],
        "beta": grpo["beta"],
        "mask_truncated_completions": (
            grpo["mask_truncated_completions"]
        ),
        "loss_type": grpo["loss_type"],
        "use_vllm": grpo["use_vllm"],
        "bf16": True,
        "fp16": False,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": config.checkpoint["save_steps"],
        "save_total_limit": 2,
        "save_only_model": False,
        "report_to": ["wandb", "tensorboard"],
        "logging_dir": str(
            output_directory / "telemetry" / "tensorboard"
        ),
        "run_name": f"{config.experiment_name}-{variant}",
        "remove_unused_columns": False,
        "push_to_hub": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "disable_tqdm": True,
        "dataloader_num_workers": 0,
    }


def build_reward_function(
    *,
    variant: PolicyVariant,
    max_completion_length: int,
    event_sink: Callable[[dict[str, Any]], None],
    group_size: int = 4,
) -> Callable[..., list[float]]:
    """Build the custom scalar reward seam consumed by maintained GRPOTrainer."""

    def action_quality_reward(
        completions: list[Any],
        completion_ids: list[Any] | None = None,
        **metadata: Any,
    ) -> list[float]:
        required = (
            "board_json",
            "teacher_action_scores_json",
            "teacher_action",
            "teacher_margin_scale",
            "record_id",
        )
        missing = [field for field in required if field not in metadata]
        if missing:
            raise RuntimeError(
                "GRPOTrainer omitted reward metadata: "
                + ", ".join(sorted(missing))
            )
        count = len(completions)
        if count == 0 or count % group_size:
            raise RuntimeError(
                "GRPOTrainer reward batch is not a whole Rollout Group"
            )
        completion_rows = (
            completion_ids
            if completion_ids is not None
            else [None] * count
        )
        if len(completion_rows) != count:
            raise RuntimeError(
                "GRPOTrainer completion IDs differ from completion count"
            )
        scores: list[float] = []
        for group_start in range(0, count, group_size):
            group_end = group_start + group_size
            record_ids = metadata["record_id"][group_start:group_end]
            if len(set(record_ids)) != 1:
                raise RuntimeError(
                    "GRPOTrainer mixed boards inside one Rollout Group"
                )
            board = json.loads(metadata["board_json"][group_start])
            raw_action_scores = json.loads(
                metadata["teacher_action_scores_json"][group_start]
            )
            action_scores = {
                str(action).upper(): value
                for action, value in raw_action_scores.items()
            }
            teacher_action = str(
                metadata["teacher_action"][group_start]
            ).upper()
            tau = float(metadata["teacher_margin_scale"][group_start])
            lengths = [
                len(row) if row is not None else 0
                for row in completion_rows[group_start:group_end]
            ]
            responses = [
                _native_policy_response(variant, _completion_text(completion))
                for completion in completions[group_start:group_end]
            ]
            rewards = teacher_guided_reward_callback(
                board=board,
                completions=[
                    TeacherGuidedCompletion(
                        variant=variant,
                        response=response,
                        truncated=length >= max_completion_length,
                    )
                    for response, length in zip(responses, lengths)
                ],
                group_size=group_size,
                teacher_action_scores=action_scores,  # type: ignore[arg-type]
                teacher_action=teacher_action,  # type: ignore[arg-type]
                tau=tau,
            )
            for response, length, reward in zip(
                responses, lengths, rewards
            ):
                event_sink(
                    {
                        "record_id": record_ids[0],
                        "response": response,
                        "response_length_tokens": length,
                        "reward": reward.total,
                        "reward_components": reward.components.resolved(),
                        "action": reward.contract.action,
                        "valid_action": reward.contract.valid_action,
                        "policy_failure": reward.contract.policy_failure,
                        "policy_failure_reason": (
                            reward.contract.policy_failure_reason
                        ),
                        "teacher_action": teacher_action,
                        "teacher_action_agreement": (
                            reward.contract.action == teacher_action
                        ),
                        "regret": reward.regret,
                    }
                )
                scores.append(reward.total)
        return scores

    return action_quality_reward


def paired_reward_comparison(
    *,
    initial: Sequence[float],
    final: Sequence[float],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Compute the ticket's deterministic paired bootstrap learning test."""
    if len(initial) != len(final) or not initial:
        raise ValueError(
            "paired reward samples must be non-empty and equal length"
        )
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    deltas = [after - before for before, after in zip(initial, final)]
    generator = random.Random(seed)
    estimates = sorted(
        statistics.fmean(
            deltas[generator.randrange(len(deltas))]
            for _ in range(len(deltas))
        )
        for _ in range(bootstrap_samples)
    )
    tail = (1.0 - confidence_level) / 2.0
    lower = _quantile(estimates, tail)
    upper = _quantile(estimates, 1.0 - tail)
    return {
        "boards": len(deltas),
        "initial_mean_reward": statistics.fmean(initial),
        "final_mean_reward": statistics.fmean(final),
        "mean_reward_delta": statistics.fmean(deltas),
        "reward_delta_ci": {
            "confidence_level": confidence_level,
            "lower": lower,
            "upper": upper,
            "method": "paired_nonparametric_bootstrap_percentile",
            "samples": bootstrap_samples,
        },
        "measurable_learning_rule": (
            "paired_reward_delta_lower_ci_gt_zero"
        ),
        "measurable_learning": lower > 0.0,
        "project_success_claimed": False,
    }


def summarize_evaluation_events(
    events: Sequence[Mapping[str, Any]],
    *,
    wall_seconds: float,
    generated_tokens: int,
) -> dict[str, Any]:
    """Aggregate every issue #9 fast-evaluation metric without exclusions."""
    if not events:
        raise ValueError("evaluation events must not be empty")
    rewards = [float(event["reward"]) for event in events]
    component_names = (
        "action_quality",
        "best_action_bonus",
        "illegal_action_penalty",
        "policy_failure_penalty",
    )
    component_means = {
        name: statistics.fmean(
            float(event["reward_components"][name]) for event in events
        )
        for name in component_names
    }
    agreement = sum(
        bool(event["teacher_action_agreement"]) for event in events
    )
    failures = [
        str(event["policy_failure_reason"])
        for event in events
        if event["policy_failure"]
    ]
    action_counts = Counter(
        str(event["action"])
        for event in events
        if event["action"] is not None
    )
    ordered_action_counts = {
        action: action_counts.get(action, 0)
        for action in ("LEFT", "RIGHT", "UP", "DOWN")
    }
    action_total = sum(ordered_action_counts.values())
    action_rates = {
        action: (
            count / action_total if action_total else 0.0
        )
        for action, count in ordered_action_counts.items()
    }
    action_entropy = -sum(
        probability * math.log(probability)
        for probability in action_rates.values()
        if probability > 0.0
    )
    lengths = sorted(
        int(event["response_length_tokens"]) for event in events
    )
    return {
        "boards": len(events),
        "reward": {
            "mean": statistics.fmean(rewards),
            "minimum": min(rewards),
            "maximum": max(rewards),
            "components_mean": component_means,
        },
        "teacher_action_agreement": {
            "count": agreement,
            "rate": agreement / len(events),
        },
        "policy_failures": {
            "count": len(failures),
            "rate": len(failures) / len(events),
            "classes": dict(sorted(Counter(failures).items())),
        },
        "action_distribution": {
            "counts": ordered_action_counts,
            "rates": action_rates,
            "unparsed_or_missing": len(events) - action_total,
        },
        "sampling": {
            "policy_action_entropy_nats": action_entropy,
            "kl": None,
            "kl_reason": (
                "GRPO beta is zero, so no reference policy is loaded"
            ),
        },
        "response_length_tokens": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p95": lengths[max(0, math.ceil(0.95 * len(lengths)) - 1)],
            "maximum": max(lengths),
        },
        "latency": {
            "wall_seconds": wall_seconds,
            "seconds_per_board": wall_seconds / len(events),
            "generated_tokens": generated_tokens,
            "tokens_per_second": (
                generated_tokens / wall_seconds
                if wall_seconds > 0.0
                else None
            ),
        },
    }


def checkpoint_contract(
    checkpoint_directory: Path,
    *,
    expected_step: int,
) -> dict[str, Any]:
    """Verify a maintained Trainer checkpoint is resumable and adapter-only."""
    expected_name = f"checkpoint-{expected_step}"
    if checkpoint_directory.name != expected_name:
        raise RuntimeError(
            f"checkpoint path must end in {expected_name}"
        )
    requirements = {
        "adapter config": [checkpoint_directory / "adapter_config.json"],
        "adapter weights": sorted(
            checkpoint_directory.glob("adapter_model.*")
        ),
        "optimizer state": [checkpoint_directory / "optimizer.pt"],
        "scheduler state": [checkpoint_directory / "scheduler.pt"],
        "RNG state": [checkpoint_directory / "rng_state.pth"],
        "Trainer state": [checkpoint_directory / "trainer_state.json"],
    }
    for label, candidates in requirements.items():
        if not candidates or not any(path.is_file() for path in candidates):
            raise RuntimeError(
                f"checkpoint is missing required {label}"
            )
    prohibited = [
        path.name
        for path in checkpoint_directory.iterdir()
        if path.name == "model.safetensors"
        or path.name.startswith("model-")
        or path.name.startswith("pytorch_model")
    ]
    if prohibited:
        raise RuntimeError(
            "checkpoint contains prohibited full-model weights: "
            + ", ".join(sorted(prohibited))
        )
    members = [
        {
            "path": path.name,
            "sha256": _file_sha256(path),
        }
        for paths in requirements.values()
        for path in paths
        if path.is_file()
    ]
    return {
        "optimizer_step": expected_step,
        "artifact_kind": "lora_adapter",
        "optimizer_state_saved": True,
        "scheduler_state_saved": True,
        "rng_state_saved": True,
        "required_members": members,
    }


def prepare_block_data_plan(
    config: TeacherGuidedBlockConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the immutable Teacher Core and fixed validation snapshot."""
    manifest_path = config.corpus_manifest_path(environ)
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except OSError as error:
        raise TeacherGuidedBlockPreflightError(
            f"cannot read Teacher Policy Corpus manifest: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise TeacherGuidedBlockPreflightError(
            "Teacher Policy Corpus manifest is not valid JSON"
        ) from error
    manifest_sha = sha256(manifest_bytes).hexdigest()
    if manifest_sha != config.corpus["manifest_sha256"]:
        raise TeacherGuidedBlockPreflightError(
            "Teacher Policy Corpus manifest SHA-256 differs from the "
            "registered block configuration"
        )
    manifest_object = _object(manifest, "Teacher Policy Corpus manifest")
    _validate_corpus_manifest(manifest_object)
    tau = float(manifest_object["calibration"]["tau"])
    if not math.isfinite(tau) or tau <= 0.0:
        raise TeacherGuidedBlockPreflightError(
            "Teacher Policy Corpus calibration tau must be positive"
        )
    pool_records, pool_evidence = _load_teacher_core(
        manifest_path=manifest_path,
        manifest=manifest_object,
        expected_count=config.dynamic_board_pool[
            "initial_teacher_core_count"
        ],
        expected_strata=config.dynamic_board_pool["strata"],
    )
    selection = SelectionConfig(
        split=str(config.validation["split"]),
        count=int(config.validation["count"]),
        strata=dict(config.validation["strata"]),
    )
    try:
        validation = select_corpus_records(
            manifest_path=manifest_path,
            manifest=manifest_object,
            selection=selection,
            seed=config.seed,
            purpose="issue-9-fixed-validation",
        )
    except GatePreflightError as error:
        raise TeacherGuidedBlockPreflightError(str(error)) from error
    isolation = _block_split_isolation(pool_records, validation.records)
    identity_payload = {
        "generation": config.dynamic_board_pool["generation"],
        "refresh_during_block": False,
        "teacher_core": pool_evidence,
    }
    pool_identity = sha256(
        json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "manifest": manifest_object,
        "tau": tau,
        "pool_records": pool_records,
        "validation_records": validation.records,
        "evidence": {
            "teacher_policy_corpus": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "teacher_policy": manifest_object["teacher_policy"],
                "split_unit": manifest_object["corpus"]["split_unit"],
            },
            "dynamic_board_pool": {
                **identity_payload,
                "identity_sha256": pool_identity,
            },
            "validation_snapshot": validation.evidence(),
            "split_isolation": isolation,
            "confirmation_seeds_consumed": False,
        },
    }


def inspect_upstream_evidence(
    config: TeacherGuidedBlockConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate immutable #7/#8 evidence without importing the GPU stack."""
    environment = os.environ if environ is None else environ
    gate_path, gate = _load_evidence(
        config.zero_shot_gate,
        configuration_path=config.configuration_path,
        environ=environment,
        label="#8 zero-shot/SFT gate result",
    )
    feasibility_path, feasibility = _load_evidence(
        config.grpo_feasibility,
        configuration_path=config.configuration_path,
        environ=environment,
        label="#7 GRPO feasibility result",
    )
    _require(gate.get("status"), "completed", "#8 result.status")
    selections = _object(
        gate.get("selected_starting_points"),
        "#8 result.selected_starting_points",
    )
    if set(selections) != {"direct_action", "reasoning"}:
        raise TeacherGuidedBlockPreflightError(
            "#8 result must select both Student Policy variants"
        )
    direct = _object(selections["direct_action"], "#8 Direct-action selection")
    reasoning = _object(selections["reasoning"], "#8 Reasoning selection")
    _validate_selected_candidate(config, "direct_action", direct)
    _validate_selected_candidate(config, "reasoning", reasoning)

    _require(feasibility.get("status"), "completed", "#7 result.status")
    decision = _object(feasibility.get("decision"), "#7 result.decision")
    _require(decision.get("go"), True, "#7 result.decision.go")
    _require(
        decision.get("precision"),
        config.model["precision"],
        "#7 result.decision.precision",
    )
    _require(
        decision.get("quantized_fallback_used"),
        False,
        "#7 result.decision.quantized_fallback_used",
    )
    _require(
        decision.get("teacher_guided_group_size"),
        config.grpo["group_size"],
        "#7 result.decision.teacher_guided_group_size",
    )
    ready = gate.get("ready_for_teacher_guided_grpo")
    if not isinstance(ready, bool):
        raise TeacherGuidedBlockPreflightError(
            "#8 result.ready_for_teacher_guided_grpo must be boolean"
        )
    risks = (
        []
        if ready
        else [
            {
                "code": "upstream_gate_not_passed",
                "message": (
                    "#8 selected non-passing candidates; #9 authorizes the "
                    "bounded comparison but this does not make the gate pass"
                ),
            }
        ]
    )
    return {
        "gate_result": {
            "path": str(gate_path),
            "sha256": config.zero_shot_gate.sha256,
            "ready_for_teacher_guided_grpo": ready,
        },
        "feasibility_result": {
            "path": str(feasibility_path),
            "sha256": config.grpo_feasibility.sha256,
            "go": True,
        },
        "selected_starting_points": {
            "direct_action": dict(direct),
            "reasoning": dict(reasoning),
        },
        "selected_group_size": decision["teacher_guided_group_size"],
        "upstream_gate_passed": ready,
        "training_authorized_by_issue_9": True,
        "risks": risks,
    }


def dry_run_block(
    *,
    config: TeacherGuidedBlockConfig,
    variant: PolicyVariant,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    evidence = inspect_upstream_evidence(config, environ=environ)
    data_plan = prepare_block_data_plan(config, environ=environ)
    starting_point = _resolve_starting_point(
        config=config,
        variant=variant,
        evidence=evidence,
        environ=environ,
    )
    return {
        "schema_version": 1,
        "status": "validated",
        "variant": variant,
        "configuration_sha256": config.input_sha256,
        **evidence,
        "data": data_plan["evidence"],
        "resolved_starting_point": starting_point,
        "scope": {
            "teacher_guided_grpo_performed": False,
            "dry_run": True,
            "upstream_gate_treated_as_passed": False,
        },
    }


def run_teacher_guided_block(
    *,
    config: TeacherGuidedBlockConfig,
    variant: PolicyVariant,
    output_directory: Path,
    resume_path: Path | None = None,
    stop_after_step: int | None = None,
) -> dict[str, Any]:
    evidence = inspect_upstream_evidence(config)
    _preflight_real_block(
        config=config,
        output_directory=output_directory,
        resume_path=resume_path,
        stop_after_step=stop_after_step,
    )
    data_plan = prepare_block_data_plan(config)
    target_step = (
        config.checkpoint["final_step"]
        if stop_after_step is None
        else stop_after_step
    )
    starting_point = _resolve_starting_point(
        config=config,
        variant=variant,
        evidence=evidence,
        environ=None,
    )
    stack = _load_runtime_stack()
    _preflight_cuda(stack)
    _validate_maintained_stack(stack, config, output_directory, variant)

    output_directory.mkdir(parents=True, exist_ok=resume_path is not None)
    telemetry_directory = output_directory / "telemetry"
    telemetry_directory.mkdir(parents=True, exist_ok=True)
    _configure_telemetry(config, telemetry_directory)
    run_identity_path = output_directory / "run_identity.json"
    resumed = resume_path is not None
    resume_evidence: dict[str, Any]
    if resumed:
        if resume_path is None:
            raise RuntimeError("internal resume path mismatch")
        run_identity = _validate_resume_contract(
            config=config,
            variant=variant,
            output_directory=output_directory,
            resume_path=resume_path,
            evidence=evidence,
            data_evidence=data_plan["evidence"],
            starting_point=starting_point,
        )
        wandb_id = str(run_identity["wandb_run_id"])
        resume_evidence = {
            "performed": True,
            "source_checkpoint": str(
                resume_path.relative_to(output_directory)
            ),
            "source_resume_manifest_sha256": _file_sha256(
                resume_path / "llm2048_resume_manifest.json"
            ),
            "source_optimizer_step": config.checkpoint[
                "controlled_interruption_step"
            ],
        }
    else:
        wandb_id = str(stack.wandb.util.generate_id())
        run_identity = {}
        resume_evidence = {
            "performed": False,
            "controlled_interruption_requested": (
                target_step
                == config.checkpoint["controlled_interruption_step"]
            ),
        }

    wandb_run: Any | None = None
    model: Any | None = None
    processor: Any | None = None
    trainer: Any | None = None
    started = time.perf_counter()
    try:
        wandb_run, project_access = _start_wandb_run(
            stack=stack,
            config=config,
            variant=variant,
            evidence=evidence,
            data_evidence=data_plan["evidence"],
            telemetry_directory=telemetry_directory,
            run_id=wandb_id,
            resume=resumed,
        )
        if not resumed:
            _write_json(
                output_directory / "resolved_config.json",
                config.resolved(),
            )
            _write_json(
                output_directory / "data_manifest.json",
                data_plan["evidence"],
            )
        model, processor, trainable_evidence = _load_starting_model(
            stack=stack,
            config=config,
            variant=variant,
            starting_point=starting_point,
        )
        validation_records = data_plan["validation_records"]
        initialization_event_path = (
            output_directory / "evaluation" / "initialization.jsonl"
        )
        if resumed:
            initialization = _read_json_object(
                output_directory / "evaluation" / "initialization.json"
            )
            if _file_sha256(initialization_event_path) != run_identity[
                "initialization_events_sha256"
            ]:
                raise TeacherGuidedBlockPreflightError(
                    "initialization evaluation changed before resume"
                )
        else:
            initialization = _evaluate_policy(
                stack=stack,
                config=config,
                model=model,
                processor=processor,
                records=validation_records,
                variant=variant,
                phase="initialization",
                output_directory=output_directory,
                tau=data_plan["tau"],
            )
            _write_json(
                output_directory / "evaluation" / "initialization.json",
                initialization,
            )
            run_identity = {
                "schema_version": 1,
                "configuration_sha256": config.input_sha256,
                "variant": variant,
                "wandb_run_id": wandb_id,
                "upstream": _upstream_identity(evidence),
                "dynamic_board_pool_identity_sha256": data_plan["evidence"][
                    "dynamic_board_pool"
                ]["identity_sha256"],
                "validation_member_ids_sha256": data_plan["evidence"][
                    "validation_snapshot"
                ]["member_ids_sha256"],
                "starting_point": starting_point,
                "initialization_events_sha256": _file_sha256(
                    initialization_event_path
                ),
            }
            _write_json(run_identity_path, run_identity)
            _log_evaluation(
                wandb_run,
                phase="initialization",
                metrics=initialization["metrics"],
            )

        training_rows, prompt_budget_audit = _training_rows(
            processor=processor,
            variant=variant,
            records=data_plan["pool_records"],
            tau=data_plan["tau"],
            max_prompt_length=config.grpo["max_prompt_length"],
        )
        prompt_budget_audit["tokenizer"] = {
            "model_id": config.model["id"],
            "revision": config.model["revision"],
        }
        prompt_audit_path = output_directory / "prompt_budget_audit.json"
        if resumed and prompt_audit_path.is_file():
            previous_prompt_audit = _read_json_object(prompt_audit_path)
            if previous_prompt_audit != prompt_budget_audit:
                raise TeacherGuidedBlockPreflightError(
                    "prompt budget audit changed before resume"
                )
        _write_json(prompt_audit_path, prompt_budget_audit)
        train_dataset = stack.dataset_class.from_list(training_rows)
        reward_events_path = (
            output_directory / "training" / "reward-events.jsonl"
        )
        reward_events_path.parent.mkdir(parents=True, exist_ok=True)
        if not resumed:
            reward_events_path.write_text("", encoding="utf-8")
        reward_events: list[dict[str, Any]] = []

        def sink(event: dict[str, Any]) -> None:
            reward_events.append(event)
            _append_json_line(reward_events_path, event)

        reward_function = build_reward_function(
            variant=variant,
            max_completion_length=config.grpo["max_completion_length"],
            event_sink=sink,
            group_size=config.grpo["group_size"],
        )
        training_arguments = stack.grpo_config_class(
            **build_grpo_arguments(
                config=config,
                output_directory=output_directory,
                variant=variant,
                target_step=target_step,
            )
        )
        _set_training_mode(stack, model)
        trainer = stack.grpo_trainer_class(
            model=model,
            args=training_arguments,
            processing_class=processor,
            reward_funcs=reward_function,
            train_dataset=train_dataset,
        )
        train_started = time.perf_counter()
        train_output = trainer.train(
            resume_from_checkpoint=(
                str(resume_path) if resume_path is not None else None
            )
        )
        train_wall_seconds = time.perf_counter() - train_started
        global_step = int(trainer.state.global_step)
        if global_step != target_step:
            raise RuntimeError(
                f"GRPOTrainer stopped at step {global_step}; "
                f"expected {target_step}"
            )
        checkpoint_directory = (
            output_directory / "trainer" / f"checkpoint-{target_step}"
        )
        checkpoint = checkpoint_contract(
            checkpoint_directory,
            expected_step=target_step,
        )
        resume_manifest = {
            "schema_version": 1,
            "configuration_sha256": config.input_sha256,
            "variant": variant,
            "upstream": _upstream_identity(evidence),
            "starting_point": starting_point,
            "dynamic_board_pool": data_plan["evidence"][
                "dynamic_board_pool"
            ],
            "validation_snapshot": data_plan["evidence"][
                "validation_snapshot"
            ],
            "resolved_config": config.resolved(),
            "prompt_budget_audit": prompt_budget_audit,
            "resume": resume_evidence,
            "checkpoint": checkpoint,
        }
        _write_json(
            checkpoint_directory / "llm2048_resume_manifest.json",
            resume_manifest,
        )
        trainer_metrics = {
            key: _json_scalar(value)
            for key, value in dict(train_output.metrics).items()
        }
        training_summary = _training_summary(
            reward_events=reward_events,
            log_history=trainer.state.log_history,
            trainer_metrics=trainer_metrics,
            wall_seconds=train_wall_seconds,
            global_step=global_step,
        )
        if target_step < config.checkpoint["final_step"]:
            result = {
                "schema_version": 1,
                "status": "paused_for_controlled_resume",
                "variant": variant,
                "optimizer_steps": global_step,
                "upstream": evidence,
                "data": data_plan["evidence"],
                "starting_point": starting_point,
                "initialization": initialization,
                "training": training_summary,
                "prompt_budget_audit": prompt_budget_audit,
                "resume": resume_evidence,
                "checkpoint": {
                    **checkpoint,
                    "path": str(
                        checkpoint_directory.relative_to(output_directory)
                    ),
                    "resume_manifest": str(
                        (
                            checkpoint_directory
                            / "llm2048_resume_manifest.json"
                        ).relative_to(output_directory)
                    ),
                },
                "scope": {
                    "maintained_trl_grpo_trainer": True,
                    "custom_training_loop": False,
                    "project_success_claimed": False,
                },
            }
        else:
            final_evaluation = _evaluate_policy(
                stack=stack,
                config=config,
                model=model,
                processor=processor,
                records=validation_records,
                variant=variant,
                phase="step_250",
                output_directory=output_directory,
                tau=data_plan["tau"],
            )
            _write_json(
                output_directory / "evaluation" / "step_250.json",
                final_evaluation,
            )
            comparison = _compare_evaluation_files(
                initialization_event_path,
                output_directory / "evaluation" / "step_250.jsonl",
                config=config,
            )
            _log_evaluation(
                wandb_run,
                phase="step_250",
                metrics=final_evaluation["metrics"],
            )
            wandb_run.log(
                {
                    "comparison/mean_reward_delta": comparison[
                        "mean_reward_delta"
                    ],
                    "comparison/reward_delta_ci_lower": comparison[
                        "reward_delta_ci"
                    ]["lower"],
                    "comparison/reward_delta_ci_upper": comparison[
                        "reward_delta_ci"
                    ]["upper"],
                    "comparison/measurable_learning": float(
                        comparison["measurable_learning"]
                    ),
                },
                step=global_step,
            )
            result = {
                "schema_version": 1,
                "status": "completed",
                "variant": variant,
                "optimizer_steps": global_step,
                "upstream": evidence,
                "data": data_plan["evidence"],
                "starting_point": starting_point,
                "initialization": initialization,
                "training": training_summary,
                "prompt_budget_audit": prompt_budget_audit,
                "resume": resume_evidence,
                "step_250": final_evaluation,
                "comparison_to_initialization": comparison,
                "checkpoint": {
                    **checkpoint,
                    "path": str(
                        checkpoint_directory.relative_to(output_directory)
                    ),
                    "resume_manifest": str(
                        (
                            checkpoint_directory
                            / "llm2048_resume_manifest.json"
                        ).relative_to(output_directory)
                    ),
                },
                "scope": {
                    "maintained_trl_grpo_trainer": True,
                    "custom_training_loop": False,
                    "dynamic_board_pool_refreshed": False,
                    "confirmation_seeds_consumed": False,
                    "project_success_claimed": False,
                },
            }
        result["telemetry"] = {
            "wandb": {
                "mode": "online",
                "entity": wandb_run.entity,
                "project": wandb_run.project,
                "project_access": project_access,
                "run_id": wandb_run.id,
                "url": wandb_run.url,
                "checkpoints_uploaded": False,
            },
            "tensorboard": {
                "path": "telemetry/tensorboard",
            },
            "wall_seconds": time.perf_counter() - started,
        }
        result["model_evidence"] = trainable_evidence
        _write_json(output_directory / "result.json", result)
        _write_json(
            output_directory / "manifest.json",
            _build_run_manifest(
                stack=stack,
                config=config,
                output_directory=output_directory,
                result=result,
                evidence=evidence,
                data_evidence=data_plan["evidence"],
            ),
        )
        wandb_run.finish(exit_code=0)
        wandb_run = None
        return result
    except Exception as error:
        if output_directory.exists():
            _write_json(
                output_directory / "failure.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "variant": variant,
                    "error_type": type(error).__name__,
                    "error": _safe_error(error),
                    "wall_seconds": time.perf_counter() - started,
                    "upstream_gate_passed": evidence[
                        "upstream_gate_passed"
                    ],
                },
            )
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except Exception:
                pass
            wandb_run = None
        raise
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except Exception:
                pass
        del trainer
        del model
        del processor
        gc.collect()
        try:
            stack.torch.cuda.empty_cache()
        except Exception:
            pass


def _preflight_real_block(
    *,
    config: TeacherGuidedBlockConfig,
    output_directory: Path,
    resume_path: Path | None,
    stop_after_step: int | None,
) -> None:
    if not os.environ.get("WANDB_API_KEY", "").strip():
        raise TeacherGuidedBlockPreflightError(
            "WANDB_API_KEY is required for the private online "
            "Teacher-guided GRPO run"
        )
    if os.environ.get("WANDB_MODE") not in (None, "online"):
        raise TeacherGuidedBlockPreflightError(
            "WANDB_MODE must be 'online' for Teacher-guided GRPO"
        )
    if os.environ.get("WANDB_LOG_MODEL", "false").lower() != "false":
        raise TeacherGuidedBlockPreflightError(
            "WANDB_LOG_MODEL must be 'false' so checkpoints are not uploaded"
        )
    interruption = config.checkpoint["controlled_interruption_step"]
    if stop_after_step is not None and stop_after_step != interruption:
        raise TeacherGuidedBlockPreflightError(
            f"--stop-after-step must be {interruption} for the controlled "
            "resume proof"
        )
    if resume_path is not None and stop_after_step is not None:
        raise TeacherGuidedBlockPreflightError(
            "--resume and --stop-after-step cannot be used together"
        )
    if resume_path is None:
        if output_directory.exists() and any(output_directory.iterdir()):
            raise TeacherGuidedBlockPreflightError(
                "output directory must be absent or empty for a new block"
            )
    else:
        if not output_directory.is_dir():
            raise TeacherGuidedBlockPreflightError(
                "output directory must exist for resume"
            )
        if not resume_path.is_dir():
            raise TeacherGuidedBlockPreflightError(
                "resume checkpoint directory is missing"
            )


def _validate_maintained_stack(
    stack: RuntimeStack,
    config: TeacherGuidedBlockConfig,
    output_directory: Path,
    variant: PolicyVariant,
) -> None:
    arguments = build_grpo_arguments(
        config=config,
        output_directory=output_directory,
        variant=variant,
        target_step=config.checkpoint["final_step"],
    )
    _require_keywords(
        stack.grpo_config_class,
        set(arguments),
        "TRL GRPOConfig",
    )
    _require_keywords(
        stack.grpo_trainer_class,
        {
            "model",
            "args",
            "processing_class",
            "reward_funcs",
            "train_dataset",
        },
        "TRL GRPOTrainer",
    )
    _require_keywords(
        stack.grpo_trainer_class.train,
        {"resume_from_checkpoint"},
        "TRL GRPOTrainer.train",
    )


def _resolve_starting_point(
    *,
    config: TeacherGuidedBlockConfig,
    variant: PolicyVariant,
    evidence: Mapping[str, Any],
    environ: Mapping[str, str] | None,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    selected = dict(evidence["selected_starting_points"][variant])
    if variant == "direct_action":
        return {
            **selected,
            "initialization": "new_zero_effect_rank_64_lora_on_selected_base",
        }
    gate_path, gate = _load_evidence(
        config.zero_shot_gate,
        configuration_path=config.configuration_path,
        environ=environment,
        label="#8 zero-shot/SFT gate result",
    )
    del gate_path
    try:
        registered_adapter = gate["sft"]["reasoning"]["adapter"]
        members = registered_adapter["members"]
        registered_path = registered_adapter["path"]
    except (KeyError, TypeError) as error:
        raise TeacherGuidedBlockPreflightError(
            "#8 result lacks immutable Reasoning adapter evidence"
        ) from error
    if selected["path"] != registered_path:
        raise TeacherGuidedBlockPreflightError(
            "#8 selected Reasoning path differs from its adapter evidence"
        )
    override = environment.get("LLM2048_REASONING_START_ADAPTER")
    adapter_path = (
        Path(override).expanduser().resolve()
        if override
        else Path(str(registered_path)).expanduser().resolve()
    )
    if not adapter_path.is_dir():
        raise TeacherGuidedBlockPreflightError(
            "selected #8 Reasoning adapter directory is missing; set "
            "LLM2048_REASONING_START_ADAPTER to an identical retained copy"
        )
    if not isinstance(members, list) or not members:
        raise TeacherGuidedBlockPreflightError(
            "#8 Reasoning adapter member evidence is malformed"
        )
    verified_members: list[dict[str, str]] = []
    for member in members:
        if not isinstance(member, dict):
            raise TeacherGuidedBlockPreflightError(
                "#8 Reasoning adapter member evidence is malformed"
            )
        member_path = adapter_path / Path(str(member.get("path"))).name
        expected_sha = member.get("sha256")
        if (
            not member_path.is_file()
            or not isinstance(expected_sha, str)
            or _file_sha256(member_path) != expected_sha
        ):
            raise TeacherGuidedBlockPreflightError(
                "selected #8 Reasoning adapter differs from its registered "
                f"member {member_path.name}"
            )
        verified_members.append(
            {"path": member_path.name, "sha256": expected_sha}
        )
    adapter_config = _read_json_object(
        adapter_path / "adapter_config.json"
    )
    if (
        adapter_config.get("r") != config.lora["rank"]
        or adapter_config.get("lora_alpha") != config.lora["alpha"]
        or adapter_config.get("base_model_name_or_path")
        != config.model["id"]
    ):
        raise TeacherGuidedBlockPreflightError(
            "selected #8 Reasoning adapter does not match the rank-64 "
            "Qwen3.5-4B contract"
        )
    return {
        **selected,
        "path": str(adapter_path),
        "registered_members": verified_members,
        "registered_members_identity_sha256": sha256(
            json.dumps(
                verified_members, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }


def _load_starting_model(
    *,
    stack: RuntimeStack,
    config: TeacherGuidedBlockConfig,
    variant: PolicyVariant,
    starting_point: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    snapshot_path = _exact_model_snapshot(stack, config)
    loader_arguments = {
        "model_name": config.model["id"],
        "revision": config.model["revision"],
        "tokenizer_name": snapshot_path,
        "max_seq_length": config.model["max_sequence_length"],
        "dtype": stack.torch.bfloat16,
        "load_in_4bit": False,
        "fast_inference": False,
        "text_only": True,
        "use_exact_model_name": True,
        "full_finetuning": False,
    }
    _require_keywords(
        stack.fast_vision_model.from_pretrained,
        set(loader_arguments),
        "FastVisionModel.from_pretrained",
    )
    model, processor = stack.fast_vision_model.from_pretrained(
        **loader_arguments
    )
    _ensure_generation_architecture(model)
    if getattr(processor, "pad_token_id", None) is None:
        processor.pad_token = processor.eos_token
    processor.padding_side = "left"
    if variant == "direct_action":
        lora_arguments = {
            "model": model,
            "finetune_vision_layers": False,
            "finetune_language_layers": True,
            "finetune_attention_modules": True,
            "finetune_mlp_modules": True,
            "r": config.lora["rank"],
            "lora_alpha": config.lora["alpha"],
            "lora_dropout": config.lora["dropout"],
            "bias": "none",
            "random_state": config.seed,
            "use_rslora": False,
            "loftq_config": None,
            "use_gradient_checkpointing": config.lora[
                "gradient_checkpointing"
            ],
        }
        _require_keywords(
            stack.fast_vision_model.get_peft_model,
            set(lora_arguments),
            "FastVisionModel.get_peft_model",
        )
        model = stack.fast_vision_model.get_peft_model(**lora_arguments)
    else:
        adapter_arguments = {
            "model": model,
            "model_id": str(starting_point["path"]),
            "is_trainable": True,
        }
        _require_keywords(
            stack.peft_model_class.from_pretrained,
            set(adapter_arguments),
            "PeftModel.from_pretrained",
        )
        model = stack.peft_model_class.from_pretrained(**adapter_arguments)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    trainable = _trainable_parameter_evidence(model)
    if trainable["vision_trainable_parameter_names"]:
        raise RuntimeError(
            "Teacher-guided GRPO exposed trainable vision parameters"
        )
    if trainable["lora_trainable_parameters"] <= 0:
        raise RuntimeError(
            "Teacher-guided GRPO did not expose trainable LoRA parameters"
        )
    if variant == "direct_action":
        lora_b_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "lora_B" in name
        ]
        if not lora_b_parameters:
            raise RuntimeError(
                "Direct-action initialization exposes no LoRA B matrices"
            )
        nonzero = sum(
            int(stack.torch.count_nonzero(parameter.detach()).item())
            for parameter in lora_b_parameters
        )
        if nonzero:
            raise RuntimeError(
                "Direct-action LoRA changes the selected unchanged base "
                "before GRPO"
            )
        trainable["starting_policy_equivalence"] = {
            "selected_candidate": "unchanged_base_model",
            "proof": "all_lora_B_parameters_zero",
            "lora_B_nonzero_parameters": 0,
        }
    else:
        trainable["starting_policy_equivalence"] = {
            "selected_candidate": "registered_issue_8_reasoning_adapter",
            "registered_members_identity_sha256": starting_point[
                "registered_members_identity_sha256"
            ],
        }
    return model, processor, trainable


def _exact_model_snapshot(
    stack: RuntimeStack,
    config: TeacherGuidedBlockConfig,
) -> str:
    try:
        return str(
            stack.snapshot_download(
                repo_id=config.model["id"],
                revision=config.model["revision"],
                local_files_only=True,
            )
        )
    except Exception as error:
        raise TeacherGuidedBlockPreflightError(
            "the exact Qwen3.5-4B revision is incomplete in the local "
            "Hugging Face cache"
        ) from error


def _set_training_mode(stack: RuntimeStack, model: Any) -> None:
    for_training = getattr(
        stack.fast_vision_model, "for_training", None
    )
    if callable(for_training):
        for_training(model)
    else:
        model.train()


def _training_rows(
    *,
    processor: Any,
    variant: PolicyVariant,
    records: Sequence[Mapping[str, Any]],
    tau: float,
    max_prompt_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    for record in records:
        prompt = _render_prompt(processor, variant, record["board"])
        encoded = processor(prompt, add_special_tokens=False)
        token_ids = encoded["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if len(token_ids) > max_prompt_length:
            raise TeacherGuidedBlockPreflightError(
                "rendered Qwen policy prompt exceeds the fixed prompt budget: "
                f"{len(token_ids)} > {max_prompt_length}"
            )
        prompt_lengths.append(len(token_ids))
        action_scores = {
            str(action).upper(): value
            for action, value in record["action_scores"].items()
        }
        rows.append(
            {
                "prompt": prompt,
                "record_id": record["record_id"],
                "board_json": json.dumps(
                    record["board"], separators=(",", ":")
                ),
                "teacher_action_scores_json": json.dumps(
                    action_scores, sort_keys=True, separators=(",", ":")
                ),
                "teacher_action": str(
                    record["teacher_action"]
                ).upper(),
                "teacher_margin_scale": tau,
            }
        )
    return rows, {
        "variant": variant,
        "boards": len(rows),
        "max_prompt_length": max_prompt_length,
        "minimum_prompt_tokens": min(prompt_lengths),
        "maximum_prompt_tokens": max(prompt_lengths),
        "over_budget": 0,
        "status": "passed",
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
                    {
                        "type": "text",
                        "text": build_policy_prompt(variant, board),
                    }
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=variant == "reasoning",
    )
    if not isinstance(rendered, str):
        raise RuntimeError(
            "Qwen processor did not render a text policy prompt"
        )
    return rendered


def _evaluate_policy(
    *,
    stack: RuntimeStack,
    config: TeacherGuidedBlockConfig,
    model: Any,
    processor: Any,
    records: Sequence[Mapping[str, Any]],
    variant: PolicyVariant,
    phase: str,
    output_directory: Path,
    tau: float,
) -> dict[str, Any]:
    stack.fast_vision_model.for_inference(model)
    processor.padding_side = "left"
    event_path = output_directory / "evaluation" / f"{phase}.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.write_text("", encoding="utf-8")
    events: list[dict[str, Any]] = []
    generated_tokens = 0
    max_tokens = config.grpo["max_completion_length"]
    batch_size = config.validation["batch_size"]
    stack.torch.cuda.empty_cache()
    stack.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        prompts = [
            _render_prompt(processor, variant, row["board"])
            for row in batch
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
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.pad_token_id,
                eos_token_id=processor.eos_token_id,
            )
        completion_rows = (
            generated[:, prompt_width:].detach().cpu().tolist()
        )
        for record, completion_ids in zip(batch, completion_rows):
            trimmed, truncated = _trim_completion_ids(
                completion_ids,
                eos_token_id=processor.eos_token_id,
                pad_token_id=processor.pad_token_id,
                max_new_tokens=max_tokens,
            )
            response = processor.decode(
                trimmed, skip_special_tokens=True
            )
            response = _native_policy_response(variant, response)
            teacher_action = cast(
                Action, str(record["teacher_action"]).upper()
            )
            action_scores = cast(
                Mapping[Action, float | None],
                {
                    str(action).upper(): value
                    for action, value in record["action_scores"].items()
                },
            )
            reward = teacher_guided_reward_callback(
                board=record["board"],
                completions=[
                    TeacherGuidedCompletion(
                        variant=variant,
                        response=response,
                        truncated=truncated,
                    )
                ],
                group_size=1,
                teacher_action_scores=action_scores,
                teacher_action=teacher_action,
                tau=tau,
            )[0]
            event = {
                "record_id": record["record_id"],
                "source_split": record["split"],
                "source_stratum": record["stratum"],
                "lineage": record["lineage"],
                "variant": variant,
                "phase": phase,
                "board": record["board"],
                "response": response,
                "response_length_tokens": len(trimmed),
                "truncated": truncated,
                "action": reward.contract.action,
                "parsed": reward.contract.parsed,
                "valid_action": reward.contract.valid_action,
                "policy_failure": reward.contract.policy_failure,
                "policy_failure_reason": (
                    reward.contract.policy_failure_reason
                ),
                "teacher_action": teacher_action,
                "teacher_action_agreement": (
                    reward.contract.action == teacher_action
                ),
                "regret": reward.regret,
                "reward": reward.total,
                "reward_components": reward.components.resolved(),
            }
            events.append(event)
            _append_json_line(event_path, event)
            generated_tokens += len(trimmed)
        del generated
        del encoded
    wall_seconds = time.perf_counter() - started
    metrics = summarize_evaluation_events(
        events,
        wall_seconds=wall_seconds,
        generated_tokens=generated_tokens,
    )
    return {
        "status": "completed",
        "variant": variant,
        "phase": phase,
        "events_path": str(event_path.relative_to(output_directory)),
        "events_sha256": _file_sha256(event_path),
        "metrics": metrics,
        "generation": {
            "batch_size": batch_size,
            "max_new_tokens": max_tokens,
            "peak_allocated_vram_bytes": (
                stack.torch.cuda.max_memory_allocated()
            ),
            "peak_reserved_vram_bytes": (
                stack.torch.cuda.max_memory_reserved()
            ),
        },
    }


def _trim_completion_ids(
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


def _training_summary(
    *,
    reward_events: Sequence[Mapping[str, Any]],
    log_history: Sequence[Mapping[str, Any]],
    trainer_metrics: Mapping[str, Any],
    wall_seconds: float,
    global_step: int,
) -> dict[str, Any]:
    reward_values = [
        float(event["reward"]) for event in reward_events
    ]
    entropy_values = [
        float(row["entropy"])
        for row in log_history
        if isinstance(row.get("entropy"), (int, float))
    ]
    kl_values = [
        float(row["kl"])
        for row in log_history
        if isinstance(row.get("kl"), (int, float))
    ]
    failure_classes = Counter(
        str(event["policy_failure_reason"])
        for event in reward_events
        if event.get("policy_failure")
    )
    return {
        "optimizer_steps": global_step,
        "wall_seconds": wall_seconds,
        "reward_events_this_invocation": len(reward_events),
        "reward_mean_this_invocation": (
            statistics.fmean(reward_values) if reward_values else None
        ),
        "policy_failure_classes_this_invocation": dict(
            sorted(failure_classes.items())
        ),
        "entropy": {
            "mean": (
                statistics.fmean(entropy_values)
                if entropy_values
                else None
            ),
            "observations": len(entropy_values),
        },
        "kl": {
            "mean": statistics.fmean(kl_values) if kl_values else None,
            "observations": len(kl_values),
            "reason_if_absent": (
                "GRPO beta is zero, so no reference policy is loaded"
                if not kl_values
                else None
            ),
        },
        "trainer_metrics": dict(trainer_metrics),
        "trainer_log_history": [
            {key: _json_scalar(value) for key, value in row.items()}
            for row in log_history
        ],
    }


def _compare_evaluation_files(
    initial_path: Path,
    final_path: Path,
    *,
    config: TeacherGuidedBlockConfig,
) -> dict[str, Any]:
    initial_events = _read_json_lines(initial_path)
    final_events = _read_json_lines(final_path)
    initial_ids = [event["record_id"] for event in initial_events]
    final_ids = [event["record_id"] for event in final_events]
    if initial_ids != final_ids:
        raise RuntimeError(
            "step-250 evaluation did not use the identical fixed board order"
        )
    comparison = paired_reward_comparison(
        initial=[float(event["reward"]) for event in initial_events],
        final=[float(event["reward"]) for event in final_events],
        bootstrap_samples=config.comparison["paired_bootstrap_samples"],
        confidence_level=config.comparison["confidence_level"],
        seed=config.seed,
    )
    comparison["interpretation"] = (
        "measurable learning versus this policy's initialization"
        if comparison["measurable_learning"]
        else "no measurable learning versus this policy's initialization"
    )
    return comparison


def _configure_telemetry(
    config: TeacherGuidedBlockConfig,
    telemetry_directory: Path,
) -> None:
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_PROJECT"] = str(
        config.telemetry["wandb_project"]
    )
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_DIR"] = str(telemetry_directory / "wandb")


def _start_wandb_run(
    *,
    stack: RuntimeStack,
    config: TeacherGuidedBlockConfig,
    variant: PolicyVariant,
    evidence: Mapping[str, Any],
    data_evidence: Mapping[str, Any],
    telemetry_directory: Path,
    run_id: str,
    resume: bool,
) -> tuple[Any, str]:
    stack.wandb.login(
        key=os.environ["WANDB_API_KEY"],
        relogin=True,
        verify=True,
    )
    api = stack.wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    if not isinstance(entity, str) or not entity:
        raise RuntimeError(
            "W&B did not resolve an authenticated default entity"
        )
    project = str(config.telemetry["wandb_project"])
    access = _verify_private_wandb_project(
        api=api,
        entity=entity,
        project=project,
    )
    run = stack.wandb.init(
        entity=entity,
        project=project,
        name=f"{config.experiment_name}-{variant}",
        id=run_id,
        resume="allow" if resume else None,
        config={
            **config.resolved(),
            "variant": variant,
            "upstream_gate_passed": evidence["upstream_gate_passed"],
            "dynamic_board_pool_identity_sha256": data_evidence[
                "dynamic_board_pool"
            ]["identity_sha256"],
        },
        dir=str(telemetry_directory / "wandb"),
        mode="online",
        tags=[
            "issue-9",
            variant,
            "bf16",
            "teacher-guided-grpo",
            "upstream-gate-failed",
            "private",
        ],
        save_code=False,
        reinit=True,
    )
    if run is None:
        raise RuntimeError("W&B did not create an online run")
    if getattr(getattr(run, "settings", None), "mode", None) != "online":
        run.finish(exit_code=1)
        raise RuntimeError("W&B did not honor online mode")
    if run.entity != entity or run.project != project:
        run.finish(exit_code=1)
        raise RuntimeError(
            "W&B run target differs from the verified private project"
        )
    return run, access


def _log_evaluation(
    wandb_run: Any,
    *,
    phase: str,
    metrics: Mapping[str, Any],
) -> None:
    prefix = f"evaluation/{phase}"
    wandb_run.log(
        {
            f"{prefix}/reward_mean": metrics["reward"]["mean"],
            f"{prefix}/teacher_action_agreement": metrics[
                "teacher_action_agreement"
            ]["rate"],
            f"{prefix}/policy_failure_rate": metrics[
                "policy_failures"
            ]["rate"],
            f"{prefix}/action_entropy_nats": metrics["sampling"][
                "policy_action_entropy_nats"
            ],
            f"{prefix}/response_length_mean": metrics[
                "response_length_tokens"
            ]["mean"],
            f"{prefix}/seconds_per_board": metrics["latency"][
                "seconds_per_board"
            ],
        }
    )


def _upstream_identity(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "gate_result": evidence["gate_result"],
        "feasibility_result": evidence["feasibility_result"],
        "upstream_gate_passed": evidence["upstream_gate_passed"],
        "risks": evidence["risks"],
    }


def _validate_resume_contract(
    *,
    config: TeacherGuidedBlockConfig,
    variant: PolicyVariant,
    output_directory: Path,
    resume_path: Path,
    evidence: Mapping[str, Any],
    data_evidence: Mapping[str, Any],
    starting_point: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected_checkpoint = (
        output_directory
        / "trainer"
        / f"checkpoint-{config.checkpoint['controlled_interruption_step']}"
    ).resolve()
    if resume_path.resolve() != expected_checkpoint:
        raise TeacherGuidedBlockPreflightError(
            "resume must use this block's controlled step-125 checkpoint"
        )
    contract = checkpoint_contract(
        resume_path,
        expected_step=config.checkpoint["controlled_interruption_step"],
    )
    state = _read_json_object(resume_path / "trainer_state.json")
    if state.get("global_step") != config.checkpoint[
        "controlled_interruption_step"
    ]:
        raise TeacherGuidedBlockPreflightError(
            "Trainer state is not at the controlled interruption step"
        )
    sidecar = _read_json_object(
        resume_path / "llm2048_resume_manifest.json"
    )
    expected = {
        "configuration_sha256": config.input_sha256,
        "variant": variant,
        "upstream": _upstream_identity(evidence),
        "starting_point": dict(starting_point),
        "dynamic_board_pool": data_evidence["dynamic_board_pool"],
        "validation_snapshot": data_evidence["validation_snapshot"],
    }
    for field, value in expected.items():
        if sidecar.get(field) != value:
            raise TeacherGuidedBlockPreflightError(
                f"resume manifest {field} differs from current inputs"
            )
    if sidecar.get("checkpoint") != contract:
        raise TeacherGuidedBlockPreflightError(
            "checkpoint members changed after the resume manifest was written"
        )
    identity = _read_json_object(output_directory / "run_identity.json")
    if (
        identity.get("configuration_sha256") != config.input_sha256
        or identity.get("variant") != variant
        or identity.get("dynamic_board_pool_identity_sha256")
        != data_evidence["dynamic_board_pool"]["identity_sha256"]
        or identity.get("validation_member_ids_sha256")
        != data_evidence["validation_snapshot"]["member_ids_sha256"]
    ):
        raise TeacherGuidedBlockPreflightError(
            "run identity differs from the resumed block"
        )
    return identity


def _build_run_manifest(
    *,
    stack: RuntimeStack,
    config: TeacherGuidedBlockConfig,
    output_directory: Path,
    result: Mapping[str, Any],
    evidence: Mapping[str, Any],
    data_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for relative in (
        "resolved_config.json",
        "data_manifest.json",
        "run_identity.json",
        "prompt_budget_audit.json",
        "evaluation/initialization.jsonl",
        "evaluation/initialization.json",
        "training/reward-events.jsonl",
        "result.json",
    ):
        path = output_directory / relative
        if path.is_file():
            artifacts.append(
                {
                    "path": relative,
                    "sha256": _file_sha256(path),
                    "wandb_uploaded": False,
                }
            )
    for relative in (
        "evaluation/step_250.jsonl",
        "evaluation/step_250.json",
    ):
        path = output_directory / relative
        if path.is_file():
            artifacts.append(
                {
                    "path": relative,
                    "sha256": _file_sha256(path),
                    "wandb_uploaded": False,
                }
            )
    for checkpoint_path in sorted(
        (output_directory / "trainer").glob("checkpoint-*")
    ):
        checkpoint_members = [
            {
                "path": str(path.relative_to(output_directory)),
                "sha256": _file_sha256(path),
            }
            for path in sorted(checkpoint_path.iterdir())
            if path.is_file()
        ]
        artifacts.append(
            {
                "path": str(checkpoint_path.relative_to(output_directory)),
                "artifact_kind": "lora_adapter_checkpoint",
                "members": checkpoint_members,
                "wandb_uploaded": False,
            }
        )
    device = stack.torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "configuration_sha256": config.input_sha256,
        "configuration": config.resolved(),
        "result_status": result["status"],
        "upstream": evidence,
        "data": data_evidence,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": stack.versions,
            "cuda": stack.torch.version.cuda,
            "cudnn": stack.torch.backends.cudnn.version(),
            "gpu": {
                "name": device.name,
                "total_memory_bytes": device.total_memory,
                "compute_capability": list(
                    stack.torch.cuda.get_device_capability(0)
                ),
            },
            "api_signatures": stack.api_signatures,
        },
        "artifacts": artifacts,
        "scope": {
            "maintained_trl_grpo_trainer": True,
            "custom_training_loop": False,
            "model_checkpoints_uploaded": False,
            "project_success_claimed": False,
        },
    }


def _load_evidence(
    reference: EvidenceReference,
    *,
    configuration_path: Path,
    environ: Mapping[str, str],
    label: str,
) -> tuple[Path, Mapping[str, Any]]:
    path = reference.resolve(
        configuration_path=configuration_path, environ=environ
    )
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise TeacherGuidedBlockPreflightError(
            f"cannot read {label}: {error}"
        ) from error
    if sha256(raw).hexdigest() != reference.sha256:
        raise TeacherGuidedBlockPreflightError(
            f"{label} SHA-256 differs from the registered configuration"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise TeacherGuidedBlockPreflightError(
            f"{label} is not valid JSON"
        ) from error
    return path, _object(value, label)


def _validate_corpus_manifest(manifest: Mapping[str, Any]) -> None:
    try:
        if manifest["schema_version"] != 1:
            raise KeyError
        corpus = manifest["corpus"]
        if corpus["split_unit"] != "trajectory_and_symmetry_orbit":
            raise KeyError
        teacher = manifest["teacher_policy"]
        if (
            teacher["search_depth"] != 2
            or teacher["source"] != "retained_100m_teacher_policy"
        ):
            raise KeyError
        calibration = manifest["calibration"]
        if (
            calibration["method"] != "median_positive_margin"
            or calibration["scope"] != "train"
        ):
            raise KeyError
    except (KeyError, TypeError) as error:
        raise TeacherGuidedBlockPreflightError(
            "Teacher Policy Corpus manifest does not match the retained "
            "Depth-2 split-isolated contract"
        ) from error


def _load_teacher_core(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    expected_count: int,
    expected_strata: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = _manifest_artifact(manifest, "train")
    source_path = manifest_path.parent / artifact["path"]
    try:
        source = source_path.open("rb")
    except OSError as error:
        raise TeacherGuidedBlockPreflightError(
            f"cannot read Teacher Core source: {error}"
        ) from error
    records: list[dict[str, Any]] = []
    source_digest = sha256()
    source_records = 0
    try:
        for line_number, raw_line in enumerate(source, 1):
            source_digest.update(raw_line)
            source_records += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise TeacherGuidedBlockPreflightError(
                    f"{source_path}:{line_number} is not valid JSON"
                ) from error
            _validate_block_corpus_row(
                row,
                expected_split="train",
                source_path=source_path,
                line_number=line_number,
            )
            if row.get("teacher_core") is True:
                records.append(row)
    finally:
        source.close()
    if source_digest.hexdigest() != artifact["sha256"]:
        raise TeacherGuidedBlockPreflightError(
            "Teacher Core source SHA-256 differs from the corpus manifest"
        )
    if source_records != artifact["records"]:
        raise TeacherGuidedBlockPreflightError(
            "Teacher Core source record count differs from the corpus manifest"
        )
    strata = dict(Counter(row["stratum"] for row in records))
    if len(records) != expected_count or strata != dict(expected_strata):
        raise TeacherGuidedBlockPreflightError(
            "Teacher Core membership differs from the configured 70,000-board "
            "Dynamic Board Pool"
        )
    ids = "\n".join(row["record_id"] for row in records) + "\n"
    return records, {
        "split": "train",
        "source_path": str(source_path),
        "source_sha256": artifact["sha256"],
        "source_records": source_records,
        "selected_records": len(records),
        "member_ids_sha256": sha256(ids.encode("utf-8")).hexdigest(),
        "strata": strata,
    }


def _manifest_artifact(
    manifest: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TeacherGuidedBlockPreflightError(
            "Teacher Policy Corpus artifacts must be a list"
        )
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("name") == name
    ]
    if len(matches) != 1:
        raise TeacherGuidedBlockPreflightError(
            f"Teacher Policy Corpus must contain exactly one {name} artifact"
        )
    artifact = matches[0]
    if (
        not isinstance(artifact.get("path"), str)
        or not isinstance(artifact.get("sha256"), str)
        or not isinstance(artifact.get("records"), int)
    ):
        raise TeacherGuidedBlockPreflightError(
            f"Teacher Policy Corpus {name} artifact is malformed"
        )
    return artifact


def _validate_block_corpus_row(
    row: Any,
    *,
    expected_split: str,
    source_path: Path,
    line_number: int,
) -> None:
    try:
        if not isinstance(row, dict) or row["split"] != expected_split:
            raise KeyError
        if row["stratum"] not in {"natural", "hard", "late"}:
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
        scores = row["action_scores"]
        if not isinstance(scores, dict) or set(scores) != {
            "left",
            "right",
            "up",
            "down",
        }:
            raise KeyError
        if any(
            value is not None
            and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            )
            for value in scores.values()
        ):
            raise KeyError
        lineage = row["lineage"]
        if (
            not isinstance(lineage, dict)
            or not isinstance(lineage["trajectory_id"], str)
            or not isinstance(lineage["orbit_id"], str)
        ):
            raise KeyError
    except (KeyError, TypeError) as error:
        raise TeacherGuidedBlockPreflightError(
            f"{source_path}:{line_number} violates the block corpus row contract"
        ) from error


def _block_split_isolation(
    pool: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pool_lineages = {
        row["lineage"]["trajectory_id"] for row in pool
    }
    pool_orbits = {row["lineage"]["orbit_id"] for row in pool}
    validation_lineages = {
        row["lineage"]["trajectory_id"] for row in validation
    }
    validation_orbits = {
        row["lineage"]["orbit_id"] for row in validation
    }
    shared_lineages = pool_lineages & validation_lineages
    shared_orbits = pool_orbits & validation_orbits
    if shared_lineages or shared_orbits:
        raise TeacherGuidedBlockPreflightError(
            "Dynamic Board Pool and validation snapshot are not "
            "trajectory/orbit isolated"
        )
    return {
        "status": "passed",
        "split_unit": "trajectory_and_symmetry_orbit",
        "shared_trajectory_lineages": 0,
        "shared_symmetry_orbits": 0,
    }


def _validate_selected_candidate(
    config: TeacherGuidedBlockConfig,
    variant: PolicyVariant,
    candidate: Mapping[str, Any],
) -> None:
    expected_model = config.model["id"]
    expected_revision = config.model["revision"]
    if variant == "direct_action":
        _require(
            candidate.get("kind"),
            "unchanged_base_model",
            "#8 Direct-action selection.kind",
        )
        _require(
            candidate.get("model_id"),
            expected_model,
            "#8 Direct-action selection.model_id",
        )
        _require(
            candidate.get("revision"),
            expected_revision,
            "#8 Direct-action selection.revision",
        )
    else:
        _require(
            candidate.get("kind"),
            "lora_adapter",
            "#8 Reasoning selection.kind",
        )
        _nonempty(candidate.get("path"), "#8 Reasoning selection.path")
        _require(
            candidate.get("base_model_id"),
            expected_model,
            "#8 Reasoning selection.base_model_id",
        )
        _require(
            candidate.get("base_model_revision"),
            expected_revision,
            "#8 Reasoning selection.base_model_revision",
        )


def _validate_model(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "id": OFFICIAL_QWEN35_4B,
        "precision": "bf16",
        "load_in_4bit": False,
        "fast_inference": False,
        "text_only": True,
        "max_sequence_length": 256,
    }
    _exact_keys(raw, {*expected, "revision"}, "model")
    for field, value in expected.items():
        _require(raw[field], value, f"model.{field}")
    revision = _nonempty(raw["revision"], "model.revision")
    if MODEL_REVISION_PATTERN.fullmatch(revision) is None:
        raise TeacherGuidedBlockConfigurationError(
            "model.revision must be a full 40-character commit SHA"
        )
    return dict(raw)


def _validate_corpus(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact_keys(
        raw,
        {
            "manifest_path",
            "manifest_environment_override",
            "manifest_sha256",
        },
        "corpus",
    )
    _nonempty(raw["manifest_path"], "corpus.manifest_path")
    _nonempty(
        raw["manifest_environment_override"],
        "corpus.manifest_environment_override",
    )
    _sha256_digest(raw["manifest_sha256"], "corpus.manifest_sha256")
    return dict(raw)


def _validate_pool(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact_keys(
        raw,
        {
            "generation",
            "initial_teacher_core_count",
            "strata",
            "refresh_during_block",
        },
        "dynamic_board_pool",
    )
    _require(raw["generation"], 0, "dynamic_board_pool.generation")
    _require(
        raw["initial_teacher_core_count"],
        70000,
        "dynamic_board_pool.initial_teacher_core_count",
    )
    _require(
        raw["strata"],
        {"natural": 35000, "hard": 21000, "late": 14000},
        "dynamic_board_pool.strata",
    )
    _require(
        raw["refresh_during_block"],
        False,
        "dynamic_board_pool.refresh_during_block",
    )
    return dict(raw)


def _validate_validation(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact_keys(raw, {"split", "count", "strata", "batch_size"}, "validation")
    _require(raw["split"], "validation", "validation.split")
    _require(raw["count"], 2000, "validation.count")
    _require(
        raw["strata"],
        {"natural": 1000, "hard": 600, "late": 400},
        "validation.strata",
    )
    _require(raw["batch_size"], 2, "validation.batch_size")
    return dict(raw)


def _validate_lora(raw: Mapping[str, Any]) -> Mapping[str, Any]:
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
    _exact_keys(raw, set(expected), "lora")
    for field, value in expected.items():
        _require(raw[field], value, f"lora.{field}")
    return dict(raw)


def _validate_grpo(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "trainer": "trl.GRPOTrainer",
        "loss_type": "grpo",
        "group_size": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "generation_batch_size": 4,
        "max_steps": 250,
        "learning_rate": 0.000005,
        "optimizer": "adamw_8bit",
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "max_prompt_length": 160,
        "max_completion_length": 96,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "beta": 0.0,
        "mask_truncated_completions": False,
        "use_vllm": False,
    }
    _exact_keys(raw, set(expected), "grpo")
    for field, value in expected.items():
        _require(raw[field], value, f"grpo.{field}")
    return dict(raw)


def _validate_checkpoint(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "save_steps": 125,
        "final_step": 250,
        "controlled_interruption_step": 125,
        "save_optimizer_state": True,
        "save_scheduler_state": True,
        "save_rng_state": True,
    }
    _exact_keys(raw, set(expected), "checkpoint")
    for field, value in expected.items():
        _require(raw[field], value, f"checkpoint.{field}")
    return dict(raw)


def _validate_comparison(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "paired_bootstrap_samples": 2000,
        "confidence_level": 0.95,
        "measurable_learning_rule": (
            "paired_reward_delta_lower_ci_gt_zero"
        ),
    }
    _exact_keys(raw, set(expected), "comparison")
    for field, value in expected.items():
        _require(raw[field], value, f"comparison.{field}")
    return dict(raw)


def _validate_telemetry(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "wandb_mode": "online",
        "wandb_entity": "auto",
        "wandb_project": "2048llm-feasibility",
        "wandb_project_visibility": "private",
        "tensorboard": True,
        "upload_model_checkpoints": False,
    }
    _exact_keys(raw, set(expected), "telemetry")
    for field, value in expected.items():
        _require(raw[field], value, f"telemetry.{field}")
    return dict(raw)


def _load_evidence_reference(
    raw: Mapping[str, Any], field: str
) -> EvidenceReference:
    _exact_keys(raw, {"path", "environment_override", "sha256"}, field)
    digest = _nonempty(raw["sha256"], f"{field}.sha256")
    _sha256_digest(digest, f"{field}.sha256")
    return EvidenceReference(
        path=_nonempty(raw["path"], f"{field}.path"),
        environment_override=_nonempty(
            raw["environment_override"], f"{field}.environment_override"
        ),
        sha256=digest,
    )


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TeacherGuidedBlockConfigurationError(
            f"{field} must be an object"
        )
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], field: str
) -> None:
    if set(value) != expected:
        raise TeacherGuidedBlockConfigurationError(
            f"{field} must contain exactly {', '.join(sorted(expected))}"
        )


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TeacherGuidedBlockConfigurationError(
            f"{field} must be a non-empty string"
        )
    return value


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise TeacherGuidedBlockConfigurationError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _sha256_digest(value: Any, field: str) -> str:
    digest = _nonempty(value, field)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise TeacherGuidedBlockConfigurationError(
            f"{field} must be a lowercase SHA-256"
        )
    return digest


def _native_policy_response(
    variant: PolicyVariant, completion: str
) -> str:
    if variant == "reasoning" and not completion.startswith("<think>"):
        return "<think>\n" + completion
    return completion


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except OSError as error:
        raise TeacherGuidedBlockPreflightError(
            f"cannot read required JSON artifact {path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise TeacherGuidedBlockPreflightError(
            f"required artifact {path} is not valid JSON"
        ) from error
    if not isinstance(value, dict):
        raise TeacherGuidedBlockPreflightError(
            f"required artifact {path} must be a JSON object"
        )
    return value


def _read_json_lines(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"{path}:{line_number} is not valid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"{path}:{line_number} must be a JSON object"
                    )
                rows.append(row)
    except OSError as error:
        raise RuntimeError(f"cannot read evaluation events: {error}") from error
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_json_line(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _safe_error(error: Exception) -> str:
    message = str(error)
    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return message[:4000]


def _require(value: Any, expected: Any, field: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise TeacherGuidedBlockConfigurationError(
            f"{field} must be {expected!r}"
        )
