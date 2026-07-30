"""Shared boundary for the Stage 2 Environment GRPO backend spike.

The module intentionally owns only the 2048 environment, the frozen Environment
Reward, response parsing, comparison manifests, and telemetry. ART and TRL keep
ownership of GRPO, gradient computation, optimizer steps, and checkpointing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Literal, Mapping, Sequence, cast

from llm2048.game import Game2048
from llm2048.policy_contracts import (
    Action,
    PolicyVariant,
    build_policy_prompt,
    change_making_actions,
    enforce_policy_response,
)


BackendName = Literal["art_local", "trl_environment_factory"]
PRIVATE_WANDB_ACCESS = frozenset({"PRIVATE", "RESTRICTED", "TEAM"})
BACKEND_FAILURE_MODES: dict[BackendName, tuple[str, ...]] = {
    "art_local": (
        "ART backend imports Megatron registration but omits megatron-core",
        "a zero-variance Rollout Group is skipped and cannot prove backward",
        "the shared-GPU LocalBackend must pause inference while Unsloth trains",
        "vLLM token-id metadata is required for each generated history",
        "ART internal model configuration is version-pinned and not public API",
    ),
    "trl_environment_factory": (
        "environment_factory is experimental in the tested TRL release",
        "the native tool loop retains prior actions and observations in the prefix",
        "the tested TRL release cannot share the proven Unsloth 2026.7 stack",
        "a missing or malformed tool call becomes a strict Policy Failure",
    ),
}


class BackendSpikeConfigurationError(ValueError):
    """Raised when the registered comparison is unsafe or ambiguous."""


class BackendSpikePreflightError(RuntimeError):
    """Raised before a real backend run when prerequisites do not hold."""


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str
    precision: str
    max_sequence_length: int


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class RolloutConfig:
    variant: PolicyVariant
    group_size: int
    horizon: int
    temperature: float
    top_p: float
    top_k: int
    max_completion_tokens: int


@dataclass(frozen=True)
class StartStateConfig:
    source: str
    rng_seed: int


@dataclass(frozen=True)
class EnvironmentRewardConfig:
    reached_2048_weight: float
    tile_progress_weight: float
    score_progress_weight: float
    game_over_without_2048_penalty: float
    policy_failure_penalty: float
    score_delta_p95: float


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    optimizer_steps: int


@dataclass(frozen=True)
class TelemetryConfig:
    wandb_mode: str
    wandb_project: str
    wandb_project_visibility: str
    tensorboard: bool
    upload_model_checkpoints: bool


@dataclass(frozen=True)
class BackendSpikeConfig:
    schema_version: int
    experiment_name: str
    seed: int
    model: ModelConfig
    lora: LoraConfig
    rollout: RolloutConfig
    start_state: StartStateConfig
    environment_reward: EnvironmentRewardConfig
    training: TrainingConfig
    telemetry: TelemetryConfig
    stacks: dict[BackendName, dict[str, str]]

    @classmethod
    def load(cls, path: Path) -> tuple["BackendSpikeConfig", str]:
        try:
            raw_bytes = path.read_bytes()
            raw = json.loads(raw_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise BackendSpikeConfigurationError(
                f"cannot load Stage 2 backend spike config: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise BackendSpikeConfigurationError("config must be a JSON object")
        _exact_keys(
            raw,
            {
                "schema_version",
                "experiment_name",
                "seed",
                "model",
                "lora",
                "rollout",
                "start_state",
                "environment_reward",
                "training",
                "telemetry",
                "stacks",
            },
            "config",
        )
        if raw["schema_version"] != 1:
            raise BackendSpikeConfigurationError("schema_version must be 1")
        model_raw = _object(raw["model"], "model")
        _exact_keys(
            model_raw,
            {"id", "revision", "precision", "max_sequence_length"},
            "model",
        )
        model = ModelConfig(
            id=_string(model_raw["id"], "model.id"),
            revision=_revision(model_raw["revision"]),
            precision=_string(model_raw["precision"], "model.precision"),
            max_sequence_length=_positive_int(
                model_raw["max_sequence_length"], "model.max_sequence_length"
            ),
        )
        if model.id != "Qwen/Qwen3-0.6B" or model.precision != "bf16":
            raise BackendSpikeConfigurationError(
                "the comparison must use the registered BF16 Qwen/Qwen3-0.6B"
            )
        lora_raw = _object(raw["lora"], "lora")
        _exact_keys(
            lora_raw,
            {"rank", "alpha", "dropout", "target_modules"},
            "lora",
        )
        targets = lora_raw["target_modules"]
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(item, str) or not item for item in targets)
        ):
            raise BackendSpikeConfigurationError(
                "lora.target_modules must be a non-empty string list"
            )
        lora = LoraConfig(
            rank=_positive_int(lora_raw["rank"], "lora.rank"),
            alpha=_positive_int(lora_raw["alpha"], "lora.alpha"),
            dropout=_number(lora_raw["dropout"], "lora.dropout"),
            target_modules=tuple(targets),
        )
        rollout_raw = _object(raw["rollout"], "rollout")
        _exact_keys(
            rollout_raw,
            {
                "variant",
                "group_size",
                "horizon",
                "temperature",
                "top_p",
                "top_k",
                "max_completion_tokens",
            },
            "rollout",
        )
        variant = _string(rollout_raw["variant"], "rollout.variant")
        if variant not in {"direct_action", "reasoning"}:
            raise BackendSpikeConfigurationError("rollout.variant is invalid")
        rollout = RolloutConfig(
            variant=cast(PolicyVariant, variant),
            group_size=_positive_int(
                rollout_raw["group_size"], "rollout.group_size"
            ),
            horizon=_positive_int(rollout_raw["horizon"], "rollout.horizon"),
            temperature=_number(
                rollout_raw["temperature"], "rollout.temperature"
            ),
            top_p=_number(rollout_raw["top_p"], "rollout.top_p"),
            top_k=_positive_int(rollout_raw["top_k"], "rollout.top_k"),
            max_completion_tokens=_positive_int(
                rollout_raw["max_completion_tokens"],
                "rollout.max_completion_tokens",
            ),
        )
        if rollout.group_size < 2 or not 2 <= rollout.horizon <= 4:
            raise BackendSpikeConfigurationError(
                "comparison requires group_size >= 2 and a 2-4 step horizon"
            )
        start_raw = _object(raw["start_state"], "start_state")
        _exact_keys(start_raw, {"source", "rng_seed"}, "start_state")
        start_state = StartStateConfig(
            source=_string(start_raw["source"], "start_state.source"),
            rng_seed=_integer(start_raw["rng_seed"], "start_state.rng_seed"),
        )
        if start_state.source != "canonical_new_game":
            raise BackendSpikeConfigurationError(
                "only the canonical new-game start is registered"
            )
        reward_raw = _object(raw["environment_reward"], "environment_reward")
        reward_keys = {
            "reached_2048_weight",
            "tile_progress_weight",
            "score_progress_weight",
            "game_over_without_2048_penalty",
            "policy_failure_penalty",
            "score_delta_p95",
        }
        _exact_keys(reward_raw, reward_keys, "environment_reward")
        reward = EnvironmentRewardConfig(
            reached_2048_weight=_number(
                reward_raw["reached_2048_weight"],
                "environment_reward.reached_2048_weight",
            ),
            tile_progress_weight=_number(
                reward_raw["tile_progress_weight"],
                "environment_reward.tile_progress_weight",
            ),
            score_progress_weight=_number(
                reward_raw["score_progress_weight"],
                "environment_reward.score_progress_weight",
            ),
            game_over_without_2048_penalty=_number(
                reward_raw["game_over_without_2048_penalty"],
                "environment_reward.game_over_without_2048_penalty",
            ),
            policy_failure_penalty=_number(
                reward_raw["policy_failure_penalty"],
                "environment_reward.policy_failure_penalty",
            ),
            score_delta_p95=_number(
                reward_raw["score_delta_p95"],
                "environment_reward.score_delta_p95",
            ),
        )
        if (
            reward.reached_2048_weight != 5.0
            or reward.tile_progress_weight != 1.0
            or reward.score_progress_weight != 0.25
            or reward.game_over_without_2048_penalty != -1.0
            or reward.policy_failure_penalty != -1.25
            or reward.score_delta_p95 <= 0
        ):
            raise BackendSpikeConfigurationError(
                "Environment Reward weights must match the frozen specification"
            )
        training_raw = _object(raw["training"], "training")
        _exact_keys(
            training_raw,
            {"learning_rate", "optimizer_steps"},
            "training",
        )
        training = TrainingConfig(
            learning_rate=_number(
                training_raw["learning_rate"], "training.learning_rate"
            ),
            optimizer_steps=_positive_int(
                training_raw["optimizer_steps"], "training.optimizer_steps"
            ),
        )
        if training.optimizer_steps != 1:
            raise BackendSpikeConfigurationError(
                "the throwaway comparison is limited to one optimizer step"
            )
        telemetry_raw = _object(raw["telemetry"], "telemetry")
        _exact_keys(
            telemetry_raw,
            {
                "wandb_mode",
                "wandb_project",
                "wandb_project_visibility",
                "tensorboard",
                "upload_model_checkpoints",
            },
            "telemetry",
        )
        telemetry = TelemetryConfig(
            wandb_mode=_string(
                telemetry_raw["wandb_mode"], "telemetry.wandb_mode"
            ),
            wandb_project=_string(
                telemetry_raw["wandb_project"], "telemetry.wandb_project"
            ),
            wandb_project_visibility=_string(
                telemetry_raw["wandb_project_visibility"],
                "telemetry.wandb_project_visibility",
            ),
            tensorboard=_boolean(
                telemetry_raw["tensorboard"], "telemetry.tensorboard"
            ),
            upload_model_checkpoints=_boolean(
                telemetry_raw["upload_model_checkpoints"],
                "telemetry.upload_model_checkpoints",
            ),
        )
        if (
            telemetry.wandb_mode != "online"
            or telemetry.wandb_project_visibility != "private"
            or not telemetry.tensorboard
            or telemetry.upload_model_checkpoints
        ):
            raise BackendSpikeConfigurationError(
                "telemetry must be online/private W&B plus local TensorBoard "
                "with checkpoint upload disabled"
            )
        stacks_raw = _object(raw["stacks"], "stacks")
        _exact_keys(
            stacks_raw,
            {"art_local", "trl_environment_factory"},
            "stacks",
        )
        stacks: dict[BackendName, dict[str, str]] = {}
        for backend_name in ("art_local", "trl_environment_factory"):
            package_map = _object(stacks_raw[backend_name], f"stacks.{backend_name}")
            if not package_map or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or not value
                for key, value in package_map.items()
            ):
                raise BackendSpikeConfigurationError(
                    f"stacks.{backend_name} must pin package versions"
                )
            stacks[cast(BackendName, backend_name)] = dict(package_map)
        return (
            cls(
                schema_version=1,
                experiment_name=_string(
                    raw["experiment_name"], "experiment_name"
                ),
                seed=_integer(raw["seed"], "seed"),
                model=model,
                lora=lora,
                rollout=rollout,
                start_state=start_state,
                environment_reward=reward,
                training=training,
                telemetry=telemetry,
                stacks=stacks,
            ),
            sha256(raw_bytes).hexdigest(),
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "model": vars(self.model),
            "lora": {
                **vars(self.lora),
                "target_modules": list(self.lora.target_modules),
            },
            "rollout": vars(self.rollout),
            "start_state": vars(self.start_state),
            "environment_reward": vars(self.environment_reward),
            "training": vars(self.training),
            "telemetry": vars(self.telemetry),
            "stacks": self.stacks,
        }


@dataclass(frozen=True)
class EnvironmentReward:
    total: float
    reached_2048: float
    tile_progress: float
    score_progress: float
    game_over_without_2048: float
    policy_failure: float

    def resolved(self) -> dict[str, float]:
        return vars(self)


@dataclass(frozen=True)
class EpisodeStep:
    step: int
    prompt: str
    response: str
    action: Action | None
    board_before: list[list[int]]
    board_after: list[list[int]]
    score_delta: int
    policy_failure_reason: str | None
    terminal: bool

    def resolved(self) -> dict[str, Any]:
        return vars(self)


class Stage2TrainingEpisode:
    """One strict Markov Policy Training Episode over the canonical game."""

    def __init__(self, config: BackendSpikeConfig) -> None:
        self._config = config
        self._game = Game2048(config.start_state.rng_seed)
        self._start_snapshot = self._game.snapshot()
        self._start_max_tile = _max_tile(self._game.board)
        self._start_score = self._game.score
        self._steps: list[EpisodeStep] = []
        self._policy_failure = False
        self._terminal_reason: str | None = None

    @property
    def start_snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._start_snapshot))

    @property
    def start_snapshot_sha256(self) -> str:
        return _snapshot_sha256(self._start_snapshot)

    @property
    def rng_seed(self) -> int:
        return self._config.start_state.rng_seed

    @property
    def steps(self) -> tuple[EpisodeStep, ...]:
        return tuple(self._steps)

    @property
    def terminal(self) -> bool:
        return self._terminal_reason is not None

    @property
    def terminal_reason(self) -> str | None:
        return self._terminal_reason

    @property
    def board(self) -> list[list[int]]:
        return [row[:] for row in self._game.board]

    def policy_prompt(self) -> str:
        if self.terminal:
            raise RuntimeError("cannot request a prompt after episode termination")
        return build_policy_prompt(self._config.rollout.variant, self._game.board)

    def submit_policy_response(
        self,
        response: str,
        *,
        truncated: bool = False,
    ) -> EpisodeStep:
        if self.terminal:
            raise RuntimeError("a Training Episode cannot retry after termination")
        board_before = self.board
        prompt = self.policy_prompt()
        contract = enforce_policy_response(
            variant=self._config.rollout.variant,
            response=response,
            truncated=truncated,
            board_change_actions=change_making_actions(board_before),
        )
        score_delta = 0
        if contract.policy_failure or contract.action is None:
            self._policy_failure = True
            self._terminal_reason = "policy_failure"
        else:
            outcome = self._game.move(contract.action)
            score_delta = outcome.score_delta
            if _max_tile(self._game.board) >= 2048:
                self._terminal_reason = "2048_success"
            elif not change_making_actions(self._game.board):
                self._terminal_reason = "game_over"
            elif len(self._steps) + 1 >= self._config.rollout.horizon:
                self._terminal_reason = "horizon"
        step = EpisodeStep(
            step=len(self._steps) + 1,
            prompt=prompt,
            response=response,
            action=contract.action,
            board_before=board_before,
            board_after=self.board,
            score_delta=score_delta,
            policy_failure_reason=contract.policy_failure_reason,
            terminal=self.terminal,
        )
        self._steps.append(step)
        return step

    def reward(self) -> EnvironmentReward:
        if not self.terminal:
            raise RuntimeError("Environment Reward requires a terminated episode")
        return environment_reward(
            config=self._config.environment_reward,
            start_max_tile=self._start_max_tile,
            final_max_tile=_max_tile(self._game.board),
            start_score=self._start_score,
            final_score=self._game.score,
            reached_2048=_max_tile(self._game.board) >= 2048,
            game_over_without_2048=self._terminal_reason == "game_over",
            policy_failure=self._policy_failure,
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "start_snapshot": self.start_snapshot,
            "start_snapshot_sha256": self.start_snapshot_sha256,
            "rng_seed": self.rng_seed,
            "steps": [step.resolved() for step in self._steps],
            "terminal_reason": self._terminal_reason,
            "reward": self.reward().resolved() if self.terminal else None,
        }


def environment_reward(
    *,
    config: EnvironmentRewardConfig,
    start_max_tile: int,
    final_max_tile: int,
    start_score: int,
    final_score: int,
    reached_2048: bool,
    game_over_without_2048: bool,
    policy_failure: bool,
) -> EnvironmentReward:
    """Apply the frozen Environment Reward from GitHub issue #1."""
    tile_progress = _clip01(
        math.log2(final_max_tile) - math.log2(start_max_tile)
    )
    score_progress = _clip01(
        (final_score - start_score) / config.score_delta_p95
    )
    components = {
        "reached_2048": config.reached_2048_weight if reached_2048 else 0.0,
        "tile_progress": config.tile_progress_weight * tile_progress,
        "score_progress": config.score_progress_weight * score_progress,
        "game_over_without_2048": (
            config.game_over_without_2048_penalty
            if game_over_without_2048
            else 0.0
        ),
        "policy_failure": (
            config.policy_failure_penalty if policy_failure else 0.0
        ),
    }
    return EnvironmentReward(total=sum(components.values()), **components)


def rollout_group_manifest(config: BackendSpikeConfig) -> dict[str, Any]:
    episodes = [
        Stage2TrainingEpisode(config) for _ in range(config.rollout.group_size)
    ]
    hashes = [episode.start_snapshot_sha256 for episode in episodes]
    seeds = [episode.rng_seed for episode in episodes]
    return {
        "group_size": len(episodes),
        "member_start_snapshot_sha256": hashes,
        "member_rng_seed": seeds,
        "shared_start_snapshot": len(set(hashes)) == 1,
        "shared_rng_seed": len(set(seeds)) == 1,
    }


def installed_versions(packages: Sequence[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def validate_stack(config: BackendSpikeConfig, backend: BackendName) -> dict[str, str]:
    expected = config.stacks[backend]
    actual = installed_versions(tuple(expected))
    mismatches = {
        package: f"expected {expected_version}, found {actual[package]}"
        for package, expected_version in expected.items()
        if actual[package] != expected_version
    }
    if mismatches:
        raise BackendSpikePreflightError(
            f"{backend} dependency pins do not match: "
            + "; ".join(f"{key}: {value}" for key, value in mismatches.items())
        )
    return actual


def verify_private_wandb_project(wandb: Any, project: str) -> tuple[str, str]:
    """Authenticate and fail before a run when the W&B project is not private."""
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise BackendSpikePreflightError("WANDB_API_KEY is required")
    if os.environ.get("WANDB_MODE", "online").lower() != "online":
        raise BackendSpikePreflightError("W&B must run online")
    if os.environ.get("WANDB_LOG_MODEL", "false").lower() not in {
        "false",
        "0",
        "no",
    }:
        raise BackendSpikePreflightError("W&B model upload must be disabled")
    wandb.login(key=key, verify=True)
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    if not entity:
        raise BackendSpikePreflightError("W&B default entity is unavailable")
    query = """
    query Stage2ProjectAccess($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) { access }
    }
    """
    try:
        variables = {"entity": entity, "project": project}
        service_api = getattr(api, "_service_api", None)
        if service_api is not None:
            response = service_api.execute_graphql(query, variables)
        else:
            # W&B 0.25, pinned by ART 0.5.18, uses its vendored GraphQL
            # document type through the public Api.client retrying client.
            from wandb_gql import gql  # type: ignore[import-not-found]

            response = api.client.execute(
                gql(query),
                variable_values=variables,
            )
        access = response["project"]["access"]
    except Exception as error:
        raise BackendSpikePreflightError(
            "cannot verify the existing W&B project access"
        ) from error
    if access not in PRIVATE_WANDB_ACCESS:
        raise BackendSpikePreflightError(
            f"W&B project access must be private, found {access!r}"
        )
    return entity, access


def validate_adapter_directory(
    adapter_directory: Path,
    *,
    expected_base_model: str,
) -> dict[str, Any]:
    """Verify a standard PEFT/Unsloth adapter directory without loading weights."""
    config_path = adapter_directory / "adapter_config.json"
    candidates = (
        adapter_directory / "adapter_model.safetensors",
        adapter_directory / "adapter_model.bin",
    )
    if not config_path.is_file() or not any(path.is_file() for path in candidates):
        raise BackendSpikePreflightError(
            "checkpoint is not a Hugging Face PEFT/Unsloth adapter directory"
        )
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackendSpikePreflightError(
            "adapter_config.json is unreadable"
        ) from error
    base_model = adapter_config.get("base_model_name_or_path")
    if base_model != expected_base_model:
        raise BackendSpikePreflightError(
            f"adapter base model is {base_model!r}, expected {expected_base_model!r}"
        )
    weight_path = next(path for path in candidates if path.is_file())
    return {
        "adapter_config": str(config_path),
        "adapter_weights": str(weight_path),
        "adapter_weights_bytes": weight_path.stat().st_size,
        "base_model_name_or_path": base_model,
        "format": "huggingface_peft_unsloth",
    }


def implementation_complexity(backend: BackendName) -> dict[str, Any]:
    """Record a reproducible, intentionally simple integration-size measure."""
    module_name = {
        "art_local": "stage2_art_backend.py",
        "trl_environment_factory": "stage2_trl_backend.py",
    }[backend]
    module_path = Path(__file__).with_name(module_name)
    source_lines = module_path.read_text(encoding="utf-8").splitlines()
    non_blank_non_comment = sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in source_lines
    )
    return {
        "candidate_module": f"src/llm2048/{module_name}",
        "physical_source_lines": len(source_lines),
        "non_blank_non_comment_source_lines": non_blank_non_comment,
        "project_owned_grpo_math": False,
        "project_owned_optimizer_loop": False,
        "maintained_training_owner": (
            "ART LocalBackend.train"
            if backend == "art_local"
            else "TRL GRPOTrainer.train"
        ),
        "known_failure_modes": list(BACKEND_FAILURE_MODES[backend]),
    }


def dry_run(
    config_path: Path,
    output_directory: Path,
    backend: BackendName,
) -> dict[str, Any]:
    config, config_sha256 = BackendSpikeConfig.load(config_path)
    group = rollout_group_manifest(config)
    if not group["shared_start_snapshot"] or not group["shared_rng_seed"]:
        raise BackendSpikePreflightError(
            "Rollout Group members do not share snapshot and RNG seed"
        )
    output_directory.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "status": "dry_run",
        "backend": backend,
        "config_sha256": config_sha256,
        "resolved_config": config.resolved(),
        "rollout_group": group,
        "environment_contract": {
            "markov_policy": True,
            "policy_failure_terminates_without_retry": True,
            "environment_reward_frozen": True,
        },
        "implementation_complexity": implementation_complexity(backend),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "source_revision": _source_revision(),
        },
    }
    _write_json(output_directory / "manifest.json", result)
    return result


def _source_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    payload = json.dumps(snapshot, separators=(",", ":"), sort_keys=True)
    return sha256(payload.encode()).hexdigest()


def _max_tile(board: Sequence[Sequence[int]]) -> int:
    return max(tile for row in board for tile in row)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise BackendSpikeConfigurationError(
            f"{name} keys must be exactly {sorted(expected)}; "
            f"found {sorted(actual)}"
        )


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BackendSpikeConfigurationError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BackendSpikeConfigurationError(f"{name} must be a non-empty string")
    return value


def _revision(value: Any) -> str:
    revision = _string(value, "model.revision")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise BackendSpikeConfigurationError(
            "model.revision must be an exact 40-character commit"
        )
    return revision


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BackendSpikeConfigurationError(f"{name} must be an integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise BackendSpikeConfigurationError(f"{name} must be positive")
    return result


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BackendSpikeConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BackendSpikeConfigurationError(f"{name} must be finite")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise BackendSpikeConfigurationError(f"{name} must be boolean")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage 2 Environment GRPO backend spike"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("art_local", "trl_environment_factory"),
        required=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    backend = cast(BackendName, arguments.backend)
    try:
        if arguments.dry_run:
            result = dry_run(arguments.config, arguments.output_dir, backend)
        elif backend == "art_local":
            from llm2048.stage2_art_backend import run_art_backend

            result = run_art_backend(arguments.config, arguments.output_dir)
        else:
            from llm2048.stage2_trl_backend import run_trl_backend

            result = run_trl_backend(arguments.config, arguments.output_dir)
    except (
        BackendSpikeConfigurationError,
        BackendSpikePreflightError,
        FileExistsError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
