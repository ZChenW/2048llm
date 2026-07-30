"""Config-driven deterministic Experiment Runner tracer."""

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
import time
from typing import Any, Sequence, cast

from llm2048.game import Game2048
from llm2048.policy_contracts import (
    ACTIONS,
    ACTION_ENVELOPE,
    Action,
    REASONING_ENVELOPE,
    REASONING_MAX_GENERATION_TOKENS,
    PolicyVariant,
    build_policy_prompt,
    change_making_actions,
    enforce_policy_response,
)
from llm2048.teacher_corpus import CorpusError, export_teacher_corpus
from llm2048.teacher_guided_rewards import (
    TeacherGuidedCompletion,
    TeacherGuidedReward,
    teacher_guided_reward_callback,
)


class ConfigurationError(ValueError):
    """Raised when an experiment configuration violates the public contract."""


@dataclass(frozen=True)
class PolicyFixtureCase:
    variant: PolicyVariant
    board: list[list[int]]
    response: str
    response_length_tokens: int
    truncated: bool

    def resolved(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "board": self.board,
            "response": self.response,
            "response_length_tokens": self.response_length_tokens,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class EnvironmentPolicyResponse:
    response: str
    response_length_tokens: int
    truncated: bool

    def resolved(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "response_length_tokens": self.response_length_tokens,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class EnvironmentGameFixture:
    variant: PolicyVariant
    responses: list[EnvironmentPolicyResponse] | None
    action_preferences: tuple[Action, ...] | None

    def resolved(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            **(
                {
                    "responses": [
                        response.resolved() for response in self.responses
                    ]
                }
                if self.responses is not None
                else {"action_preferences": self.action_preferences}
            ),
        }


@dataclass(frozen=True)
class TeacherGuidedRolloutGroupFixture:
    group_size: int
    corpus_manifest_path: str
    corpus_manifest_sha256: str
    tau: float
    board: list[list[int]]
    teacher_action_scores: dict[Action, float | None]
    teacher_action: Action
    candidates: list[PolicyFixtureCase]

    def resolved(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "corpus_manifest": {
                "path": self.corpus_manifest_path,
                "sha256": self.corpus_manifest_sha256,
                "calibration": {
                    "method": "median_positive_margin",
                    "scope": "train",
                    "tau": self.tau,
                },
            },
            "board": self.board,
            "teacher_judgment": {
                "action_scores": self.teacher_action_scores,
                "teacher_action": self.teacher_action,
            },
            "candidates": [
                {
                    "variant": case.variant,
                    "response": case.response,
                    "response_length_tokens": case.response_length_tokens,
                    "truncated": case.truncated,
                }
                for case in self.candidates
            ],
        }


def _validate_board(board: Any, field: str) -> list[list[int]]:
    if (
        not isinstance(board, list)
        or len(board) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in board)
        or any(
            not isinstance(tile, int) or isinstance(tile, bool) or tile < 0
            for row in board
            for tile in row
        )
    ):
        raise ConfigurationError(
            f"{field} must be a 4x4 matrix of non-negative integers"
        )
    return board


def _validate_policy_board(board: Any, field: str) -> list[list[int]]:
    validated = _validate_board(board, field)
    if any(
        tile != 0 and (tile < 2 or tile & (tile - 1) != 0)
        for row in validated
        for tile in row
    ):
        raise ConfigurationError(f"{field} tiles must be 0 or powers of two")
    return validated


def _load_policy_cases(
    raw_cases: Any,
    total_steps: int,
) -> list[PolicyFixtureCase]:
    if not isinstance(raw_cases, list) or len(raw_cases) != total_steps:
        raise ConfigurationError(
            "fixture.policy_cases must contain one object per step"
        )

    cases: list[PolicyFixtureCase] = []
    for index, raw_case in enumerate(raw_cases):
        field = f"fixture.policy_cases[{index}]"
        if not isinstance(raw_case, dict):
            raise ConfigurationError(f"{field} must be an object")

        variant = raw_case.get("variant")
        if variant not in ("direct_action", "reasoning"):
            raise ConfigurationError(
                f"{field}.variant must be 'direct_action' or 'reasoning'"
            )
        board = _validate_policy_board(raw_case.get("board"), f"{field}.board")
        response = raw_case.get("response")
        if not isinstance(response, str):
            raise ConfigurationError(f"{field}.response must be a string")
        response_length_tokens = raw_case.get("response_length_tokens")
        if (
            not isinstance(response_length_tokens, int)
            or isinstance(response_length_tokens, bool)
            or response_length_tokens < 0
        ):
            raise ConfigurationError(
                f"{field}.response_length_tokens must be a non-negative integer"
            )
        if (
            variant == "reasoning"
            and response_length_tokens > REASONING_MAX_GENERATION_TOKENS
        ):
            raise ConfigurationError(
                f"{field}.response_length_tokens must not exceed "
                f"{REASONING_MAX_GENERATION_TOKENS} for a Reasoning Policy"
            )
        truncated = raw_case.get("truncated")
        if not isinstance(truncated, bool):
            raise ConfigurationError(f"{field}.truncated must be a boolean")
        cases.append(
            PolicyFixtureCase(
                variant=variant,
                board=board,
                response=response,
                response_length_tokens=response_length_tokens,
                truncated=truncated,
            )
        )
    return cases


def _load_environment_game(
    raw_game: Any,
    total_steps: int,
) -> EnvironmentGameFixture:
    if not isinstance(raw_game, dict):
        raise ConfigurationError("fixture.environment_game must be an object")
    variant = raw_game.get("variant")
    if variant not in ("direct_action", "reasoning"):
        raise ConfigurationError(
            "fixture.environment_game.variant must be "
            "'direct_action' or 'reasoning'"
        )
    raw_responses = raw_game.get("responses")
    raw_preferences = raw_game.get("action_preferences")
    if raw_responses is not None and raw_preferences is not None:
        raise ConfigurationError(
            "fixture.environment_game must define responses or "
            "action_preferences, not both"
        )
    if raw_preferences is not None:
        if (
            variant != "direct_action"
            or not isinstance(raw_preferences, list)
            or len(raw_preferences) != 4
            or set(raw_preferences) != ACTIONS
        ):
            raise ConfigurationError(
                "fixture.environment_game.action_preferences must contain "
                "LEFT, RIGHT, UP, and DOWN once for a Direct-action Policy"
            )
        return EnvironmentGameFixture(
            variant=variant,
            responses=None,
            action_preferences=tuple(raw_preferences),
        )
    if not isinstance(raw_responses, list) or len(raw_responses) != total_steps:
        raise ConfigurationError(
            "fixture.environment_game.responses must contain one object per step"
        )
    responses: list[EnvironmentPolicyResponse] = []
    for index, raw_response in enumerate(raw_responses):
        field = f"fixture.environment_game.responses[{index}]"
        if not isinstance(raw_response, dict):
            raise ConfigurationError(f"{field} must be an object")
        response = raw_response.get("response")
        if not isinstance(response, str):
            raise ConfigurationError(f"{field}.response must be a string")
        response_length_tokens = raw_response.get("response_length_tokens")
        if (
            not isinstance(response_length_tokens, int)
            or isinstance(response_length_tokens, bool)
            or response_length_tokens < 0
        ):
            raise ConfigurationError(
                f"{field}.response_length_tokens must be a non-negative integer"
            )
        if (
            variant == "reasoning"
            and response_length_tokens > REASONING_MAX_GENERATION_TOKENS
        ):
            raise ConfigurationError(
                f"{field}.response_length_tokens must not exceed "
                f"{REASONING_MAX_GENERATION_TOKENS} for a Reasoning Policy"
            )
        truncated = raw_response.get("truncated")
        if not isinstance(truncated, bool):
            raise ConfigurationError(f"{field}.truncated must be a boolean")
        responses.append(
            EnvironmentPolicyResponse(
                response=response,
                response_length_tokens=response_length_tokens,
                truncated=truncated,
            )
        )
    return EnvironmentGameFixture(
        variant=variant,
        responses=responses,
        action_preferences=None,
    )


def _finite_number(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ConfigurationError(f"{field} must be a finite number")
    return float(value)


def _load_teacher_guided_group(
    raw_group: Any,
    *,
    config_path: Path,
    total_steps: int,
) -> TeacherGuidedRolloutGroupFixture:
    field = "fixture.teacher_guided_rollout_group"
    if not isinstance(raw_group, dict):
        raise ConfigurationError(f"{field} must be an object")
    expected_keys = {
        "group_size",
        "corpus_manifest",
        "board",
        "teacher_judgment",
        "candidates",
    }
    if set(raw_group) != expected_keys:
        raise ConfigurationError(
            f"{field} must contain exactly {', '.join(sorted(expected_keys))}"
        )

    group_size = raw_group["group_size"]
    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size <= 0
    ):
        raise ConfigurationError(f"{field}.group_size must be a positive integer")
    if group_size != total_steps:
        raise ConfigurationError(
            f"{field}.group_size must equal total_steps"
        )

    board = _validate_policy_board(raw_group["board"], f"{field}.board")
    raw_candidates = raw_group["candidates"]
    if not isinstance(raw_candidates, list) or len(raw_candidates) != group_size:
        raise ConfigurationError(
            f"{field}.candidates must match the configured group_size"
        )
    candidate_keys = {
        "variant",
        "response",
        "response_length_tokens",
        "truncated",
    }
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict) or set(candidate) != candidate_keys:
            raise ConfigurationError(
                f"{field}.candidates[{index}] must contain exactly "
                f"{', '.join(sorted(candidate_keys))}"
            )
    cases = _load_policy_cases(
        [
            {
                **candidate,
                "board": board,
            }
            for candidate in raw_candidates
        ],
        total_steps,
    )

    manifest_display = raw_group["corpus_manifest"]
    if not isinstance(manifest_display, str) or not manifest_display:
        raise ConfigurationError(f"{field}.corpus_manifest must be a path string")
    manifest_path = Path(manifest_display)
    if not manifest_path.is_absolute():
        manifest_path = config_path.resolve().parent / manifest_path
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except OSError as error:
        raise ConfigurationError(
            f"cannot read Teacher Policy Corpus manifest: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"Teacher Policy Corpus manifest is not valid JSON: {error}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ConfigurationError(
            "Teacher Policy Corpus manifest schema_version must be 1"
        )
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ConfigurationError(
            "Teacher Policy Corpus manifest calibration must be an object"
        )
    if (
        calibration.get("method") != "median_positive_margin"
        or calibration.get("scope") != "train"
    ):
        raise ConfigurationError(
            "Teacher Policy Corpus manifest must use train "
            "median_positive_margin calibration"
        )
    tau = _finite_number(
        calibration.get("tau"),
        "Teacher Policy Corpus manifest calibration.tau",
    )
    if tau <= 0:
        raise ConfigurationError(
            "Teacher Policy Corpus manifest calibration.tau must be positive"
        )

    judgment = raw_group["teacher_judgment"]
    if not isinstance(judgment, dict) or set(judgment) != {
        "action_scores",
        "teacher_action",
    }:
        raise ConfigurationError(
            f"{field}.teacher_judgment must contain exactly "
            "action_scores and teacher_action"
        )
    raw_scores = judgment["action_scores"]
    corpus_actions = ("up", "down", "left", "right")
    if not isinstance(raw_scores, dict) or set(raw_scores) != set(corpus_actions):
        raise ConfigurationError(
            f"{field}.teacher_judgment.action_scores must contain "
            "up, down, left, and right"
        )
    legal_actions = set(change_making_actions(board))
    scores: dict[Action, float | None] = {}
    for corpus_action in corpus_actions:
        action = cast(Action, corpus_action.upper())
        raw_score = raw_scores[corpus_action]
        if action in legal_actions:
            scores[action] = _finite_number(
                raw_score,
                f"{field}.teacher_judgment.action_scores.{corpus_action}",
            )
        elif raw_score is not None:
            raise ConfigurationError(
                f"{field}.teacher_judgment.action_scores.{corpus_action} "
                "must be null for an illegal action"
            )
        else:
            scores[action] = None
    raw_teacher_action = judgment["teacher_action"]
    if raw_teacher_action not in corpus_actions:
        raise ConfigurationError(
            f"{field}.teacher_judgment.teacher_action must be "
            "up, down, left, or right"
        )
    teacher_action = cast(Action, raw_teacher_action.upper())
    teacher_score = scores[teacher_action]
    legal_scores = [score for score in scores.values() if score is not None]
    if not legal_scores:
        raise ConfigurationError(
            f"{field}.board must have at least one legal action"
        )
    if teacher_score is None or teacher_score != max(legal_scores):
        raise ConfigurationError(
            f"{field}.teacher_judgment.teacher_action must be top-ranked"
        )

    return TeacherGuidedRolloutGroupFixture(
        group_size=group_size,
        corpus_manifest_path=manifest_display,
        corpus_manifest_sha256=sha256(manifest_bytes).hexdigest(),
        tau=tau,
        board=board,
        teacher_action_scores=scores,
        teacher_action=teacher_action,
        candidates=cases,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_name: str
    seed: int
    total_steps: int
    board: list[list[int]] | None
    policy_actions: list[str] | None
    policy_cases: list[PolicyFixtureCase] | None
    environment_game: EnvironmentGameFixture | None
    teacher_guided_rollout_group: TeacherGuidedRolloutGroupFixture | None
    rewards: list[float]
    wandb_project: str

    @classmethod
    def load(cls, path: Path) -> tuple["ExperimentConfig", str]:
        try:
            raw_bytes = path.read_bytes()
        except OSError as error:
            raise ConfigurationError(f"cannot read configuration: {error}") from error

        try:
            raw = json.loads(raw_bytes)
        except json.JSONDecodeError as error:
            raise ConfigurationError(f"configuration is not valid JSON: {error}") from error

        if not isinstance(raw, dict):
            raise ConfigurationError("configuration must be a JSON object")

        required = {
            "schema_version",
            "experiment_name",
            "seed",
            "total_steps",
            "fixture",
            "telemetry",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ConfigurationError(
                f"configuration is missing required fields: {', '.join(missing)}"
            )
        if raw["schema_version"] != 1:
            raise ConfigurationError("schema_version must be 1")
        if not isinstance(raw["experiment_name"], str) or not raw["experiment_name"]:
            raise ConfigurationError("experiment_name must be a non-empty string")
        if not isinstance(raw["seed"], int) or isinstance(raw["seed"], bool):
            raise ConfigurationError("seed must be an integer")
        if (
            not isinstance(raw["total_steps"], int)
            or isinstance(raw["total_steps"], bool)
            or raw["total_steps"] <= 0
        ):
            raise ConfigurationError("total_steps must be a positive integer")

        fixture = raw["fixture"]
        if not isinstance(fixture, dict):
            raise ConfigurationError("fixture must be an object")

        total_steps = raw["total_steps"]
        environment_game_raw = fixture.get("environment_game")
        teacher_guided_raw = fixture.get("teacher_guided_rollout_group")
        policy_cases_raw = fixture.get("policy_cases")
        if environment_game_raw is not None:
            if teacher_guided_raw is not None or policy_cases_raw is not None:
                raise ConfigurationError(
                    "fixture must not combine environment_game with "
                    "policy_cases or teacher_guided_rollout_group"
                )
            environment_game = _load_environment_game(
                environment_game_raw,
                total_steps,
            )
            teacher_guided_rollout_group = None
            policy_cases = None
            board = None
            policy_actions = None
        elif teacher_guided_raw is not None:
            if set(fixture) != {"teacher_guided_rollout_group"}:
                raise ConfigurationError(
                    "fixture.teacher_guided_rollout_group must be the only "
                    "Teacher-guided fixture input"
                )
            if policy_cases_raw is not None:
                raise ConfigurationError(
                    "fixture must not combine policy_cases with "
                    "teacher_guided_rollout_group"
                )
            teacher_guided_rollout_group = _load_teacher_guided_group(
                teacher_guided_raw,
                config_path=path,
                total_steps=total_steps,
            )
            environment_game = None
            policy_cases = teacher_guided_rollout_group.candidates
            board = None
            policy_actions = None
        elif policy_cases_raw is not None:
            environment_game = None
            teacher_guided_rollout_group = None
            policy_cases = _load_policy_cases(policy_cases_raw, total_steps)
            board = None
            policy_actions = None
        else:
            environment_game = None
            teacher_guided_rollout_group = None
            policy_cases = None
            board = _validate_board(fixture.get("board"), "fixture.board")
            policy_actions = fixture.get("policy_actions")
            if (
                not isinstance(policy_actions, list)
                or len(policy_actions) != total_steps
                or any(not isinstance(action, str) for action in policy_actions)
            ):
                raise ConfigurationError(
                    "fixture.policy_actions must contain one string per step"
                )
            unsupported_action = next(
                (
                    action
                    for action in policy_actions
                    if action not in ACTIONS
                ),
                None,
            )
            if unsupported_action is not None:
                raise ConfigurationError(
                    "fixture.policy_actions contains unsupported action "
                    f"{unsupported_action!r}"
                )

        rewards = fixture.get("rewards")
        if teacher_guided_rollout_group is not None:
            if rewards is not None:
                raise ConfigurationError(
                    "fixture.rewards is computed for a Teacher-guided Rollout Group"
                )
            rewards = [0.0] * total_steps
        elif rewards is None and (
            policy_cases is not None or environment_game is not None
        ):
            rewards = [0.0] * total_steps
        if (
            not isinstance(rewards, list)
            or len(rewards) != total_steps
            or any(
                not isinstance(reward, (int, float)) or isinstance(reward, bool)
                for reward in rewards
            )
        ):
            raise ConfigurationError("fixture.rewards must contain one number per step")

        telemetry = raw["telemetry"]
        if not isinstance(telemetry, dict):
            raise ConfigurationError("telemetry must be an object")
        wandb_project = telemetry.get("wandb_project")
        if not isinstance(wandb_project, str) or not wandb_project:
            raise ConfigurationError(
                "telemetry.wandb_project must be a non-empty string"
            )

        return (
            cls(
                schema_version=1,
                experiment_name=raw["experiment_name"],
                seed=raw["seed"],
                total_steps=total_steps,
                board=board,
                policy_actions=policy_actions,
                policy_cases=policy_cases,
                environment_game=environment_game,
                teacher_guided_rollout_group=teacher_guided_rollout_group,
                rewards=[float(reward) for reward in rewards],
                wandb_project=wandb_project,
            ),
            sha256(raw_bytes).hexdigest(),
        )

    def resolved(self) -> dict[str, Any]:
        fixture: dict[str, Any]
        if self.environment_game is not None:
            fixture = {
                "environment_game": self.environment_game.resolved(),
                "rewards": self.rewards,
            }
        elif self.teacher_guided_rollout_group is not None:
            fixture = {
                "teacher_guided_rollout_group": (
                    self.teacher_guided_rollout_group.resolved()
                )
            }
        elif self.policy_cases is None:
            fixture = {
                "board": self.board,
                "policy_actions": self.policy_actions,
                "rewards": self.rewards,
            }
        else:
            fixture = {
                "policy_cases": [case.resolved() for case in self.policy_cases],
                "rewards": self.rewards,
            }
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "total_steps": self.total_steps,
            "fixture": fixture,
            "telemetry": {
                "wandb_mode": "offline",
                "wandb_project": self.wandb_project,
                "tensorboard": True,
            },
            "checkpoint_every": 1,
            **(
                {
                    "policy_contracts": {
                        "action_envelope": ACTION_ENVELOPE,
                        "reasoning_envelope": REASONING_ENVELOPE,
                        "reasoning_max_generation_tokens": (
                            REASONING_MAX_GENERATION_TOKENS
                        ),
                    }
                }
                if (
                    self.policy_cases is not None
                    or self.environment_game is not None
                )
                else {}
            ),
        }

    def experiment_id(self) -> str:
        encoded = json.dumps(
            self.resolved(), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, path)


def _append_json_line(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_members(
    artifact_directory: Path, output_directory: Path, pattern: str
) -> list[dict[str, str]]:
    return [
        {
            "path": str(path.relative_to(output_directory)),
            "sha256": _file_sha256(path),
        }
        for path in sorted(artifact_directory.rglob(pattern))
        if path.is_file()
    ]


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"required package {package!r} is not installed; install project dependencies"
        ) from error


def _git_output(*arguments: str) -> str | None:
    source_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=source_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _source_revision() -> str | None:
    return _git_output("rev-parse", "HEAD")


def _source_dirty() -> bool | None:
    status = _git_output("status", "--porcelain")
    return None if status is None else bool(status)


class Telemetry:
    def __init__(
        self,
        output_directory: Path,
        config: ExperimentConfig,
        experiment_id: str,
    ) -> None:
        try:
            import wandb
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
        except ImportError as error:
            raise RuntimeError(
                "W&B and TensorBoard are required; install project dependencies"
            ) from error

        self._wandb_base = output_directory / "telemetry" / "wandb"
        self._tensorboard_directory = output_directory / "telemetry" / "tensorboard"
        self._wandb_base.mkdir(parents=True, exist_ok=True)
        self._tensorboard_directory.mkdir(parents=True, exist_ok=True)

        settings = wandb.Settings(mode="offline", silent=True)
        self._wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.experiment_name,
            id=experiment_id[:32],
            dir=str(self._wandb_base),
            config=config.resolved(),
            settings=settings,
            reinit=True,
        )
        self._event_writer = EventFileWriter(str(self._tensorboard_directory))

    def _log(self, step: int, values: dict[str, float]) -> None:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        self._wandb_run.log(values, step=step)
        self._event_writer.add_event(
            Event(
                wall_time=time.time(),
                step=step,
                summary=Summary(
                    value=[
                        Summary.Value(tag=tag, simple_value=value)
                        for tag, value in values.items()
                    ]
                ),
            )
        )
        self._event_writer.flush()

    def log_training(self, step: int, reward: float, reward_total: float) -> None:
        self._log(
            step,
            {
                "train/reward": reward,
                "train/reward_total": reward_total,
            },
        )

    def log_evaluation(self, step: int, mean_reward: float) -> None:
        self._log(step, {"eval/mean_reward": mean_reward})

    def log_policy_training(
        self,
        step: int,
        variant: PolicyVariant,
        *,
        reward: float,
        reward_total: float,
        parsed: bool,
        truncated: bool,
        illegal_action: bool,
        valid_action: bool,
        policy_failure: bool,
        response_length_tokens: int,
        teacher_guided_score: TeacherGuidedReward | None = None,
    ) -> None:
        prefix = f"policy/{variant}"
        values = {
            "train/reward": reward,
            "train/reward_total": reward_total,
            f"{prefix}/parsed": float(parsed),
            f"{prefix}/truncated": float(truncated),
            f"{prefix}/illegal_action": float(illegal_action),
            f"{prefix}/valid_action": float(valid_action),
            f"{prefix}/policy_failure": float(policy_failure),
            f"{prefix}/response_length_tokens": float(response_length_tokens),
        }
        if teacher_guided_score is not None:
            components = teacher_guided_score.components
            values.update(
                {
                    "teacher_guided/action_quality": components.action_quality,
                    "teacher_guided/best_action_bonus": (
                        components.best_action_bonus
                    ),
                    "teacher_guided/illegal_action_penalty": (
                        components.illegal_action_penalty
                    ),
                    "teacher_guided/policy_failure_penalty": (
                        components.policy_failure_penalty
                    ),
                }
            )
            if teacher_guided_score.regret is not None:
                values["teacher_guided/regret"] = teacher_guided_score.regret
        self._log(step, values)

    def log_teacher_guided_group(
        self,
        step: int,
        *,
        mean_reward: float,
        metrics: dict[str, Any],
    ) -> None:
        component_means = metrics["reward_component_means"]
        values = {
            "teacher_guided/group/mean_reward": mean_reward,
            "teacher_guided/group/action_quality_mean": component_means[
                "action_quality"
            ],
            "teacher_guided/group/best_action_bonus_mean": component_means[
                "best_action_bonus"
            ],
            "teacher_guided/group/illegal_action_penalty_mean": component_means[
                "illegal_action_penalty"
            ],
            "teacher_guided/group/policy_failure_penalty_mean": component_means[
                "policy_failure_penalty"
            ],
            "teacher_guided/group/teacher_action_agreement_rate": metrics[
                "teacher_action_agreement_rate"
            ],
        }
        valid_legal_mean_regret = metrics["valid_legal_mean_regret"]
        if valid_legal_mean_regret is not None:
            values["teacher_guided/group/valid_legal_mean_regret"] = (
                valid_legal_mean_regret
            )
        self._log(step, values)

    def close(self) -> tuple[Path, Path]:
        self._event_writer.close()
        self._wandb_run.finish()
        offline_runs = sorted(self._wandb_base.rglob("offline-run-*"))
        if not offline_runs:
            raise RuntimeError("W&B offline run directory was not created")
        return offline_runs[-1].parent, self._tensorboard_directory


def _load_checkpoint(path: Path, experiment_id: str) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot load checkpoint: {error}") from error
    if checkpoint.get("experiment_id") != experiment_id:
        raise ConfigurationError("checkpoint does not belong to this experiment")
    return checkpoint


def _policy_event(
    case: PolicyFixtureCase,
    *,
    reward: float,
    reward_total: float,
    step: int,
) -> dict[str, Any]:
    board_change_actions = change_making_actions(case.board)
    contract = enforce_policy_response(
        variant=case.variant,
        response=case.response,
        truncated=case.truncated,
        board_change_actions=board_change_actions,
    )
    return {
        "action": contract.action,
        "board": case.board,
        "change_making_actions": board_change_actions,
        "max_generation_tokens": (
            REASONING_MAX_GENERATION_TOKENS
            if case.variant == "reasoning"
            else None
        ),
        "parsed": contract.parsed,
        "policy_failure": contract.policy_failure,
        "policy_failure_reason": contract.policy_failure_reason,
        "policy_reasoning_trace": contract.policy_reasoning_trace,
        "prompt": build_policy_prompt(case.variant, case.board),
        "response": case.response,
        "response_length_tokens": case.response_length_tokens,
        "reward": reward,
        "reward_total": reward_total,
        "step": step,
        "truncated": case.truncated,
        "valid_action": contract.valid_action,
        "variant": case.variant,
    }


def _teacher_guided_policy_event(
    case: PolicyFixtureCase,
    score: TeacherGuidedReward,
    *,
    group: TeacherGuidedRolloutGroupFixture,
    reward_total: float,
    step: int,
) -> dict[str, Any]:
    event = _policy_event(
        case,
        reward=score.total,
        reward_total=reward_total,
        step=step,
    )
    event.update(
        {
            "action": score.contract.action,
            "parsed": score.contract.parsed,
            "policy_failure": score.contract.policy_failure,
            "policy_failure_reason": score.contract.policy_failure_reason,
            "policy_reasoning_trace": score.contract.policy_reasoning_trace,
            "valid_action": score.contract.valid_action,
            "teacher_action": group.teacher_action,
            "teacher_action_scores": group.teacher_action_scores,
            "selected_action_score": score.selected_action_score,
            "teacher_top1_score": score.teacher_top1_score,
            "regret": score.regret,
            "reward_components": score.components.resolved(),
            "teacher_margin_scale": group.tau,
        }
    )
    return event


def _policy_metrics(events_path: Path) -> dict[str, dict[str, float | int]]:
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    metrics: dict[str, dict[str, float | int]] = {}
    for variant in ("direct_action", "reasoning"):
        variant_events = [event for event in events if event["variant"] == variant]
        responses = len(variant_events)
        if responses == 0:
            continue
        metrics[variant] = {
            "responses": responses,
            "parse_rate": (
                sum(bool(event["parsed"]) for event in variant_events) / responses
            ),
            "truncation_rate": (
                sum(bool(event["truncated"]) for event in variant_events) / responses
            ),
            "illegal_action_rate": (
                sum(
                    event["policy_failure_reason"] == "illegal_action"
                    for event in variant_events
                )
                / responses
            ),
            "valid_action_rate": (
                sum(bool(event["valid_action"]) for event in variant_events)
                / responses
            ),
            "policy_failure_rate": (
                sum(bool(event["policy_failure"]) for event in variant_events)
                / responses
            ),
            "mean_response_length_tokens": (
                sum(event["response_length_tokens"] for event in variant_events)
                / responses
            ),
        }
    return metrics


def _game_summary(
    game: Game2048,
    termination_reason: str,
    policy_failure: bool,
) -> dict[str, Any]:
    tiles = [tile for row in game.board for tile in row if tile != 0]
    histogram = {
        str(value): tiles.count(value)
        for value in sorted(set(tiles))
    }
    return {
        "2048_success": max(tiles) >= 2048,
        "empty_cells": 16 - len(tiles),
        "maximum_tile": max(tiles),
        "moves": game.moves,
        "policy_failure": policy_failure,
        "score": game.score,
        "termination_reason": termination_reason,
        "tile_histogram": histogram,
    }


def _environment_response(
    fixture: EnvironmentGameFixture,
    board: Sequence[Sequence[int]],
    zero_based_step: int,
) -> EnvironmentPolicyResponse:
    if fixture.responses is not None:
        return fixture.responses[zero_based_step]
    if fixture.action_preferences is None:
        raise RuntimeError("environment game policy script is missing")
    change_actions = set(change_making_actions(board))
    action = next(
        (
            preference
            for preference in fixture.action_preferences
            if preference in change_actions
        ),
        None,
    )
    if action is None:
        raise RuntimeError("scripted policy was invoked after game over")
    return EnvironmentPolicyResponse(
        response=f"<action>{action}</action>",
        response_length_tokens=3,
        truncated=False,
    )


def _replay_environment_prefix(
    config: ExperimentConfig,
    completed_steps: int,
) -> Game2048:
    if config.environment_game is None:
        raise RuntimeError("environment game fixture is missing")
    game = Game2048(config.seed)
    for zero_based_step in range(completed_steps):
        response = _environment_response(
            config.environment_game,
            game.board,
            zero_based_step,
        )
        contract = enforce_policy_response(
            variant=config.environment_game.variant,
            response=response.response,
            truncated=response.truncated,
            board_change_actions=change_making_actions(game.board),
        )
        if contract.policy_failure or contract.action is None:
            raise ConfigurationError(
                "checkpoint continues beyond a terminated environment game"
            )
        game.move(contract.action)
        if (
            max(tile for row in game.board for tile in row) >= 2048
            or not change_making_actions(game.board)
        ) and zero_based_step + 1 < completed_steps:
            raise ConfigurationError(
                "checkpoint continues beyond a terminated environment game"
            )
    return game


def _teacher_guided_group_metrics(
    scores: Sequence[TeacherGuidedReward],
    group: TeacherGuidedRolloutGroupFixture,
) -> dict[str, Any]:
    regrets = [score.regret for score in scores if score.regret is not None]
    return {
        "group_size": group.group_size,
        "reward_component_means": {
            component: (
                sum(
                    getattr(score.components, component)
                    for score in scores
                )
                / group.group_size
            )
            for component in (
                "action_quality",
                "best_action_bonus",
                "illegal_action_penalty",
                "policy_failure_penalty",
            )
        },
        "teacher_action_agreement_rate": (
            sum(
                score.contract.action == group.teacher_action
                for score in scores
            )
            / group.group_size
        ),
        "teacher_margin_scale": group.tau,
        "valid_legal_mean_regret": (
            sum(regrets) / len(regrets) if regrets else None
        ),
    }


def run_experiment(
    config_path: Path,
    output_directory: Path,
    resume_path: Path | None,
    stop_after_step: int | None,
) -> dict[str, Any]:
    config, input_sha256 = ExperimentConfig.load(config_path)
    experiment_id = config.experiment_id()
    teacher_guided_scores: list[TeacherGuidedReward] | None = None
    if config.teacher_guided_rollout_group is not None:
        group = config.teacher_guided_rollout_group
        teacher_guided_scores = teacher_guided_reward_callback(
            board=group.board,
            completions=[
                TeacherGuidedCompletion(
                    variant=case.variant,
                    response=case.response,
                    truncated=case.truncated,
                )
                for case in group.candidates
            ],
            group_size=group.group_size,
            teacher_action_scores=group.teacher_action_scores,
            teacher_action=group.teacher_action,
            tau=group.tau,
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_directory / "checkpoints" / "latest.json"
    events_path = output_directory / "events.jsonl"
    result_path = output_directory / "result.json"
    invocations_path = output_directory / "invocations.jsonl"

    checkpoint: dict[str, Any] | None = None
    if resume_path is None:
        if events_path.exists() or checkpoint_path.exists():
            raise ConfigurationError(
                "output directory already contains run state; pass --resume"
            )
        completed_steps = 0
        reward_total = 0.0
    else:
        checkpoint = _load_checkpoint(resume_path, experiment_id)
        completed_steps = checkpoint["completed_steps"]
        reward_total = checkpoint["reward_total"]
        existing_events = (
            events_path.read_text(encoding="utf-8").splitlines()
            if events_path.exists()
            else []
        )
        if len(existing_events) != completed_steps:
            raise ConfigurationError(
                "checkpoint completed_steps does not match the event history"
            )

    game: Game2048 | None = None
    if config.environment_game is not None:
        if checkpoint is None:
            game = Game2048(config.seed)
        else:
            snapshot = checkpoint.get("game_snapshot")
            if not isinstance(snapshot, dict):
                raise ConfigurationError(
                    "environment game checkpoint is missing game_snapshot"
                )
            try:
                game = Game2048.from_snapshot(snapshot)
            except ValueError as error:
                raise ConfigurationError(str(error)) from error
            replayed_game = _replay_environment_prefix(config, completed_steps)
            if game.snapshot() != replayed_game.snapshot():
                raise ConfigurationError(
                    "checkpoint game_snapshot does not match the seeded "
                    "response history"
                )

    target_step = config.total_steps
    if stop_after_step is not None:
        if stop_after_step <= completed_steps or stop_after_step > config.total_steps:
            raise ConfigurationError(
                "--stop-after-step must be greater than completed progress "
                "and no greater than total_steps"
            )
        target_step = stop_after_step

    start_step = completed_steps
    termination_reason: str | None = None
    policy_failure = False
    telemetry = Telemetry(output_directory, config, experiment_id)
    try:
        for zero_based_step in range(completed_steps, target_step):
            step = zero_based_step + 1
            reward = (
                teacher_guided_scores[zero_based_step].total
                if teacher_guided_scores is not None
                else config.rewards[zero_based_step]
            )
            reward_total += reward
            response_length_tokens: int | None = None
            event: dict[str, Any]
            if game is not None:
                if config.environment_game is None:
                    raise RuntimeError("environment game fixture is missing")
                response = _environment_response(
                    config.environment_game,
                    game.board,
                    zero_based_step,
                )
                board = [row[:] for row in game.board]
                board_change_actions = change_making_actions(board)
                contract = enforce_policy_response(
                    variant=config.environment_game.variant,
                    response=response.response,
                    truncated=response.truncated,
                    board_change_actions=board_change_actions,
                )
                event = {
                    "action": contract.action,
                    "board": board,
                    "board_after_move": None,
                    "next_board": board,
                    "parsed": contract.parsed,
                    "policy_failure": contract.policy_failure,
                    "policy_failure_reason": contract.policy_failure_reason,
                    "policy_reasoning_trace": contract.policy_reasoning_trace,
                    "prompt": build_policy_prompt(
                        config.environment_game.variant,
                        board,
                    ),
                    "response": response.response,
                    "response_length_tokens": response.response_length_tokens,
                    "reward": reward,
                    "reward_total": reward_total,
                    "score_delta": 0,
                    "spawned_tile": None,
                    "step": step,
                    "truncated": response.truncated,
                    "valid_action": contract.valid_action,
                    "variant": config.environment_game.variant,
                }
                response_length_tokens = response.response_length_tokens
                if contract.policy_failure:
                    termination_reason = "policy_failure"
                    policy_failure = True
                else:
                    if contract.action is None:
                        raise RuntimeError("valid policy response has no action")
                    outcome = game.move(contract.action)
                    event["board_after_move"] = outcome.board_after_move
                    event["next_board"] = outcome.next_board
                    event["score_delta"] = outcome.score_delta
                    event["spawned_tile"] = outcome.spawned_tile.resolved()
                    if max(tile for row in game.board for tile in row) >= 2048:
                        termination_reason = "2048_success"
                    elif not change_making_actions(game.board):
                        termination_reason = "game_over"
            elif config.policy_cases is None:
                if config.policy_actions is None or config.board is None:
                    raise RuntimeError("legacy fixture is missing policy inputs")
                event = {
                    "action": config.policy_actions[zero_based_step],
                    "board": config.board,
                    "reward": reward,
                    "reward_total": reward_total,
                    "step": step,
                }
            else:
                case = config.policy_cases[zero_based_step]
                if (
                    teacher_guided_scores is not None
                    and config.teacher_guided_rollout_group is not None
                ):
                    event = _teacher_guided_policy_event(
                        case,
                        teacher_guided_scores[zero_based_step],
                        group=config.teacher_guided_rollout_group,
                        reward_total=reward_total,
                        step=step,
                    )
                else:
                    event = _policy_event(
                        case,
                        reward=reward,
                        reward_total=reward_total,
                        step=step,
                    )
                response_length_tokens = case.response_length_tokens
            _append_json_line(events_path, event)
            if config.policy_cases is None and game is None:
                telemetry.log_training(step, reward, reward_total)
            else:
                if response_length_tokens is None:
                    raise RuntimeError("policy response length is missing")
                telemetry.log_policy_training(
                    step,
                    event["variant"],
                    reward=reward,
                    reward_total=reward_total,
                    parsed=bool(event["parsed"]),
                    truncated=bool(event["truncated"]),
                    illegal_action=(
                        event["policy_failure_reason"] == "illegal_action"
                    ),
                    valid_action=bool(event["valid_action"]),
                    policy_failure=bool(event["policy_failure"]),
                    response_length_tokens=response_length_tokens,
                    teacher_guided_score=(
                        teacher_guided_scores[zero_based_step]
                        if teacher_guided_scores is not None
                        else None
                    ),
                )
            completed_steps = step
            _write_json(
                checkpoint_path,
                {
                    "completed_steps": completed_steps,
                    "experiment_id": experiment_id,
                    **(
                        {"game_snapshot": game.snapshot()}
                        if game is not None
                        else {}
                    ),
                    "reward_total": reward_total,
                    "schema_version": 1,
                },
            )
            if termination_reason is not None:
                break
        if completed_steps == config.total_steps:
            if (
                teacher_guided_scores is not None
                and config.teacher_guided_rollout_group is not None
            ):
                telemetry.log_teacher_guided_group(
                    completed_steps,
                    mean_reward=reward_total / completed_steps,
                    metrics=_teacher_guided_group_metrics(
                        teacher_guided_scores,
                        config.teacher_guided_rollout_group,
                    ),
                )
            telemetry.log_evaluation(completed_steps, reward_total / completed_steps)
    finally:
        wandb_directory, tensorboard_directory = telemetry.close()

    if (
        game is not None
        and termination_reason is None
        and completed_steps == config.total_steps
    ):
        termination_reason = "script_exhausted"
    status = (
        "completed"
        if completed_steps == config.total_steps or termination_reason is not None
        else "paused"
    )
    result = {
        "completed_steps": completed_steps,
        "metrics": {
            "mean_reward": reward_total / completed_steps,
            "reward_total": reward_total,
        },
        "status": status,
    }
    if config.policy_cases is not None or game is not None:
        result["policy_metrics"] = _policy_metrics(events_path)
    if game is not None:
        if termination_reason is None:
            termination_reason = "paused"
        result["game"] = _game_summary(
            game,
            termination_reason,
            policy_failure,
        )
    if (
        teacher_guided_scores is not None
        and config.teacher_guided_rollout_group is not None
        and completed_steps == config.total_steps
    ):
        result["teacher_guided_rollout_group"] = (
            _teacher_guided_group_metrics(
                teacher_guided_scores,
                config.teacher_guided_rollout_group,
            )
        )
    _write_json(result_path, result)
    _append_json_line(
        invocations_path,
        {
            "end_step": completed_steps,
            "start_step": start_step,
            "steps_executed": completed_steps - start_step,
        },
    )

    relative_wandb = wandb_directory.relative_to(output_directory)
    relative_tensorboard = tensorboard_directory.relative_to(output_directory)
    manifest = {
        "schema_version": 1,
        "experiment": {
            "id": experiment_id,
            "name": config.experiment_name,
            "seed": config.seed,
            "total_steps": config.total_steps,
        },
        "configuration": config.resolved(),
        "input": {
            "path": str(config_path),
            "sha256": input_sha256,
        },
        "environment": {
            "packages": {
                "python": platform.python_version(),
                "tensorboard": _package_version("tensorboard"),
                "wandb": _package_version("wandb"),
            },
            "source_dirty": _source_dirty(),
            "source_revision": _source_revision(),
        },
        "artifacts": [
            {
                "equivalence": "content",
                "name": "checkpoint",
                "path": str(checkpoint_path.relative_to(output_directory)),
                "sha256": _file_sha256(checkpoint_path),
            },
            {
                "equivalence": "content",
                "name": "events",
                "path": str(events_path.relative_to(output_directory)),
                "sha256": _file_sha256(events_path),
            },
            {
                "equivalence": "attempt_audit",
                "name": "invocations",
                "path": str(invocations_path.relative_to(output_directory)),
                "sha256": _file_sha256(invocations_path),
            },
            {
                "equivalence": "content",
                "name": "result",
                "path": str(result_path.relative_to(output_directory)),
                "sha256": _file_sha256(result_path),
            },
            {
                "equivalence": "semantic_metrics",
                "identity": f"{experiment_id}:tensorboard",
                "members": _artifact_members(
                    tensorboard_directory,
                    output_directory,
                    "events.out.tfevents.*",
                ),
                "name": "tensorboard",
                "path": str(relative_tensorboard),
            },
            {
                "equivalence": "semantic_metrics",
                "identity": f"{experiment_id}:wandb",
                "members": _artifact_members(
                    wandb_directory, output_directory, "run-*.wandb"
                ),
                "name": "wandb",
                "path": str(relative_wandb),
            },
        ],
    }
    _write_json(output_directory / "manifest.json", manifest)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible 2048 Student Policy experiment"
    )
    configuration = parser.add_mutually_exclusive_group(required=True)
    configuration.add_argument("--config", type=Path)
    configuration.add_argument("--teacher-corpus-config", type=Path)
    configuration.add_argument("--grpo-smoke-config", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        if parsed.grpo_smoke_config is not None:
            if parsed.resume is not None or parsed.stop_after_step is not None:
                raise ConfigurationError(
                    "--resume and --stop-after-step are not supported for GRPO smoke runs"
                )
            from llm2048.grpo_smoke import (
                GrpoSmokeConfig,
                GrpoSmokeConfigurationError,
                run_real_smoke,
            )

            try:
                smoke_config, smoke_config_sha256 = GrpoSmokeConfig.load(
                    parsed.grpo_smoke_config
                )
            except GrpoSmokeConfigurationError as error:
                raise ConfigurationError(str(error)) from error
            if not parsed.dry_run:
                result = run_real_smoke(
                    config=smoke_config,
                    input_sha256=smoke_config_sha256,
                    output_directory=parsed.output_dir,
                )
            else:
                result = {"status": "validated", **smoke_config.resolved()}
        elif parsed.teacher_corpus_config is not None:
            if parsed.dry_run:
                raise ConfigurationError(
                    "--dry-run is supported only for GRPO smoke runs"
                )
            if parsed.resume is not None or parsed.stop_after_step is not None:
                raise ConfigurationError(
                    "--resume and --stop-after-step are not supported for corpus exports"
                )
            result = export_teacher_corpus(
                config_path=parsed.teacher_corpus_config,
                output_directory=parsed.output_dir,
            )
        else:
            if parsed.dry_run:
                raise ConfigurationError(
                    "--dry-run is supported only for GRPO smoke runs"
                )
            result = run_experiment(
                config_path=parsed.config,
                output_directory=parsed.output_dir,
                resume_path=parsed.resume,
                stop_after_step=parsed.stop_after_step,
            )
    except (ConfigurationError, CorpusError, RuntimeError) as error:
        print(f"experiment runner error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
