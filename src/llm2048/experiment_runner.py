"""Config-driven deterministic Experiment Runner tracer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence


class ConfigurationError(ValueError):
    """Raised when an experiment configuration violates the public contract."""


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_name: str
    seed: int
    total_steps: int
    board: list[list[int]]
    policy_actions: list[str]
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
        board = fixture.get("board")
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
                "fixture.board must be a 4x4 matrix of non-negative integers"
            )

        policy_actions = fixture.get("policy_actions")
        rewards = fixture.get("rewards")
        total_steps = raw["total_steps"]
        if (
            not isinstance(policy_actions, list)
            or len(policy_actions) != total_steps
            or any(not isinstance(action, str) for action in policy_actions)
        ):
            raise ConfigurationError(
                "fixture.policy_actions must contain one string per step"
            )
        supported_actions = {"LEFT", "RIGHT", "UP", "DOWN"}
        unsupported_action = next(
            (
                action
                for action in policy_actions
                if action not in supported_actions
            ),
            None,
        )
        if unsupported_action is not None:
            raise ConfigurationError(
                "fixture.policy_actions contains unsupported action "
                f"{unsupported_action!r}"
            )
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
                rewards=[float(reward) for reward in rewards],
                wandb_project=wandb_project,
            ),
            sha256(raw_bytes).hexdigest(),
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "total_steps": self.total_steps,
            "fixture": {
                "board": self.board,
                "policy_actions": self.policy_actions,
                "rewards": self.rewards,
            },
            "telemetry": {
                "wandb_mode": "offline",
                "wandb_project": self.wandb_project,
                "tensorboard": True,
            },
            "checkpoint_every": 1,
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


def run_experiment(
    config_path: Path,
    output_directory: Path,
    resume_path: Path | None,
    stop_after_step: int | None,
) -> dict[str, Any]:
    config, input_sha256 = ExperimentConfig.load(config_path)
    experiment_id = config.experiment_id()
    output_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_directory / "checkpoints" / "latest.json"
    events_path = output_directory / "events.jsonl"
    result_path = output_directory / "result.json"
    invocations_path = output_directory / "invocations.jsonl"

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

    target_step = config.total_steps
    if stop_after_step is not None:
        if stop_after_step <= completed_steps or stop_after_step > config.total_steps:
            raise ConfigurationError(
                "--stop-after-step must be greater than completed progress "
                "and no greater than total_steps"
            )
        target_step = stop_after_step

    start_step = completed_steps
    telemetry = Telemetry(output_directory, config, experiment_id)
    try:
        for zero_based_step in range(completed_steps, target_step):
            step = zero_based_step + 1
            action = config.policy_actions[zero_based_step]
            reward = config.rewards[zero_based_step]
            reward_total += reward
            _append_json_line(
                events_path,
                {
                    "action": action,
                    "board": config.board,
                    "reward": reward,
                    "reward_total": reward_total,
                    "step": step,
                },
            )
            telemetry.log_training(step, reward, reward_total)
            completed_steps = step
            _write_json(
                checkpoint_path,
                {
                    "completed_steps": completed_steps,
                    "experiment_id": experiment_id,
                    "reward_total": reward_total,
                    "schema_version": 1,
                },
            )
        if completed_steps == config.total_steps:
            telemetry.log_evaluation(completed_steps, reward_total / completed_steps)
    finally:
        wandb_directory, tensorboard_directory = telemetry.close()

    status = "completed" if completed_steps == config.total_steps else "paused"
    result = {
        "completed_steps": completed_steps,
        "metrics": {
            "mean_reward": reward_total / completed_steps,
            "reward_total": reward_total,
        },
        "status": status,
    }
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
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        result = run_experiment(
            config_path=parsed.config,
            output_directory=parsed.output_dir,
            resume_path=parsed.resume,
            stop_after_step=parsed.stop_after_step,
        )
    except (ConfigurationError, RuntimeError) as error:
        print(f"experiment runner error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
