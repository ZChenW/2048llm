"""Maintained ART LocalBackend candidate for Stage 2 Environment GRPO."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any

from llm2048.policy_contracts import change_making_actions, enforce_policy_response
from llm2048.stage2_backend_spike import (
    BackendSpikeConfig,
    BackendSpikePreflightError,
    Stage2TrainingEpisode,
    _source_revision,
    _write_json,
    implementation_complexity,
    rollout_group_manifest,
    validate_adapter_directory,
    validate_stack,
    verify_private_wandb_project,
)


class _GpuMemorySampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_used_bytes = 0
        self.baseline_used_bytes = _nvidia_used_bytes()

    def __enter__(self) -> "_GpuMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _sample(self) -> None:
        while not self._stop.wait(0.2):
            self.peak_used_bytes = max(
                self.peak_used_bytes,
                _nvidia_used_bytes(),
            )


def _nvidia_used_bytes() -> int:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    total_mib = sum(
        int(line.strip())
        for line in output.splitlines()
        if line.strip().isdigit()
    )
    return total_mib * 1024 * 1024


def _internal_model_config(config: BackendSpikeConfig) -> dict[str, Any]:
    return {
        "init_args": {
            "revision": config.model.revision,
            "use_exact_model_name": True,
            "load_in_4bit": False,
            "load_in_16bit": True,
            "max_seq_length": config.model.max_sequence_length,
            "random_state": config.seed,
        },
        "engine_args": {
            "revision": config.model.revision,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.5,
            "max_model_len": config.model.max_sequence_length,
            "max_num_seqs": config.rollout.group_size,
            "enable_sleep_mode": True,
        },
        "peft_args": {
            "r": config.lora.rank,
            "lora_alpha": config.lora.alpha,
            "lora_dropout": config.lora.dropout,
            "target_modules": list(config.lora.target_modules),
            "random_state": config.seed,
            "use_gradient_checkpointing": "unsloth",
        },
        "trainer_args": {
            "num_generations": config.rollout.group_size,
            "per_device_train_batch_size": config.rollout.group_size,
            "gradient_accumulation_steps": 1,
            "max_steps": config.training.optimizer_steps,
            "learning_rate": config.training.learning_rate,
            "logging_steps": 1,
            "report_to": "none",
            "save_strategy": "no",
        },
        "chat_template_kwargs": {"enable_thinking": False},
        "rollout_weights_mode": "lora",
        "allow_unvalidated_arch": True,
    }


def _install_art_registry_compatibility(
    config: BackendSpikeConfig,
) -> bool:
    """Bypass ART 0.5.18's undeclared Megatron import for Unsloth mode."""
    module_name = "art.megatron.model_support"
    if module_name in sys.modules:
        return False
    compatibility_module = ModuleType(module_name)

    def default_target_modules_for_model(
        base_model: str,
        *,
        allow_unvalidated_arch: bool,
    ) -> list[str]:
        if (
            base_model != config.model.id
            or not allow_unvalidated_arch
        ):
            raise BackendSpikePreflightError(
                "ART compatibility boundary received an unregistered model"
            )
        return list(config.lora.target_modules)

    compatibility_module.default_target_modules_for_model = (  # type: ignore[attr-defined]
        default_target_modules_for_model
    )
    sys.modules[module_name] = compatibility_module
    return True


def _training_completion_token_count(choice: Any) -> int:
    """Validate the token logprobs consumed by ART's maintained tokenizer."""
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None)
    if not isinstance(content, list) or not content:
        raise BackendSpikePreflightError(
            "ART vLLM response omitted generated-token logprobs"
        )
    if any(
        not isinstance(getattr(token, "token", None), str)
        or not token.token.startswith("token_id:")
        for token in content
    ):
        raise BackendSpikePreflightError(
            "ART vLLM did not return logprob tokens as token IDs"
        )
    return len(content)


async def _rollout(
    *,
    art: Any,
    history_class: Any,
    model: Any,
    config: BackendSpikeConfig,
    member_index: int,
) -> Any:
    episode = Stage2TrainingEpisode(config)
    histories: list[Any] = []
    completion_tokens = 0
    rollout_started = time.monotonic()
    client = model.openai_client()
    for _ in range(config.rollout.horizon):
        prompt = episode.policy_prompt()
        completion = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=config.rollout.max_completion_tokens,
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            logprobs=True,
            extra_body={
                "top_k": config.rollout.top_k,
                "return_tokens_as_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        choice = completion.choices[0]
        content = choice.message.content or ""
        completion_tokens += _training_completion_token_count(choice)
        history = history_class(
            messages_and_choices=[
                {"role": "user", "content": prompt},
                choice,
            ]
        )
        histories.append(history)
        episode.submit_policy_response(content)
        if episode.terminal:
            break

    if not episode.terminal:
        raise RuntimeError("registered ART rollout did not terminate")
    reward = episode.reward()
    primary = histories[0]
    trajectory = art.Trajectory(
        messages_and_choices=primary.messages_and_choices,
        additional_histories=histories[1:],
        reward=reward.total,
        metrics={
            "environment_steps": len(episode.steps),
            "policy_failure": float(episode.terminal_reason == "policy_failure"),
            "reached_2048_reward": reward.reached_2048,
            "tile_progress_reward": reward.tile_progress,
            "score_progress_reward": reward.score_progress,
            "game_over_without_2048_reward": reward.game_over_without_2048,
            "policy_failure_reward": reward.policy_failure,
            "completion_tokens": completion_tokens,
            "rollout_seconds": time.monotonic() - rollout_started,
        },
        metadata={
            "member_index": member_index,
            "rng_seed": episode.rng_seed,
            "start_snapshot_sha256": episode.start_snapshot_sha256,
            "terminal_reason": episode.terminal_reason,
        },
    )
    return trajectory.finish()


async def _run(
    config: BackendSpikeConfig,
    output_directory: Path,
    config_sha256: str,
) -> dict[str, Any]:
    import art  # type: ignore[import-not-found]

    registry_compatibility_installed = _install_art_registry_compatibility(
        config
    )
    from art.dev import InternalModelConfig  # type: ignore[import-not-found]
    from art.local import LocalBackend  # type: ignore[import-not-found]
    from art.trajectories import History  # type: ignore[import-not-found]
    from torch.utils.tensorboard import SummaryWriter
    import torch
    import wandb

    versions = validate_stack(config, "art_local")
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_LOG_MODEL"] = "false"
    entity, project_access = verify_private_wandb_project(
        wandb,
        config.telemetry.wandb_project,
    )
    os.environ["WANDB_ENTITY"] = entity
    art_directory = output_directory / "art"
    tensorboard_directory = output_directory / "tensorboard"
    output_directory.mkdir(parents=True, exist_ok=False)
    tensorboard_directory.mkdir()
    group_evidence = rollout_group_manifest(config)
    if (
        not group_evidence["shared_start_snapshot"]
        or not group_evidence["shared_rng_seed"]
    ):
        raise BackendSpikePreflightError("Rollout Group common randomness failed")

    # A failed spike attempt must not resume the same W&B run ID as a later
    # attempt. The output directory is already required to be fresh.
    model_name = f"{config.experiment_name}-art-{output_directory.name}"
    model = art.TrainableModel(
        name=model_name,
        project=config.telemetry.wandb_project,
        entity=entity,
        base_model=config.model.id,
        _internal_config=InternalModelConfig(**_internal_model_config(config)),
    )
    model.update_wandb_config(
        {
            "backend": "art_local",
            "config_sha256": config_sha256,
            "model_revision": config.model.revision,
            "upload_model_checkpoints": False,
        }
    )
    backend = LocalBackend(in_process=True, path=str(art_directory))
    run_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with _GpuMemorySampler() as gpu_sampler:
        try:
            await model.register(backend)
            rollout_started = time.monotonic()
            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        (
                            _rollout(
                                art=art,
                                history_class=History,
                                model=model,
                                config=config,
                                member_index=index,
                            )
                            for index in range(config.rollout.group_size)
                        ),
                        metadata={
                            "rng_seed": config.start_state.rng_seed,
                            "start_snapshot_sha256": group_evidence[
                                "member_start_snapshot_sha256"
                            ][0],
                        },
                    ),
                ),
                pbar_desc="issue-12 ART rollout",
                max_exceptions=config.rollout.group_size,
            )
            rollout_seconds = time.monotonic() - rollout_started
            if len(groups) != 1 or len(groups[0].trajectories) != config.rollout.group_size:
                raise BackendSpikePreflightError(
                    "ART did not produce the complete registered Rollout Group"
                )
            trajectory_seeds = {
                trajectory.metadata["rng_seed"]
                for trajectory in groups[0].trajectories
            }
            trajectory_snapshots = {
                trajectory.metadata["start_snapshot_sha256"]
                for trajectory in groups[0].trajectories
            }
            if len(trajectory_seeds) != 1 or len(trajectory_snapshots) != 1:
                raise BackendSpikePreflightError(
                    "ART Rollout Group members diverged in start randomness"
                )
            training_started = time.monotonic()
            train_result = await backend.train(
                model,
                groups,
                learning_rate=config.training.learning_rate,
                save_checkpoint=True,
            )
            training_seconds = time.monotonic() - training_started
            if train_result.step != 1 or not train_result.checkpoint_path:
                raise BackendSpikePreflightError(
                    "ART did not execute and save optimizer step 1"
                )
            if (
                train_result.metrics.get("data/step_num_groups_trainable", 0.0)
                < 1.0
                or "loss/grad_norm" not in train_result.metrics
            ):
                raise BackendSpikePreflightError(
                    "ART did not observe a trainable Rollout Group and backward pass"
                )
            await model.log(
                groups,
                metrics=train_result.metrics,
                step=train_result.step,
                split="train",
            )
            wandb_run = model._get_wandb_run()
            wandb_url = wandb_run.url if wandb_run is not None else None
            if wandb_run is not None:
                wandb_run.finish()
        finally:
            await backend.close()

        checkpoint = Path(train_result.checkpoint_path)
        adapter = validate_adapter_directory(
            checkpoint,
            expected_base_model=config.model.id,
        )

        # A new backend and model instance must discover step 1 and serve its LoRA.
        resumed_model = art.TrainableModel(
            name=model_name,
            project=config.telemetry.wandb_project,
            entity=entity,
            base_model=config.model.id,
            _internal_config=InternalModelConfig(**_internal_model_config(config)),
        )
        resumed_backend = LocalBackend(in_process=True, path=str(art_directory))
        try:
            await resumed_model.register(resumed_backend)
            resumed_step = await resumed_model.get_step()
            if resumed_step != train_result.step:
                raise BackendSpikePreflightError(
                    "ART checkpoint resume did not preserve the optimizer step"
                )
            resume_episode = Stage2TrainingEpisode(config)
            resume_started = time.monotonic()
            resume_completion = await resumed_model.openai_client().chat.completions.create(
                model=resumed_model.get_inference_name(),
                messages=[
                    {"role": "user", "content": resume_episode.policy_prompt()}
                ],
                max_completion_tokens=config.rollout.max_completion_tokens,
                temperature=0.0,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            resume_seconds = time.monotonic() - resume_started
            resumed_response = resume_completion.choices[0].message.content or ""
            contract = enforce_policy_response(
                variant=config.rollout.variant,
                response=resumed_response,
                truncated=False,
                board_change_actions=change_making_actions(resume_episode.board),
            )
        finally:
            await resumed_backend.close()

    completion_tokens = sum(
        int(trajectory.metrics["completion_tokens"])
        for trajectory in groups[0].trajectories
    )
    environment_steps = [
        int(trajectory.metrics["environment_steps"])
        for trajectory in groups[0].trajectories
    ]
    rewards = [trajectory.reward for trajectory in groups[0].trajectories]
    metrics = {
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "baseline_gpu_used_bytes": gpu_sampler.baseline_used_bytes,
        "peak_gpu_used_bytes": gpu_sampler.peak_used_bytes,
        "peak_gpu_delta_bytes": max(
            0,
            gpu_sampler.peak_used_bytes - gpu_sampler.baseline_used_bytes,
        ),
        "rollout_seconds": rollout_seconds,
        "rollout_completion_tokens": completion_tokens,
        "rollout_tokens_per_second": (
            completion_tokens / rollout_seconds if rollout_seconds else 0.0
        ),
        "optimizer_seconds": training_seconds,
        "optimizer_steps_per_second": (
            train_result.step / training_seconds if training_seconds else 0.0
        ),
        "wall_seconds": time.monotonic() - run_started,
        "resume_generation_seconds": resume_seconds,
        **{f"art/{key}": value for key, value in train_result.metrics.items()},
    }
    writer = SummaryWriter(log_dir=str(tensorboard_directory))
    try:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, train_result.step)
        writer.add_scalar(
            "environment/reward_mean",
            sum(rewards) / len(rewards),
            train_result.step,
        )
        writer.add_scalar(
            "environment/policy_failure_rate",
            sum(
                trajectory.metadata["terminal_reason"] == "policy_failure"
                for trajectory in groups[0].trajectories
            )
            / len(groups[0].trajectories),
            train_result.step,
        )
    finally:
        writer.close()

    result = {
        "schema_version": 1,
        "status": "completed",
        "backend": "art_local",
        "config_sha256": config_sha256,
        "resolved_config": config.resolved(),
        "dependency_versions": versions,
        "compatibility": {
            "art_megatron_registry_import_shim": (
                registry_compatibility_installed
            ),
            "frozen_lora_targets_only": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "source_revision": _source_revision(),
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "rollout_group": {
            **group_evidence,
            "environment_steps": environment_steps,
            "rewards": rewards,
            "all_members_completed_2_to_4_steps": all(
                2 <= steps <= 4 for steps in environment_steps
            ),
        },
        "gradient_update": {
            "optimizer_step": train_result.step,
            "backward_observed": "loss/grad_norm" in train_result.metrics,
            "trainer_metrics": train_result.metrics,
        },
        "checkpoint": {
            **adapter,
            "path": str(checkpoint),
            "resume_step": resumed_step,
            "resumed_response": resumed_response,
            "resumed_response_parsed": contract.parsed,
            "resumed_response_policy_failure": contract.policy_failure,
        },
        "telemetry": {
            "wandb_entity": entity,
            "wandb_project": config.telemetry.wandb_project,
            "wandb_project_access": project_access,
            "wandb_url": wandb_url,
            "model_checkpoint_uploaded": False,
            "tensorboard_directory": str(tensorboard_directory),
        },
        "metrics": metrics,
        "environment_contract": {
            "markov_policy": True,
            "separate_history_per_environment_step": True,
            "policy_failure_terminates_without_retry": True,
            "environment_reward_frozen": True,
        },
        "implementation_complexity": implementation_complexity("art_local"),
    }
    _write_json(output_directory / "result.json", result)
    with suppress(Exception):
        wandb.finish()
    return result


def run_art_backend(
    config_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    config, config_sha256 = BackendSpikeConfig.load(config_path)
    return asyncio.run(_run(config, output_directory, config_sha256))
