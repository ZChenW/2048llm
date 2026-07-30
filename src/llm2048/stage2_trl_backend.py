"""Maintained TRL environment_factory candidate for Stage 2 Environment GRPO."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

from llm2048.stage2_art_backend import _GpuMemorySampler
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


_ACTIVE_CONFIG: BackendSpikeConfig | None = None
_CREATED_ENVIRONMENTS: list["Trl2048Environment"] = []
_TOOL_INSTRUCTION = (
    "\nDo not return the Policy Response as assistant text. You must call "
    "submit_policy_response exactly once, setting response to the exact "
    "Policy Response string required above."
)


class Trl2048Environment:
    """Stateful TRL tool environment backed by the canonical Training Episode."""

    def __init__(self) -> None:
        if _ACTIVE_CONFIG is None:
            raise RuntimeError("TRL environment factory is not configured")
        self._config = _ACTIVE_CONFIG
        self._episode: Stage2TrainingEpisode | None = None
        self.policy_observation_count = 0
        self.rollout_started_at: float | None = None
        self.rollout_completed_at: float | None = None
        _CREATED_ENVIRONMENTS.append(self)

    def reset(self, rng_seed: int | None = None, **_: object) -> str:
        if rng_seed is not None and rng_seed != self._config.start_state.rng_seed:
            raise RuntimeError("TRL reset received an unregistered RNG seed")
        self._episode = Stage2TrainingEpisode(self._config)
        self.policy_observation_count = 1
        self.rollout_started_at = time.monotonic()
        self.rollout_completed_at = None
        return _TOOL_INSTRUCTION

    def submit_policy_response(self, response: str) -> str:
        """Submit one strict Policy Response and return the next board observation.

        Args:
            response: Exactly one canonical ``<action>...</action>`` response.

        Returns:
            The next Markov Policy prompt, or a terminal status string.
        """
        if self._episode is None:
            raise RuntimeError("environment must be reset before a move")
        step = self._episode.submit_policy_response(response)
        if step.terminal:
            return f"Training Episode terminated: {self._episode.terminal_reason}."
        self.policy_observation_count += 1
        return self._episode.policy_prompt() + _TOOL_INSTRUCTION

    def get_reward(self) -> float:
        if self._episode is None:
            raise RuntimeError("environment must be reset before reward")
        if not self._episode.terminal:
            # Stopping without a Policy Response is a strict format failure.
            self._episode.submit_policy_response("")
        self.rollout_completed_at = time.monotonic()
        return self._episode.reward().total

    def _evidence(self) -> dict[str, Any]:
        if self._episode is None:
            return {"reset": False}
        if not self._episode.terminal:
            self.get_reward()
        return {
            **self._episode.resolved(),
            "policy_observation_count": self.policy_observation_count,
            "rollout_seconds": (
                self.rollout_completed_at - self.rollout_started_at
                if self.rollout_started_at is not None
                and self.rollout_completed_at is not None
                else None
            ),
            # TRL appends every tool call/result to the conversation. A second
            # observation therefore has prior history in the model prefix.
            "markov_prefix_preserved": self.policy_observation_count <= 1,
        }


def _run(
    config: BackendSpikeConfig,
    output_directory: Path,
    config_sha256: str,
) -> dict[str, Any]:
    global _ACTIVE_CONFIG
    import torch
    from datasets import Dataset  # type: ignore[import-untyped]
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    import wandb

    versions = validate_stack(config, "trl_environment_factory")
    if "environment_factory" not in inspect.signature(
        GRPOTrainer.__init__
    ).parameters:
        raise BackendSpikePreflightError(
            "installed maintained TRL has no environment_factory"
        )
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_LOG_MODEL"] = "false"
    entity, project_access = verify_private_wandb_project(
        wandb,
        config.telemetry.wandb_project,
    )
    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_PROJECT"] = config.telemetry.wandb_project
    os.environ["WANDB_NAME"] = f"{config.experiment_name}-trl"
    output_directory.mkdir(parents=True, exist_ok=False)
    trainer_directory = output_directory / "trainer"
    tensorboard_directory = output_directory / "tensorboard"
    group_evidence = rollout_group_manifest(config)

    start_episode = Stage2TrainingEpisode(config)
    dataset = Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": start_episode.policy_prompt(),
                    }
                ]
            ],
            "rng_seed": [config.start_state.rng_seed],
        }
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.id,
        revision=config.model.revision,
    )
    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    arguments = GRPOConfig(  # type: ignore[call-arg]
        output_dir=str(trainer_directory),
        run_name=f"{config.experiment_name}-trl",
        max_steps=config.training.optimizer_steps,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=config.rollout.group_size,
        generation_batch_size=config.rollout.group_size,
        max_completion_length=(
            config.rollout.horizon * config.rollout.max_completion_tokens
        ),
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
        top_k=config.rollout.top_k,
        max_tool_calling_iterations=config.rollout.horizon,
        beta=0.0,
        bf16=True,
        use_vllm=False,
        save_strategy="steps",
        save_steps=1,
        save_only_model=False,
        logging_steps=1,
        logging_first_step=True,
        report_to=["wandb", "tensorboard"],
        logging_dir=str(tensorboard_directory),
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        num_completions_to_print=config.rollout.group_size,
        seed=config.seed,
        data_seed=config.seed,
    )
    _ACTIVE_CONFIG = config
    _CREATED_ENVIRONMENTS.clear()
    trainer = GRPOTrainer(  # type: ignore[call-arg]
        model=model,
        args=arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        environment_factory=Trl2048Environment,
    )
    run_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with _GpuMemorySampler() as gpu_sampler:
        train_result = trainer.train()
        wall_seconds = time.monotonic() - run_started
    training_logs = list(trainer.state.log_history)
    backward_observed = any(
        "loss" in entry and "grad_norm" in entry
        for entry in training_logs
    )
    if train_result.global_step != 1 or not backward_observed:
        raise BackendSpikePreflightError(
            "TRL did not record a real backward and optimizer step"
        )
    checkpoint = trainer_directory / "checkpoint-1"
    if not checkpoint.is_dir():
        raise BackendSpikePreflightError("TRL did not save checkpoint-1")
    adapter = validate_adapter_directory(
        checkpoint,
        expected_base_model=config.model.id,
    )
    optimizer_state = checkpoint / "optimizer.pt"
    trainer_state = checkpoint / "trainer_state.json"
    if not optimizer_state.is_file() or not trainer_state.is_file():
        raise BackendSpikePreflightError(
            "TRL checkpoint does not contain optimizer and trainer resume state"
        )
    run = wandb.run
    wandb_url = run.url if run is not None else None
    if run is not None:
        run.finish()

    # Load both model and optimizer/trainer state through the maintained resume API.
    resumed_base = AutoModelForCausalLM.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    resumed_model = PeftModel.from_pretrained(
        resumed_base,
        str(checkpoint),
        is_trainable=True,
    )
    resume_arguments = replace(
        arguments,
        output_dir=str(output_directory / "resume-trainer"),
        max_steps=-1,
        num_train_epochs=0.0,
        steps_per_generation=None,
        report_to=[],
        run_name=f"{config.experiment_name}-trl-resume",
    )

    resumed_trainer = GRPOTrainer(  # type: ignore[call-arg,arg-type]
        model=resumed_model,  # type: ignore[arg-type]
        args=resume_arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=Trl2048Environment,
    )
    resume_result = resumed_trainer.train(resume_from_checkpoint=str(checkpoint))
    if resume_result.global_step != 1:
        raise BackendSpikePreflightError(
            "TRL resume did not preserve global optimizer step 1"
        )

    environments = [
        environment._evidence() for environment in _CREATED_ENVIRONMENTS
    ]
    completed = [
        environment
        for environment in environments
        if environment.get("terminal_reason") is not None
    ]
    steps = [len(environment.get("steps", [])) for environment in completed]
    rewards = [
        float(environment["reward"]["total"])
        for environment in completed
        if environment.get("reward") is not None
    ]
    snapshot_hashes = {
        environment["start_snapshot_sha256"]
        for environment in completed
        if "start_snapshot_sha256" in environment
    }
    rng_seeds = {
        environment["rng_seed"]
        for environment in completed
        if "rng_seed" in environment
    }
    history_breach = any(
        not bool(environment.get("markov_prefix_preserved", True))
        for environment in completed
    )
    all_members_completed_2_to_4_steps = (
        len(steps) >= config.rollout.group_size
        and all(2 <= value <= 4 for value in steps)
    )
    rollout_contract_failure = not all_members_completed_2_to_4_steps
    rollout_seconds_samples = [
        float(environment["rollout_seconds"])
        for environment in completed
        if environment.get("rollout_seconds") is not None
    ]
    last_step_log = next(
        (
            entry
            for entry in reversed(training_logs)
            if "completions/mean_length" in entry
        ),
        {},
    )
    completion_tokens = (
        float(last_step_log.get("completions/mean_length", 0.0))
        * config.rollout.group_size
    )
    rollout_seconds = (
        max(rollout_seconds_samples) if rollout_seconds_samples else 0.0
    )
    optimizer_seconds = float(last_step_log.get("step_time", 0.0))
    metrics = {
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "baseline_gpu_used_bytes": gpu_sampler.baseline_used_bytes,
        "peak_gpu_used_bytes": gpu_sampler.peak_used_bytes,
        "peak_gpu_delta_bytes": max(
            0,
            gpu_sampler.peak_used_bytes - gpu_sampler.baseline_used_bytes,
        ),
        "wall_seconds": wall_seconds,
        "rollout_seconds": rollout_seconds,
        "rollout_completion_tokens": completion_tokens,
        "rollout_tokens_per_second": (
            completion_tokens / rollout_seconds if rollout_seconds else 0.0
        ),
        "optimizer_seconds": optimizer_seconds,
        "optimizer_steps_per_second": (
            train_result.global_step / optimizer_seconds
            if optimizer_seconds
            else 0.0
        ),
        "train_runtime": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_steps_per_second": float(
            train_result.metrics.get("train_steps_per_second", 0.0)
        ),
        "train_samples_per_second": float(
            train_result.metrics.get("train_samples_per_second", 0.0)
        ),
    }
    result = {
        "schema_version": 1,
        "status": "completed_with_contract_failure"
        if history_breach or rollout_contract_failure
        else "completed",
        "backend": "trl_environment_factory",
        "config_sha256": config_sha256,
        "resolved_config": config.resolved(),
        "dependency_versions": versions,
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
            "observed_start_snapshot_sha256": sorted(snapshot_hashes),
            "observed_rng_seeds": sorted(rng_seeds),
            "environment_steps": steps,
            "rewards": rewards,
            "all_members_completed_2_to_4_steps": (
                all_members_completed_2_to_4_steps
            ),
            "rollout_contract_failure": rollout_contract_failure,
        },
        "gradient_update": {
            "optimizer_step": int(train_result.global_step),
            "backward_observed": backward_observed,
            "trainer_metrics": train_result.metrics,
            "trainer_log_history": training_logs,
        },
        "checkpoint": {
            **adapter,
            "path": str(checkpoint),
            "optimizer_state": str(optimizer_state),
            "trainer_state": str(trainer_state),
            "resume_step": int(resume_result.global_step),
            "resume_probe_stopped_before_new_training": True,
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
            "environment_factory_experimental": True,
            "markov_policy": not history_breach,
            "multi_turn_markov_policy_demonstrated": (
                not history_breach and not rollout_contract_failure
            ),
            "history_retained_by_native_tool_loop": history_breach,
            "tool_interface_suffix_only": True,
            "policy_failure_terminates_without_retry": True,
            "environment_reward_frozen": True,
        },
        "implementation_complexity": implementation_complexity(
            "trl_environment_factory"
        ),
    }
    _write_json(output_directory / "result.json", result)
    return result


def run_trl_backend(
    config_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    config, config_sha256 = BackendSpikeConfig.load(config_path)
    return _run(config, output_directory, config_sha256)
