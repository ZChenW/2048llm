"""Real Unsloth and TRL Teacher-guided GRPO feasibility smoke path."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Mapping, cast

from llm2048.policy_contracts import (
    Action,
    PolicyVariant,
    build_policy_prompt,
    change_making_actions,
    enforce_policy_response,
)
from llm2048.teacher_guided_rewards import (
    TeacherGuidedCompletion,
    teacher_guided_reward_callback,
)


OFFICIAL_QWEN35_4B = "Qwen/Qwen3.5-4B"
MODEL_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


class GrpoSmokeConfigurationError(ValueError):
    """Raised when the real GRPO smoke configuration is not safe and explicit."""


class GrpoSmokePreflightError(RuntimeError):
    """Raised before model loading when real-run prerequisites are absent."""


@dataclass(frozen=True)
class RuntimeStack:
    torch: Any
    wandb: Any
    dataset_class: Any
    fast_vision_model: Any
    snapshot_download: Any
    peft_model_class: Any
    grpo_config_class: Any
    grpo_trainer_class: Any
    versions: dict[str, str]
    api_signatures: dict[str, str]


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
class GrpoConfig:
    trainer: str
    loss_type: str
    num_generations: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    generation_batch_size: int
    max_steps: int
    learning_rate: float
    optimizer: str
    max_prompt_length: int
    max_completion_length: int
    temperature: float
    beta: float
    mask_truncated_completions: bool
    safe_group8_reserved_headroom_gib: float
    use_vllm: bool

    def resolved(self) -> dict[str, Any]:
        return {
            "trainer": self.trainer,
            "loss_type": self.loss_type,
            "num_generations": self.num_generations,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "generation_batch_size": self.generation_batch_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
            "temperature": self.temperature,
            "beta": self.beta,
            "mask_truncated_completions": self.mask_truncated_completions,
            "safe_group8_reserved_headroom_gib": (
                self.safe_group8_reserved_headroom_gib
            ),
            "use_vllm": self.use_vllm,
        }


@dataclass(frozen=True)
class TeacherGuidedExample:
    variant: PolicyVariant
    board: list[list[int]]
    teacher_action_scores: dict[Action, float | None]
    teacher_action: Action
    teacher_margin_scale: float

    def resolved(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "board": self.board,
            "teacher_action_scores": self.teacher_action_scores,
            "teacher_action": self.teacher_action,
            "teacher_margin_scale": self.teacher_margin_scale,
        }


@dataclass(frozen=True)
class SmokeTelemetryConfig:
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
class GrpoSmokeConfig:
    schema_version: int
    experiment_name: str
    seed: int
    model: ModelConfig
    lora: LoraConfig
    grpo: GrpoConfig
    teacher_guided_example: TeacherGuidedExample
    telemetry: SmokeTelemetryConfig

    @classmethod
    def load(cls, path: Path) -> tuple["GrpoSmokeConfig", str]:
        try:
            raw_bytes = path.read_bytes()
        except OSError as error:
            raise GrpoSmokeConfigurationError(
                f"cannot read GRPO smoke configuration: {error}"
            ) from error
        try:
            raw = json.loads(raw_bytes)
        except json.JSONDecodeError as error:
            raise GrpoSmokeConfigurationError(
                f"GRPO smoke configuration is not valid JSON: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise GrpoSmokeConfigurationError(
                "GRPO smoke configuration must be a JSON object"
            )
        _require_exact_keys(
            raw,
            {
                "schema_version",
                "experiment_name",
                "seed",
                "model",
                "lora",
                "grpo",
                "teacher_guided_example",
                "telemetry",
            },
            "GRPO smoke configuration",
        )
        if raw["schema_version"] != 1:
            raise GrpoSmokeConfigurationError("schema_version must be 1")
        experiment_name = _nonempty_string(
            raw["experiment_name"], "experiment_name"
        )
        seed = _integer(raw["seed"], "seed")
        model = _load_model(_object(raw["model"], "model"))
        lora = _load_lora(_object(raw["lora"], "lora"))
        grpo = _load_grpo(_object(raw["grpo"], "grpo"))
        example = _load_teacher_guided_example(
            _object(raw["teacher_guided_example"], "teacher_guided_example")
        )
        telemetry = _load_telemetry(_object(raw["telemetry"], "telemetry"))
        if model.max_sequence_length <= grpo.max_completion_length:
            raise GrpoSmokeConfigurationError(
                "model.max_sequence_length must leave room for the prompt"
            )
        return (
            cls(
                schema_version=1,
                experiment_name=experiment_name,
                seed=seed,
                model=model,
                lora=lora,
                grpo=grpo,
                teacher_guided_example=example,
                telemetry=telemetry,
            ),
            sha256(raw_bytes).hexdigest(),
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "model": self.model.resolved(),
            "lora": self.lora.resolved(),
            "grpo": self.grpo.resolved(),
            "teacher_guided_example": self.teacher_guided_example.resolved(),
            "telemetry": self.telemetry.resolved(),
        }


def preflight_real_smoke(
    config: GrpoSmokeConfig,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fail closed before creating artifacts, importing the stack, or loading."""
    environment = os.environ if environ is None else environ
    if not environment.get("WANDB_API_KEY", "").strip():
        raise GrpoSmokePreflightError(
            "WANDB_API_KEY is required for the private online W&B smoke run"
        )
    wandb_mode = environment.get("WANDB_MODE")
    if wandb_mode is not None and wandb_mode != "online":
        raise GrpoSmokePreflightError(
            "WANDB_MODE must be 'online' for the real GRPO smoke run"
        )
    wandb_log_model = environment.get("WANDB_LOG_MODEL")
    if wandb_log_model is not None and wandb_log_model.lower() != "false":
        raise GrpoSmokePreflightError(
            "WANDB_LOG_MODEL must be 'false' so checkpoints are not uploaded"
        )


def run_real_smoke(
    *,
    config: GrpoSmokeConfig,
    input_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Execute the real one-step feasibility run through maintained GRPOTrainer."""
    preflight_real_smoke(config)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise GrpoSmokePreflightError(
            "output directory must be absent or empty for a GRPO smoke run"
        )
    stack = _load_runtime_stack()
    _preflight_cuda(stack)

    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "resolved_config.json", config.resolved())
    telemetry_directory = output_directory / "telemetry"
    telemetry_directory.mkdir(parents=True, exist_ok=True)
    adapter_directory = output_directory / "adapter"
    failure_path = output_directory / "failure.json"

    _configure_wandb_environment(config, telemetry_directory)
    wandb_run: Any | None = None
    wandb_project_access: str | None = None
    model: Any | None = None
    processor: Any | None = None
    training_started = time.perf_counter()
    phase_results: list[dict[str, Any]] = []
    group8_result: dict[str, Any] = {
        "status": "not_considered",
        "reason": "group 4 must complete before headroom is evaluated",
    }
    try:
        wandb_run, wandb_project_access = _start_online_wandb_run(
            stack=stack,
            config=config,
            telemetry_directory=telemetry_directory,
        )
        stack.torch.cuda.reset_peak_memory_stats()
        model, processor = _load_bf16_model(stack, config)
        trainable_evidence = _trainable_parameter_evidence(model)
        if trainable_evidence["vision_trainable_parameter_names"]:
            raise RuntimeError(
                "Unsloth exposed trainable vision parameters despite the "
                "language-only LoRA configuration"
            )
        if trainable_evidence["lora_trainable_parameters"] <= 0:
            raise RuntimeError("Unsloth did not expose trainable LoRA parameters")

        train_dataset = stack.dataset_class.from_list(
            [
                _training_row(
                    config.teacher_guided_example,
                    processor,
                    config.grpo.max_prompt_length,
                )
            ]
        )
        group4_result = _train_one_group(
            stack=stack,
            config=config,
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            output_directory=output_directory,
            group_size=4,
        )
        phase_results.append(group4_result)

        device_total = stack.torch.cuda.get_device_properties(0).total_memory
        group4_headroom = device_total - group4_result["peak_reserved_vram_bytes"]
        safe_headroom = int(
            config.grpo.safe_group8_reserved_headroom_gib * 1024**3
        )
        if group4_headroom >= safe_headroom:
            try:
                group8_result = _train_one_group(
                    stack=stack,
                    config=config,
                    model=model,
                    processor=processor,
                    train_dataset=train_dataset,
                    output_directory=output_directory,
                    group_size=8,
                )
                phase_results.append(group8_result)
            except Exception as error:
                group8_result = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": _safe_error(error),
                    "peak_allocated_vram_bytes": (
                        stack.torch.cuda.max_memory_allocated()
                    ),
                    "peak_reserved_vram_bytes": (
                        stack.torch.cuda.max_memory_reserved()
                    ),
                }
                stack.torch.cuda.empty_cache()
        else:
            group8_result = {
                "status": "not_attempted",
                "reason": (
                    "group 4 reserved-VRAM headroom was below the safety "
                    "threshold"
                ),
                "reserved_headroom_bytes": group4_headroom,
                "required_reserved_headroom_bytes": safe_headroom,
            }

        model.save_pretrained(adapter_directory)
        processor.save_pretrained(adapter_directory)
        _write_json(
            adapter_directory / "llm2048_base_model.json",
            {
                "id": config.model.id,
                "revision": config.model.revision,
                "precision": config.model.precision,
            },
        )
        _assert_adapter_only_checkpoint(adapter_directory)

        del model
        model = None
        del processor
        processor = None
        gc.collect()
        stack.torch.cuda.empty_cache()
        reload_result = _reload_and_validate_policy_response(
            stack=stack,
            config=config,
            adapter_directory=adapter_directory,
        )
        if not reload_result["policy_contract"]["valid_action"]:
            raise RuntimeError(
                "fresh adapter reload did not produce a valid Policy Response"
            )

        selected_group_size = (
            8 if group8_result.get("status") == "completed" else 4
        )
        telemetry_results = [*phase_results, group8_result, reload_result]
        peak_allocated = max(
            result["peak_allocated_vram_bytes"]
            for result in telemetry_results
            if "peak_allocated_vram_bytes" in result
        )
        peak_reserved = max(
            result["peak_reserved_vram_bytes"]
            for result in telemetry_results
            if "peak_reserved_vram_bytes" in result
        )
        result = {
            "schema_version": 1,
            "status": "completed",
            "decision": {
                "go": True,
                "precision": "bf16",
                "teacher_guided_group_size": selected_group_size,
                "quantized_fallback_used": False,
            },
            "model_evidence": trainable_evidence,
            "phases": {
                "group_4": group4_result,
                "group_8": group8_result,
                "adapter_reload": reload_result,
            },
            "telemetry": {
                "peak_allocated_vram_bytes": peak_allocated,
                "peak_reserved_vram_bytes": peak_reserved,
                "wall_seconds": time.perf_counter() - training_started,
                "wandb": {
                    "mode": "online",
                    "entity": wandb_run.entity,
                    "project_access": wandb_project_access,
                    "run_id": wandb_run.id,
                    "url": wandb_run.url,
                    "checkpoints_uploaded": False,
                },
            },
            "versions": stack.versions,
        }
        wandb_run.log(
            {
                "feasibility/peak_allocated_vram_bytes": peak_allocated,
                "feasibility/peak_reserved_vram_bytes": peak_reserved,
                "feasibility/selected_group_size": selected_group_size,
                "feasibility/group_8_completed": (
                    1.0
                    if group8_result.get("status") == "completed"
                    else 0.0
                ),
                "feasibility/reload_valid_policy_response": 1.0,
            }
        )
        _write_json(output_directory / "result.json", result)
        manifest = _build_manifest(
            config=config,
            input_sha256=input_sha256,
            output_directory=output_directory,
            result=result,
            stack=stack,
        )
        _write_json(output_directory / "manifest.json", manifest)
        wandb_run.finish(exit_code=0)
        wandb_run = None
        return result
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "decision": {
                "go": False,
                "precision_attempted": "bf16",
                "quantized_fallback_used": False,
            },
            "error_type": type(error).__name__,
            "error": _safe_error(error),
            "completed_phases": phase_results,
            "group_8": group8_result,
            "telemetry": _failure_telemetry(stack, training_started),
            "versions": stack.versions,
        }
        _write_json(failure_path, failure)
        _write_json(output_directory / "result.json", failure)
        if wandb_run is not None:
            try:
                wandb_run.log({"feasibility/completed": 0.0})
                wandb_run.finish(exit_code=1)
            except Exception:
                pass
            wandb_run = None
        raise RuntimeError(
            "BF16 GRPO smoke failed; review "
            f"{failure_path} before considering a quantized fallback"
        ) from error
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except Exception:
                pass


def _load_runtime_stack() -> RuntimeStack:
    try:
        unsloth = import_module("unsloth")
        torch = import_module("torch")
        wandb = import_module("wandb")
        datasets = import_module("datasets")
        huggingface_hub = import_module("huggingface_hub")
        peft = import_module("peft")
        trl = import_module("trl")
    except ImportError as error:
        raise GrpoSmokePreflightError(
            "the real smoke requires torch, Unsloth Core, datasets, TRL, and W&B"
        ) from error

    try:
        fast_vision_model = unsloth.FastVisionModel
        dataset_class = datasets.Dataset
        snapshot_download = huggingface_hub.snapshot_download
        peft_model_class = peft.PeftModel
        grpo_config_class = trl.GRPOConfig
        grpo_trainer_class = trl.GRPOTrainer
    except AttributeError as error:
        raise GrpoSmokePreflightError(
            "the installed maintained stack does not expose the required "
            "FastVisionModel and GRPOTrainer APIs"
        ) from error

    api_callables = {
        "FastVisionModel.from_pretrained": fast_vision_model.from_pretrained,
        "FastVisionModel.get_peft_model": fast_vision_model.get_peft_model,
        "FastVisionModel.for_inference": fast_vision_model.for_inference,
        "PeftModel.from_pretrained": peft_model_class.from_pretrained,
        "GRPOConfig": grpo_config_class,
        "GRPOTrainer": grpo_trainer_class,
    }
    api_signatures = {
        name: _callable_signature(function)
        for name, function in api_callables.items()
    }
    for name, function, required_keywords in (
        (
            "FastVisionModel.from_pretrained",
            fast_vision_model.from_pretrained,
            {
                "model_name",
                "max_seq_length",
                "load_in_4bit",
                "fast_inference",
            },
        ),
        (
            "FastVisionModel.get_peft_model",
            fast_vision_model.get_peft_model,
            {
                "finetune_vision_layers",
                "finetune_language_layers",
                "finetune_attention_modules",
                "finetune_mlp_modules",
                "r",
                "lora_alpha",
            },
        ),
        (
            "PeftModel.from_pretrained",
            peft_model_class.from_pretrained,
            {"model", "model_id", "is_trainable"},
        ),
        (
            "GRPOTrainer",
            grpo_trainer_class,
            {
                "model",
                "args",
                "processing_class",
                "reward_funcs",
                "train_dataset",
            },
        ),
    ):
        _require_keywords(function, required_keywords, name)

    return RuntimeStack(
        torch=torch,
        wandb=wandb,
        dataset_class=dataset_class,
        fast_vision_model=fast_vision_model,
        snapshot_download=snapshot_download,
        peft_model_class=peft_model_class,
        grpo_config_class=grpo_config_class,
        grpo_trainer_class=grpo_trainer_class,
        versions={
            package: _installed_version(package)
            for package in (
                "torch",
                "unsloth",
                "unsloth_zoo",
                "trl",
                "transformers",
                "datasets",
                "huggingface_hub",
                "peft",
                "wandb",
                "tensorboard",
            )
        },
        api_signatures=api_signatures,
    )


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


def _require_keywords(
    function: Any, required_keywords: set[str], name: str
) -> None:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as error:
        raise GrpoSmokePreflightError(
            f"cannot inspect the installed {name} API"
        ) from error
    supports_var_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    missing = required_keywords - set(signature.parameters)
    if missing and not supports_var_keywords:
        raise GrpoSmokePreflightError(
            f"installed {name} API is missing required parameters: "
            f"{', '.join(sorted(missing))}"
        )


def _supports_keyword(function: Any, keyword: str) -> bool:
    signature = inspect.signature(function)
    return keyword in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _preflight_cuda(stack: RuntimeStack) -> None:
    if not stack.torch.cuda.is_available():
        raise GrpoSmokePreflightError(
            "a CUDA GPU is required for the real BF16 GRPO smoke run"
        )
    if not stack.torch.cuda.is_bf16_supported():
        raise GrpoSmokePreflightError(
            "the active CUDA GPU does not support BF16"
        )


def _configure_wandb_environment(
    config: GrpoSmokeConfig, telemetry_directory: Path
) -> None:
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_PROJECT"] = config.telemetry.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_DIR"] = str(telemetry_directory / "wandb")


def _start_online_wandb_run(
    *,
    stack: RuntimeStack,
    config: GrpoSmokeConfig,
    telemetry_directory: Path,
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
    project_access = _verify_private_wandb_project(
        api=api,
        entity=entity,
        project=config.telemetry.wandb_project,
    )
    run = stack.wandb.init(
        entity=entity,
        project=config.telemetry.wandb_project,
        name=config.experiment_name,
        config=config.resolved(),
        dir=str(telemetry_directory / "wandb"),
        mode="online",
        tags=["issue-7", "bf16", "grpo-feasibility", "private"],
        save_code=False,
        reinit=True,
    )
    if run is None:
        raise RuntimeError("W&B did not create an online run")
    run_mode = getattr(getattr(run, "settings", None), "mode", None)
    if run_mode != "online":
        run.finish(exit_code=1)
        raise RuntimeError(
            "W&B did not honor the required online telemetry mode"
        )
    if not isinstance(getattr(run, "entity", None), str) or not run.entity:
        run.finish(exit_code=1)
        raise RuntimeError("W&B did not resolve an authenticated default entity")
    if run.entity != entity or run.project != config.telemetry.wandb_project:
        run.finish(exit_code=1)
        raise RuntimeError("W&B run target differs from the verified private project")
    return run, project_access


def _verify_private_wandb_project(
    *,
    api: Any,
    entity: str,
    project: str,
) -> str:
    query = """
    query ProjectAccess($entity: String!, $project: String!) {
      project(entityName: $entity, name: $project) {
        access
      }
    }
    """
    response = api._service_api.execute_graphql(  # noqa: SLF001
        query,
        {"entity": entity, "project": project},
    )
    project_response = (
        response.get("project") if isinstance(response, dict) else None
    )
    access = (
        project_response.get("access")
        if isinstance(project_response, dict)
        else None
    )
    if access not in {"PRIVATE", "TEAM", "RESTRICTED"}:
        raise RuntimeError(
            "W&B project is not private; expected PRIVATE, TEAM, or RESTRICTED "
            f"access but received {access!r}"
        )
    return cast(str, access)


def _load_bf16_model(
    stack: RuntimeStack, config: GrpoSmokeConfig
) -> tuple[Any, Any]:
    loader = stack.fast_vision_model.from_pretrained
    snapshot_path = _exact_snapshot_path(stack, config)
    loader_arguments = {
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
    _require_keywords(loader, set(loader_arguments), "FastVisionModel.from_pretrained")
    model, processor = loader(**loader_arguments)
    _ensure_generation_architecture(model)
    peft_loader = stack.fast_vision_model.get_peft_model
    peft_arguments = {
        "model": model,
        "finetune_vision_layers": False,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "bias": "none",
        "random_state": config.seed,
        "use_rslora": False,
        "loftq_config": None,
        "use_gradient_checkpointing": config.lora.gradient_checkpointing,
    }
    _require_keywords(
        peft_loader,
        set(peft_arguments),
        "FastVisionModel.get_peft_model",
    )
    model = peft_loader(**peft_arguments)
    return model, processor


def _ensure_generation_architecture(model: Any) -> None:
    architectures = getattr(model.config, "architectures", None)
    if architectures:
        return
    model.config.architectures = [type(model).__name__]


def _exact_snapshot_path(
    stack: RuntimeStack,
    config: GrpoSmokeConfig,
) -> str:
    try:
        path = stack.snapshot_download(
            repo_id=config.model.id,
            revision=config.model.revision,
            local_files_only=True,
        )
    except Exception as error:
        raise GrpoSmokePreflightError(
            "the exact Qwen model revision is not complete in the local "
            "Hugging Face cache; run the documented hf download command"
        ) from error
    return str(path)


def _trainable_parameter_evidence(model: Any) -> dict[str, Any]:
    trainable_names: list[str] = []
    vision_parameter_names: list[str] = []
    vision_names: list[str] = []
    total = 0
    lora_total = 0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(marker in lowered for marker in ("visual", "vision", "image")):
            vision_parameter_names.append(name)
        if not parameter.requires_grad:
            continue
        trainable_names.append(name)
        parameter_count = int(parameter.numel())
        total += parameter_count
        if "lora_" in name.lower():
            lora_total += parameter_count
        if any(marker in lowered for marker in ("visual", "vision", "image")):
            vision_names.append(name)
    return {
        "trainable_parameters": total,
        "lora_trainable_parameters": lora_total,
        "trainable_parameter_names": trainable_names,
        "vision_parameter_names": vision_parameter_names,
        "vision_modules_loaded": bool(vision_parameter_names),
        "vision_trainable_parameter_names": vision_names,
    }


def _policy_messages(example: TeacherGuidedExample) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_policy_prompt(
                        example.variant,
                        example.board,
                    ),
                }
            ],
        }
    ]


def _training_row(
    example: TeacherGuidedExample,
    processor: Any,
    max_prompt_length: int,
) -> dict[str, str]:
    prompt = processor.apply_chat_template(
        _policy_messages(example),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str):
        raise RuntimeError("Qwen processor did not render a text training prompt")
    encoded = processor(prompt, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    prompt_tokens = len(input_ids)
    if prompt_tokens > max_prompt_length:
        raise GrpoSmokePreflightError(
            "rendered Qwen prompt exceeds the configured token budget: "
            f"{prompt_tokens} > {max_prompt_length}"
        )
    return {"prompt": prompt}


def _train_one_group(
    *,
    stack: RuntimeStack,
    config: GrpoSmokeConfig,
    model: Any,
    processor: Any,
    train_dataset: Any,
    output_directory: Path,
    group_size: int,
) -> dict[str, Any]:
    reward_events: list[dict[str, Any]] = []
    train_started = time.perf_counter()
    first_reward_at: float | None = None
    generated_tokens = 0

    def action_quality_reward(
        completions: list[Any],
        completion_ids: list[Any] | None = None,
        **_: Any,
    ) -> list[float]:
        nonlocal first_reward_at, generated_tokens
        completion_texts = [_completion_text(value) for value in completions]
        completion_id_rows = (
            completion_ids
            if completion_ids is not None
            else [None] * len(completion_texts)
        )
        lengths = [
            len(row) if row is not None else 0 for row in completion_id_rows
        ]
        generated_tokens += sum(lengths)
        if first_reward_at is None:
            first_reward_at = time.perf_counter()
        scores = teacher_guided_reward_callback(
            board=config.teacher_guided_example.board,
            completions=[
                TeacherGuidedCompletion(
                    variant=config.teacher_guided_example.variant,
                    response=text,
                    truncated=length >= config.grpo.max_completion_length,
                )
                for text, length in zip(completion_texts, lengths)
            ],
            group_size=len(completion_texts),
            teacher_action_scores=(
                config.teacher_guided_example.teacher_action_scores
            ),
            teacher_action=config.teacher_guided_example.teacher_action,
            tau=config.teacher_guided_example.teacher_margin_scale,
        )
        for text, length, score in zip(completion_texts, lengths, scores):
            reward_events.append(
                {
                    "response": text,
                    "response_length_tokens": length,
                    "reward": score.total,
                    "reward_components": score.components.resolved(),
                    "action": score.contract.action,
                    "valid_action": score.contract.valid_action,
                    "policy_failure_reason": (
                        score.contract.policy_failure_reason
                    ),
                    "regret": score.regret,
                }
            )
        return [score.total for score in scores]

    os.environ["TENSORBOARD_LOGGING_DIR"] = str(
        output_directory
        / "telemetry"
        / "tensorboard"
        / f"group-{group_size}"
    )
    grpo_arguments = _grpo_arguments(
        stack=stack,
        config=config,
        output_directory=output_directory,
        group_size=group_size,
    )
    training_args = stack.grpo_config_class(**grpo_arguments)
    trainer = stack.grpo_trainer_class(
        model=model,
        args=training_args,
        processing_class=processor,
        reward_funcs=action_quality_reward,
        train_dataset=train_dataset,
    )
    stack.torch.cuda.empty_cache()
    stack.torch.cuda.reset_peak_memory_stats()
    train_output = trainer.train()
    wall_seconds = time.perf_counter() - train_started
    global_step = int(trainer.state.global_step)
    if global_step != 1:
        raise RuntimeError(
            f"GRPOTrainer completed {global_step} optimizer steps instead of 1"
        )
    if not reward_events:
        raise RuntimeError("GRPOTrainer did not project Action Quality Reward")
    peak_allocated = stack.torch.cuda.max_memory_allocated()
    peak_reserved = stack.torch.cuda.max_memory_reserved()
    generation_seconds = (
        first_reward_at - train_started
        if first_reward_at is not None
        else 0.0
    )
    metrics = {
        key: _json_scalar(value)
        for key, value in dict(train_output.metrics).items()
    }
    reward_path = output_directory / f"group-{group_size}-reward-events.jsonl"
    for event in reward_events:
        _append_json_line(reward_path, event)
    result = {
        "status": "completed",
        "group_size": group_size,
        "optimizer_steps": global_step,
        "generated_tokens": generated_tokens,
        "generation_seconds_until_reward": generation_seconds,
        "generation_tokens_per_second": (
            generated_tokens / generation_seconds
            if generation_seconds > 0
            else None
        ),
        "training_wall_seconds": wall_seconds,
        "training_steps_per_second": metrics.get("train_steps_per_second"),
        "training_samples_per_second": metrics.get(
            "train_samples_per_second"
        ),
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "trainer_metrics": metrics,
        "reward_events": len(reward_events),
    }
    del trainer
    gc.collect()
    stack.torch.cuda.empty_cache()
    return result


def _grpo_arguments(
    *,
    stack: RuntimeStack,
    config: GrpoSmokeConfig,
    output_directory: Path,
    group_size: int,
) -> dict[str, Any]:
    group_output = output_directory / f"group-{group_size}-trainer"
    arguments: dict[str, Any] = {
        "output_dir": str(group_output),
        "learning_rate": config.grpo.learning_rate,
        "optim": config.grpo.optimizer,
        "per_device_train_batch_size": (
            config.grpo.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": group_size,
        "num_generations": group_size,
        "generation_batch_size": group_size,
        "max_completion_length": config.grpo.max_completion_length,
        "max_steps": config.grpo.max_steps,
        "temperature": config.grpo.temperature,
        "beta": config.grpo.beta,
        "mask_truncated_completions": (
            config.grpo.mask_truncated_completions
        ),
        "loss_type": config.grpo.loss_type,
        "use_vllm": False,
        "bf16": True,
        "fp16": False,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "no",
        "report_to": ["wandb", "tensorboard"],
        "run_name": f"{config.experiment_name}-group-{group_size}",
        "remove_unused_columns": False,
        "push_to_hub": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "disable_tqdm": True,
    }
    config_class = stack.grpo_config_class
    _require_keywords(
        config_class,
        set(arguments) - {"generation_batch_size"},
        "GRPOConfig",
    )
    if not _supports_keyword(config_class, "generation_batch_size"):
        raise GrpoSmokePreflightError(
            "installed GRPOConfig lacks generation_batch_size; refusing to "
            "guess group scheduling semantics"
        )
    if _supports_keyword(config_class, "max_prompt_length"):
        arguments["max_prompt_length"] = config.grpo.max_prompt_length
    return arguments


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        final_message = completion[-1]
        if isinstance(final_message, dict):
            content = final_message.get("content")
            if isinstance(content, str):
                return content
    raise RuntimeError("GRPOTrainer returned an unsupported completion format")


def _assert_adapter_only_checkpoint(adapter_directory: Path) -> None:
    adapter_config = adapter_directory / "adapter_config.json"
    adapter_weights = list(adapter_directory.glob("adapter_model.*"))
    if not adapter_config.is_file() or not adapter_weights:
        raise RuntimeError("the saved checkpoint does not contain a LoRA adapter")
    prohibited = [
        path.name
        for path in adapter_directory.iterdir()
        if path.name == "model.safetensors"
        or path.name.startswith("model-")
        or path.name.startswith("pytorch_model")
    ]
    if prohibited:
        raise RuntimeError(
            "the local save unexpectedly contains full model checkpoints: "
            + ", ".join(sorted(prohibited))
        )


def _load_exact_adapter(
    *,
    stack: RuntimeStack,
    config: GrpoSmokeConfig,
    adapter_directory: Path,
) -> tuple[Any, Any]:
    loader = stack.fast_vision_model.from_pretrained
    snapshot_path = _exact_snapshot_path(stack, config)
    reload_arguments = {
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
        loader,
        set(reload_arguments),
        "FastVisionModel.from_pretrained",
    )
    model, processor = loader(**reload_arguments)
    _ensure_generation_architecture(model)
    adapter_loader = stack.peft_model_class.from_pretrained
    adapter_arguments = {
        "model": model,
        "model_id": str(adapter_directory),
        "is_trainable": False,
    }
    _require_keywords(
        adapter_loader,
        set(adapter_arguments),
        "PeftModel.from_pretrained",
    )
    return adapter_loader(**adapter_arguments), processor


def _reload_and_validate_policy_response(
    *,
    stack: RuntimeStack,
    config: GrpoSmokeConfig,
    adapter_directory: Path,
) -> dict[str, Any]:
    stack.torch.cuda.reset_peak_memory_stats()
    reload_started = time.perf_counter()
    model, processor = _load_exact_adapter(
        stack=stack,
        config=config,
        adapter_directory=adapter_directory,
    )
    stack.fast_vision_model.for_inference(model)
    messages = _policy_messages(config.teacher_guided_example)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    generation_started = time.perf_counter()
    with stack.torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=config.grpo.max_completion_length,
            do_sample=False,
            use_cache=True,
        )
    generation_seconds = time.perf_counter() - generation_started
    completion_ids = generated[:, prompt_tokens:]
    generated_tokens = int(completion_ids.shape[-1])
    response = processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
    )[0]
    contract = enforce_policy_response(
        variant=config.teacher_guided_example.variant,
        response=response,
        truncated=generated_tokens >= config.grpo.max_completion_length,
        board_change_actions=change_making_actions(
            config.teacher_guided_example.board
        ),
    )
    result = {
        "status": "completed",
        "response": response,
        "response_length_tokens": generated_tokens,
        "generation_seconds": generation_seconds,
        "generation_tokens_per_second": (
            generated_tokens / generation_seconds
            if generation_seconds > 0
            else None
        ),
        "reload_and_generation_wall_seconds": (
            time.perf_counter() - reload_started
        ),
        "peak_allocated_vram_bytes": stack.torch.cuda.max_memory_allocated(),
        "peak_reserved_vram_bytes": stack.torch.cuda.max_memory_reserved(),
        "policy_contract": {
            "action": contract.action,
            "parsed": contract.parsed,
            "valid_action": contract.valid_action,
            "policy_failure": contract.policy_failure,
            "policy_failure_reason": contract.policy_failure_reason,
        },
    }
    del model
    del processor
    gc.collect()
    stack.torch.cuda.empty_cache()
    return result


def _failure_telemetry(
    stack: RuntimeStack, started: float
) -> dict[str, Any]:
    telemetry: dict[str, Any] = {
        "wall_seconds": time.perf_counter() - started,
    }
    try:
        telemetry.update(
            {
                "peak_allocated_vram_bytes": (
                    stack.torch.cuda.max_memory_allocated()
                ),
                "peak_reserved_vram_bytes": (
                    stack.torch.cuda.max_memory_reserved()
                ),
            }
        )
    except Exception:
        pass
    return telemetry


def _build_manifest(
    *,
    config: GrpoSmokeConfig,
    input_sha256: str,
    output_directory: Path,
    result: dict[str, Any],
    stack: RuntimeStack,
) -> dict[str, Any]:
    device_properties = stack.torch.cuda.get_device_properties(0)
    artifacts: list[dict[str, Any]] = []
    for name, path in (
        ("resolved_config", output_directory / "resolved_config.json"),
        ("result", output_directory / "result.json"),
    ):
        artifacts.append(
            {
                "name": name,
                "path": str(path.relative_to(output_directory)),
                "sha256": _file_sha256(path),
            }
        )
    for path in sorted(
        output_directory.glob("group-*-reward-events.jsonl")
    ):
        artifacts.append(
            {
                "name": path.stem,
                "path": str(path.relative_to(output_directory)),
                "sha256": _file_sha256(path),
            }
        )
    artifacts.append(
        {
            "name": "adapter",
            "path": "adapter",
            "members": _directory_members(
                output_directory / "adapter",
                output_directory,
            ),
            "wandb_uploaded": False,
        }
    )
    artifacts.append(
        {
            "name": "tensorboard",
            "path": "telemetry/tensorboard",
            "members": _directory_members(
                output_directory / "telemetry" / "tensorboard",
                output_directory,
            ),
        }
    )
    return {
        "schema_version": 1,
        "input": {"sha256": input_sha256},
        "configuration": config.resolved(),
        "result_status": result["status"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": stack.versions,
            "cuda": stack.torch.version.cuda,
            "cudnn": stack.torch.backends.cudnn.version(),
            "gpu": {
                "name": device_properties.name,
                "total_memory_bytes": device_properties.total_memory,
                "compute_capability": list(
                    stack.torch.cuda.get_device_capability(0)
                ),
            },
            "api_signatures": stack.api_signatures,
        },
        "metric_definitions": {
            "generation_tokens_per_second": (
                "generated completion token count divided by elapsed generation "
                "time; the training phase interval ends at reward callback entry"
            ),
            "training_steps_per_second": (
                "maintained Transformers Trainer train_steps_per_second metric"
            ),
            "peak_allocated_vram_bytes": (
                "maximum bytes allocated according to torch.cuda"
            ),
            "peak_reserved_vram_bytes": (
                "maximum bytes reserved according to torch.cuda"
            ),
        },
        "artifacts": artifacts,
    }


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


def _directory_members(
    directory: Path, output_directory: Path
) -> list[dict[str, str]]:
    if not directory.exists():
        return []
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
        raise GrpoSmokeConfigurationError(f"{field} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], field: str
) -> None:
    if set(value) != expected:
        raise GrpoSmokeConfigurationError(
            f"{field} must contain exactly {', '.join(sorted(expected))}"
        )


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise GrpoSmokeConfigurationError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GrpoSmokeConfigurationError(f"{field} must be an integer")
    return value


def _finite_number(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise GrpoSmokeConfigurationError(f"{field} must be a finite number")
    return float(value)


def _require(value: Any, expected: Any, field: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise GrpoSmokeConfigurationError(
            f"{field} must be {expected!r} for the BF16 feasibility smoke"
        )


def _load_model(raw: Mapping[str, Any]) -> ModelConfig:
    expected = {
        "id",
        "revision",
        "precision",
        "load_in_4bit",
        "fast_inference",
        "text_only",
        "max_sequence_length",
    }
    _require_exact_keys(raw, expected, "model")
    _require(raw["id"], OFFICIAL_QWEN35_4B, "model.id")
    revision = _nonempty_string(raw["revision"], "model.revision")
    if MODEL_REVISION_PATTERN.fullmatch(revision) is None:
        raise GrpoSmokeConfigurationError(
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


def _load_lora(raw: Mapping[str, Any]) -> LoraConfig:
    expected = {
        "rank",
        "alpha",
        "dropout",
        "finetune_vision_layers",
        "finetune_language_layers",
        "finetune_attention_modules",
        "finetune_mlp_modules",
        "gradient_checkpointing",
    }
    _require_exact_keys(raw, expected, "lora")
    required_values = {
        "rank": 64,
        "alpha": 64,
        "dropout": 0.0,
        "finetune_vision_layers": False,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
        "gradient_checkpointing": "unsloth",
    }
    for field, expected_value in required_values.items():
        _require(raw[field], expected_value, f"lora.{field}")
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


def _load_grpo(raw: Mapping[str, Any]) -> GrpoConfig:
    expected = {
        "trainer",
        "loss_type",
        "num_generations",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "generation_batch_size",
        "max_steps",
        "learning_rate",
        "optimizer",
        "max_prompt_length",
        "max_completion_length",
        "temperature",
        "beta",
        "mask_truncated_completions",
        "safe_group8_reserved_headroom_gib",
        "use_vllm",
    }
    _require_exact_keys(raw, expected, "grpo")
    required_values = {
        "trainer": "trl.GRPOTrainer",
        "loss_type": "grpo",
        "num_generations": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "generation_batch_size": 4,
        "max_steps": 1,
        "optimizer": "adamw_8bit",
        "max_prompt_length": 160,
        "max_completion_length": 96,
        "temperature": 1.0,
        "beta": 0.0,
        "mask_truncated_completions": False,
        "safe_group8_reserved_headroom_gib": 2.0,
        "use_vllm": False,
    }
    for field, expected_value in required_values.items():
        _require(raw[field], expected_value, f"grpo.{field}")
    learning_rate = _finite_number(raw["learning_rate"], "grpo.learning_rate")
    if learning_rate <= 0:
        raise GrpoSmokeConfigurationError(
            "grpo.learning_rate must be positive"
        )
    return GrpoConfig(
        trainer="trl.GRPOTrainer",
        loss_type="grpo",
        num_generations=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=4,
        max_steps=1,
        learning_rate=learning_rate,
        optimizer="adamw_8bit",
        max_prompt_length=160,
        max_completion_length=96,
        temperature=1.0,
        beta=0.0,
        mask_truncated_completions=False,
        safe_group8_reserved_headroom_gib=2.0,
        use_vllm=False,
    )


def _load_teacher_guided_example(
    raw: Mapping[str, Any],
) -> TeacherGuidedExample:
    expected = {
        "variant",
        "board",
        "teacher_action_scores",
        "teacher_action",
        "teacher_margin_scale",
    }
    _require_exact_keys(raw, expected, "teacher_guided_example")
    _require(raw["variant"], "direct_action", "teacher_guided_example.variant")
    board_value = raw["board"]
    if (
        not isinstance(board_value, list)
        or len(board_value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in board_value)
        or any(
            not isinstance(tile, int)
            or isinstance(tile, bool)
            or tile < 0
            or (tile != 0 and (tile < 2 or tile & (tile - 1) != 0))
            for row in board_value
            for tile in row
        )
    ):
        raise GrpoSmokeConfigurationError(
            "teacher_guided_example.board must be a 4x4 matrix of 0 or powers of two"
        )
    board = cast(list[list[int]], board_value)
    raw_scores = _object(
        raw["teacher_action_scores"],
        "teacher_guided_example.teacher_action_scores",
    )
    if set(raw_scores) != {"LEFT", "RIGHT", "UP", "DOWN"}:
        raise GrpoSmokeConfigurationError(
            "teacher_guided_example.teacher_action_scores must score every action"
        )
    scores: dict[Action, float | None] = {}
    for raw_action, raw_score in raw_scores.items():
        action = cast(Action, raw_action)
        scores[action] = (
            None
            if raw_score is None
            else _finite_number(
                raw_score,
                f"teacher_guided_example.teacher_action_scores.{raw_action}",
            )
        )
    teacher_action_value = raw["teacher_action"]
    if teacher_action_value not in scores:
        raise GrpoSmokeConfigurationError(
            "teacher_guided_example.teacher_action must be a supported action"
        )
    teacher_action = cast(Action, teacher_action_value)
    top_score = scores[teacher_action]
    if top_score is None or any(
        score is not None and score > top_score for score in scores.values()
    ):
        raise GrpoSmokeConfigurationError(
            "teacher_guided_example.teacher_action must have the top finite score"
        )
    change_actions = set(change_making_actions(board))
    if teacher_action not in change_actions:
        raise GrpoSmokeConfigurationError(
            "teacher_guided_example.teacher_action must change the board"
        )
    for action in change_actions:
        if scores[action] is None:
            raise GrpoSmokeConfigurationError(
                "every change-making action must have a Teacher Policy score"
            )
    tau = _finite_number(
        raw["teacher_margin_scale"],
        "teacher_guided_example.teacher_margin_scale",
    )
    if tau <= 0:
        raise GrpoSmokeConfigurationError(
            "teacher_guided_example.teacher_margin_scale must be positive"
        )
    return TeacherGuidedExample(
        variant="direct_action",
        board=board,
        teacher_action_scores=scores,
        teacher_action=teacher_action,
        teacher_margin_scale=tau,
    )


def _load_telemetry(raw: Mapping[str, Any]) -> SmokeTelemetryConfig:
    expected = {
        "wandb_mode",
        "wandb_entity",
        "wandb_project",
        "wandb_project_visibility",
        "tensorboard",
        "upload_model_checkpoints",
    }
    _require_exact_keys(raw, expected, "telemetry")
    _require(raw["wandb_mode"], "online", "telemetry.wandb_mode")
    _require(raw["wandb_entity"], "auto", "telemetry.wandb_entity")
    project = _nonempty_string(raw["wandb_project"], "telemetry.wandb_project")
    _require(
        raw["wandb_project_visibility"],
        "private",
        "telemetry.wandb_project_visibility",
    )
    _require(raw["tensorboard"], True, "telemetry.tensorboard")
    _require(
        raw["upload_model_checkpoints"],
        False,
        "telemetry.upload_model_checkpoints",
    )
    return SmokeTelemetryConfig(
        wandb_mode="online",
        wandb_entity="auto",
        wandb_project=project,
        wandb_project_visibility="private",
        tensorboard=True,
        upload_model_checkpoints=False,
    )
