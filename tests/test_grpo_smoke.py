from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llm2048.grpo_smoke import (
    GrpoSmokeConfig,
    GrpoSmokePreflightError,
    TeacherGuidedExample,
    _load_bf16_model,
    _load_exact_adapter,
    _start_online_wandb_run,
    _training_row,
    _verify_private_wandb_project,
)


class _ServiceApi:
    def __init__(self, access: str) -> None:
        self.access = access
        self.variables: dict[str, str] | None = None

    def execute_graphql(
        self,
        query: str,
        variables: dict[str, str],
    ) -> dict[str, dict[str, str]]:
        self.variables = variables
        if "access" not in query:
            raise AssertionError("project access must be queried")
        return {"project": {"access": self.access}}


class _Processor:
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count
        self.template_kwargs: dict[str, object] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        **kwargs: object,
    ) -> str:
        if not messages:
            raise AssertionError("policy messages must not be empty")
        self.template_kwargs = kwargs
        return "<rendered-policy-prompt>"

    def __call__(
        self,
        prompt: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        if prompt != "<rendered-policy-prompt>":
            raise AssertionError("the rendered prompt must be tokenized")
        if add_special_tokens:
            raise AssertionError("the chat template already includes special tokens")
        return {"input_ids": list(range(self.token_count))}


class _Wandb:
    def __init__(self, access: str) -> None:
        self.service_api = _ServiceApi(access)
        self.default_entity = "research-team"
        self.init_called = False

    def login(self, **_: object) -> None:
        pass

    def Api(self) -> SimpleNamespace:
        return SimpleNamespace(
            _service_api=self.service_api,
            default_entity=self.default_entity,
        )

    def init(self, **kwargs: object) -> SimpleNamespace:
        self.init_called = True
        return SimpleNamespace(
            entity=kwargs["entity"],
            project=kwargs["project"],
            id="test-run",
            url="https://wandb.invalid/test-run",
            settings=SimpleNamespace(mode="online"),
        )


def _config() -> GrpoSmokeConfig:
    repository = Path(__file__).resolve().parents[1]
    config, _ = GrpoSmokeConfig.load(
        repository / "configs" / "qwen35_4b_grpo_smoke.json"
    )
    return config


def _example() -> TeacherGuidedExample:
    config = _config()
    return config.teacher_guided_example


class _FastVisionModel:
    def __init__(self) -> None:
        self.arguments: dict[str, object] | None = None
        self.base_model = _TextModel()
        self.processor = object()

    def from_pretrained(
        self,
        *,
        model_name: str,
        revision: str,
        tokenizer_name: str,
        max_seq_length: int,
        dtype: object,
        load_in_4bit: bool,
        fast_inference: bool,
        text_only: bool,
        use_exact_model_name: bool,
        full_finetuning: bool,
    ) -> tuple[object, object]:
        self.arguments = {
            "model_name": model_name,
            "revision": revision,
            "tokenizer_name": tokenizer_name,
            "max_seq_length": max_seq_length,
            "dtype": dtype,
            "load_in_4bit": load_in_4bit,
            "fast_inference": fast_inference,
            "text_only": text_only,
            "use_exact_model_name": use_exact_model_name,
            "full_finetuning": full_finetuning,
        }
        return self.base_model, self.processor

    def get_peft_model(
        self,
        *,
        model: object,
        finetune_vision_layers: bool,
        finetune_language_layers: bool,
        finetune_attention_modules: bool,
        finetune_mlp_modules: bool,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        bias: str,
        random_state: int,
        use_rslora: bool,
        loftq_config: object,
        use_gradient_checkpointing: str,
    ) -> object:
        return model


class _TextModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(architectures=None)


class _PeftModel:
    def __init__(self) -> None:
        self.arguments: dict[str, object] | None = None
        self.adapter_model = object()

    def from_pretrained(
        self,
        *,
        model: object,
        model_id: str,
        is_trainable: bool,
    ) -> object:
        self.arguments = {
            "model": model,
            "model_id": model_id,
            "is_trainable": is_trainable,
        }
        return self.adapter_model


class AdapterReloadTests(unittest.TestCase):
    def test_adapter_reload_pins_the_original_base_model_revision(self) -> None:
        config = _config()
        fast_vision_model = _FastVisionModel()
        peft_model = _PeftModel()
        dtype = object()

        def snapshot_download(
            *,
            repo_id: str,
            revision: str,
            local_files_only: bool,
        ) -> str:
            self.assertEqual(repo_id, "Qwen/Qwen3.5-4B")
            self.assertEqual(
                revision,
                "c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97",
            )
            self.assertTrue(local_files_only)
            return "/cache/exact-qwen-snapshot"

        stack = SimpleNamespace(
            torch=SimpleNamespace(bfloat16=dtype),
            fast_vision_model=fast_vision_model,
            snapshot_download=snapshot_download,
            peft_model_class=peft_model,
        )
        adapter_directory = Path("/tmp/adapter-fixture")

        model, processor = _load_exact_adapter(
            stack=stack,  # type: ignore[arg-type]
            config=config,
            adapter_directory=adapter_directory,
        )

        self.assertIs(model, peft_model.adapter_model)
        self.assertIs(processor, fast_vision_model.processor)
        self.assertEqual(
            fast_vision_model.arguments,
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "revision": (
                    "c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97"
                ),
                "tokenizer_name": "/cache/exact-qwen-snapshot",
                "max_seq_length": 256,
                "dtype": dtype,
                "load_in_4bit": False,
                "fast_inference": False,
                "text_only": True,
                "use_exact_model_name": True,
                "full_finetuning": False,
            },
        )
        self.assertEqual(
            peft_model.arguments,
            {
                "model": fast_vision_model.base_model,
                "model_id": str(adapter_directory),
                "is_trainable": False,
            },
        )


class ModelLoadCompatibilityTests(unittest.TestCase):
    def test_text_only_model_exposes_architecture_metadata_for_generation(
        self,
    ) -> None:
        config = _config()
        fast_vision_model = _FastVisionModel()

        def snapshot_download(**_: object) -> str:
            return "/cache/exact-qwen-snapshot"

        stack = SimpleNamespace(
            torch=SimpleNamespace(bfloat16=object()),
            fast_vision_model=fast_vision_model,
            snapshot_download=snapshot_download,
        )

        model, _ = _load_bf16_model(
            stack,  # type: ignore[arg-type]
            config,
        )

        self.assertEqual(model.config.architectures, ["_TextModel"])


class TrainingPromptTests(unittest.TestCase):
    def test_direct_action_prompt_disables_qwen_thinking(self) -> None:
        processor = _Processor(token_count=87)

        row = _training_row(_example(), processor, max_prompt_length=160)

        self.assertEqual(row, {"prompt": "<rendered-policy-prompt>"})
        self.assertEqual(
            processor.template_kwargs,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )

    def test_prompt_over_token_budget_fails_before_training(self) -> None:
        processor = _Processor(token_count=161)

        with self.assertRaisesRegex(
            GrpoSmokePreflightError,
            "rendered Qwen prompt exceeds",
        ):
            _training_row(_example(), processor, max_prompt_length=160)


class WandbProjectAccessTests(unittest.TestCase):
    def test_public_project_is_rejected_before_a_run_is_created(self) -> None:
        wandb = _Wandb("PUBLIC")
        stack = SimpleNamespace(wandb=wandb)
        repository = Path(__file__).resolve().parents[1]
        config, _ = GrpoSmokeConfig.load(
            repository / "configs" / "qwen35_4b_grpo_smoke.json"
        )

        with (
            patch.dict("os.environ", {"WANDB_API_KEY": "secret"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "W&B project is not private"),
        ):
            _start_online_wandb_run(
                stack=stack,  # type: ignore[arg-type]
                config=config,
                telemetry_directory=Path("/tmp/unused-wandb-test"),
            )

        self.assertFalse(wandb.init_called)

    def test_private_project_access_is_verified_against_the_resolved_run(self) -> None:
        service_api = _ServiceApi("PRIVATE")
        api = SimpleNamespace(_service_api=service_api)

        access = _verify_private_wandb_project(
            api=api,
            entity="research-team",
            project="2048llm",
        )

        self.assertEqual(access, "PRIVATE")
        self.assertEqual(
            service_api.variables,
            {"entity": "research-team", "project": "2048llm"},
        )

    def test_public_project_access_fails_closed(self) -> None:
        api = SimpleNamespace(_service_api=_ServiceApi("PUBLIC"))

        with self.assertRaisesRegex(RuntimeError, "W&B project is not private"):
            _verify_private_wandb_project(
                api=api,
                entity="research-team",
                project="2048llm",
            )


if __name__ == "__main__":
    unittest.main()
