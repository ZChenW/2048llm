from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from llm2048.stage2_backend_spike import (
    BackendSpikeConfig,
    BackendSpikeConfigurationError,
    BackendSpikePreflightError,
    Stage2TrainingEpisode,
    dry_run,
    environment_reward,
    implementation_complexity,
    rollout_group_manifest,
    validate_adapter_directory,
)
from llm2048.policy_contracts import change_making_actions
import llm2048.stage2_trl_backend as trl_backend
from llm2048.stage2_art_backend import (
    _install_art_registry_compatibility,
    _logprob_calculation_chunk_size,
    _training_completion_token_count,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY / "configs" / "stage2_backend_spike.json"


class BackendSpikeConfigTests(unittest.TestCase):
    def test_registered_config_is_strict_and_frozen(self) -> None:
        config, digest = BackendSpikeConfig.load(CONFIG_PATH)

        self.assertEqual(config.model.id, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.rollout.horizon, 3)
        self.assertEqual(config.training.optimizer_steps, 1)
        self.assertEqual(config.environment_reward.reached_2048_weight, 5.0)
        self.assertEqual(len(digest), 64)

    def test_environment_reward_weight_change_is_rejected(self) -> None:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        raw["environment_reward"]["reached_2048_weight"] = 4.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(
                BackendSpikeConfigurationError,
                "frozen specification",
            ):
                BackendSpikeConfig.load(path)


class CanonicalEpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, _ = BackendSpikeConfig.load(CONFIG_PATH)

    def test_rollout_group_shares_start_snapshot_and_rng_seed(self) -> None:
        manifest = rollout_group_manifest(self.config)

        self.assertEqual(manifest["group_size"], 4)
        self.assertTrue(manifest["shared_start_snapshot"])
        self.assertTrue(manifest["shared_rng_seed"])
        self.assertEqual(
            len(set(manifest["member_start_snapshot_sha256"])),
            1,
        )
        self.assertEqual(set(manifest["member_rng_seed"]), {12012})

    def test_real_three_step_environment_fixture_uses_markov_prompts(self) -> None:
        episode = Stage2TrainingEpisode(self.config)

        for _ in range(3):
            action = change_making_actions(episode.board)[0]
            step = episode.submit_policy_response(f"<action>{action}</action>")
            self.assertIn(str(step.board_before).replace(" ", ""), step.prompt)

        self.assertTrue(episode.terminal)
        self.assertEqual(episode.terminal_reason, "horizon")
        self.assertEqual(len(episode.steps), 3)
        self.assertGreaterEqual(episode.reward().total, 0.0)

    def test_policy_failure_terminates_without_mask_retry_or_move(self) -> None:
        episode = Stage2TrainingEpisode(self.config)
        start_board = episode.board

        step = episode.submit_policy_response("LEFT")

        self.assertTrue(step.terminal)
        self.assertEqual(step.policy_failure_reason, "missing_action")
        self.assertEqual(episode.terminal_reason, "policy_failure")
        self.assertEqual(episode.board, start_board)
        self.assertEqual(episode.reward().policy_failure, -1.25)
        with self.assertRaisesRegex(RuntimeError, "cannot retry"):
            episode.submit_policy_response("<action>LEFT</action>")

    def test_success_component_dominates_bounded_progress(self) -> None:
        reward = environment_reward(
            config=self.config.environment_reward,
            start_max_tile=512,
            final_max_tile=2048,
            start_score=100,
            final_score=10000,
            reached_2048=True,
            game_over_without_2048=False,
            policy_failure=False,
        )

        self.assertEqual(reward.reached_2048, 5.0)
        self.assertEqual(reward.tile_progress, 1.0)
        self.assertEqual(reward.score_progress, 0.25)
        self.assertEqual(reward.total, 6.25)


class TrlEnvironmentContractTests(unittest.TestCase):
    def test_only_policy_tool_is_public_and_three_steps_are_real(self) -> None:
        config, _ = BackendSpikeConfig.load(CONFIG_PATH)
        trl_backend._ACTIVE_CONFIG = config
        environment = trl_backend.Trl2048Environment()
        public_methods = {
            name
            for name, _ in inspect.getmembers(
                environment,
                predicate=inspect.ismethod,
            )
            if name not in {"reset", "get_reward"} and not name.startswith("_")
        }

        self.assertEqual(public_methods, {"submit_policy_response"})
        self.assertEqual(
            tuple(
                inspect.signature(
                    environment.submit_policy_response
                ).parameters
            ),
            ("response",),
        )
        instruction = environment.reset(
            rng_seed=config.start_state.rng_seed
        )
        self.assertIn(
            "Do not return the Policy Response as assistant text",
            instruction,
        )
        self.assertIn("setting response to the exact Policy Response", instruction)
        for step_index in range(config.rollout.horizon):
            assert environment._episode is not None
            action = change_making_actions(environment._episode.board)[0]
            observation = environment.submit_policy_response(
                f"<action>{action}</action>"
            )
            if step_index + 1 < config.rollout.horizon:
                self.assertIn("Board (4x4 JSON array", observation)
            else:
                self.assertIn("terminated: horizon", observation)

        evidence = environment._evidence()
        self.assertEqual(len(evidence["steps"]), 3)
        self.assertFalse(evidence["markov_prefix_preserved"])
        self.assertIsInstance(environment.get_reward(), float)


class ArtifactContractTests(unittest.TestCase):
    def test_art_logprob_chunk_size_tiles_registered_sequence(self) -> None:
        config, _ = BackendSpikeConfig.load(CONFIG_PATH)

        chunk_size = _logprob_calculation_chunk_size(config)

        self.assertEqual(chunk_size, 512)
        self.assertEqual(
            config.model.max_sequence_length % chunk_size,
            0,
        )

    def test_art_training_requires_token_id_logprobs(self) -> None:
        valid = SimpleNamespace(
            logprobs=SimpleNamespace(
                content=[
                    SimpleNamespace(token="token_id:17"),
                    SimpleNamespace(token="token_id:23"),
                ]
            )
        )
        self.assertEqual(_training_completion_token_count(valid), 2)

        with self.assertRaisesRegex(
            BackendSpikePreflightError,
            "omitted generated-token logprobs",
        ):
            _training_completion_token_count(
                SimpleNamespace(logprobs=None)
            )
        with self.assertRaisesRegex(
            BackendSpikePreflightError,
            "did not return logprob tokens as token IDs",
        ):
            _training_completion_token_count(
                SimpleNamespace(
                    logprobs=SimpleNamespace(
                        content=[SimpleNamespace(token="plain text")]
                    )
                )
            )

    def test_art_registry_compatibility_exposes_only_frozen_targets(
        self,
    ) -> None:
        config, _ = BackendSpikeConfig.load(CONFIG_PATH)
        module_name = "art.megatron.model_support"
        previous = sys.modules.pop(module_name, None)
        try:
            self.assertTrue(_install_art_registry_compatibility(config))
            compatibility = importlib.import_module(module_name)
            target_resolver = compatibility.default_target_modules_for_model

            self.assertEqual(
                target_resolver(
                    config.model.id,
                    allow_unvalidated_arch=True,
                ),
                list(config.lora.target_modules),
            )
            with self.assertRaises(BackendSpikePreflightError):
                target_resolver(
                    "other/model",
                    allow_unvalidated_arch=True,
                )
        finally:
            sys.modules.pop(module_name, None)
            if previous is not None:
                sys.modules[module_name] = previous

    def test_dry_run_persists_shared_group_evidence_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = dry_run(CONFIG_PATH, output, "art_local")

            self.assertEqual(result["status"], "dry_run")
            self.assertTrue(result["rollout_group"]["shared_start_snapshot"])
            self.assertFalse(
                result["implementation_complexity"]["project_owned_optimizer_loop"]
            )
            self.assertTrue((output / "manifest.json").is_file())

    def test_backend_complexity_records_known_failure_modes(self) -> None:
        art = implementation_complexity("art_local")
        trl = implementation_complexity("trl_environment_factory")

        self.assertGreater(art["non_blank_non_comment_source_lines"], 0)
        self.assertGreater(trl["non_blank_non_comment_source_lines"], 0)
        self.assertIn(
            "zero-variance",
            " ".join(art["known_failure_modes"]),
        )
        self.assertIn(
            "retains prior actions",
            " ".join(trl["known_failure_modes"]),
        )

    def test_adapter_directory_requires_peft_files_and_matching_qwen_base(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {"base_model_name_or_path": "Qwen/Qwen3-0.6B"}
                ),
                encoding="utf-8",
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")

            result = validate_adapter_directory(
                adapter,
                expected_base_model="Qwen/Qwen3-0.6B",
            )

            self.assertEqual(result["format"], "huggingface_peft_unsloth")

    def test_adapter_directory_rejects_wrong_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "other/model"}),
                encoding="utf-8",
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            with self.assertRaises(BackendSpikePreflightError):
                validate_adapter_directory(
                    adapter,
                    expected_base_model="Qwen/Qwen3-0.6B",
                )


if __name__ == "__main__":
    unittest.main()
