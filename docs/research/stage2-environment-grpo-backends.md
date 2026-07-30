# Stage 2 Environment GRPO backend API research

Date: 2026-07-31
Scope: GitHub issue #12, ART LocalBackend versus TRL `environment_factory`

The `research` skill normally delegates source reading to a background agent.
The global agent-thread limit rejected that delegation, so this note was
researched directly from the maintainers' documentation, tagged source, and
published package metadata. No secondary tutorials were used.

## Common comparison boundary

The repository's frozen design requires a Markov Policy: every Student Policy
decision receives only the current board, while the trainer may retain the
trajectory externally. Policy Failure must terminate without masking, repair,
fallback, or retry. The frozen Environment Reward is:

`5.0 * reached_2048 + 1.0 * tile_progress + 0.25 * score_progress
- 1.0 * game_over_without_2048 - 1.25 * policy_failure`.

The spike therefore shares one exact Qwen revision, BF16 LoRA shape, start
snapshot, PRNG seed, horizon, Rollout Group size, response parser, and reward
fixture. Backend code may orchestrate the canonical environment, but neither
candidate may replace its maintained optimizer, advantage calculation, or
trainer loop.

## ART LocalBackend

- ART's maintained training contract is a caller-provided rollout that returns
  rewarded `Trajectory` objects grouped as `TrajectoryGroup`; `backend.train`
  owns GRPO, optimization, checkpoint save, and loading the new LoRA for
  inference. The local backend runs vLLM plus an Unsloth or torchtune trainer.
  [ART training loop](https://art.openpipe.ai/fundamentals/training-loop),
  [ART client](https://art.openpipe.ai/fundamentals/art-client),
  [ART backend](https://art.openpipe.ai/fundamentals/art-backend).
- `LocalBackend` supports shared-GPU mode, which pauses inference during
  training. Dedicated `PipelineTrainer` mode requires separate inference and
  training GPUs, but that restriction does not apply to the ordinary shared
  backend used by this one-GPU spike.
  [ART backend](https://art.openpipe.ai/fundamentals/art-backend).
- ART `additional_histories` tokenizes related histories separately and
  distributes training weight across them. That is the maintained mechanism
  needed here: each 2048 decision can be generated from a fresh current-board
  prompt while all decisions remain in one rewarded Training Episode.
  [ART additional histories](https://art.openpipe.ai/features/additional-histories).
- ART documents warm-starting a `TrainableModel` from a standard local
  Hugging Face PEFT/Unsloth adapter directory. Its own Unsloth service saves
  checkpoints through the maintained trainer's `save_model`, so the selected
  checkpoint format has a direct compatibility path.
  [ART client: existing SFT LoRA](https://art.openpipe.ai/fundamentals/art-client).
- ART logs reward, loss, gradient norm, time, data volume, and throughput when
  `model.log` is called. The spike adds only the same scalar values to a local
  TensorBoard writer; that is telemetry, not a trainer loop.
  [ART metric tracking](https://art.openpipe.ai/features/tracking-metrics).
- The current published backend extra is `openpipe-art[backend]==0.5.18`.
  Its package metadata pins Torch 2.10.0, Transformers 5.2.0, TRL 0.20.0,
  Unsloth 2026.3.3, Unsloth Zoo 2026.3.1, and W&B 0.25.0. ART also ships a
  separately managed vLLM runtime. These pins require an isolated environment
  rather than mutation of the proven `td2048` environment.
  [PyPI release metadata](https://pypi.org/pypi/openpipe-art/0.5.18/json).
- The 0.5.18 local-backend import path loads ART's Megatron model registry
  even when the selected trainer is Unsloth, but the `backend` extra does not
  declare Megatron. Its separately advertised `megatron` extra requests
  `megatron-core==0.16.0rc0` and `megatron-bridge==0.4.0rc0`; neither is
  available for this Python 3.11 stack, while the available older bridge
  conflicts with ART's Transformers 5.2 pin. The spike installs a narrow
  compatibility module before ART imports that registry: it returns the
  already frozen LoRA target modules and changes no trainer, GRPO, optimizer,
  or rollout behavior. This workaround is counted against ART's integration
  complexity.
  [ART 0.5.18 package metadata](https://pypi.org/pypi/openpipe-art/0.5.18/json),
  [Megatron Core 0.16.0](https://pypi.org/project/megatron-core/0.16.0/).

## TRL `environment_factory`

- Current TRL creates one environment instance per rollout, discovers its
  public methods as tools, calls `reset`, executes the multi-turn tool loop,
  and optionally calls environment-owned `get_reward`. A deterministic shared
  key is required when all members of a Rollout Group must share an initial
  state. Transformers 5.2 or newer is required.
  [TRL GRPO environments](https://huggingface.co/docs/trl/main/grpo_trainer#environments).
- TRL's native tool loop appends the assistant tool call and tool result to the
  accumulated completion before generating again. That is useful for general
  agents, but a second 2048 decision sees prior interaction in its prefix. The
  `environment_factory` API offers no supported hook that replaces that prefix
  with a fresh current-board-only prompt. A custom `rollout_func` could take
  control, but TRL labels it experimental and it would defeat this ticket's
  comparison of the maintained `environment_factory` loop.
  [TRL v1.9.2 tool loop source](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py),
  [TRL environment factory versus rollout function](https://huggingface.co/docs/trl/en/openenv#environment-factory-vs-rollout-func).
- TRL 1.9.2 exposes `environment_factory` but emits an explicit warning that
  the feature is experimental. It does support PEFT configuration and Trainer
  checkpoint/resume, so a real integration run can still measure gradient,
  optimizer, checkpoint, telemetry, and the Markov-prefix failure rather than
  rejecting it from documentation alone.
  [TRL v1.9.2 package metadata](https://pypi.org/pypi/trl/1.9.2/json),
  [TRL v1.9.2 source](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py).
- The repository's proven Unsloth 2026.7.4 stack pins TRL at no newer than
  0.24.0. Installed TRL 0.24.0 has no `environment_factory` parameter, whereas
  the maintained TRL 1.9.2 API does. The candidate must therefore run in a
  separate environment without Unsloth; installing it over `td2048` would
  invalidate the working Stage 1 stack.
  [Unsloth release metadata](https://pypi.org/pypi/unsloth/2026.7.4/json),
  [TRL v1.9.2 release metadata](https://pypi.org/pypi/trl/1.9.2/json).

## Measured outcome

The comparison selected exactly ART LocalBackend for the next bounded tracer.
ART completed four three-step Training Episodes with nonzero reward variance,
ran its maintained backward/optimizer path, saved a standard adapter, and
served it from a fresh backend. TRL produced direct assistant text instead of
tool calls: every candidate became a one-step Policy Failure, reward variance,
loss, and gradient norm were zero, and the required multi-turn Markov contract
was not demonstrated.

The evidence, effective-configuration confounders, telemetry supplements,
measurement boundaries, and the block on long-run promotion are recorded in
[ADR 0002](../adr/0002-select-art-localbackend-for-stage2-spike.md).
