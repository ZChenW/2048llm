# Select ART LocalBackend for the bounded Stage 2 tracer

Status: Accepted
Date: 2026-07-31
Decision owner: GitHub issue #12

## Context

Stage 2 Environment GRPO needs a maintained trainer that can generate a short
2048 Training Episode from a current-board-only Markov Policy observation,
compute the frozen Environment Reward, update a LoRA adapter, checkpoint it,
and resume it. The spike compared ART `LocalBackend` 0.5.18 with TRL 1.9.2
`environment_factory` on the same RTX 5070 using:

- `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, BF16, rank-16 LoRA;
- one canonical new-game snapshot, environment seed 12012, group size four,
  three-step horizon, and independent registered policy sampling;
- the same strict Policy Response parser and frozen Environment Reward; and
- private W&B plus local TensorBoard, with model upload disabled.

The registered config SHA-256 was
`ea1fe6f57e1a33ddfa8477081cd1ba6ef384dd5ac4b792f7398ce62ddce8f258`.
Neither integration owns GRPO mathematics or an optimizer loop.

## Decision

Select exactly **ART LocalBackend** for the next bounded Stage 2 tracer.
Do not use TRL `environment_factory`.

This selection does not authorize a long Environment GRPO run. ART remains
blocked from long-run promotion until a follow-up either demonstrates
optimizer and RNG continuity across restart or explicitly accepts
adapter-and-step-only recovery. The spike demonstrated a standard PEFT adapter,
step discovery, and fresh-backend inference; ART's public path did not expose
optimizer/RNG resume evidence.

## Measured evidence

### ART LocalBackend: accepted for the bounded tracer

The final run is
[attempt-8](https://wandb.ai/zichenw66-umass-amherst/2048llm-feasibility/runs/issue-12-stage2-backend-spike-art-stage2-art-attempt-8).
Its immutable local `result.json` SHA-256 is
`d8c305fa17931f53dcd4bb55a27a7ded79cb8ffff55aec6e56ba7c979cb3b17f`.

- All four Rollout Group members shared the registered snapshot and environment
  seed, used disjoint policy-sampling schedules, and completed exactly three
  real environment steps. Rewards were `[0, 0.015625, 0, 0]`.
- The maintained trainer reported one train request, one trainable Rollout
  Group, 96 trainer tokens, 12 effective gradient steps, loss
  `0.0536774996`, and gradient norm `3.9927591483`.
- Rollout throughput was `34.2877` completion tokens/s. The train request took
  `27.8936` s: `0.03585` backend train requests/s and `0.43021` effective
  gradient steps/s.
- Peak observed GPU delta was 8,076,132,352 bytes; peak Torch allocated and
  reserved memory were 3,505,018,368 and 3,609,198,592 bytes.
- End-to-end wall time was 169.3641 s from initial ART registration through
  rollout, training, save, fresh backend registration, checkpoint discovery,
  and resumed inference.
- Checkpoint `0001` is a 40,422,168-byte standard Hugging Face PEFT/Unsloth
  adapter. A new backend discovered step 1 and generated the valid response
  `<action>UP</action>`. Retained local adapter weights changed from checkpoint
  `0000` SHA-256
  `d9418c9c0a2d19760177715d46d1da4f591a89f6594c32901be590094004a080`
  to checkpoint `0001` SHA-256
  `e556f34f9a46009429127b11d7cbf0b1eee86b538e8f9a86ded8c64e6ce4a38c`.

ART 0.5.18's public `LocalBackend.train` does not expose a
gradient-accumulation-sequences control. The maintained trainer therefore
performed one gradient step for each of the 12 separately tokenized histories.
This is an effective-configuration confounder, not hidden parity with TRL's
single zero-signal step, and it must remain visible in later measurements.

The maintained trainers also chose different supported effective settings.
ART used `paged_adamw_8bit`, per-device batch size four, gradient accumulation
one, gradient checkpointing, BF16, and no internal maximum-step limit. TRL
used `adamw_torch_fused`, per-device batch size one, gradient accumulation
one, gradient checkpointing, BF16, and `max_steps=1`. These optimizer and batch
differences are recorded confounders; the project did not patch either
maintained trainer to force unsupported parity.

ART required two local compatibility measures: a narrow model-registry shim
that returns only the frozen LoRA targets because the backend extra omits its
Megatron dependency, and the maintained PyTorch sampler because FlashInfer
0.6.6 rejects compute capability 12.0. Neither measure replaces the trainer,
optimizer, GRPO calculation, or rollout.

### TRL environment_factory: explicit asymmetric no-go

The measured run is
[attempt-4](https://wandb.ai/zichenw66-umass-amherst/2048llm-feasibility/runs/r668mpou).
Its immutable local `result.json` SHA-256 is
`48cd0547648a951d24e410ee04e02d1fafc4a105b49121186aeecccb376e83cc`.

- The maintained model returned direct assistant text instead of calling the
  environment tool. Tool-call frequency was zero, so all four candidates
  terminated as Policy Failure after one step. It did not meet the required
  two-to-four-step rollout contract and did not demonstrate a multi-turn
  Markov Policy.
- Rewards were `[-1.25, -1.25, -1.25, -1.25]`; reward standard deviation,
  loss, and gradient norm were all zero. TRL invoked its backward/optimizer
  path and advanced global step 1, but there was no optimization signal and no
  evidence of a policy weight update.
- It saved the same-size PEFT adapter plus optimizer, scheduler, RNG, and
  trainer state. A maintained zero-epoch resume loaded checkpoint 1 and
  preserved global step 1, but this does not compensate for the rollout and
  zero-signal failures.
- The recorded rollout throughput was 54.2666 completion tokens/s, the logged
  step time was 0.7289 s, peak GPU delta was 415,236,096 bytes, and the old
  `wall_seconds` value was 7.5597 s.

That old TRL wall boundary covers only `GRPOTrainer.train`; it excludes initial
model/trainer setup and the resume probe. It is therefore not directly
comparable to ART's 169.3641 s end-to-end value. The implementation now emits
both an end-to-end boundary and an explicitly named trainer-only boundary for
future runs. No historical time is retroactively estimated.

The GPU boundaries also differ. ART's external sampler covered registration,
training, and fresh resume; TRL's covered only `GRPOTrainer.train`. TRL's Torch
peak was reset before training and read after resume, while ART reset before
registration. The raw values document each run, but this decision makes no
like-for-like VRAM ranking from them.

TRL's maintained tool loop would append prior tool calls and observations to
later model prefixes. Because attempt-4 never reached a second observation,
its immutable result's `markov_policy=true` boolean was erroneous:
`multi_turn_markov_policy_demonstrated=false` is the supported evidence. This
decision supersedes the erroneous boolean with status `not_demonstrated`;
future result code emits `markov_policy=false` for the same outcome. Neither
artifact claims an observed history breach.

## Telemetry audit

ART's original private W&B run contains total reward, every frozen reward
component, loss, gradient norm, data volume, and time. Review found that the
original local TensorBoard event lacked component tags, so a CPU-only
supplement copied the already-recorded W&B component means into a second local
event. Its manifest records the source result and W&B-summary hashes; no
training or checkpoint changed.

TRL's original TensorBoard event is under `trainer/runs/...`, not the
prospective `tensorboard` path in its old result. A CPU-only supplement wrote
all five component means to `runs/stage2-trl-attempt-4/tensorboard` and appended
the same keys plus source-result SHA to the existing private W&B summary.
The derivation is exact: four one-step Policy Failures, four rewards equal to
the frozen `-1.25` failure component, zero reward variance, and no other
negative component. Both supplement manifests are retained beside their run
artifacts and state `training_rerun=false`.

Future runs fail closed when ART has any member outside two-to-four steps,
write all reward components directly to both telemetry sinks, report the
actual TensorBoard event files, distinguish a backward invocation from a
nonzero learning signal, and disclose measurement boundaries.

## Consequences

- Issue #13's bounded tracer may integrate ART `LocalBackend`, preserving one
  fresh current-board prompt per decision through `additional_histories`.
- The tracer must retain the compatibility shim, native sampler selection,
  per-request policy seeds, component telemetry, and two-to-four-step guard.
- A long-run Stage 2 ticket cannot begin from this decision alone. It needs
  explicit optimizer/RNG restart continuity evidence or an accepted
  adapter-only recovery design.
- TRL `environment_factory` is not a fallback for this design. Reconsidering
  TRL would require a maintained API that both reliably invokes the environment
  and permits current-board-only prefixes; a project-owned trainer or GRPO
  loop remains out of scope.
