# 2048llm

This repository trains and evaluates a 2048 Student Policy against a
TD-learning Teacher Policy.

## Experiment Runner tracer

Install the lightweight, model-free runner in the existing environment:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m pip install -e '.[dev]'
```

Run the deterministic fixture:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m llm2048.experiment_runner \
  --config tests/fixtures/experiment_runner_tracer.json \
  --output-dir runs/experiment-runner-tracer
```

The run writes a resolved manifest, deterministic events and result, an
atomic checkpoint, a W&B offline run, and local TensorBoard events. It does
not load a model or require a GPU.

Run the deterministic Policy Response contract fixture:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m llm2048.experiment_runner \
  --config tests/fixtures/policy_response_contracts.json \
  --output-dir runs/policy-response-contracts
```

Each `fixture.policy_cases` entry is one independent Markov Policy decision.
It supplies the latest 4×4 board, captured response and token length,
and truncation status. The runner derives change-making actions canonically
from tile movement and merges; it does not spawn tiles, calculate scores, or
put those actions in the prompt. Direct-action responses must be exactly
`<action>ACTION</action>`; Reasoning responses must be exactly
`<think>POLICY_REASONING_TRACE</think><action>ACTION</action>` and use a fixed
96-token maximum generation budget. Policy Reasoning Traces require an ASCII
letter and reject non-ASCII alphabetic characters. `result.json` reports
parse, truncation, illegal-action, valid-action, Policy Failure, and mean
response-length metrics for each variant.

Run one deterministic complete 2048 game with the model-free scripted
Direct-action Policy:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m llm2048.experiment_runner \
  --config tests/fixtures/environment_game_complete.json \
  --output-dir runs/environment-game-complete
```

An `environment_game` starts from the standard seeded two-tile reset. It can
use captured `responses` to exercise the strict Policy Response contract or a
Direct-action `action_preferences` script that selects from the current board
alone. Each legal action uses canonical compress-once, merge-once, merge-score,
and seeded 90% 2-tile / 10% 4-tile spawning. An illegal, malformed, or
truncated response ends the game immediately as Policy Failure.

Environment events record the current board-only Markov Policy prompt,
pre-spawn and post-spawn boards, score delta, and spawned tile. `result.json`
records score, moves, maximum tile, empty cells, tile histogram, termination
reason, Policy Failure, and 2048 Success. Checkpoints contain the board, score,
move count, and complete RNG state; resume verifies the snapshot is exactly
reachable by replaying the configured seed and prior responses before
restoring it.

Run the deterministic Teacher-guided Rollout Group fixture:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m llm2048.experiment_runner \
  --config tests/fixtures/teacher_guided_rollout_group.json \
  --output-dir runs/teacher-guided-rollout-group
```

The group configuration names one shared board, its configured candidate count,
corpus-style Teacher action scores, and a Teacher Policy Corpus manifest. The
runner reads the Teacher Margin Scale (`tau`) from that manifest, applies the
strict Policy Response contract, and reports Action Quality Reward, best-action
bonus, illegal-action penalty, and other Policy Failure penalty separately in
run events, aggregate results, offline W&B, and TensorBoard.

Run the CLI contract tests:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m unittest discover -s tests -v
```

## Qwen3.5-4B zero-shot gate and conditional masked SFT

The pre-RL gate compares the Direct-action Policy with the native Qwen
Reasoning Policy on the same deterministic 500-board slice of the production
Teacher Policy Corpus test split. The slice preserves the corpus 50% Natural
State / 30% Hard State / 20% Late State mix. Selection is a stable hash of the
configured seed, purpose, and record ID; the data manifest records the selected
member checksum.

Point an isolated worktree at the retained corpus and validate all source
hashes, counts, Corpus Strata, Trajectory Lineage, Symmetry Orbit isolation,
and the 500/900/100 selections without loading the model:

```bash
export LLM2048_TEACHER_CORPUS_MANIFEST="$PWD/runs/teacher-corpus-depth2-production/manifest.json"

/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m llm2048.experiment_runner \
  --zero-shot-sft-config configs/qwen35_4b_zero_shot_sft_gate.json \
  --output-dir runs/qwen35-4b-zero-shot-sft-gate \
  --dry-run
```

The real run requires the same pinned BF16 Unsloth stack and private online
W&B project as the GRPO feasibility smoke. The Reasoning Policy gate enables
Qwen's native thinking mode; the Direct-action Policy disables it. A policy
variant passes at 95% parse rate and at most 2% illegal actions. A passing
variant keeps the unchanged base model and performs no SFT.

For each failing variant, the runner derives exactly 900 train and 100
validation examples from their corresponding corpus splits. The Teacher
Policy supplies only the final action. The Reasoning Policy target contains a
non-Teacher placeholder trace whose token labels are masked; `<think>` and
`</think>` envelope tokens plus the final `<action>…</action>` remain
supervised. Maintained Transformers `Trainer` and Unsloth language-only
rank-64 LoRA are used; there is no custom training loop and no GRPO in this
ticket.

Load the credential without printing it, keep checkpoint upload disabled, and
run:

```bash
export WANDB_MODE=online
export WANDB_LOG_MODEL=false

/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m llm2048.experiment_runner \
  --zero-shot-sft-config configs/qwen35_4b_zero_shot_sft_gate.json \
  --output-dir runs/qwen35-4b-zero-shot-sft-gate-phase-1
```

To avoid retaining Trainer CUDA state while reloading the final Reasoning
adapter, the first command exits successfully with status
`awaiting_fresh_process_finalization`. It emits `handoff.json` and an immutable
`finalization-lock.json`; it does not create or depend on `failure.json`.
Complete the mandatory final gate in a new process:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m llm2048.experiment_runner \
  --zero-shot-sft-config configs/qwen35_4b_zero_shot_sft_gate.json \
  --finalize-from runs/qwen35-4b-zero-shot-sft-gate-phase-1 \
  --finalization-lock \
    runs/qwen35-4b-zero-shot-sft-gate-phase-1/finalization-lock.json \
  --output-dir runs/qwen35-4b-zero-shot-sft-gate-final
```

Together the two successful phases write per-board before/after gate events,
data/config provenance, private W&B and local TensorBoard metrics,
adapter-only checkpoints for failing variants, exact-base reload evidence,
and the selected starting point for each Student Policy.

The completed 500-board gate found that neither variant is ready for
Teacher-guided GRPO. Direct-action zero-shot parsed 100% but produced 10.2%
illegal actions; its masked-SFT adapter regressed to 16.4%, so the selected
candidate remains the unchanged pinned base. Native Reasoning zero-shot
truncated all responses at 96 tokens. Its adapter reload produced complete
8-token envelopes without truncation, but all 500 contained an empty or
non-English Policy Reasoning Trace and therefore parsed at 0%. It remains the
best provisional Reasoning candidate, but does not pass the gate.

Before the planned two-phase handoff was added, two historical combined
train/evaluate attempts retained enough CUDA state to OOM during the final
Reasoning adapter reload. The exceptional recovery command below preserves
that lineage, verifies an immutable SHA-256 lock over the failed source
configuration, data, fixed gate events, and both adapters, and cannot invoke
training. It is evidence recovery, not the supported default workflow:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m llm2048.experiment_runner \
  --zero-shot-sft-config configs/qwen35_4b_zero_shot_sft_gate.json \
  --recovery-from runs/qwen35-4b-zero-shot-sft-gate-attempt-2 \
  --recovery-lock runs/qwen35-4b-zero-shot-sft-gate-attempt-2.recovery-lock.json \
  --output-dir runs/qwen35-4b-zero-shot-sft-gate-attempt-2-recovery
```

The private online recovery telemetry is available in W&B run
[`esgsgfrl`](https://wandb.ai/zichenw66-umass-amherst/2048llm-feasibility/runs/esgsgfrl);
adapter upload remained disabled and the same metrics were written to local
TensorBoard events.

The expensive 900/100 SFT sequence was not repeated solely to exercise the
new handoff: attempt 2 already completed the identical deterministic training
phase and the recovery run completed the identical fresh-process finalizer.
The phase-selection regression test additionally executes the real
orchestration seam with mocked model operations, proving that phase 1 returns
the awaiting status, writes both handoff files, never reloads the Reasoning
adapter, and rejects any planned-finalization dependency on `failure.json`.

## Qwen3.5-4B GRPO feasibility smoke

Install the pinned RTX 50-series training stack in the `td2048` environment:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m pip install -e '.[dev,grpo]'
```

The smoke configuration pins the public, post-trained
`Qwen/Qwen3.5-4B` repository at commit
`c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97`. Its 15 files total
9,342,907,713 bytes. Cache that exact revision before the run:

```bash
HF_HUB_DISABLE_XET=1 \
  /home/chakew/miniconda3/envs/td2048/bin/hf download \
  Qwen/Qwen3.5-4B \
  --revision c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97
```

Validate the immutable BF16/rank-64/group-4 plan without importing the GPU
stack, contacting W&B, or creating run artifacts:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m llm2048.experiment_runner \
  --grpo-smoke-config configs/qwen35_4b_grpo_smoke.json \
  --output-dir runs/qwen35-4b-grpo-smoke \
  --dry-run
```

For the real run, first create the `2048llm-feasibility` W&B project with
Private, Team, or Restricted visibility. Authenticate outside source control
by exporting `WANDB_API_KEY` in the shell. An optional `WANDB_ENTITY` selects
a team; otherwise W&B uses the account's default entity. Before creating a
run, the runner requires online mode and verifies that the existing project
has private access. It keeps W&B model upload disabled and writes local
TensorBoard events:

```bash
export WANDB_API_KEY
export WANDB_MODE=online
export WANDB_LOG_MODEL=false

/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m llm2048.experiment_runner \
  --grpo-smoke-config configs/qwen35_4b_grpo_smoke.json \
  --output-dir runs/qwen35-4b-grpo-smoke
```

The real path uses Unsloth `FastVisionModel` in text-only mode, so the vision
tower is not loaded or trained, together with maintained TRL `GRPOTrainer`;
project code supplies only the 2048 prompt, Policy Response validation, and
Action Quality Reward. It attempts BF16 first, records allocated and reserved
VRAM plus generation/training throughput, saves only a local LoRA adapter,
reloads it onto the same pinned base-model revision, and tests group 8 only
when group 4 leaves at least 2 GiB of reserved-VRAM headroom. A BF16 failure
is written as explicit no-go evidence before any separately reviewed
quantized fallback.

## Depth-2 Teacher Policy corpus

The Experiment Runner also owns the reproducible, leakage-safe corpus export
path. Both profiles compile the checked-in C++ exporter, verify the retained
100M Teacher Policy checkpoint checksum, assign complete trajectories to fixed
splits before sampling, and independently validate every output row:

```bash
# Fast end-to-end contract check (30 records).
python -m llm2048.experiment_runner \
  --teacher-corpus-config configs/teacher_corpus_depth2_smoke.json \
  --output-dir runs/teacher-corpus-depth2-smoke

# Opt-in production export (100k train / 10k validation / 10k test).
python -m llm2048.experiment_runner \
  --teacher-corpus-config configs/teacher_corpus_depth2_production.json \
  --output-dir runs/teacher-corpus-depth2-production
```

The production train split is exactly 50% Natural States, 30% Hard States, and
20% Late States; each Late State has a maximum tile of at least 512. These
Corpus Strata are exclusive with precedence `late > hard > natural`. The
manifest records artifact checksums, a 70k train-only immutable Teacher Core
checksum, and the Teacher Margin Scale (`tau`), calibrated from train only.
Records carry identity Trajectory Lineage and a canonical Symmetry Orbit ID
for leakage checks; augmentation remains an online-training concern.
Corpus JSONL remains under the ignored `runs/` directory and must not be
committed.
