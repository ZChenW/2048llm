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

Run the CLI contract tests:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m unittest discover -s tests -v
```

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
