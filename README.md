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

The production train split is exactly 50% natural, 30% hard, and 20% late;
late states have a maximum tile of at least 512. Natural, hard, and late are
exclusive with precedence `late > hard > natural`. The manifest records
artifact checksums, a 70k train-only immutable Teacher core checksum, and
`tau`, calibrated as the median strictly positive top-1 minus top-2 margin
from train only. Records carry identity symmetry lineage and a canonical D4
orbit ID for leakage checks; augmentation remains an online-training concern.
Corpus JSONL remains under the ignored `runs/` directory and must not be
committed.
