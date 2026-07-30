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
  llm2048-experiment \
  --config tests/fixtures/experiment_runner_tracer.json \
  --output-dir runs/experiment-runner-tracer
```

The run writes a resolved manifest, deterministic events and result, an
atomic checkpoint, a W&B offline run, and local TensorBoard events. It does
not load a model or require a GPU.

Run the CLI contract tests:

```bash
/home/chakew/miniconda3/bin/conda run -n td2048 \
  python -m unittest discover -s tests -v
```
