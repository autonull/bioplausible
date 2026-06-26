# Circle Packing Demo

This directory contains the runnable circle-packing benchmark included in the public ASI-Evolve release. The task is to place 26 circles inside a unit square and maximize the sum of their radii.

## Included Files

```text
circle_packing_demo/
|-- config.yaml
|-- eval.sh
|-- evaluator.py
|-- initial_program
|-- init_cognition.py
|-- input.md
`-- prompts/
    |-- analyzer.jinja2
    `-- researcher.jinja2
```

## What This Demo Shows

- How the general ASI-Evolve pipeline can be adapted to a concrete optimization task.
- How the cognition store can inject domain priors before and during evolution.
- How the database and sampling policy can be used for iterative program search.
- How the evaluation loop can be wrapped as a benchmark-specific script.

## Quick Start

From the repository root:

```bash
pip install -r requirements.txt
python experiments/circle_packing_demo/init_cognition.py
python main.py \
  --experiment circle_packing_demo \
  --steps 10 \
  --sample-n 3 \
  --eval-script /absolute/path/to/experiments/circle_packing_demo/eval.sh
```

Use an absolute path for `--eval-script`. The engineer executes the evaluator from per-step working directories, so absolute paths avoid path-resolution issues.

## Notes on the Configuration

- `diff_based_evolution` is disabled in this demo so the researcher produces full rewrites.
- The database uses the island sampler with MAP-Elites-style feature bins over `complexity` and `diversity`.
- Weights & Biases logging is enabled in offline mode by default.
- The cognition retriever uses a slightly higher threshold than the repository defaults to keep retrieved items focused.

## Saved Best Programs

The repository-level directory `experiments/best/circle_packing/` stores selected high-scoring programs from ablation runs. Those files are useful as reference artifacts, but this demo is self-contained and can be run independently.

## Benchmark Context

The target score used throughout this demo is `2.635`, following the `n=26` circle-packing result highlighted in the AlphaEvolve paper. ASI-Evolve uses this task as a lightweight but meaningful benchmark for comparing evolution dynamics and framework components.
