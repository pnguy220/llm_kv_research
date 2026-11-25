# LLM Research: Arithmetic Representations and KV Cache

This repository contains my experiments with Qwen2.5-0.5B. I am investigating
whether a model's internal representations of arithmetic operations can be
identified and substituted across prompts, with the longer-term aim of
understanding whether repeated computation could be reused to reduce work or
memory during inference.

The experiments so far measure the model's KV cache, compare internal
representations, train small probes to read operation and operand information,
and intervene on cached or hidden-state representations. Results are
exploratory: they show behavior on the tested prompts and setup, not a general
reuse method or a demonstrated reduction in inference cost.

## Project files

- `inspect_cache.py` — Experiment 001: inspect KV tensors and their growth.
- `experiment_002_*.py` — Experiment 002: compare representations for
  controlled arithmetic prompts.
- `experiment_003_*.py` — Experiment 003: test what operation and operand
  information can be read from hidden states.
- `experiment_004_kv_swap.py` — Experiment 004: substitute operator K/V cache
  entries and observe answer behavior.
- `experiment_005_activation_patch.py` — Experiment 005: replace selected
  hidden-state activations and observe answer behavior.
- `EXPERIMENT_SUMMARY.md` — Plain-language summary of the experiments and
  current findings.
- `PROJECT_NOTES.md` — Detailed research log, measurements, decisions, and
  next steps.
- `results/` — Experiment CSV files and plots.

## Running the experiments

The scripts use the existing Qwen2.5-0.5B checkpoint and Python environment.
The model experiments expect CUDA and use FP16; they do not download a model
when run because model loading uses the local Hugging Face cache.

In the configured WSL environment:

```bash
llmvenv
python inspect_cache.py
```

Run another experiment by substituting its script name, for example:

```bash
python experiment_005_activation_patch.py
```

Most experiment scripts save uniquely named CSVs and plots in `results/` to
avoid replacing earlier runs. Dependencies used by the scripts include
PyTorch, Transformers, and Matplotlib; use the project's existing environment.

## Current scope

The work currently focuses on small arithmetic examples with Qwen2.5-0.5B.
Experiments 001–005 have been run, but the research question remains open:
whether internal information about a requested operation can be transferred
reliably across inputs and contexts in a way that supports actual reuse.
