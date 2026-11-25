# Results folders

Existing result files are grouped by experiment:

- `experiment_001_kv_cache/` — cache measurements and context-length plot.
- `experiment_002_similarity/` — prompt comparisons, similarity measurements,
  and plots from the semantic-similarity experiments.
- `experiment_003_probes/` — operation, wording-generalization, operand, and
  token-position probe outputs.
- `experiment_004_kv_cache_swap/` — operator K/V substitution measurements
  and plot.
- `experiment_005_activation_patching/` — hidden-state patching measurements
  and plot.

The experiment scripts currently continue to write new outputs directly into
this `results/` directory. Move new files into the appropriate folder after
each run if desired. Numbered filename suffixes distinguish repeat runs.
