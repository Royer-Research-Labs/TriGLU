# Results, provenance, and measurement protocol

This protocol defines the minimum evidence needed to review or reuse a reported result.
Generated charts and summary tables are presentation artifacts; the resolved config,
metric stream, data identity, and environment metadata are the underlying record.

## Quality metrics

Evaluation aggregates summed next-token cross-entropy over the exact number of held-out
tokens, then reports mean loss and perplexity. Next-token accuracy is secondary and uses
the same token denominator. Compare every plan at the same training-token checkpoints;
do not select a different best checkpoint for each architecture unless that selection
rule is declared in advance.

Each run directory must retain:

- the fully resolved YAML/JSON configuration and explicit layer list;
- JSONL training/evaluation metrics;
- checkpoints containing model, optimizer, step, token count, and random states;
- Python, PyTorch, CUDA, device, dtype, and compile metadata;
- the token-file manifest and hashes used by the run.

For multi-seed comparisons, retain every run rather than only aggregate values. Report the
per-architecture sample count, seeds, mean, and sample standard deviation. Baseline-relative
per-run metrics must use a baseline trained with the same seed.

## Throughput

The training loop logs observed optimizer-step throughput, including batch sampling but
excluding evaluation and checkpoint overhead. The standalone benchmark separately
measures architecture throughput on preallocated synthetic token batches so data loading
is not attributed to either mixer. Warmup and compilation iterations are excluded from
standalone measurements, and CUDA synchronizes around each timed region.

Report at least the median steady-state tokens/second together with all raw samples,
batch size, sequence length, gradient accumulation, dtype, compile setting, device, and
PyTorch version. Keep those settings identical across plans; never silently reduce the
batch after an out-of-memory error.

Inference reports prefill and cached-decode throughput separately. Only attention layers
have growing logical key/value content; TriGLU layers retain position metadata only.
The benchmark preallocates attention cache capacity to avoid timing repeated history
copies. Record context length, generated length, batch size, peak allocated memory,
logically used cache bytes, and allocated cache capacity.

Context lengths above a checkpoint's training configuration may be used to measure
architecture-level runtime and memory scaling when the benchmark records both configured
and effective context capacity. Such measurements must be labeled as efficiency
extrapolations and must not be presented as evidence of modeling quality beyond the
trained context window.

Repeated context benchmarks must be retained as separate raw artifacts. For derived
tables and figures, the bundled reporter selects the artifact with the greatest number
of measured iterations for each architecture/context pair; ties retain the first artifact
in deterministic path order.

## Mechanistic diagnostics

Schema-v2 rank analyses record centered channel-covariance spectra at seven points in
each decoder block: block input, normalized mixer input, mixer update, post-mixer
residual, normalized FFN input, FFN update, and block output. Compact summaries include
entropy effective rank, participation ratio, stable rank, rank deltas, update/residual
scale, and update/residual direction.

Layer sensitivity zeros one trained residual update at a time while retaining all other
sublayers. This is an out-of-distribution intervention. A large loss delta shows that the
checkpoint depends on that update under the intervention; a small delta does not prove
that the component was unnecessary during training or that several such updates can be
removed jointly.

## Derived reports

`python -m triglu.report --suite <suite>` writes a reproducible report under
`results/generated/<suite>-report/`:

- `summary.csv` contains one row per run;
- `summary_by_architecture.csv` contains multi-seed means and sample standard deviations;
- `summary_confirmatory.csv` excludes the exploratory screening seed when confirmatory
  runs are available;
- incomplete runs are excluded from every aggregate and recorded in
  `incomplete_runs.csv`;
- `training_curves.csv`, `layer_diagnostics.csv`, `layer_transitions.csv`, and
  `layer_sensitivity.csv` provide tidy source data for figures;
- `context_benchmarks.csv` and the `context_*` SVGs summarize labeled context-scaling
  benchmark families; and
- SVG files visualize quality/throughput, learning curves, activation geometry, update
  scale, and intervention sensitivity.

The reporter can be rerun after new raw artifacts arrive. Derived files should not be
edited by hand.
