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
have growing logical key/value content; TriGLU and the explicitly labeled token-local
attention-slot controls retain position metadata only. `TriGLUFFN` owns no cache
interface, while the attention sublayers in its all-attention control retain their full
KV state. An `ffn_only` structural-control block also retains position metadata only; it
has no attention/replacement mixer and must not be counted as a token-local replacement
mixer. The benchmark preallocates attention cache capacity to avoid timing repeated
history copies. Record context length, generated length, batch size, peak allocated
memory, logically used cache bytes, and allocated cache capacity.

Context lengths above a checkpoint's training configuration may be used to measure
architecture-level runtime and memory scaling when the benchmark records both configured
and effective context capacity. Such measurements must be labeled as efficiency
extrapolations and must not be presented as evidence of modeling quality beyond the
trained context window.

Repeated context benchmarks must be retained as separate raw artifacts. For derived
tables and figures, the bundled reporter selects the artifact with the greatest number
of measured iterations for each architecture/context pair; ties retain the first artifact
in deterministic path order. Schema-v2 benchmark artifacts must record physical depth, the exact
mixer/block plan, global FFN type, effective per-layer FFN-width schedule, total FFN
hidden width, and effective residual-initialization depth. Legacy artifacts without a
width schedule normalize to `[ffn_hidden_size] * n_layers`; missing residual-init depth
normalizes to physical depth. New nonuniform models must record their explicit schedule.
These checks prevent FFN-form and residual-topology controls from being mistaken for the
conventional all-attention baseline.

## Mechanistic diagnostics

Schema-v3 rank analyses record centered channel-covariance spectra at seven points in
each ordinary decoder block: block input, normalized mixer input, mixer update,
post-mixer residual, normalized FFN input, FFN update, and block output. An `ffn_only`
block has only four real stages—block input, normalized FFN input, FFN update, and block
output—so mixer rows are omitted rather than synthesized as zero-rank operations. Its
FFN rank delta is measured directly from block input to output. Compact summaries
include entropy effective rank, participation ratio, stable rank, rank deltas,
update/residual scale, and update/residual direction.

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

Schema-v6 summary tables expose physical `n_layers`, `attention_layers`,
`token_local_layers`, `ffn_only_layers`, `replacement_mixers`,
`residual_updates_per_forward`, `structural_controls`, `ffn_type`, the legacy/default
scalar `ffn_hidden_size`, the exact
`ffn_width_schedule`, `ffn_total_hidden_size`, and `residual_init_depth`, plus a count for
each supported replacement type. `ffn_only_layers` are deliberately excluded from
`token_local_layers` and `replacement_mixers`: they contain a token-local operation but
are structural block controls, not attention-slot mixers. This keeps attention-slot,
FFN-form, and residual-topology controls distinguishable. The derived
`experiment_family` field separates `primary`, `attention_slot_control`,
`ffn_form_control`, and `residual_topology_control` rows, and the generated Markdown
report presents those families in separate tables. Replicas grouped under one
architecture must also agree on parameter count, physical depth and exact layer plan,
width schedule, initialization depth, context, token/evaluation budgets, and validation
data identity.
Architecture names are derived from the run directory after removing a trailing
`_seed<integer>` replication suffix. Automatic baseline-relative metrics prefer the
canonical conventional-SwiGLU all-attention architecture; a lower-loss all-attention
FFN control cannot silently become the baseline.

The reporter can be rerun after new raw artifacts arrive. Derived files should not be
edited by hand.
