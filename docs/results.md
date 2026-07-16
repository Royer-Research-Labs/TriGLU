# Results: quality retention and long-context efficiency

In an 89M-parameter decoder trained for one billion tokens, replacing five of twenty
attention layers with TriGLU retained the all-attention baseline's validation quality in
two tested placements. The hybrids used 25% less KV-cache capacity and improved 4K
prefill and cached-decode throughput. Their advantage increased in long-context runtime
benchmarks, while a batch-one 1K training benchmark showed that TriGLU is not uniformly
faster in every operating regime.

The evidence below separates confirmatory quality measurements, controlled efficiency
benchmarks, exploratory placement results, and mechanistic diagnostics.

## What was tested

The primary suite contains 15 completed 20-layer, width-512 runs at 4,096-token context:

- nine seed-1337 screening runs spanning replacement ratios and placements;
- two additional attention-baseline runs at seeds 2357 and 7331; and
- two additional runs at each confirmatory seed for `15a5t_front_blend` and
  `15a5t_final_attention`.

Every run has 89,018,880 trainable parameters, sees 1,000,013,824 FineWeb-Edu training
tokens, and evaluates the same 1,048,576 validation targets. Seeds 2357 and 7331 form the
confirmatory subset; seed 1337 remains exploratory because it was used to select the
focused architectures.

## Confirmatory quality result

| Architecture | Seeds | Mean validation loss | Loss SD | Matched loss delta | Mean perplexity | Observed train tok/s | Throughput vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20a0t` | 2357, 7331 | 3.463930 | 0.007754 | 0 | 31.9428 | 206,231 | 1.000× |
| `15a5t_front_blend` | 2357, 7331 | 3.463040 | 0.009051 | -0.000890 | 31.9145 | 220,562 | 1.069× |
| `15a5t_final_attention` | 2357, 7331 | 3.464796 | 0.011435 | +0.000866 | 31.9710 | 221,023 | 1.072× |

Replacing five of twenty attention layers retained validation quality to within 0.0009
mean loss in both tested placements. The observed training runs were about 7% faster.
With only two independent confirmatory seeds, these measurements support a quality-
retention result, not a claim that either hybrid improves quality or a formal statistical
equivalence bound.

The broader seed-1337 sweep shows that replacement ratio alone is not sufficient:
`15a5t_tail_block` and uniformly spaced `15a5t` trail the selected placements, and more
aggressive 9A/11T, 10A/10T, and 5A/15T plans trade progressively more quality for speed.
See the [screening table](screening-results.md) for every plan.

## Controlled 4K efficiency result

The standalone benchmark excludes data loading and compilation warmup. It measures
training, prompt prefill, and eager cached decode separately on preallocated synthetic
tokens.

| Architecture | Train tok/s | Train vs baseline | Prefill tok/s | Prefill vs baseline | Decode tok/s | Decode vs baseline | KV capacity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20a0t` | 176,225 | 1.000× | 648,932 | 1.000× | 87.33 | 1.000× | 0.156 GiB |
| `15a5t_front_blend` | 178,289 | 1.012× | 706,108 | 1.088× | 95.80 | 1.097× | 0.117 GiB |
| `15a5t_final_attention` | 191,674 | 1.088× | 683,478 | 1.053× | 94.68 | 1.084× | 0.117 GiB |

Both hybrids reduce allocated KV-cache capacity by exactly 25%, matching the reduction
from twenty to fifteen attention layers. Prefill improves by 5–9% and cached decode by
8–10%. Controlled training timing is more sensitive to placement and compiler scheduling:
the two equal-ratio hybrids span 1–9% improvement, so the repository reports both rather
than selecting one number.

## Context scaling

The context sweep fixes batch size at one and compares `20a0t` with
`15a5t_front_blend`. The 1K and 2K points use 50 measured iterations after 10 warmups;
the longer contexts use 10 measured iterations after 3 warmups.

| Context | Training change | Prefill change | Decode change | Prefill-memory change | KV-cache change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | -27.9% | +0.9% | +7.7% | -3.5% | -25% |
| 2K | +2.9% | -1.6% | +5.8% | -6.8% | -25% |
| 4K | +8.1% | +11.9% | +5.8% | -10.1% | -25% |
| 8K | +12.1% | +16.0% | +9.3% | -13.0% | -25% |
| 16K | +15.8% | +19.8% | +6.9% | -15.8% | -25% |

At batch size one, the short 1K training case is dominated by utilization and kernel
overhead; replacing attention is slower there. The training crossover occurs between 1K
and 2K, while the clearest training and prefill gains appear from 4K onward. Decode is
consistently faster and KV capacity is consistently lower.

The models were trained at 4K. The 8K and 16K measurements extend RoPE positions and
cache capacity only to measure architecture-level runtime and memory scaling. They do
not establish language-modeling quality beyond the trained window.

## Mechanistic interpretation

The stage-separated effective-rank trajectories are non-monotonic. Attention and TriGLU
updates both expand rank at some layers and contract it at others; large contractions are
localized rather than evidence of one persistent depth threshold. The tail-block plan
also performs worse than the selected blended placements.

These observations do not support a model in which attention becomes replaceable after
one universal rank-collapse point. The current evidence instead supports a narrower
placement hypothesis: early and periodically retained attention layers provide cross-token
communication, while selected later attention sublayers can be replaced by token-local
gating without measurable quality loss at this scale. Rank, update scale, and intervention
sensitivity remain diagnostics, not an identified causal mechanism.

## Implications for component users

TriGLU is best treated as a hybrid building block, not as a complete substitute for token
mixing. The results suggest a practical initial integration strategy:

- retain attention throughout the network, with an attention-rich lower stack;
- begin with roughly one TriGLU layer for every three retained attention layers;
- benchmark at the deployment context and batch size, because the speed crossover is
  hardware- and workload-dependent; and
- validate placement rather than assuming that equal replacement ratios are equivalent.

TriGLU preserves the mixer's input/output width and projection parameter count, but its
token-local computation cannot recover information that was never communicated by a
retained attention layer. Results from different wrappers, scales, datasets, or placement
plans should therefore be treated as new experiments.

## Supported and unsupported claims

The completed experiment supports the following statements at the tested scale:

- selected 15-attention/5-TriGLU hybrids retain the 20-attention baseline's validation
  quality after one billion training tokens;
- placement materially affects quality at a fixed replacement ratio;
- replacing 25% of attention layers removes 25% of KV-cache capacity; and
- long-context training and prefill efficiency improve as attention's context-dependent
  work becomes more significant.

It does not establish:

- quality equivalence at substantially larger parameter counts or training budgets;
- language-modeling quality beyond 4K context;
- downstream task or long-context retrieval performance;
- that attention can be removed entirely; or
- rank collapse or any other single mechanism as the cause of successful replacement.

## Reproducibility artifacts

The reporting pipeline writes source tables, editable SVG figures, benchmark records,
and mechanistic diagnostics under `results/generated/` (gitignored; regenerated with
`python -m triglu.report --suite 20l_4k_1b`). The complete evidence set behind this
document — every completed run's resolved config, metric stream, environment metadata,
and data hashes, the raw evaluation/benchmark/rank-analysis JSON, and the generated
`20l_4k_1b-report/` tables and figures — is published as the
[v0.1.0 evidence archive](https://github.com/Royer-Research-Labs/TriGLU/releases/tag/v0.1.0).
Inside that archive, start with:

- `results/generated/20l_4k_1b-report/README.md`
- `results/generated/20l_4k_1b-report/summary_confirmatory.csv`
- `results/generated/20l_4k_1b-report/context_benchmarks.csv`
- `results/generated/20l_4k_1b-report/layer_transitions.csv`

Large checkpoints and tokenized corpora are intentionally separate from the source
distribution. The [results and provenance protocol](results-schema.md) defines the raw
records required to reproduce or audit a published comparison.
