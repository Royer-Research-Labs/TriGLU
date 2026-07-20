# Exploratory 20-layer screening results

This page records the seed-1337 screening sweep that selected architectures for focused
replication. It is an exploratory snapshot, not a confirmatory multi-seed conclusion.
Seeds 2357 and 7331 are assigned to the attention baseline, front-blend hybrid, and
final-attention hybrid by `scripts/run_replication_seeds.sh`.

## Fixed comparison settings

- FineWeb-Edu `sample-10BT`, pinned revision
  `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`;
- data-manifest SHA256
  `ad4a56313b9024327991917698f44f68a2ae3d7e3e4132d74b9078ebd72f4b1f`;
- 20 layers, width 512, 8 heads, context 4,096;
- 89,018,880 trainable parameters in every plan;
- 1,000,013,824 training tokens and 1,048,576 validation targets per evaluation;
- BF16 with `torch.compile` on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition;
- PyTorch `2.13.0+cu130`.

## Screening outcomes

The throughput column is the median of the latter half of training-log measurements. It
includes batch sampling and reflects observed run conditions; it is not a controlled
synthetic benchmark.

| Plan | Attention | TriGLU | Validation loss | Perplexity | Accuracy | Observed train tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `15a5t_final_attention` | 15 | 5 | 3.44594 | 31.3727 | 38.069% | 221,343 |
| `15a5t_front_blend` | 15 | 5 | 3.44745 | 31.4202 | 38.031% | 214,337 |
| `15a5t_late_alternating` | 15 | 5 | 3.45228 | 31.5723 | 38.039% | 216,408 |
| `20a0t` | 20 | 0 | 3.45719 | 31.7276 | 37.948% | 205,314 |
| `15a5t_tail_block` | 15 | 5 | 3.46864 | 32.0931 | 37.861% | 221,423 |
| `9a11t_front_blend` | 9 | 11 | 3.47840 | 32.4077 | 37.715% | 236,676 |
| `15a5t` | 15 | 5 | 3.50036 | 33.1275 | 37.455% | 220,069 |
| `10a10t` | 10 | 10 | 3.51170 | 33.5051 | 37.364% | 232,409 |
| `5a15t` | 5 | 15 | 3.51643 | 33.6641 | 37.301% | 253,544 |

The screening seed places the final-attention and front-blend plans slightly ahead of the
attention baseline, while more aggressive replacement increases observed throughput at a
quality cost. These differences motivated focused replication; selection on this same
seed prevents it from serving as independent confirmation.

The table above uses the RoPE TriGLU mixer, the original component form. The sweeps below
use the no-RoPE variant, which the confirmatory experiments found to be the stronger
attention replacement (see [`results.md`](results.md)); they are single-seed screens under
the same fixed settings.

## No-RoPE replacement-amount ladder (seed 1337)

Nested placement, each larger plan replacing a superset of the smaller plan's layers:

| Plan | Attention | Replaced | Validation loss | Perplexity | Accuracy | Observed train tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `15a5_triglu_no_rope_nested` | 15 | 5 | 3.43916 | 31.1608 | 38.142% | 215,348 |
| `12a8_triglu_no_rope_nested` | 12 | 8 | 3.44622 | 31.3815 | 38.135% | 227,156 |
| `9a11_triglu_no_rope_nested` | 9 | 11 | 3.45172 | 31.5546 | 37.978% | 239,619 |
| `20a0t` (baseline) | 20 | 0 | 3.45719 | 31.7276 | 37.948% | 205,314 |
| `18a2_triglu_no_rope_nested` | 18 | 2 | 3.46181 | 31.8747 | 37.965% | 204,660 |
| `6a14_triglu_no_rope_nested` | 6 | 14 | 3.53050 | 34.1410 | 37.256% | 253,986 |

Replacement quality is non-monotonic in amount: too little (two layers) is
indistinguishable from the baseline, a broad sweet spot spans roughly five to eleven
replacements, and quality falls off a cliff between eleven (55%) and fourteen (70%)
replacements while throughput keeps rising. The 55% plan carried forward to confirmatory
replication.

## No-RoPE placement sweep at 25% replacement (seed 1337)

Five replacements held fixed while their positions move:

| Plan | Zero-based no-RoPE layers | Validation loss | Perplexity | Accuracy | Observed train tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `15a5_triglu_no_rope_front_blend` | 8, 12, 15, 17, 19 | 3.43753 | 31.1101 | 38.155% | 214,357 |
| `15a5_triglu_no_rope_nested` | 7, 11, 15, 17, 19 | 3.43916 | 31.1608 | 38.142% | 215,348 |
| `15a5_triglu_no_rope_tail_block` | 15, 16, 17, 18, 19 | 3.45096 | 31.5306 | 38.055% | 215,087 |
| `15a5_triglu_no_rope_early_intrusion` | 3, 12, 15, 17, 19 (front-blend, layer 8 → 3) | 3.45966 | 31.8062 | 37.939% | 215,797 |
| `15a5_triglu_no_rope_repeating` | 3, 7, 11, 15, 19 | 3.48761 | 32.7077 | 37.652% | 214,804 |

At a fixed amount, throughput is nearly constant while quality spans 0.05 loss: attention-
rich-front and nested placements lead, and evenly-repeating placement trails. Front-blend
and early-intrusion were carried to confirmatory replication, where the early-intrusion
penalty held (see [`results.md`](results.md)).

## Evidence availability

Mutable local outputs under `runs/` and `results/generated/` are gitignored to prevent
partial runs and large checkpoints from being published accidentally. A release that
reports these results must include the completed runs' resolved configs, metric streams,
environment metadata, data hashes, raw evaluation/benchmark/analysis JSON, and generated
report as a curated release artifact or archival deposit. The required contents are
specified in [`results-schema.md`](results-schema.md).
