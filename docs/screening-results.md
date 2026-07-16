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

## Evidence availability

Mutable local outputs under `runs/` and `results/generated/` are gitignored to prevent
partial runs and large checkpoints from being published accidentally. A release that
reports these results must include the completed runs' resolved configs, metric streams,
environment metadata, data hashes, raw evaluation/benchmark/analysis JSON, and generated
report as a curated release artifact or archival deposit. The required contents are
specified in [`results-schema.md`](results-schema.md).
