# TriGLU: a token-local gated alternative to selected attention layers

TriGLU—the **Triple-Product Gated Linear Unit**—is a token-local channel mixer designed
for use in place of selected causal-attention sublayers. It applies three projected
factors at each token independently and requires no KV cache. This repository provides
both a small, readable PyTorch implementation and a controlled decoder-only Transformer
harness for measuring quality, training throughput, inference throughput, memory, and
KV-cache trade-offs on one NVIDIA GPU.

The scientific question is deliberately narrow:

> As standard causal-attention sublayers are progressively replaced with TriGLU,
> how much language-modeling quality is retained, and what training/inference
> efficiency is gained?

TriGLU originated as an ablation in which cross-token mixing was removed from a richer
experimental mixer. The token-local remainder retained enough empirical promise to merit
an independent test. The richer mixer is not included here: this repository isolates the
remaining equation and asks what it contributes under controlled conditions.

Within each fixed-seed comparison, the experiments vary only the explicit
attention/TriGLU layer plan. Comparisons between the 12-layer and 20-layer suites are not
controlled ablations because their depth, context, and training budget differ.

## Repository scope and evidence

The model, deterministic data preparation, training, checkpoint/resume, evaluation,
mechanistic analysis, reporting, and benchmark paths are executable. CPU unit tests and
a tiny offline smoke configuration are included. Installation and tests never launch a
full experiment suite.

Only results accompanied by resolved configs, metric streams, data hashes, and runtime
metadata are treated as reproduced evidence. The legacy figures in `assets/` do not have
that complete provenance and are excluded from the reported comparisons. The publication
requirements are defined in [`docs/results-schema.md`](docs/results-schema.md).

### Reading guide

- **Component users:** start with [The exact TriGLU component](#the-exact-triglu-component),
  then consult the [framework-independent pseudocode](pseudocode/triglu_layer.pseudo) and
  [equation tests](tests/test_triglu.py).
- **Reviewers:** read the [experimental design](docs/design.md), the explicit configs under
  `configs/`, the [reproduced results](docs/results.md), and the
  [results/provenance protocol](docs/results-schema.md).
- **Reproducers:** follow [Install and verify](#install-and-verify), prepare the pinned data,
  and run only the experiment suite relevant to the intended comparison.

## Evidence summary

In the completed 89M-parameter experiment, two hybrids that replaced five of twenty
attention layers retained the all-attention baseline's validation quality after one
billion FineWeb-Edu training tokens. Both used an attention-rich lower stack and retained
attention across later depth.

Across the two confirmatory seeds, `15a5t_front_blend` differs from the attention baseline
by -0.00089 mean validation loss and `15a5t_final_attention` by +0.00087. Their observed
training throughput is approximately 7% higher. Controlled 4K benchmarks show 5–9% faster
prefill, 8–10% faster cached decode, and exactly 25% less allocated KV-cache capacity.
For the front-blend hybrid, controlled training and prefill gains rise to 15.8% and 19.8%
at 16K; those contexts are efficiency extrapolations because quality training stopped at
4K.

These results position TriGLU as a hybrid component for selectively reducing attention,
especially at longer contexts. They do not show that TriGLU improves quality or can replace
all token mixing. Placement matters, and the mechanistic diagnostics do not identify one
universal rank-collapse threshold. Read the [complete results and limitations](docs/results.md)
and the [exploratory screening table](docs/screening-results.md).

Large checkpoints, prepared corpora, and mutable run directories are excluded from the
source distribution. The [results/provenance protocol](docs/results-schema.md) specifies
the records needed to audit or reproduce reported comparisons.

## The exact TriGLU component

For input `x` with shape `[batch, tokens, channels]`, `C = channels`:

```text
k, g, v = split_3(Linear_C_to_3C(x))
k       = RoPE(k, offset=position)
y       = k * SiLU(g) * v
y       = Linear_C_to_C(y)
```

Both products are elementwise. RoPE is applied to the full `C`-wide `k` stream only.
There is no attention matrix, convolution, recurrence, reduction over tokens, or other
cross-token operation. Position affects a token's channel transform, but one token cannot
read another token inside this component.

The mixer itself does not require or apply normalization. The evaluated decoder wrapper
passes `RMSNorm(x)`; another integration is responsible for defining its own input
normalization policy.

The name describes its relationship to the GLU family. A SwiGLU combines two projected
factors, `SiLU(g) * v`; TriGLU adds the third projected factor `k`, giving the exact
triple product `RoPE(k) * SiLU(g) * v`. “Tri” refers to these three factors, not three
separate gates. The terminology follows the original
[Gated Linear Unit](https://arxiv.org/abs/1612.08083) and subsequent
[Transformer GLU-variant](https://arxiv.org/abs/2002.05202) literature.

At cached inference time a TriGLU layer's cache entry is the tuple
`(None, None, next_position)`: it stores no growing key/value tensors. Only retained
attention layers carry KV state.

This is not sigmoid gating, linear attention, or an attention approximation. The readable
implementation is
[`TriGLU`](src/triglu/layers.py), and direct equation tests are in
[`tests/test_triglu.py`](tests/test_triglu.py).

### Component contract

`TriGLU` is the mixer only: it does not contain normalization, a residual connection, or
the decoder FFN. This keeps the component independently testable and allows users to place
it inside an existing block topology without inheriting repository-specific scaffolding.

| Property | Contract |
| --- | --- |
| Input/output | `[batch, tokens, channels]` with unchanged shape |
| Cross-token mixing | None |
| Position input | Absolute `cache_position`, applied through full-width RoPE on `k` |
| Training parameters | One `C → 3C` projection and one `C → C` projection |
| Autoregressive state | Position integer only; no K/V tensors |
| Return value | `(output, cache_or_none)` |

Minimal construction inside another PyTorch model is:

```python
from triglu import RopeModule, TriGLU

rope = RopeModule(config.d_model, theta=config.rope_theta)
mixer = TriGLU(config, rope)
y, next_cache = mixer(x, cache_position=position, use_cache=use_cache)
```

`config` must expose `d_model`, `bias`, and `rope_theta` as used by this repository's
`ModelConfig`. The experimental claims in this repository apply to the decoder wrapper
below; integrations using a different wrapper are valid uses of the mixer but constitute
separate experiments.

## Controlled decoder architecture

Both mixer choices use the same conventional pre-norm wrapper:

```text
h     = x + mixer(RMSNorm(x))
x_out = h + SwiGLU(RMSNorm(h))
```

The rest of the model is intentionally conservative:

- standard multi-head causal self-attention through PyTorch scaled-dot-product attention;
- RoPE on attention queries/keys, with head dimension 64 in the supplied experiment configs;
- `torch.nn.RMSNorm`, standard SwiGLU, and residual connections;
- tied token-embedding/output-head weights;
- zero dropout by default (the `dropout` setting reaches only attention weights;
  TriGLU and FFN paths have no dropout, so a nonzero value would regularize the
  retained attention layers alone);
- GPT-style depth scaling for residual output projections;
- AdamW (fused on supported CUDA builds), cosine decay with warmup, and gradient clipping;
- automatic CUDA BF16 selection when supported, otherwise FP32;
- optional `torch.compile`, disabled in correctness tests.

The component boundary is intentional. Holding the wrapper fixed ensures that replacing
attention does not simultaneously change normalization, residual flow, or FFN placement.
See [`docs/design.md`](docs/design.md) for the experimental controls and explicit layer
placements.

No GQA/MQA, sliding attention, additional experimental temporal mixer, alternative
gate/FFN family, custom positional encoding or normalization, convolution, state-space
layer, MoE, shared-weight bank, or proprietary architecture component is included.

## Install and verify

Python 3.10+ and PyTorch 2.4+ are required.

For component use:

```bash
python -m pip install -e .
```

For repository development and data preparation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data]"
pytest
```

The data extras are needed only for downloading/tokenizing public corpora; the model,
tests, synthetic smoke run, and prepared `.bin` files do not import them.

Run the short offline smoke test (three CPU optimizer steps, evaluation, and checkpoint):

```bash
python -m triglu.train \
  --config configs/smoke.yaml \
  --output-dir runs/smoke
```

A fresh (non-resume) invocation refuses to start when its output directory already
contains run records; re-running the smoke test into the same directory requires
`--overwrite-run`, which discards the previous records deliberately.

Then exercise the saved checkpoint and all three benchmark paths with deliberately tiny
settings:

```bash
python -m triglu.evaluate \
  --checkpoint runs/smoke/latest.pt \
  --synthetic --batches 2 --device cpu

python -m triglu.benchmark \
  --checkpoint runs/smoke/latest.pt \
  --batch-size 1 --sequence-length 8 --prompt-length 8 --decode-tokens 4 \
  --warmup 1 --iterations 2 --device cpu \
  --output runs/smoke/benchmark.json
```

## Prepare FineWeb-Edu data for the reference suite

Both suites use tiktoken's GPT-2 encoding (50,257 emitted IDs) with one GPT-2
EOS token appended after every document, including the last. The model vocabulary is padded to 50,304 for hardware
alignment; IDs 50,257–50,303 are never emitted. Documents are concatenated without PAD,
then sampled as contiguous next-token windows. The validation prefix is prepared first,
and its documents never appear in the training stream.

Prepare exactly 200 million training tokens and 5 million validation tokens from the
FineWeb-Edu 10BT sample. The pinned immutable revision explicitly declares the
`sample-10BT` configuration in its dataset metadata:

```bash
python -m triglu.prepare_data \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --revision fc9850dff5e2d0f8f776efe41b24a1c49556cfc5 \
  --split train --text-column text \
  --train-tokens 200000000 \
  --val-tokens 5000000 \
  --output-dir data/fineweb_edu_10bt
```

The dataset is public, so authentication is optional. Setting `HF_TOKEN` avoids the Hub's
unauthenticated-request warning and provides higher rate limits.

This is the only dataset preparation required for the four-run reference comparison. It
writes `train.bin`, `val.bin`, `manifest.json`, and `SHA256SUMS`. The manifest records
the source revision, tokenizer/version, split policy, document counts, exact token counts,
and file hashes. Keep it with every published result.

Review the upstream dataset licenses and terms before redistribution. Prepared corpora and
checkpoints are ignored by git.

## Lightweight 12-layer reference experiment

All four reference configs use the same 12-layer, width-512 decoder:

```yaml
vocab_size: 50304
n_layers: 12
d_model: 512
n_heads: 8
ffn_hidden_size: 1376
context_length: 1024
dropout: 0.0
```

Every plan has exactly 63,713,792 trainable parameters; attention and TriGLU use
equal-size input/output projections, so parameter count does not move with the ratio.

Each run consumes 1,526 optimizer steps × 65,536 tokens/step = **100,007,936 training
tokens**, with the same batch sequence, initialization seed, optimizer, schedule, and
evaluation cadence. Each in-training evaluation covers exactly **204,800 target tokens**.
The explicit plans are:

| Config | Attention | TriGLU | Placement summary |
| --- | ---: | ---: | --- |
| `12a0t.yaml` | 12 | 0 | full-attention baseline |
| `9a3t.yaml` | 9 | 3 | TriGLU at zero-based layers 3, 7, 11 |
| `6a6t.yaml` | 6 | 6 | alternating attention/TriGLU |
| `3a9t.yaml` | 3 | 9 | attention at layers 0, 4, 8 |

The TriGLU sets are nested: every more aggressive plan converts a superset of the
same layers. This keeps placement changes from obscuring the replacement-ratio trend.

Run each command explicitly on one GPU; they are not chained or launched automatically:

```bash
python -m triglu.train --config configs/12l_1k_100m/12a0t.yaml
python -m triglu.train --config configs/12l_1k_100m/9a3t.yaml
python -m triglu.train --config configs/12l_1k_100m/6a6t.yaml
python -m triglu.train --config configs/12l_1k_100m/3a9t.yaml
```

These runs are grouped under `runs/12l_1k_100m/<plan>/`.

The default effective batch is 64 sequences (65,536 tokens): microbatch 4 × gradient
accumulation 16. If that microbatch does not fit the target GPU, change it before **all**
runs and compensate accumulation so the effective token batch stays fixed. Never compare
throughput from silently different batch sizes. Exact runtime depends heavily on GPU and
compile/kernel versions; budget roughly 400 MB for prepared FineWeb token files plus one
final checkpoint and small run metadata per plan.

### Scaled 20-layer, 4K-context experiment

The configs under `configs/20l_4k_1b/` form a separate 1B-token suite. They use 20
layers at width 512, native 4,096-token sequences, microbatch 16, and gradient
accumulation 1. This preserves the shared 65,536 tokens per optimizer update while
moving the comparison into a regime where attention's quadratic work is material. Their
outputs are grouped under `runs/20l_4k_1b/<plan>/`.

Every plan has 89,018,880 trainable parameters and evaluates 1,048,576 target tokens at
each evaluation point. The completed screening runs allocated up to approximately
45.2 GiB of CUDA memory and took roughly 75–90 minutes per plan on an NVIDIA RTX PRO 6000
Blackwell Workstation Edition. Treat those values as observed reference points, not
portable requirements: compiler, kernel, and GPU versions materially affect both memory
and runtime. The prepared 1B-token stream occupies approximately 2.01 GB on disk.

Prepare the separate 1B-token stream before running these configs:

```bash
python -m triglu.prepare_data \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --revision fc9850dff5e2d0f8f776efe41b24a1c49556cfc5 \
  --split train --text-column text \
  --train-tokens 1000000000 \
  --val-tokens 5000000 \
  --output-dir data/fineweb_edu_10bt_1b
```

The main ratio sweep uses nested replacements. The placement ablation holds the 15/5
ratio fixed, while `9a11t_front_blend` tests a more aggressive attention-rich-front
schedule:

| Config | Attention | TriGLU | Zero-based TriGLU layers |
| --- | ---: | ---: | --- |
| `20a0t.yaml` | 20 | 0 | none |
| `15a5t.yaml` | 15 | 5 | 3, 7, 11, 15, 19 |
| `10a10t.yaml` | 10 | 10 | 1, 3, 5, ..., 19 |
| `9a11t_front_blend.yaml` | 9 | 11 | 6, 7, 9–11, 13–15, 17–19 |
| `5a15t.yaml` | 5 | 15 | all except 0, 4, 8, 12, 16 |
| `15a5t_front_blend.yaml` | 15 | 5 | 8, 12, 15, 17, 19 |
| `15a5t_late_alternating.yaml` | 15 | 5 | 11, 13, 15, 17, 19 |
| `15a5t_tail_block.yaml` | 15 | 5 | 15, 16, 17, 18, 19 |
| `15a5t_final_attention.yaml` | 15 | 5 | 8, 12, 15, 17, 18 |

Run each experiment explicitly; no command launches the suite automatically:

```bash
python -m triglu.train --config configs/20l_4k_1b/20a0t.yaml
python -m triglu.train --config configs/20l_4k_1b/15a5t.yaml
python -m triglu.train --config configs/20l_4k_1b/10a10t.yaml
python -m triglu.train --config configs/20l_4k_1b/9a11t_front_blend.yaml
python -m triglu.train --config configs/20l_4k_1b/5a15t.yaml
python -m triglu.train --config configs/20l_4k_1b/15a5t_front_blend.yaml
python -m triglu.train --config configs/20l_4k_1b/15a5t_late_alternating.yaml
python -m triglu.train --config configs/20l_4k_1b/15a5t_tail_block.yaml
python -m triglu.train --config configs/20l_4k_1b/15a5t_final_attention.yaml
```

Microbatch 16 is an intended starting point for a high-memory GPU. Before launching all
nine plans, smoke-test `20a0t` on the target GPU. If it does not fit, reduce `batch_size`
identically in every config; increasing gradient accumulation preserves the effective
batch but changes the suite's accumulation-1 execution design and must be disclosed.

The supplied configs set `checkpoint_interval: 0`. This disables periodic recovery
snapshots and retains only `latest.pt` at normal completion. For a long or preemptible run,
set a positive interval before starting; for example, `checkpoint_interval: 500` retains a
numbered recovery checkpoint every 500 optimizer steps in addition to updating `latest.pt`.

Resume a run without changing its model config:

```bash
python -m triglu.train \
  --config configs/12l_1k_100m/6a6t.yaml \
  --resume runs/12l_1k_100m/6a6t/latest.pt
```

Without `--resume`, the trainer treats an output directory that already contains run
records as protected evidence and exits; `--overwrite-run` is the explicit opt-out.

### Replicate the leading 20-layer plans

The seed-1337 screening sweep selected `20a0t`, `15a5t_front_blend`, and
`15a5t_final_attention` for focused replication. Launch seeds 2357 and 7331 for all three
architectures sequentially on one GPU:

```bash
bash scripts/run_replication_seeds.sh
```

The launcher uses the base configs with only `training.seed` and `output_dir`
overridden. Runs are written beside the originals as, for example,
`runs/20l_4k_1b/15a5t_final_attention_seed2357/`. Completed runs are skipped safely;
an existing incomplete directory stops the launcher rather than silently overwriting it.
Set `PYTHON_BIN` when the active interpreter is not named `python`:

```bash
PYTHON_BIN=/path/to/environment/bin/python bash scripts/run_replication_seeds.sh
```

After all six runs, the launcher regenerates the suite report. `summary.csv` retains each
seed separately, while `summary_by_architecture.csv` reports the mean and sample standard
deviation. Per-run baseline deltas always use the `20a0t` run with the matching seed.
Because seed 1337 was used to select the focused architectures, treat it as exploratory;
use seeds 2357 and 7331 as the confirmatory subset and report the pooled summary only as a
secondary descriptive view.

## Evaluation

Evaluate the fixed FineWeb-Edu validation prefix for the reference suite:

```bash
python -m triglu.evaluate \
  --checkpoint runs/12l_1k_100m/6a6t/latest.pt \
  --data data/fineweb_edu_10bt/val.bin \
  --batch-size 4 --sequence-length 1024 --batches 50 \
  --device cuda --dtype bfloat16 \
  --output results/generated/6a6t-fineweb.json
```

Evaluation reports token-weighted mean next-token loss, perplexity, and accuracy over the
same denominator. Use the same number of held-out tokens for every plan.

## Mechanistic rank and layer-sensitivity analysis

`triglu.analyze_rank` measures where each block contracts or expands the residual stream
without assuming that rank contraction is the mechanism. It is strictly post-training:
the command loads a checkpoint without changing it and runs an uncompiled eager model with
forward hooks. Run it after GPU training is finished so it does not contend with training.

Analyze the 20-layer attention baseline at native 4K context:

```bash
python -m triglu.analyze_rank \
  --checkpoint runs/20l_4k_1b/20a0t/latest.pt \
  --data data/fineweb_edu_10bt_1b/val.bin \
  --batch-size 1 --sequence-length 4096 \
  --rank-batches 4 --rank-samples-per-batch 512 \
  --head-samples-per-batch 128 --sensitivity-batches 8 \
  --include-ffn-sensitivity \
  --device cuda --dtype bfloat16 \
  --output results/generated/20l_4k_1b-20a0t-rank-analysis.json
```

For a cheaper structural comparison, collect spectra from one selected hybrid while
skipping the repeated loss interventions:

```bash
python -m triglu.analyze_rank \
  --checkpoint runs/20l_4k_1b/15a5t_final_attention/latest.pt \
  --data data/fineweb_edu_10bt_1b/val.bin \
  --batch-size 1 --sequence-length 4096 \
  --rank-batches 4 --rank-samples-per-batch 512 \
  --head-samples-per-batch 128 --skip-sensitivity \
  --device cuda --dtype bfloat16 \
  --output results/generated/20l_4k_1b-15a5t-final-attention-rank-analysis.json
```

The schema-v2 JSON separates the complete pre-norm block path into `block_input`,
`mixer_norm_input`, `mixer_update`, `post_mixer_residual`, `ffn_norm_input`, `ffn_update`,
and `block_output`. Every stage receives a complete centered channel-covariance spectrum
and numerical rank, stable rank, participation ratio, and entropy effective rank. Derived
stage transitions show whether the mixer and FFN separately contract or expand the stream.
Mixer and FFN update/residual RMS ratios and cosine novelty are reported independently.

By default, sensitivity analysis zeros every attention or TriGLU mixer update one layer at
a time. `--include-ffn-sensitivity` adds the matched FFN interventions, roughly doubling
the intervention phase. All interventions consume the same deterministic validation prefix.
They measure model dependence under an out-of-distribution intervention; they are evidence
about component importance, not a claim that the ablated network is itself well trained.

Raw causal-attention matrix rank is intentionally not the primary statistic. Its
triangular mask commonly makes it algebraically full-rank even when its update subspace is
numerically redundant. Rank, update scale, and intervention sensitivity describe different
properties and must be interpreted jointly; none is sufficient on its own to establish a
replacement mechanism.

## Generate result tables and charts

After training and any optional evaluations, benchmarks, or rank analyses, generate a
self-contained report for one suite:

```bash
python -m triglu.report \
  --suite 20l_4k_1b \
  --runs-root runs \
  --results-root results/generated
```

This discovers all `runs/20l_4k_1b/*/metrics.jsonl` files and matching resolved configs.
It also joins benchmark, standalone-evaluation, and rank-analysis JSON by checkpoint or
config plan name. Incomplete runs are excluded from aggregates and listed separately.
The generated directory `results/generated/20l_4k_1b-report/` contains:

```text
README.md                    # compact Markdown results table and figure links
report.json                  # machine-readable derived summary
summary.csv                  # final quality, throughput, and baseline-relative metrics
summary_by_architecture.csv  # multi-seed means and sample standard deviations
summary_confirmatory.csv     # non-screening seeds only, when available
incomplete_runs.csv          # excluded partial runs, when present
training_curves.csv          # every logged validation point
layer_diagnostics.csv        # tidy layer/stage rank metrics when analyses exist
layer_transitions.csv        # mixer-vs-FFN rank deltas, scale, and direction
layer_sensitivity.csv        # tidy mixer/FFN intervention metrics
quality_vs_throughput.svg    # quality vs observed training-log throughput
validation_loss_curves.svg   # equal-token learning curves
validation_loss_curves_zoomed.svg # final 20% with a data-derived tight y-axis
layer_effective_rank.svg     # cross-plan block-output rank trajectories
layer_stages-<plan>.svg       # stage-separated rank trajectory for each plan
rank_delta-<plan>.svg         # rank change caused by each residual sublayer
update_scale-<plan>.svg       # mixer and FFN update/residual RMS ratios
layer_sensitivity.svg        # intervention loss deltas by depth
context_benchmarks.csv       # throughput, memory, and KV cache by context length
context_*_throughput.svg     # training, prefill, and decode context-scaling figures
context_kv_cache.svg         # allocated attention-cache capacity by context
```

SVG output uses only the Python standard library and remains editable in common vector
graphics tools. Regenerate these derived files after adding or replacing raw artifacts;
publish the raw JSON/JSONL and resolved configs alongside the figures.

## Fair training and inference benchmarks

Benchmark architecture throughput on preallocated synthetic token batches, excluding data
I/O and warmup/compilation iterations:

### Lightweight 12-layer suite

```bash
python -m triglu.benchmark \
  --config configs/12l_1k_100m/12a0t.yaml \
  --batch-size 4 --sequence-length 1024 \
  --prompt-length 768 --decode-tokens 128 \
  --warmup 5 --iterations 20 --device cuda --dtype bfloat16 --compile \
  --output results/generated/12a0t-benchmark.json

python -m triglu.benchmark \
  --config configs/12l_1k_100m/9a3t.yaml \
  --batch-size 4 --sequence-length 1024 \
  --prompt-length 768 --decode-tokens 128 \
  --warmup 5 --iterations 20 --device cuda --dtype bfloat16 --compile \
  --output results/generated/9a3t-benchmark.json

python -m triglu.benchmark \
  --config configs/12l_1k_100m/6a6t.yaml \
  --batch-size 4 --sequence-length 1024 \
  --prompt-length 768 --decode-tokens 128 \
  --warmup 5 --iterations 20 --device cuda --dtype bfloat16 --compile \
  --output results/generated/6a6t-benchmark.json

python -m triglu.benchmark \
  --config configs/12l_1k_100m/3a9t.yaml \
  --batch-size 4 --sequence-length 1024 \
  --prompt-length 768 --decode-tokens 128 \
  --warmup 5 --iterations 20 --device cuda --dtype bfloat16 --compile \
  --output results/generated/3a9t-benchmark.json
```

The benchmark reports training, prefill, and cached-decode throughput separately, along
with raw timings, peak CUDA allocation, logically used and preallocated KV-cache bytes,
model/plan, dtype, compile mode, PyTorch/CUDA versions, and device identity. Decode uses a
fixed-capacity attention cache so timings do not include repeated full-history
concatenation. It stays eager because the initialized attention extent changes each step
and compiler specialization would otherwise become part of the measurement.

### Primary 20-layer, 4K-context comparison

Use identical settings for the attention baseline and the two focused hybrids:

```bash
for plan in 20a0t 15a5t_front_blend 15a5t_final_attention
do
  python -m triglu.benchmark \
    --config "configs/20l_4k_1b/${plan}.yaml" \
    --batch-size 1 --sequence-length 4096 \
    --prompt-length 3072 --decode-tokens 512 \
    --warmup 5 --iterations 20 --device cuda --dtype bfloat16 --compile \
    --output "results/generated/20l_4k_1b-${plan}-benchmark.json"
done
```

These benchmark results, rather than differences between training-log throughput values,
are the primary evidence for architecture-level efficiency claims.

### Context-scaling benchmark through 16K

After the primary 4K comparison, run the baseline and leading front-blend hybrid at 1K,
2K, 4K, 8K, and 16K:

```bash
bash scripts/run_context_benchmarks.sh
```

The sweep uses batch size 1, a fixed 128-token decode suffix, and otherwise fills each
context with prompt tokens. It measures training, prefill, cached decode, peak allocation,
and KV-cache capacity at every length. Outputs are written as
`results/generated/20l_4k_1b-context-<length>-<plan>-benchmark.json`; completed files are
skipped safely, and the suite report is regenerated after the sweep. If the active Python
executable is not named `python`, use:

```bash
PYTHON_BIN=/path/to/environment/bin/python bash scripts/run_context_benchmarks.sh
```

The 8K and 16K measurements use a benchmark-only context-capacity override. RoPE positions
and static attention caches extend to the requested length, but parameter shapes and layer
plans do not change. Because the models were trained at 4K, results above 4K establish
architecture-level runtime and memory scaling only—not language-modeling quality at those
lengths. The default sweep uses 3 warmup and 10 measured iterations; `WARMUP`, `ITERATIONS`,
and `DECODE_TOKENS` can be overridden consistently for all runs.

To verify noisy 1K/2K measurements without overwriting the original sweep, run the
targeted longer validation (10 warmup and 50 measured iterations by default):

```bash
PYTHON_BIN=/path/to/environment/bin/python bash scripts/run_short_context_validation.sh
```

Validation artifacts use a separate filename and benchmark label. When more than one
labeled measurement exists for an architecture/context pair, the generated report uses
the artifact with the most measured iterations and retains every raw JSON file.

## Optional additional experiments

The 12-layer reference comparison does **not** require Wikipedia. An optional
out-of-domain check can evaluate selected checkpoints on a fixed English-Wikipedia stream.
Keep these results separate from the FineWeb-Edu validation results.

Prepare the November 2023 English dump. The one-token train output is unused; this command
reuses the two-split preparer to create the validation stream:

```bash
python -m triglu.prepare_data \
  --dataset wikimedia/wikipedia \
  --dataset-config 20231101.en \
  --revision e6057dc557255a03c9c3c47ceab0eb44353b1bc5 \
  --split train --text-column text \
  --train-tokens 1 \
  --val-tokens 5000000 \
  --output-dir data/wiki_20231101_en
```

Then evaluate each selected checkpoint with the same evaluation settings:

```bash
python -m triglu.evaluate \
  --checkpoint runs/12l_1k_100m/6a6t/latest.pt \
  --data data/wiki_20231101_en/val.bin \
  --batch-size 4 --sequence-length 1024 --batches 50 \
  --device cuda --dtype bfloat16 \
  --output results/generated/optional-6a6t-wikipedia.json
```

## Outputs and fair comparison

A training run writes:

```text
runs/<suite>/<plan>/
  resolved_config.yaml
  environment.json
  data_provenance.json
  data_manifest.json
  metrics.jsonl
  latest.pt
  checkpoint_step_XXXXXXXX.pt  # only when checkpoint_interval > 0
```

Compare plans at equal `tokens_seen`, with the same token-file hashes and metric cadence.
For quality, report validation loss/perplexity first and token accuracy second. For speed,
publish the raw benchmark JSON rather than only a chart. Do not combine the 12-layer and
20-layer suites into a single controlled comparison: depth, context, and token budget all
differ.

## Limitations

- TriGLU cannot move information between positions. Hybrid stacks rely entirely on
  their retained attention layers for token mixing; an all-TriGLU model is only a
  diagnostic negative control.
- The 100M-token suite is intentionally small. It tests trends and engineering trade-offs,
  not frontier-scale model quality.
- Dataset order, GPU kernels, compiler behavior, and hardware affect absolute throughput.
  Reproduce comparisons on one software/hardware setup and publish that metadata.
- Fixed seeds make initialization and the sampled batch sequence identical across plans;
  they do not make CUDA kernels bitwise deterministic. Compare trends and repeated
  seeds, not bitwise-identical loss curves.
- Screening runs use one seed. The two additional focused seeds reduce uncertainty but
  remain a small confirmatory sample; report each seed, dispersion, and confirmatory-only
  conclusions rather than presenting three runs as definitive.

## Citation

```bibtex
@misc{royer_triglu,
  author = {Nick Royer},
  title  = {TriGLU: Controlled Attention-Replacement Ablations},
  year   = {2026},
  note   = {Royer Research Labs, LLC},
  url    = {https://github.com/Royer-Research-Labs/TriGLU}
}
```

## License

Copyright © 2026 Royer Research Labs, LLC. Principal author: Nick Royer.

TriGLU's software—including source code, tests, scripts, configurations, workflows, and
pseudocode—is licensed under the [Apache License 2.0](LICENSE). Documentation prose and
original result figures and tables are licensed under the
[Creative Commons Attribution 4.0 International License](LICENSES/CC-BY-4.0.txt). See the
[licensing scope](LICENSES/README.md) and [`NOTICE`](NOTICE) for details and attribution.

Dataset content, tokenizer assets, and other third-party material retain their upstream
licenses and terms.
