# Results: how much attention a small decoder needs, and where

In an 89M-parameter, 20-layer decoder trained on one billion FineWeb-Edu tokens at 4K
context, a substantial fraction of the causal-attention layers can be replaced by cheap
token-local gated blocks at quality parity, and the *placement and amount* of that
replacement — not the exact token-local formulation — are the primary levers. The
triple-product form (TriGLU) contributes in two specific places: it uniquely holds quality
at aggressive replacement, and it is the strongest FFN we tested. The two contributions do
not stack.

Losses are token-weighted mean next-token cross-entropy on the fixed 1,048,576-target
FineWeb-Edu validation prefix, lower is better. Unless a finding is marked exploratory
(single seed), each mean pools three seeds (1337, 2357, 7331); seed 1337 is the screening
seed used to select placements, so the seed-split view in
`summary_confirmatory.csv` is the conservative reading. Efficiency claims use the
controlled standalone benchmark (data loading and compilation warmup excluded, CUDA
synchronized around each timed region), which is the primary efficiency evidence rather
than training-log throughput.

## What was tested

Every run uses the same width-512 decoder (20 layers except the deliberately collapsed
9-layer topology controls), the same tokenizer, sampled token sequence, optimizer,
schedule, and one-billion-token budget. Parameter counts are exactly 89,018,880 except
where a control's documented near-match differs: the two-factor SwiGLU attention-slot
controls carry +512 weights per replaced layer (89,021,440 at 25%, 89,024,512 at 55%,
≤0.007%), the grouped collapses land within +1,024 (89,019,392 / 89,019,904), the
parallel blocks drop one norm per block (89,008,640), and the shallow capacity anchor is
54,224,384 by design. Exact figures per config are in
[`design.md`](design.md). The runs group into labeled families, recorded in `summary.csv`
under `experiment_family`:

- **primary** — the attention baseline and the original RoPE-TriGLU placement suite;
- **attention-slot controls** — parameter-matched token-local mixers in the same slots:
  no-RoPE TriGLU, a two-factor SwiGLU, and an MB-MLP-style `GELU(k·g)·v`, at 25% and 55%
  replacement and across placements;
- **FFN-form control** — runs whose FFN is the triple-product form rather than SwiGLU:
  the all-attention FFN control and the combination probe that also replaces attention
  slots;
- **residual-topology controls** — parameter-matched 9-layer collapses and a single-norm
  parallel block.

## Attention is replaceable — and placement/amount dominate the mixer

At 25% replacement (five of twenty layers, attention-rich-front placement), every
parameter-matched token-local mixer matched or beat the all-attention baseline, and the
simplest one did best:

| 25% replacement (3 seeds) | Mean loss | SD | Δ vs attention |
| --- | ---: | ---: | ---: |
| attention baseline (`20a0t`) | 3.46168 | 0.00672 | — |
| two-factor SwiGLU control | 3.43265 | 0.00226 | **−0.02903** |
| no-RoPE TriGLU | 3.44105 | 0.00324 | −0.02064 |
| MB-MLP control | 3.44558 | 0.00327 | −0.01611 |

The RoPE-TriGLU variant at the same positions sits at baseline parity (the v0.1.0 result;
see `summary_confirmatory.csv` and [`screening-results.md`](screening-results.md)) — so at
25% the distinctive parts of TriGLU, the rotation and the third factor, are not what buys
the quality. The simplest gated form suffices.

That changes under aggressive replacement. At 55% (eleven of twenty layers, nested
placement), only the triple product holds the line:

| 55% replacement (3 seeds) | Mean loss | SD | Δ vs attention |
| --- | ---: | ---: | ---: |
| attention baseline (`20a0t`) | 3.46168 | 0.00672 | — |
| no-RoPE TriGLU (`9a11`) | 3.44714 | 0.00403 | **−0.01454** |
| two-factor SwiGLU (`9a11`) | 3.46212 | 0.01049 | +0.00044 |

No-RoPE TriGLU still beats the baseline, while the two-factor control loses its 25%
advantage and regresses to attention parity. The third multiplicative factor earns its
keep precisely where replacement is hardest. No-RoPE TriGLU's matched-seed margin over
attention is consistent across all three seeds (−0.0055, −0.0253, −0.0129); the SwiGLU
control's margin is seed-sensitive (+0.0113, −0.0194, +0.0094 — it beats attention at one
seed and trails at two), which is why its mean lands at parity with a larger dispersion.

## Efficiency of the 55% hybrid

Removing 55% of the attention layers removes the same fraction of KV-cache capacity and a
large share of attention compute. Controlled benchmarks of the no-RoPE 55% hybrid against
the attention baseline:

| Context | Training | Prefill | Cached decode | KV-cache capacity |
| ---: | ---: | ---: | ---: | ---: |
| 4K | +18.9% | +40.4% | +50.7% | −55% |
| 8K | +32.2% | +37.5% | +55.8% | −55% |
| 16K | +48.8% | +54.8% | +50.3% | −55% |

KV-cache capacity is exactly 55% lower at every length (nine of twenty attention layers).
The model was trained at 4K; the 8K and 16K points extend RoPE positions and cache capacity
only to measure architecture-level runtime and memory scaling, and do not establish
language-modeling quality beyond the trained window. The 25% RoPE-TriGLU hybrids from
v0.1.0 show the smaller-but-real version of the same pattern (≈25% less KV; +8% controlled
training and decode at 4K with prefill up to +16%, widening with context); the front-blend
hybrid's context rows are in `context_benchmarks.csv`, and both hybrids' quality and 4K
benchmarks remain in `summary_confirmatory.csv`.

## TriGLU is also the best FFN tested

With all twenty attention layers retained and only the FFN changed from SwiGLU to the
no-RoPE triple-product form, the model reaches the single best quality in the study:

| FFN form (3 seeds, all attention retained) | Mean loss | SD | Δ vs baseline |
| --- | ---: | ---: | ---: |
| standard SwiGLU FFN (`20a0t`) | 3.46168 | 0.00672 | — |
| triple-product FFN | 3.43129 | 0.00401 | **−0.03039** |

This is a quality result, not an efficiency one: it retains the full KV cache and buys
no speed. Its cost is small — controlled training throughput is within 1.4% of the
baseline at every context (−1.1% at 4K), though the observed training-log throughput ran
about 6% lower. It is reported as a separate direction — the triple-product gate is a
strong drop-in FFN — not folded into the attention-replacement story.

## The two roles do not stack

Using the triple product in **both** roles at once — 55% no-RoPE replacement *and* a
matched triple-product FFN — does not combine the wins (exploratory, seed 1337):

| Seed-1337 comparison | Loss |
| --- | ---: |
| attention baseline | 3.45719 |
| replacement only (55% no-RoPE, SwiGLU FFN) | 3.45172 |
| FFN only (all attention, triple-product FFN) | 3.43492 |
| **both together** | 3.45642 |

The combination lands near baseline — worse than either lever alone. The FFN quality gain
depends on the attention layers staying intact; strip 55% of them and it largely
evaporates. The practical consequence is that the recommended configurations are
goal-specific and mutually exclusive: the 55% replacement hybrid for efficiency at parity,
or the all-attention triple-product FFN for maximum quality.

## The hybrid is more than parameter budget

A skeptic's first question is whether the hybrid is merely "nine attention layers plus
extra channel capacity." On quality, it is not. Collapsing the 20-layer hybrid into a
parameter-matched 9-layer model, with the freed budget grouped into fat FFNs at the same
depths, is measurably worse (3 seeds):

| Structure control | Mean loss | SD | Structure gap |
| --- | ---: | ---: | ---: |
| 20-layer SwiGLU hybrid | 3.46212 | 0.01049 | — |
| → collapsed to 9 layers, fat FFNs | 3.50353 | 0.00519 | **+0.04141** |
| 20-layer no-RoPE hybrid | 3.44714 | 0.00403 | — |
| → collapsed to 9 layers, fat FFNs | 3.46899 | 0.00611 | **+0.02185** |

Adding the capacity is necessary but not sufficient: a 9-layer model at standard FFN width
(54.2M parameters, one seed) reaches only 3.57945, so most of the gap between it and the
89M hybrid is capacity, but a replicated remainder (0.022–0.041) belongs to distributed
structure and depth.

The collapses are, however, *faster* — the shorter 9-block serial path buys them
+39–50% controlled cached-decode throughput at every context, and the SwiGLU collapse
also trains +12% faster at 4K (the TriGLU-FFN collapse is mixed: +2.5% at 4K, −3.4% at
8K/16K). Collapsing therefore trades quality for speed rather than being a refuted
alternative: it is a legitimate point on the speed/quality frontier, and the hybrid's
replicated quality advantage at matched parameters is a genuine structural effect that
comes at a real efficiency cost relative to the collapse — not a free win.

## Placement and replacement amount

At a fixed 25% replacement, placement moves quality. The replicated contrast: attention-rich
front placement (3.44105, 3 seeds) beats early intrusion — the same plan with its earliest
replacement moved from layer 8 down to layer 3 — at 3.45332 (3 seeds), a +0.012 penalty for
that single early swap. The single-seed placement ordering agrees: front-blend (3.43753) ≈
nested (3.43916) < tail-block (3.45096) < early intrusion < evenly-repeating (3.48761). Across replacement amount, the
nested ladder degrades smoothly until a sharp cliff between 55% and 70% replacement; the
full ladder is tabulated in [`screening-results.md`](screening-results.md).

## Merging attention and the FFN is roughly neutral

A single-norm parallel block — `x + mixer(norm(x)) + ffn(norm(x))`, the standard
"faster training" lever — is close to a wash here on both axes (exploratory, seed 1337):
loss moves within ±0.008 (all-attention 3.45056 vs 3.45719; hybrid 3.45931 vs 3.45172), and
controlled throughput moves within ±5%. Merging blocks is therefore neither a competing
speed lever that would explain away the replacement gains nor a free addition to them — the
55% hybrid's efficiency is its own, distinct lever.

## Mechanistic diagnostics

Stage-separated effective-rank trajectories are non-monotonic: attention and token-local
updates both expand rank at some depths and contract it at others, and large contractions
are localized rather than evidence of one persistent depth threshold. These diagnostics do
not support a model in which attention becomes replaceable after a single universal
rank-collapse point. They describe the geometry of sampled activations and dependence under
an out-of-distribution zeroing intervention; they remain diagnostics, not an identified
causal mechanism. The placement results are the load-bearing evidence, not the rank
measurements.

## Recommended configurations

The findings support two goal-specific integrations, which do not combine:

- **Efficiency at quality parity:** replace up to ~55% of attention layers with no-RoPE
  TriGLU using an attention-rich-front, nested placement; expect parity-or-better quality,
  large prefill/decode/KV savings that grow with context, and no benefit from also changing
  the FFN.
- **Maximum quality:** retain all attention and use the triple-product form as the FFN;
  expect the best quality measured here at a small throughput cost and no KV savings.

In both cases, validate placement at the deployment context and batch size, and treat a
different wrapper, scale, dataset, or placement plan as a new experiment: a token-local
block cannot recover information a retained attention layer never communicated.

## Supported and unsupported claims

Supported at the tested scale:

- a small decoder's attention is substantially replaceable — 25% by any parameter-matched
  token-local mixer, 55% by no-RoPE TriGLU — at validation-quality parity or better;
- placement and replacement amount are the primary levers; the mixer formulation matters
  mainly under aggressive replacement, where the triple product uniquely holds quality;
- the triple-product form is the strongest FFN tested (a separate, quality-only result);
- the hybrid's advantage over a parameter-matched collapse is partly structural, not only
  capacity; and
- replacing 55% of attention removes 55% of KV-cache capacity and yields large,
  context-growing prefill/decode/training throughput gains.

Not established:

- that TriGLU is the uniquely best token-local mixer, or that its equation is novel (the
  recommended no-RoPE form is a GQU-family gate; see [`related-work.md`](related-work.md));
- quality equivalence at substantially larger scale, or language-modeling quality beyond 4K
  context;
- that the two roles combine, or that attention can be removed entirely; or
- rank collapse or any single mechanism as the cause of successful replacement.

## Reproducibility artifacts

The reporting pipeline writes source tables, editable SVG figures, benchmark records, and
mechanistic diagnostics under `results/generated/` (gitignored; regenerated with
`python -m triglu.report --suite 20l_4k_1b`). The complete evidence set behind this
document — every completed run's resolved config, metric stream, environment metadata, and
data hashes, the raw evaluation/benchmark/rank-analysis JSON, and the generated
`20l_4k_1b-report/` tables and figures — is published as the
[v0.2.0 evidence archive](https://github.com/Royer-Research-Labs/TriGLU/releases/tag/v0.2.0).
Inside that archive, start with:

- `results/generated/20l_4k_1b-report/README.md`
- `results/generated/20l_4k_1b-report/summary.csv` and `summary_by_architecture.csv`
- `results/generated/20l_4k_1b-report/summary_confirmatory.csv`
- `results/generated/20l_4k_1b-report/context_benchmarks.csv`
- `results/generated/20l_4k_1b-report/layer_transitions.csv`

Large checkpoints and tokenized corpora are intentionally separate from the source
distribution. The [results and provenance protocol](results-schema.md) defines the raw
records required to reproduce or audit a published comparison.
