# Experimental design and integration boundary

This document distinguishes the independently usable TriGLU mixer from the decoder
wrapper used to evaluate it. That distinction is important for both audiences:

- component users can integrate the exact mixer into another architecture without
  importing the experimental harness; and
- reviewers can identify which conclusions are supported by controlled comparisons in
  this repository.

## Scientific variable

Within a fixed-seed primary comparison, every plan uses the same decoder width, depth,
feed-forward network, normalization, residual topology, tokenizer, sampled token
sequence, optimizer, schedule, and token budget. The only architecture variable is the
explicit per-layer choice between causal attention and `TriGLU`. Focused replications
change the seed for every compared architecture together and report matched-seed deltas.

Three secondary controls hold the selected layer positions fixed and change only the
token-local replacement formulation: TriGLU without RoPE, an MB-MLP-style triple-branch
product, and a near-parameter-matched two-factor SwiGLU. They test which part of the
TriGLU formulation matters; they are not part of the primary replacement-ratio or
placement ablations.

A separate all-attention FFN-form control leaves every mixer as causal attention and
changes only the second sublayer from conventional SwiGLU to a no-RoPE triple-product
FFN. It tests the form's competitiveness in the FFN role, not its ability to replace
attention. It is outside both the primary attention-replacement comparisons and the
attention-slot differentiation controls.

A family of post-hoc residual-topology controls deliberately relaxes the fixed-wrapper
rule: parameter-matched single-residual SwiGLU blocks at selected positions, two
nine-block grouped-width collapses of the hybrid schedule, a conventional shallow
capacity anchor, and single-norm parallel blocks. They test possible architectural
explanations of the placement result and cannot be used as evidence about TriGLU versus
another mixer.

The resolved configuration records the complete layer list. Ratios are labels, not a
hidden placement algorithm.

## Authoritative TriGLU equation

TriGLU means **Triple-Product Gated Linear Unit**. For an input token `x_t` with width
`C`, the recommended form computes

```text
(k_t, g_t, v_t) = split_3(W_c x_t + b_c)
z_t              = k_t * SiLU(g_t) * v_t
y_t              = W_o z_t + b_o
```

All products are elementwise. `W_c` is one fused `C -> 3C` projection and `W_o` is
`C -> C`. No operation reads another token. This is therefore a token-local, channel-wise
gated-product block—not linear attention and not an attention approximation. Relative to
the two-factor `SiLU(g_t) * v_t` SwiGLU form, TriGLU adds the third projected factor `k_t`.
“Tri” names the three multiplicative factors, not three gates, and is independent of RoPE.

An **optional RoPE variant** inserts `k_t = RoPE(k_t, position=t)` before the product,
rotating the full `C`-wide `k` vector only (position-dependent but still token-local). This
was TriGLU's ablation origin and the v0.1.0 default; the controlled experiments in this
repository found the no-RoPE form to be the better attention replacement, so the RoPE form
is retained only as the origin and as the control that isolates the positional branch.
In the layer plan the recommended form is `triglu_no_rope` and the origin variant is
`triglu`; the historical layer-type names are kept so v0.1.0 configs remain valid.

During cached decoding a TriGLU layer stores no keys or values. Its cache tuple contains
only the next integer position: `(None, None, next_position)`.

Normalization is outside this equation. The evaluated wrapper supplies an RMS-normalized
residual stream, while another integration may define a different normalization policy.

## Component boundary

The authoritative component begins with the fused `C → 3C` projection and ends with the
`C → C` output projection. It accepts and returns one tensor of shape `[B, T, C]` plus the
standard cache metadata used by the model API. It does not own RMSNorm, residual addition,
or an FFN.

Those operations belong to the surrounding decoder block. An integration may choose a
different block topology, but results from that integration should not be attributed to
the controlled architecture evaluated here without a separate ablation.

## Evaluated decoder wrapper

The research harness places every attention-slot component in the same conventional pre-norm
decoder wrapper:

```text
h       = x + mixer(RMSNorm(x))
x_out   = h + SwiGLU(RMSNorm(h))
```

Attention uses standard multi-head causal scaled-dot-product attention and per-head
RoPE. TriGLU uses the authoritative equation above. These two primary mixer choices have
the same projection parameter count (`4*C*C`, plus `4*C` when bias is enabled).
This wrapper defines the primary and attention-slot-control experiments. The
all-attention FFN-form control below preserves the two-sublayer topology but substitutes
the explicitly documented function in the second sublayer.

Using the same wrapper is an experimental control, not a requirement of the TriGLU
equation. A fused form such as `x + MLP(mixer(norm(x)))` changes normalization, residual
flow, and the FFN input together with the mixer. It therefore answers a different
architectural question and is outside the comparisons reported by this repository. The
framework-independent reference in [`pseudocode/triglu_layer.pseudo`](../pseudocode/triglu_layer.pseudo)
shows the mixer and evaluated wrapper as separate functions.

## Conservative model choices

- pre-norm `torch.nn.RMSNorm`;
- ordinary multi-head attention (no GQA, MQA, or local windows);
- PyTorch scaled-dot-product attention with causal masking;
- per-head RoPE for attention; the recommended TriGLU form applies no rotation (the
  full-width-RoPE TriGLU is an optional origin-variant control);
- standard SwiGLU channel MLP;
- tied input embeddings and language-model head;
- GPT-style depth scaling on residual output projections;
- no dropout by default.

The primary suites add no learned positional mode, convolution, recurrence, state-space
layer, temporal mixer beyond standard attention, custom normalization, custom FFN
family, MoE, or shared parameter bank. The secondary controls described below add only
the explicitly named attention-slot equations; their surrounding decoder FFN remains the
same conventional SwiGLU. The separately labeled FFN-form control is documented in its
own section because it changes that otherwise fixed sublayer.

## Layer placement: 12-layer suite

Reference plans keep attention layers distributed through depth so cross-token mixing
occurs at multiple scales:

| Label | Explicit layer types, bottom to top |
| --- | --- |
| `12A-0T` | `A A A A A A A A A A A A` |
| `9A-3T` | `A A A T A A A T A A A T` |
| `6A-6T` | `A T A T A T A T A T A T` |
| `3A-9T` | `A T T T A T T T A T T T` |

These conversions are nested: TriGLU blocks at layers `{3, 7, 11}` are joined by
`{1, 5, 9}`, then `{2, 6, 10}`. Each more aggressive plan therefore replaces a
superset of the same attention layers while the retained attention remains spread
through depth.

## Layer placement: 20-layer suite

The 4K-context suite separates replacement ratio from placement. Zero-based TriGLU layer
indices completely specify each plan:

| Label | Attention | TriGLU | TriGLU layers |
| --- | ---: | ---: | --- |
| `20a0t` | 20 | 0 | none |
| `15a5t` | 15 | 5 | 3, 7, 11, 15, 19 |
| `10a10t` | 10 | 10 | 1, 3, 5, ..., 19 |
| `9a11t_front_blend` | 9 | 11 | 6, 7, 9–11, 13–15, 17–19 |
| `5a15t` | 5 | 15 | all except 0, 4, 8, 12, 16 |
| `15a5t_front_blend` | 15 | 5 | 8, 12, 15, 17, 19 |
| `15a5t_late_alternating` | 15 | 5 | 11, 13, 15, 17, 19 |
| `15a5t_tail_block` | 15 | 5 | 15–19 |
| `15a5t_final_attention` | 15 | 5 | 8, 12, 15, 17, 18 |

The last four rows hold the 15-attention/5-TriGLU ratio fixed and test placement. In
particular, `15a5t_final_attention` differs from `15a5t_front_blend` only by swapping the
types of layers 18 and 19.

## Prior-art differentiation controls

The secondary controls use the same 20-layer model and the
`15a5t_front_blend` replacement positions `{8, 12, 15, 17, 19}`. Retained attention
layers, wrapper, data order, optimizer, schedule, and one-billion-token budget stay
fixed. The four comparison formulations are:

| Config | Token-local replacement | Model parameters |
| --- | --- | ---: |
| `15a5t_front_blend.yaml` | `W_o[RoPE(k) * SiLU(g) * v]` | 89,018,880 |
| `15a5_triglu_no_rope_front_blend.yaml` | `W_o[k * SiLU(g) * v]` | 89,018,880 |
| `15a5_mb_mlp_front_blend.yaml` | `W_o[GELU(k * g) * v]` | 89,018,880 |
| `15a5_swiglu_front_blend.yaml` | `W_o[SiLU(g) * v]` | 89,021,440 |

The no-RoPE control is the authoritative `TriGLU` module with its optional rotation
disabled, so it isolates position dependence without changing learned parameters. The
MB-MLP control retains the same-width `C → 3C → C` projection budget but changes both
activation placement and positional treatment; it is a published-equation control, not
a one-variable activation test.

The SwiGLU attention-slot control uses hidden width 683 because
`3 * 512 * 683 = 1,049,088`, only 512 projection weights more per replaced layer than
`4 * 512^2 = 1,048,576`. All parameters are active. Its quality is therefore
near-parameter-matched, while throughput should be interpreted cautiously because 683
is not a hardware-friendly width.

Implementation and interpretation links for the closest published precedents are in
[`docs/related-work.md`](related-work.md). These controls do not change the
authoritative TriGLU equation or the conclusions supported by the already completed
primary suite.

## All-attention FFN-form control

The config
[`20a0t_triglu_no_rope_ffn.yaml`](../configs/20l_4k_1b/ablations/20a0t_triglu_no_rope_ffn.yaml)
keeps the baseline's 20 attention mixers and replaces every conventional SwiGLU FFN with
a distinct `TriGLUFFN` module. For normalized FFN input `x`, the two forms are:

```text
SwiGLU:
g = W_gate x
v = W_up x
y = W_down[SiLU(g) * v]

TriGLU-form FFN:
(k, g, v) = split_3(W_c x)
y = W_down[k * SiLU(g) * v]
```

The TriGLU-form FFN deliberately omits RoPE and has no position argument, cross-token
operation, or cache state. Its projection width is an FFN hidden width rather than the
residual width used by the authoritative attention-slot `TriGLU` component. It is
therefore an equation-form control in a different architectural role, not a modified
definition of the public mixer.

For residual width `C`, standard SwiGLU width `H`, and triple-product width `T`, the
bias-free projection counts are `3*C*H` and `4*C*T`, respectively. The supplied control
uses `C = 512`, `H = 1376`, and `T = 1032`:

```text
3 * 512 * 1376 = 2,113,536 weights per SwiGLU block
4 * 512 * 1032 = 2,113,536 weights per TriGLU-form block
```

Because `bias: false` and every tensor outside the FFN is unchanged, the two complete
all-attention models are exactly matched at 89,018,880 parameters. RMSNorm placement,
residual flow, attention, initialization policy, embeddings, output head, data order,
optimizer, schedule, and token budget remain fixed.

This control cannot support an attention-replacement or KV-cache claim: every attention
sublayer and its KV state remain present. Its scoped question is whether the
triple-product equation is competitive with conventional SwiGLU as a decoder FFN.
Conventional SwiGLU remains the default and the sole FFN in all primary and
attention-slot-control experiments.

Run the recorded seed directly with:

```bash
python -m triglu.train \
  --config configs/20l_4k_1b/ablations/20a0t_triglu_no_rope_ffn.yaml
```

For three matched seeds and the guarded 4K/8K/16K benchmark cleanup, use
`scripts/run_final_replications_and_benchmarks.sh` as documented in the repository
README. The 8K and 16K measurements are efficiency extrapolations above the 4K training
context; they do not measure language-model quality at those lengths.

## Combined replacement and FFN-form control

The attention-slot replacement and the FFN-form control each improve quality on their own.
The config
[`9a11_triglu_no_rope_nested_triglu_ffn.yaml`](../configs/20l_4k_1b/ablations/9a11_triglu_no_rope_nested_triglu_ffn.yaml)
tests whether they compose: it uses the flagship 55% no-RoPE nested replacement placement
*and* the no-RoPE triple-product FFN in every block, at FFN width 1032. Because
`4 * 512 * 1032 = 3 * 512 * 1376`, the model has exactly the 9a11 hybrid's 89,018,880
parameters, so the only change from the replacement-only flagship is the FFN equation.

Its scoped question is whether the two independently measured wins add. It is an
exploratory seed-1337 probe, interpreted against the same-seed replacement-only and
FFN-only runs rather than the pooled means.

## Residual-topology stress controls

These controls are explicitly outside the primary controlled decoder wrapper.

### Exactly matched single-residual block

At front-blend positions `{8, 12, 15, 17, 19}`, the config
[`15a5_wide_swiglu_single_residual_front_blend.yaml`](../configs/20l_4k_1b/ablations/15a5_wide_swiglu_single_residual_front_blend.yaml)
replaces

```text
h = x + mixer(RMSNorm_1(x))
y = h + SwiGLU_1376(RMSNorm_2(h))
```

with

```text
y = x + SwiGLU_2059(RMSNorm(x)).
```

The new `ffn_only` block has no mixer, no mixer normalization, and no mixer residual
addition. It returns only position metadata under the model's unified cache API.
For bias-free residual width `C = 512`, the whole-block parameter counts are exactly
equal:

```text
ordinary block = 4*C^2 + 3*C*1376 + 2*C = 3,163,136
FFN-only block =         3*C*2059 +   C = 3,163,136
```

The 20-block model therefore remains exactly 89,018,880 parameters. The experiment
changes attention count, normalization count, residual depth, and the allocation of
channel capacity together. Its scoped question is whether a single wider channel update
can explain the retained capacity; a positive result would not identify which of those
joint changes is causal. Parameter equality does not imply equal initial update RMS or
runtime because the two residual paths have been consolidated into one wider FFN.

### Nine-block grouped-width collapse

The config
[`9l_9a0t_grouped_swiglu_9a11_nested_collapse.yaml`](../configs/20l_4k_1b/ablations/9l_9a0t_grouped_swiglu_9a11_nested_collapse.yaml)
starts from attention positions `{0, 1, 2, 3, 4, 5, 8, 12, 16}` in the nested
9-attention/11-token-local plan. Each retained attention absorbs the complete parameter
budget of the blocks following it up to the next retained attention:

| New block | Source blocks | SwiGLU width |
| ---: | --- | ---: |
| 0–4 | `[0]`, `[1]`, `[2]`, `[3]`, `[4]` | 1376 each |
| 5 | `[5, 6, 7]` | 5495 |
| 6 | `[8, 9, 10, 11]` | 7554 |
| 7 | `[12, 13, 14, 15]` | 7554 |
| 8 | `[16, 17, 18, 19]` | 7554 |

Width 7554 exactly matches four original blocks. The three-block group cannot be matched
exactly at integer hidden width: 5495 is 512 parameters above, while 5494 would be 1,024
below. The complete model is therefore 89,019,392 parameters, only 512 (+0.0006%) above
the 89,018,880 reference.

This model has nine attention sublayers, nine KV caches, nine FFNs, and 18 residual
additions. It has no TriGLU or other attention-replacement block. The resolved config
sets `residual_init_depth: 20`, so every surviving attention and FFN output projection
retains the source model's `init_std / sqrt(2*20)` per-weight standard deviation rather
than receiving the additional depth-based increase implied by nine physical blocks.
This does not equalize the residual-update RMS of width-1376 and width-7554 FFNs; wider
hidden activations can increase output variance even at the same down-projection weight
standard deviation. That width-dependent update scale is an explicit limitation of this
stress test and should be measured with the residual-update diagnostics.

Both controls use the same data order, optimizer, schedule, 4K training context, and
one-billion-token budget as their references. Seed 1337 is an exploratory screen.
Quality claims require matched-seed replication; 8K and 16K measurements remain
architecture-efficiency extrapolations.

### Nine-layer FFN-form and shallow controls

Two follow-on seed-1337 controls make the grouped-collapse result easier to interpret.
The grouped no-RoPE TriGLU-FFN model retains the same nine attention blocks, source
groups, 20-layer residual initialization, and training configuration as the grouped
SwiGLU model. Its FFN widths are
`[1032, 1032, 1032, 1032, 1032, 4121, 5666, 5666, 5665]`. Since a no-RoPE
triple-product FFN has `4*C*H` projection weights while SwiGLU has `3*C*H`, this is
the nearest integer-width match to the grouped SwiGLU projection budget. The complete
models have 89,019,904 and 89,019,392 parameters respectively, a difference of 512
(0.0006%). The comparison changes the FFN equation while holding the collapsed topology
and initialization fixed; it does not test TriGLU in an attention slot.

The uniform shallow control is a conventional nine-layer all-attention decoder with a
width-1376 SwiGLU in every block, 18 residual additions, and normal nine-layer
residual-projection initialization. It has 54,224,384 parameters. This control shows the
quality and efficiency of simply training a much smaller ordinary decoder at the same
token budget. It does not, by itself, isolate physical layer count: relative to the
grouped-width controls, both FFN capacity allocation and residual initialization differ.
The distinction is intentional and must remain visible in result tables through the
width schedule, parameter count, and residual initialization depth.

| Config | Physical blocks | FFN form and widths | Init depth | Parameters |
| --- | ---: | --- | ---: | ---: |
| `9l_9a0t_grouped_swiglu_9a11_nested_collapse` | 9 | SwiGLU, 1376–7554 | 20 | 89,019,392 |
| `9l_9a0t_grouped_triglu_no_rope_ffn` | 9 | no-RoPE triple product, 1032–5666 | 20 | 89,019,904 |
| `9l_9a0t_standard_swiglu` | 9 | SwiGLU, 1376 throughout | 9 | 54,224,384 |

### Single-norm parallel block

`block_mode: parallel` replaces the canonical two-norm sequential wrapper with a
single-norm parallel block in which the mixer and FFN read one shared normalization of the
block input and write into a single residual add:

```text
sequential (default):          parallel:
  h = x + mixer(RMSNorm_1(x))     n = RMSNorm(x)
  y = h + FFN(RMSNorm_2(h))       y = x + mixer(n) + FFN(n)
```

Two configs use it, both otherwise identical to their sequential references:
[`20a0t_parallel_block.yaml`](../configs/20l_4k_1b/ablations/20a0t_parallel_block.yaml)
(all attention, isolating the merge itself) and
[`9a11_triglu_no_rope_nested_parallel_block.yaml`](../configs/20l_4k_1b/ablations/9a11_triglu_no_rope_nested_parallel_block.yaml)
(the 55% hybrid, testing whether merging and replacement compose). Dropping the second
RMSNorm removes exactly `d_model` weights per block — 10,240 total, about 0.01%, in the
parallel model's own disfavor — so the comparison is effectively parameter-matched. The
residual-projection initialization is unchanged because each block still contributes two
residual updates.

This is the standard "faster training" lever; its scoped question is the speed/quality
trade of merging the sublayers, not attention replacement. Both are exploratory seed-1337
probes. Because a parallel block has a single normalization rather than the two-norm
sequential path, the mechanistic rank analysis (which hooks the sequential mixer and FFN
norms separately) does not apply to these runs.

## Exploratory placement/amount sweep (no-RoPE variant)

The single-seed sweep under `configs/20l_4k_1b/placement_amount/` varies replacement
amount and placement with the no-RoPE control while holding the 20-layer geometry,
data order, optimizer, schedule, and token budget fixed. The five nested plans are
strictly nested: each larger plan replaces a superset of the smaller plan's layers.

| Config | Attention | Replaced | Zero-based no-RoPE layers |
| --- | ---: | ---: | --- |
| `18a2_triglu_no_rope_nested.yaml` | 18 | 2 | 15, 19 |
| `15a5_triglu_no_rope_nested.yaml` | 15 | 5 | 7, 11, 15, 17, 19 |
| `12a8_triglu_no_rope_nested.yaml` | 12 | 8 | 6, 7, 10, 11, 14, 15, 17, 19 |
| `9a11_triglu_no_rope_nested.yaml` | 9 | 11 | 6, 7, 9–11, 13–15, 17–19 |
| `6a14_triglu_no_rope_nested.yaml` | 6 | 14 | 6–19 |
| `15a5_triglu_no_rope_repeating.yaml` | 15 | 5 | 3, 7, 11, 15, 19 |
| `15a5_triglu_no_rope_tail_block.yaml` | 15 | 5 | 15, 16, 17, 18, 19 |
| `15a5_triglu_no_rope_early_intrusion.yaml` | 15 | 5 | 3, 12, 15, 17, 19 |
| `15a5_swiglu_repeating.yaml` (SwiGLU mixer) | 15 | 5 | 3, 7, 11, 15, 19 |
| `9a11_swiglu_nested.yaml` (SwiGLU mixer) | 9 | 11 | 6, 7, 9–11, 13–15, 17–19 |

The early-intrusion plan is the front-blend mask with its earliest replacement moved from
layer 8 to layer 3 (a single swap; later replicated at three seeds). The two SwiGLU-mixer
probes reuse the no-RoPE plans' masks to test whether placement sensitivity and the
55%-replacement result persist for the simplest gated formulation.

These plans are exploratory single-seed probes unless replicated, subject to the same
screening-seed caveats as the original sweep: they locate trends and select candidates for
focused
replication; they do not by themselves support confirmatory claims.

## Post-hoc no-RoPE placement and amount study

The follow-up configs under `configs/20l_4k_1b/placement_amount/` use the no-RoPE
formulation `W_o[k * SiLU(g) * v]` throughout. They retain the same seed-1337 data order,
model dimensions, optimizer, schedule, wrapper, evaluation set, and one-billion-token
budget. The study is explicitly exploratory because its component and masks were chosen
after observing the primary and prior-art-control results.

The staged amount ladder is:

| Attention / token-local | No-RoPE TriGLU layers |
| --- | --- |
| 18 / 2 | `{15, 19}` |
| 15 / 5 | `{7, 11, 15, 17, 19}` |
| 12 / 8 | `{6, 7, 10, 11, 14, 15, 17, 19}` |
| 9 / 11 | `{6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19}` |
| 6 / 14 | `{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}` |

Each replacement set is a strict superset of the previous set. The last transition
removes the only late attention layers—`{8, 12, 16}`—from the 9A/11T plan, leaving a
six-attention prefix followed by fourteen token-local blocks.

The fixed 15A/5T placement comparison uses five distinct masks:

| Placement | No-RoPE TriGLU layers |
| --- | --- |
| selected front blend | `{8, 12, 15, 17, 19}` |
| nested-ladder rung | `{7, 11, 15, 17, 19}` |
| repeating | `{3, 7, 11, 15, 19}` |
| tail block | `{15, 16, 17, 18, 19}` |
| early intrusion (front blend, 8 → 3) | `{3, 12, 15, 17, 19}` |

Nestedness makes marginal changes along this one removal path readable, but replacement
count still changes together with the identities of the newly converted layers. The
same-count comparison is the direct placement control. A late loss increase would locate
a candidate knee; it would not prove that periodic attention refresh is the cause.
Follow-up replication should be limited to the knee and its immediate neighboring plans
rather than treating every post-hoc screen as independent confirmation.

## Interpretation boundary

TriGLU and all three secondary controls are token local: none can transmit information
between positions. Every hybrid therefore relies on retained attention layers for
cross-token communication. An all-token-local stack is useful as a diagnostic negative
control, not as a general language-model architecture.

Effective-rank measurements describe the geometry of sampled activations; zero-update
interventions describe dependence on a trained component under an out-of-distribution
perturbation. Neither measurement alone identifies why a placement succeeds. Mechanistic
claims should be supported by stage-separated rank, update scale/direction, intervention
sensitivity, and controlled training outcomes together.
