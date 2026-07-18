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

The resolved configuration records the complete layer list. Ratios are labels, not a
hidden placement algorithm.

## Authoritative TriGLU equation

TriGLU means **Triple-Product Gated Linear Unit**. For an input token `x_t` with width
`C`, it computes

```text
(k_t, g_t, v_t) = split_3(W_c x_t + b_c)
k_t              = RoPE(k_t, position=t)
z_t              = k_t * SiLU(g_t) * v_t
y_t              = W_o z_t + b_o
```

All products are elementwise. `W_c` is one fused `C -> 3C` projection and `W_o` is
`C -> C`. RoPE rotates the full `C`-wide `k` vector and is not applied to `g` or `v`.
No operation reads another token. This is therefore a position-dependent, token-local,
channel-wise gated-product block—not linear attention and not an attention approximation.
Relative to the two-factor `SiLU(g_t) * v_t` SwiGLU form, TriGLU adds the third
projected factor `k_t`. “Tri” names the three multiplicative factors, not three gates.

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
- per-head RoPE for attention and full-width RoPE for TriGLU;
- standard SwiGLU channel MLP;
- tied input embeddings and language-model head;
- GPT-style depth scaling on residual output projections;
- no dropout by default.

The primary suites add no learned positional mode, convolution, recurrence, state-space
layer, temporal mixer beyond standard attention, custom normalization, custom FFN
family, MoE, or shared parameter bank. The secondary controls described below add only
the explicitly named attention-slot equations; the surrounding decoder FFN remains the
same conventional SwiGLU in every run.

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

These plans are exploratory single-seed probes, subject to the same screening-seed
caveats as the original sweep: they locate trends and select candidates for focused
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

The fixed 15A/5T placement comparison uses four distinct masks:

| Placement | No-RoPE TriGLU layers |
| --- | --- |
| selected front blend | `{8, 12, 15, 17, 19}` |
| nested-ladder rung | `{7, 11, 15, 17, 19}` |
| repeating | `{3, 7, 11, 15, 19}` |
| tail block | `{15, 16, 17, 18, 19}` |

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
