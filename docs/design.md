# Experimental design and integration boundary

This document distinguishes the independently usable TriGLU mixer from the decoder
wrapper used to evaluate it. That distinction is important for both audiences:

- component users can integrate the exact mixer into another architecture without
  importing the experimental harness; and
- reviewers can identify which conclusions are supported by controlled comparisons in
  this repository.

## Scientific variable

Within a fixed-seed comparison in an experiment suite, every plan uses the same decoder
width, depth, feed-forward network, normalization, residual topology, tokenizer, sampled
token sequence, optimizer, schedule, and token budget. The only architecture variable is
the explicit per-layer choice between causal attention and `TriGLU`. Focused replications
change the seed for every compared architecture together and report matched-seed deltas.

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

The research harness places either mixer in the same conventional pre-norm
decoder wrapper:

```text
h       = x + mixer(RMSNorm(x))
x_out   = h + SwiGLU(RMSNorm(h))
```

Attention uses standard multi-head causal scaled-dot-product attention and per-head
RoPE. TriGLU uses the authoritative equation above. Both mixer choices have the
same projection parameter count (`4*C*C`, plus `4*C` when bias is enabled).

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

No learned positional mode, convolution, recurrence, state-space layer, additional
temporal mixer beyond standard attention, custom normalization, custom FFN family, MoE,
shared parameter bank, or other experimental component is implemented.

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

## Interpretation boundary

TriGLU cannot transmit information between positions. Every hybrid therefore relies on
retained attention layers for cross-token communication. An all-TriGLU stack is useful as
a diagnostic negative control, not as a general language-model architecture.

Effective-rank measurements describe the geometry of sampled activations; zero-update
interventions describe dependence on a trained component under an out-of-distribution
perturbation. Neither measurement alone identifies why a placement succeeds. Mechanistic
claims should be supported by stage-separated rank, update scale/direction, intervention
sensitivity, and controlled training outcomes together.
