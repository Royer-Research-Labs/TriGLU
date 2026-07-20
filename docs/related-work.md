# Related work and contribution boundary

Last reviewed: 2026-07-18.

This repository does not claim that multiplicative gating, three projected feature
products, or token-local attention replacement originated with TriGLU. It also does not
claim novelty for the equation. The canonical RoPE form and the recommended no-RoPE form,

```text
y_t = W_o[RoPE_t(W_k x_t) * SiLU(W_g x_t) * W_v x_t]      # canonical / origin
y_t = W_o[         W_k x_t  * SiLU(W_g x_t) * W_v x_t]      # recommended (no-RoPE)
```

are both triple-branch gated products with an output projection; the no-RoPE form is
structurally a member of the Gated Quadratic Unit family (see below). The contribution is
therefore not the formula but **the controlled causal-language-model study** built around
it: separating what fills a vacated attention slot from where and how much attention is
replaced, and measuring placement, replacement amount, quality retention, throughput, and
KV-cache trade-offs under parameter-matched, seed-matched conditions. None of the
observations below is a patentability or freedom-to-operate opinion.

## Closest component and experiment precedents

| Work | Material overlap | Distinction from this study |
| --- | --- | --- |
| [Gated Quadratic Unit (GQU)](https://arxiv.org/abs/2602.14495) | Three learned branches with one activated branch, a triple elementwise product, and an output projection — the same structure as the recommended no-RoPE TriGLU form, and its closest algebraic precedent. | Evaluated as a shallow function-approximation unit, not as an attention-slot component in a causal decoder; no hybrid language model, and no placement or replacement-amount study. |
| [MB-MLP](https://aclanthology.org/2026.acl-long.853/) ([earlier OpenReview record](https://openreview.net/forum?id=Mu18gwLAnk)) | Three Q/K/V projections fused as `GELU(Q * K) * V`, explicitly removing token interaction from attention. | Multivariate time-series forecasting, different activation placement, no RoPE, and no causal-language-model experiment. |
| [Deconstructing Attention](https://aclanthology.org/2025.ijcnlp-long.40/) | A parameter-matched token-local `W_o[SiLU(W_g x) * W_v x]` replaces attention in uniform and hybrid language models — exactly this repository's two-factor SwiGLU control, and the closest precedent for the hybrid-replacement question. | Two active factors rather than three; it does not vary the token-local formulation or map placement against replacement amount, where this study finds the third factor matters only under aggressive replacement. |
| [PADRe](https://arxiv.org/abs/2407.11306) | General polynomial attention replacements built from projected branches and Hadamard products. | A broader framework evaluated primarily in vision; it does not run the controlled placement/amount/formulation study in a causal decoder. |
| [H3](https://arxiv.org/abs/2212.14052) and [Hyena](https://proceedings.mlr.press/v202/poli23a.html) | Multiplicatively gated projected streams, efficient attention replacement, and hybrid language models. | Their defining operators perform temporal mixing through state-space operations or long convolutions. TriGLU contains no such operation. |
| [SwiGLU](https://arxiv.org/abs/2002.05202) and the original [GLU](https://proceedings.mlr.press/v70/dauphin17a.html) | Establish the two-factor gated product underlying `SiLU(g) * v`. | Conventionally used as an FFN/channel transformation, not this three-factor positional attention-slot component. |
| [ShishuLM](https://arxiv.org/abs/2510.13860) | From-scratch hybrid language models with attention removed from selected decoder blocks and replaced by MLP capacity. | Uses ordinary MLP-only blocks rather than the TriGLU equation and studies a different placement/parameter-sharing design. |

H3 and Hyena are also relevant to the component's history: TriGLU was independently
obtained by removing temporal mixing from a richer experimental mixer. Algebraically
replacing the temporal operators in those published architectures with identities can
also expose projected multiplicative products. That observation is a structural
comparison, not a claim that either paper evaluated TriGLU.

## Placement and amount precedents

Attention-layer placement and amount have been studied before, in three families. None
varies placement, amount, and the token-local formulation inside one parameter-matched,
seed-matched frame, which is what the sweeps here do — but each anticipates part of the
question, and the directional agreement below is external support for, not a threat to,
the placement findings.

| Work | Material overlap | Distinction from this study |
| --- | --- | --- |
| [Sandwich Transformers](https://arxiv.org/abs/1911.03864) | Reorders attention/FFN sublayers; attention-heavy bottom with FFN-heavy top improves language modeling. | Reorders sublayers of a fixed budget rather than replacing attention in place; no cache/throughput accounting, single formulation. |
| [Pay Attention when Required](https://arxiv.org/abs/2009.04534) | Replaces ~63% of self-attention blocks with feed-forward blocks via searched placement at maintained quality — the closest single precedent in spirit. | Searched orderings with conventional FFNs only; no controlled formulation comparison, matched-seed replication, or KV-cache analysis. |
| [Not All Attention is Needed](https://arxiv.org/abs/2406.15786) | Drops redundant attention layers from trained models (about half in Llama-2-70B) with minimal loss. | Post-hoc pruning of trained models, not from-scratch training with explicit placement plans. |
| [Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887) | Deep-layer pruning shows later layers are more expendable — consistent with late replacement working here. | Removes whole blocks post-hoc; not an attention-slot substitution study. |
| [Jamba](https://arxiv.org/abs/2403.19887), [Waleffe et al.](https://arxiv.org/abs/2406.07887), and hybrid-design analyses ([1](https://arxiv.org/abs/2510.04800), [2](https://arxiv.org/pdf/2407.05489)) | Attention:SSM ratio and interleaving studies; small attention fractions (down to ~7%) suffice, with attention often placed mid-stack and no consensus on optimal position. | Their replacement layers are other **sequence mixers** — the vacated slots still mix tokens. Here the replaced slots perform no token mixing at all, so placement measures where any cross-token communication is required. |

Recent hybrid analyses continue this line ([component ablation](https://arxiv.org/abs/2603.22473),
[memory-recall behavior](https://arxiv.org/abs/2510.26912)); none isolates token-local
replacement formulation from placement and amount.

## Claims this repository does not make

The cited work rules out broad descriptions such as:

- the first triple-product gated unit;
- the first QKV-like block without token interaction;
- the first token-local attention replacement;
- the first hybrid language model combining attention and gated MLP layers;
- the first study of attention-layer placement or amount; or
- a new general class of polynomial attention alternatives.

A narrower description is supported:

> TriGLU is a token-local triple-product GLU — a GQU-family gate — evaluated as a
> parameter-matched replacement for selected attention sublayers, and as an FFN, in
> causal language models. The contribution is the controlled study built around it: how
> much attention a decoder needs and where, how placement and replacement amount trade
> off against quality and efficiency, and which token-local formulations matter under
> which conditions — not novelty of the equation.

## Direct differentiation controls

The secondary configs under `configs/20l_4k_1b/ablations/` hold the selected
front-blend layer positions fixed and compare:

1. `k * SiLU(g) * v`, removing RoPE while retaining TriGLU's other operations;
2. `GELU(k * g) * v`, adapting the MB-MLP fusion equation to the same-width slot; and
3. `SiLU(g) * v`, using a genuine two-factor SwiGLU with a near-matched projection
   budget.

The first is a single-variable positional ablation. The MB-MLP comparison changes both
activation placement and positional treatment, so it is a published-equation control,
not a single-variable activation ablation. The SwiGLU comparison changes factor-space
width from 512 to 683 to match projection parameters within 0.003% of the complete model;
it should be interpreted as a parameter-budget control rather than an identical-width
control.

These controls test whether the exact TriGLU formulation is informative. Together with
the exploratory placement/amount sweep tabulated in [`docs/design.md`](design.md), they
also test whether placement and replacement-amount effects persist across token-local
formulations rather than belonging to any single equation. They cannot establish legal
novelty, and a literature/public-code search cannot exclude unpublished implementations
or patent applications.
