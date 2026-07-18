# Related work and contribution boundary

Last reviewed: 2026-07-17.

This repository does not claim that multiplicative gating, three projected feature
products, or token-local attention replacement originated with TriGLU. Its contribution
is the controlled causal-language-model evaluation of this exact component:

```text
y_t = W_o[RoPE_t(W_k x_t) * SiLU(W_g x_t) * W_v x_t]
```

The full-width rotation on one branch is position dependent but token local. In the
sources reviewed below, no earlier work used that complete equation as a same-width
attention-slot component and studied its placement, quality retention, throughput, and
KV-cache trade-offs in a conventional causal decoder. That narrow observation is not a
patentability or freedom-to-operate opinion.

## Closest component and experiment precedents

| Work | Material overlap | Distinction from this study |
| --- | --- | --- |
| [Gated Quadratic Unit (GQU)](https://arxiv.org/abs/2602.14495) | Three learned branches with one activated branch, a triple elementwise product, and an output projection: the closest algebraic precedent. | Evaluated as a shallow function-approximation unit; no one-branch RoPE, attention-slot integration, hybrid causal decoder, or language-model placement study. |
| [MB-MLP](https://aclanthology.org/2026.acl-long.853/) ([earlier OpenReview record](https://openreview.net/forum?id=Mu18gwLAnk)) | Three Q/K/V projections fused as `GELU(Q * K) * V`, explicitly removing token interaction from attention. | Multivariate time-series forecasting, different activation placement, no RoPE, and no causal-language-model experiment. |
| [Deconstructing Attention](https://aclanthology.org/2025.ijcnlp-long.40/) | A parameter-matched token-local `W_o[SiLU(W_g x) * W_v x]` replaces attention in uniform and hybrid language models. | Two active factors rather than three and no positional rotation in the token-local block. |
| [PADRe](https://arxiv.org/abs/2407.11306) | General polynomial attention replacements built from projected branches and Hadamard products. | A broader framework evaluated primarily in vision; it does not isolate TriGLU's exact activation and one-branch rotation. |
| [H3](https://arxiv.org/abs/2212.14052) and [Hyena](https://proceedings.mlr.press/v202/poli23a.html) | Multiplicatively gated projected streams, efficient attention replacement, and hybrid language models. | Their defining operators perform temporal mixing through state-space operations or long convolutions. TriGLU contains no such operation. |
| [SwiGLU](https://arxiv.org/abs/2002.05202) and the original [GLU](https://proceedings.mlr.press/v70/dauphin17a.html) | Establish the two-factor gated product underlying `SiLU(g) * v`. | Conventionally used as an FFN/channel transformation, not this three-factor positional attention-slot component. |
| [ShishuLM](https://arxiv.org/abs/2510.13860) | From-scratch hybrid language models with attention removed from selected decoder blocks and replaced by MLP capacity. | Uses ordinary MLP-only blocks rather than the TriGLU equation and studies a different placement/parameter-sharing design. |

H3 and Hyena are also relevant to the component's history: TriGLU was independently
obtained by removing temporal mixing from a richer experimental mixer. Algebraically
replacing the temporal operators in those published architectures with identities can
also expose projected multiplicative products. That observation is a structural
comparison, not a claim that either paper evaluated TriGLU.

## Claims this repository does not make

The cited work rules out broad descriptions such as:

- the first triple-product gated unit;
- the first QKV-like block without token interaction;
- the first token-local attention replacement;
- the first hybrid language model combining attention and gated MLP layers; or
- a new general class of polynomial attention alternatives.

A narrower description is supported:

> TriGLU is a token-local triple-product GLU evaluated as a parameter-matched
> replacement for selected attention sublayers in causal language models. Its
> distinguishing formulation combines a one-branch positional rotation with
> `RoPE(k) * SiLU(g) * v`; this repository measures how placement affects retained
> language-modeling quality and efficiency.

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
