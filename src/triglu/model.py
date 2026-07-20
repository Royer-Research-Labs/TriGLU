"""Decoder-only causal language model for controlled mixer and FFN ablations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .layers import DecoderBlock, FFNOnlyBlock, LayerCache, RMSNorm


@dataclass
class LMOutput:
    """Outputs from :class:`DecoderLM`."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    caches: tuple[LayerCache, ...] | None = None


class DecoderLM(nn.Module):
    """A conservative decoder-only LM with a configurable layer plan."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            (
                FFNOnlyBlock(config, config.ffn_hidden_size_for_layer(layer_index))
                if layer_type == "ffn_only"
                else DecoderBlock(
                    config,
                    layer_type,
                    config.ffn_hidden_size_for_layer(layer_index),
                )
            )
            for layer_index, layer_type in enumerate(config.layer_types)
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self._scale_residual_projections()
        # Tie after initialization so the shared parameter is initialized once
        # through the token embedding and remains a single trainable tensor.
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def _scale_residual_projections(self) -> None:
        # GPT-style scaling keeps the residual-stream variance stable with depth.
        residual_std = self.config.init_std / math.sqrt(
            2 * self.config.effective_residual_init_depth
        )
        for block in self.blocks:
            if isinstance(block, DecoderBlock):
                nn.init.normal_(
                    block.mixer.proj_out.weight,
                    mean=0.0,
                    std=residual_std,
                )
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _cache_position(caches: Sequence[LayerCache]) -> int:
        positions: list[int] = []
        for index, cache in enumerate(caches):
            if cache is None or len(cache) != 3:
                raise ValueError(
                    f"cache for layer {index} must be a (key, value, position) tuple"
                )
            position = cache[2]
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError(
                    f"cache position for layer {index} must be a non-negative integer"
                )
            positions.append(position)
        if len(set(positions)) != 1:
            raise ValueError(f"all layer caches must have the same position, got {positions}")
        return positions[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: Sequence[LayerCache] | None = None,
        use_cache: bool = False,
        cache_position: int | None = 0,
        *,
        labels: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
    ) -> LMOutput:
        """Run the model.

        ``targets`` and ``labels`` are aliases for an already shifted target
        tensor: each entry is compared with the logit at the same position.
        Supplying caches means ``input_ids`` should contain only the uncached
        suffix.  The default cache position is inferred when caches are present.
        """

        if targets is not None and labels is not None:
            raise ValueError("pass only one of targets or labels")
        targets = labels if labels is not None else targets

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, sequence], got {input_ids.shape}"
            )
        if input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one token")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"input_ids must use an integer dtype, got {input_ids.dtype}")

        if logits_to_keep is not None:
            if (
                isinstance(logits_to_keep, bool)
                or not isinstance(logits_to_keep, int)
                or logits_to_keep <= 0
            ):
                raise ValueError(
                    "logits_to_keep must be a positive integer or None, got "
                    f"{logits_to_keep!r}"
                )
            if logits_to_keep > input_ids.size(1):
                raise ValueError(
                    "logits_to_keep cannot exceed the input sequence length: "
                    f"got {logits_to_keep} for length {input_ids.size(1)}"
                )
            if targets is not None and logits_to_keep < input_ids.size(1):
                raise ValueError(
                    "targets/labels cannot be used with truncated logits; "
                    "omit logits_to_keep or request the full sequence length"
                )

        if caches is not None:
            if len(caches) != self.config.n_layers:
                raise ValueError(
                    f"expected {self.config.n_layers} layer caches, got {len(caches)}"
                )
            inferred_position = self._cache_position(caches)
            if cache_position in (None, 0):
                cache_position = inferred_position
            elif cache_position != inferred_position:
                raise ValueError(
                    "cache_position does not match supplied caches: "
                    f"got {cache_position}, expected {inferred_position}"
                )
        elif cache_position is None:
            cache_position = 0

        if (
            isinstance(cache_position, bool)
            or not isinstance(cache_position, int)
            or cache_position < 0
        ):
            raise ValueError(
                f"cache_position must be a non-negative integer, got {cache_position!r}"
            )
        end_position = cache_position + input_ids.size(1)
        if end_position > self.config.context_length:
            raise ValueError(
                "sequence exceeds configured context length: "
                f"positions [0, {end_position}) vs limit {self.config.context_length}"
            )

        x = self.token_embedding(input_ids)
        next_caches: list[LayerCache] | None = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            layer_cache = caches[index] if caches is not None else None
            x, next_cache = block(
                x,
                cache=layer_cache,
                cache_position=cache_position,
                use_cache=use_cache,
            )
            if next_caches is not None:
                assert next_cache is not None
                next_caches.append(next_cache)

        hidden_for_logits = x if logits_to_keep is None else x[:, -logits_to_keep:]
        logits = self.lm_head(self.final_norm(hidden_for_logits))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    "targets/labels must have the same shape as input_ids, got "
                    f"{targets.shape} and {input_ids.shape}"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1).long(),
                ignore_index=-100,
            )

        return LMOutput(
            logits=logits,
            loss=loss,
            caches=tuple(next_caches) if next_caches is not None else None,
        )

    def num_parameters(self, *, exclude_embeddings: bool = False) -> int:
        """Return the number of unique trainable parameters."""

        parameters = sum(parameter.numel() for parameter in self.parameters())
        if exclude_embeddings:
            parameters -= self.token_embedding.weight.numel()
        return parameters

    def allocate_static_cache(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[LayerCache, ...]:
        """Preallocate fixed-capacity inference caches for every model layer.

        Attention layers receive ``[B, H, context, head_dim]`` K/V storage.
        Token-local replacement layers retain a position-only cache tuple.
        Pass the result back through ``caches=`` with ``use_cache=True`` under
        ``torch.no_grad()`` or ``torch.inference_mode()``.
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        parameter = self.token_embedding.weight
        resolved_device = parameter.device if device is None else torch.device(device)
        resolved_dtype = parameter.dtype if dtype is None else dtype
        if not isinstance(resolved_dtype, torch.dtype) or not resolved_dtype.is_floating_point:
            raise TypeError(f"cache dtype must be floating point, got {resolved_dtype!r}")

        caches: list[LayerCache] = []
        for block in self.blocks:
            if block.layer_type != "attention":
                caches.append((None, None, 0))
                continue
            shape = (
                batch_size,
                self.config.n_heads,
                self.config.context_length,
                self.config.head_dim,
            )
            key = torch.empty(shape, device=resolved_device, dtype=resolved_dtype)
            value = torch.empty(shape, device=resolved_device, dtype=resolved_dtype)
            caches.append((key, value, 0))
        return tuple(caches)


# A discoverable project-specific alias; both names refer to the same model.
TriGLULM = DecoderLM
