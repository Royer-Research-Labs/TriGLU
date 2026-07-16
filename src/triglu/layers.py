"""Building blocks for the controlled Transformer/TriGLU ablation."""

from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .rope import RopeModule


LayerCache: TypeAlias = tuple[torch.Tensor | None, torch.Tensor | None, int]


# Use PyTorch's standard implementation so normalization is not an experimental
# variable in the controlled ablation.  The alias keeps the package API concise.
RMSNorm = nn.RMSNorm


class SwiGLU(nn.Module):
    """Conventional SwiGLU feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.d_model, config.ffn_hidden_size, bias=config.bias
        )
        self.up_proj = nn.Linear(
            config.d_model, config.ffn_hidden_size, bias=config.bias
        )
        self.down_proj = nn.Linear(
            config.ffn_hidden_size, config.d_model, bias=config.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TriGLU(nn.Module):
    """The authoritative token-local Triple-Product Gated Linear Unit.

    TriGLU extends the two-factor SwiGLU form with a third projected factor:
    ``RoPE(K) * SiLU(G) * V``. It performs no cross-token reduction or
    aggregation. RoPE changes each token independently and is applied to K only.
    """

    def __init__(self, config: ModelConfig, rope: RopeModule | None):
        super().__init__()
        self.act = nn.SiLU()
        C = config.d_model

        self.rope = rope
        self.c_proj = nn.Linear(C, 3 * C, bias=config.bias)
        self.proj_out = nn.Linear(C, C, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache | None = None,
        cache_position: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        if cache is not None:
            if len(cache) != 3:
                raise ValueError("a TriGLU cache must be a (None, None, position) tuple")
            cached_k, cached_v, cached_position = cache
            if cached_k is not None or cached_v is not None:
                raise ValueError("a TriGLU cache stores no key/value tensors")
            if cached_position != cache_position:
                raise ValueError(
                    "TriGLU cache position mismatch: "
                    f"cache has {cached_position}, forward requested {cache_position}"
                )

        # K, G, V = split(W_kgv x), with three full-width C streams.
        k, g, v = self.c_proj(x).chunk(3, dim=-1)

        # RoPE rotates K only; it changes each token independently and does not
        # mix tokens.
        if self.rope is not None:
            k = self.rope(k, offset=cache_position)

        # y = W_o(K ⊙ SiLU(G) ⊙ V).
        y = k * self.act(g)
        y = y * v
        y = self.proj_out(y)

        # TriGLU stores no K/V history, only the next absolute position.
        triglu_cache = (
            (None, None, cache_position + k.size(1)) if use_cache else None
        )
        return y, triglu_cache


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention using PyTorch SDPA."""

    def __init__(self, config: ModelConfig, rope: RopeModule | None = None) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.context_length = config.context_length
        self.dropout = config.dropout
        self.rope = rope or RopeModule(config.head_dim, theta=config.rope_theta)
        self.c_proj = nn.Linear(
            config.d_model, 3 * config.d_model, bias=config.bias
        )
        self.proj_out = nn.Linear(config.d_model, config.d_model, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache | None = None,
        cache_position: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        batch_size, seq_len, channels = x.shape
        q, k, v = self.c_proj(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, seq_len, self.n_heads, self.head_dim
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        q = self.rope(q, offset=cache_position)
        k = self.rope(k, offset=cache_position)

        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        static_cache = False
        if cache is not None:
            if len(cache) != 3:
                raise ValueError("an attention cache must be a (key, value, position) tuple")
            past_k, past_v, cached_position = cache
            if (past_k is None) != (past_v is None):
                raise ValueError("attention cache must contain both key and value, or neither")
            if cached_position != cache_position:
                raise ValueError(
                    "attention cache position mismatch: "
                    f"cache has {cached_position}, forward requested {cache_position}"
                )
            if past_k is not None:
                expected_prefix = (batch_size, self.n_heads)
                if past_k.shape[:2] != expected_prefix or past_v.shape[:2] != expected_prefix:
                    raise ValueError("attention cache batch/head dimensions do not match input")
                if past_k.size(-1) != self.head_dim or past_v.size(-1) != self.head_dim:
                    raise ValueError("attention cache head dimension does not match model")
                if past_k.shape != past_v.shape:
                    raise ValueError("attention cache key/value shapes must match")

                cache_length = past_k.size(-2)
                # Dynamic caches have exactly one stored entry per prior token.
                # Static caches reserve the model's complete context capacity.
                # A full cache (position == capacity) must classify as static so
                # the capacity check below fails loudly instead of the dynamic
                # path silently concatenating past the model's context.
                static_cache = (
                    cache_length == self.context_length
                    and cache_position <= cache_length
                )
                if not static_cache and cache_length != cache_position:
                    raise ValueError(
                        "attention cache must either have dynamic length equal to "
                        "cache_position or full static context capacity; got "
                        f"length {cache_length} and position {cache_position}"
                    )

        next_position = cache_position + seq_len
        if static_cache:
            if not use_cache:
                raise ValueError("a static attention cache requires use_cache=True")
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "static attention caches are inference-only; use torch.no_grad() "
                    "or torch.inference_mode()"
                )
            if next_position > self.context_length:
                raise ValueError(
                    "static attention cache capacity exceeded: "
                    f"requested position {next_position}, capacity {self.context_length}"
                )
            assert past_k is not None and past_v is not None
            # Copy into stable storage and expose only initialized positions to
            # SDPA. Unwritten capacity must never participate in attention.
            past_k[:, :, cache_position:next_position].copy_(k)
            past_v[:, :, cache_position:next_position].copy_(v)
            k = past_k[:, :, :next_position]
            v = past_v[:, :, :next_position]

        has_prefix = cache_position > 0 and past_k is not None
        if has_prefix and not static_cache:
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        # `is_causal=True` uses a top-left triangular mask.  That is ideal for
        # full-sequence training, but incorrect for a short query appended to a
        # longer cache, so cached decoding uses an absolute-position mask.
        attn_mask = None
        is_causal = not has_prefix
        if has_prefix and seq_len == 1:
            # A single newly appended query can attend every cached key.  Avoid
            # materializing an all-true mask in the latency-sensitive decode path.
            is_causal = False
        elif has_prefix:
            query_positions = torch.arange(
                cache_position,
                cache_position + seq_len,
                device=x.device,
            )
            key_positions = torch.arange(k.size(-2), device=x.device)
            attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.proj_out(y)

        if use_cache and static_cache:
            # Return the capacity tensors, not their initialized prefix views,
            # so every decode step retains the same allocation and data pointer.
            assert past_k is not None and past_v is not None
            next_cache = (past_k, past_v, next_position)
        else:
            next_cache = (k, v, next_position) if use_cache else None
        return y, next_cache


class DecoderBlock(nn.Module):
    """Shared two-sublayer pre-norm wrapper for either mixer choice.

    Keeping this wrapper identical is essential: the layer plan changes only
    the attention-replacement component, not normalization or residual topology.
    """

    def __init__(self, config: ModelConfig, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.norm_1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm_2 = RMSNorm(config.d_model, eps=config.norm_eps)
        if layer_type == "attention":
            self.mixer: nn.Module = CausalSelfAttention(config)
        elif layer_type == "triglu":
            triglu_rope = RopeModule(config.d_model, theta=config.rope_theta)
            self.mixer = TriGLU(config, triglu_rope)
        else:
            raise ValueError(f"unsupported layer type {layer_type!r}")
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache | None = None,
        cache_position: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        mixed, next_cache = self.mixer(
            self.norm_1(x),
            cache=cache,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        x = x + mixed
        x = x + self.ffn(self.norm_2(x))
        return x, next_cache
