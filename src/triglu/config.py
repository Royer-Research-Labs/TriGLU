"""Configuration for the controlled decoder-only language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


_LAYER_TYPES = frozenset({"attention", "triglu"})


@dataclass
class ModelConfig:
    """Architecture settings for :class:`triglu.DecoderLM`.

    The defaults describe the canonical 12-layer, width-512 model.  A different
    depth should also supply ``layer_types`` explicitly; requiring one entry per
    layer keeps the attention/TriGLU plan visible in every experiment.
    """

    vocab_size: int = 50_304
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    ffn_hidden_size: int = 1_376
    context_length: int = 1_024
    # Reaches only attention weights inside SDPA. TriGLU, SwiGLU, and the
    # residual stream carry no dropout, so a nonzero value regularizes the
    # retained attention layers alone — keep 0.0 for controlled comparisons.
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_embeddings: bool = True
    layer_types: list[str] = field(
        default_factory=lambda: ["attention"] * 12
    )

    def __post_init__(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "ffn_hidden_size": self.ffn_hidden_size,
            "context_length": self.context_length,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads "
                f"({self.d_model} is not divisible by {self.n_heads})"
            )

        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError(
                "the per-head rotary dimension must be even, "
                f"got head_dim={head_dim}"
            )
        if self.d_model % 2 != 0:
            raise ValueError(
                "d_model must be even for full-width TriGLU RoPE, "
                f"got d_model={self.d_model}"
            )

        if not isinstance(self.bias, bool):
            raise TypeError(f"bias must be a boolean, got {self.bias!r}")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError(f"dropout must be numeric, got {self.dropout!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if isinstance(self.rope_theta, bool) or not isinstance(
            self.rope_theta, (int, float)
        ):
            raise TypeError(f"rope_theta must be numeric, got {self.rope_theta!r}")
        if self.rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if isinstance(self.norm_eps, bool) or not isinstance(self.norm_eps, (int, float)):
            raise TypeError(f"norm_eps must be numeric, got {self.norm_eps!r}")
        if self.norm_eps <= 0.0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
        if isinstance(self.init_std, bool) or not isinstance(self.init_std, (int, float)):
            raise TypeError(f"init_std must be numeric, got {self.init_std!r}")
        if self.init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {self.init_std}")
        if not isinstance(self.tie_embeddings, bool):
            raise TypeError(
                f"tie_embeddings must be a boolean, got {self.tie_embeddings!r}"
            )
        if not self.tie_embeddings:
            raise ValueError("tie_embeddings must remain true in this controlled model")

        if not isinstance(self.layer_types, list):
            raise TypeError("layer_types must be an explicit list of layer type names")
        # Copy the list so callers cannot mutate the source object after validation.
        self.layer_types = list(self.layer_types)
        if len(self.layer_types) != self.n_layers:
            raise ValueError(
                "layer_types must contain exactly one entry per layer: "
                f"expected {self.n_layers}, got {len(self.layer_types)}"
            )
        if any(not isinstance(layer_type, str) for layer_type in self.layer_types):
            raise TypeError("every layer_types entry must be a string")
        invalid = sorted(set(self.layer_types) - _LAYER_TYPES)
        if invalid:
            allowed = ", ".join(sorted(_LAYER_TYPES))
            raise ValueError(
                f"unsupported layer type(s) {invalid}; expected only {allowed}"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, serialization-friendly copy of the configuration."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Construct a config, rejecting misspelled or unsupported fields."""

        if not isinstance(values, Mapping):
            raise TypeError(f"config values must be a mapping, got {type(values).__name__}")
        return cls(**dict(values))
