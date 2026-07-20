"""Configuration for the controlled decoder-only language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


_LAYER_TYPES = frozenset(
    {
        "attention",
        "triglu",
        "triglu_no_rope",
        "mb_mlp",
        "swiglu_mixer",
        "ffn_only",
    }
)
_FFN_TYPES = frozenset({"swiglu", "triglu_no_rope"})
_BLOCK_MODES = frozenset({"sequential", "parallel"})


@dataclass
class ModelConfig:
    """Architecture settings for :class:`triglu.DecoderLM`.

    The defaults describe the canonical 12-layer, width-512 model.  A different
    depth should also supply ``layer_types`` explicitly; requiring one entry per
    layer keeps the attention/replacement plan visible in every experiment.
    """

    vocab_size: int = 50_304
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    ffn_hidden_size: int = 1_376
    context_length: int = 1_024
    # Reaches only attention weights inside SDPA. Token-local mixers and the
    # residual stream carry no dropout, so a nonzero value regularizes retained
    # attention layers alone — keep 0.0 for controlled comparisons.
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_embeddings: bool = True
    # Required only by the two-factor SwiGLU attention-slot control. Keeping
    # this width explicit makes its near-parameter-matching choice part of the
    # resolved experiment record instead of an implementation-side convention.
    swiglu_mixer_hidden_size: int | None = None
    layer_types: list[str] = field(
        default_factory=lambda: ["attention"] * 12
    )
    # Appended after the v0.1 fields to preserve positional construction.
    # Conventional SwiGLU remains the default; the no-RoPE TriGLU option is a
    # separate FFN-form control and does not change the attention plan.
    ffn_type: str = "swiglu"
    # Optional layerwise widths support explicitly labeled topology controls.
    # Primary experiments leave this unset and use ``ffn_hidden_size`` at every
    # layer. Keeping the override in the resolved config makes width allocation
    # reviewable rather than an implementation-side convention.
    ffn_hidden_sizes: list[int] | None = None
    # By default residual projections use the physical depth in the GPT-style
    # initialization scale. A collapsed-depth control can retain the reference
    # model's scale by recording its reference depth explicitly.
    residual_init_depth: int | None = None
    # Block topology. "sequential" is the canonical two-norm pre-norm wrapper
    # used by every primary experiment. "parallel" is a labeled speed/quality
    # control: mixer and FFN read one shared norm of the block input and write
    # into a single residual add — x + mixer(norm(x)) + ffn(norm(x)).
    block_mode: str = "sequential"

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

        if not isinstance(self.ffn_type, str):
            raise TypeError(f"ffn_type must be a string, got {self.ffn_type!r}")
        if self.ffn_type not in _FFN_TYPES:
            allowed = ", ".join(sorted(_FFN_TYPES))
            raise ValueError(
                f"unsupported FFN type {self.ffn_type!r}; expected one of {allowed}"
            )

        if self.ffn_hidden_sizes is not None:
            if not isinstance(self.ffn_hidden_sizes, list):
                raise TypeError(
                    "ffn_hidden_sizes must be null or an explicit list of "
                    "positive integer widths"
                )
            self.ffn_hidden_sizes = list(self.ffn_hidden_sizes)
            if len(self.ffn_hidden_sizes) != self.n_layers:
                raise ValueError(
                    "ffn_hidden_sizes must contain exactly one entry per layer: "
                    f"expected {self.n_layers}, got {len(self.ffn_hidden_sizes)}"
                )
            for index, width in enumerate(self.ffn_hidden_sizes):
                if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                    raise ValueError(
                        "every ffn_hidden_sizes entry must be a positive integer; "
                        f"layer {index} has {width!r}"
                    )

        if self.residual_init_depth is not None and (
            isinstance(self.residual_init_depth, bool)
            or not isinstance(self.residual_init_depth, int)
            or self.residual_init_depth <= 0
        ):
            raise ValueError(
                "residual_init_depth must be null or a positive integer, got "
                f"{self.residual_init_depth!r}"
            )

        if not isinstance(self.block_mode, str):
            raise TypeError(f"block_mode must be a string, got {self.block_mode!r}")
        if self.block_mode not in _BLOCK_MODES:
            allowed = ", ".join(sorted(_BLOCK_MODES))
            raise ValueError(
                f"unsupported block_mode {self.block_mode!r}; expected one of {allowed}"
            )
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
        if "ffn_only" in self.layer_types and self.ffn_hidden_sizes is None:
            raise ValueError(
                "ffn_hidden_sizes must be explicit when layer_types contains "
                "'ffn_only'"
            )
        if self.block_mode == "parallel" and "ffn_only" in self.layer_types:
            raise ValueError(
                "block_mode 'parallel' is incompatible with 'ffn_only' layers, "
                "which already have no mixer sublayer to parallelize"
            )

        uses_swiglu_mixer = "swiglu_mixer" in self.layer_types
        hidden_size = self.swiglu_mixer_hidden_size
        if uses_swiglu_mixer:
            if (
                isinstance(hidden_size, bool)
                or not isinstance(hidden_size, int)
                or hidden_size <= 0
            ):
                raise ValueError(
                    "swiglu_mixer_hidden_size must be a positive integer when "
                    "layer_types contains 'swiglu_mixer'"
                )
        elif hidden_size is not None:
            raise ValueError(
                "swiglu_mixer_hidden_size must be null unless layer_types contains "
                "'swiglu_mixer'"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def ffn_hidden_size_for_layer(self, layer_index: int) -> int:
        """Return the validated FFN width for one zero-based layer."""

        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.n_layers
        ):
            raise IndexError(
                f"layer_index must be in [0, {self.n_layers}), got {layer_index!r}"
            )
        if self.ffn_hidden_sizes is None:
            return self.ffn_hidden_size
        return self.ffn_hidden_sizes[layer_index]

    @property
    def effective_ffn_hidden_sizes(self) -> list[int]:
        """Return a fresh list containing every physical layer's FFN width."""

        if self.ffn_hidden_sizes is None:
            return [self.ffn_hidden_size] * self.n_layers
        return list(self.ffn_hidden_sizes)

    @property
    def effective_residual_init_depth(self) -> int:
        """Depth used by GPT-style residual projection initialization."""

        return (
            self.n_layers
            if self.residual_init_depth is None
            else self.residual_init_depth
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, serialization-friendly copy of the configuration."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Construct a config, rejecting misspelled or unsupported fields."""

        if not isinstance(values, Mapping):
            raise TypeError(f"config values must be a mapping, got {type(values).__name__}")
        return cls(**dict(values))
