"""Public API for the TriGLU research model."""

from .config import ModelConfig
from .layers import (
    CausalSelfAttention,
    DecoderBlock,
    LayerCache,
    TriGLU,
    RMSNorm,
    SwiGLU,
)
from .model import DecoderLM, TriGLULM, LMOutput
from .rope import RopeModule, RotaryEmbedding

__all__ = [
    "CausalSelfAttention",
    "DecoderBlock",
    "DecoderLM",
    "LayerCache",
    "TriGLULM",
    "TriGLU",
    "LMOutput",
    "ModelConfig",
    "RMSNorm",
    "RopeModule",
    "RotaryEmbedding",
    "SwiGLU",
]
