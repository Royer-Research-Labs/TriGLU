from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from triglu import TriGLU, ModelConfig, RopeModule


def tiny_triglu_config(*, bias: bool = True) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        n_layers=1,
        d_model=8,
        n_heads=1,
        ffn_hidden_size=16,
        context_length=16,
        bias=bias,
        layer_types=["triglu"],
    )


class VisibleOffsetRope(nn.Module):
    """A token-local test double that makes the K-only transformation obvious."""

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        positions = torch.arange(
            offset, offset + x.size(1), device=x.device, dtype=x.dtype
        ).view(1, -1, 1)
        return x + positions


def test_triglu_matches_authoritative_equation_and_rotates_only_k() -> None:
    torch.manual_seed(7)
    config = tiny_triglu_config()
    rope = VisibleOffsetRope()
    mixer = TriGLU(config, rope)
    x = torch.randn(2, 3, config.d_model)

    actual, cache = mixer(x, cache_position=5)

    projected = F.linear(x, mixer.c_proj.weight, mixer.c_proj.bias)
    k, g, v = projected.chunk(3, dim=-1)
    expected = F.linear(
        rope(k, offset=5) * F.silu(g) * v,
        mixer.proj_out.weight,
        mixer.proj_out.bias,
    )
    torch.testing.assert_close(actual, expected)
    assert cache is None


def test_triglu_is_token_local() -> None:
    torch.manual_seed(8)
    config = tiny_triglu_config()
    mixer = TriGLU(config, RopeModule(config.d_model)).eval()
    x = torch.randn(1, 5, config.d_model)
    changed = x.clone()
    changed[:, 2] += 100.0

    before, _ = mixer(x)
    after, _ = mixer(changed)

    torch.testing.assert_close(before[:, :2], after[:, :2])
    torch.testing.assert_close(before[:, 3:], after[:, 3:])
    assert not torch.equal(before[:, 2], after[:, 2])


def test_triglu_chunks_match_full_sequence_and_cache_has_no_kv() -> None:
    torch.manual_seed(9)
    config = tiny_triglu_config(bias=False)
    mixer = TriGLU(config, RopeModule(config.d_model)).eval()
    x = torch.randn(2, 6, config.d_model)

    full, full_cache = mixer(x, cache_position=3, use_cache=True)
    first, first_cache = mixer(x[:, :2], cache_position=3, use_cache=True)
    second, second_cache = mixer(
        x[:, 2:], cache=first_cache, cache_position=5, use_cache=True
    )

    torch.testing.assert_close(full, torch.cat((first, second), dim=1))
    assert first_cache == (None, None, 5)
    assert second_cache == (None, None, 9)
    assert full_cache == (None, None, 9)


def test_triglu_parameter_count() -> None:
    for bias in (False, True):
        config = tiny_triglu_config(bias=bias)
        mixer = TriGLU(config, RopeModule(config.d_model))
        expected = 4 * config.d_model**2 + (4 * config.d_model if bias else 0)
        assert sum(parameter.numel() for parameter in mixer.parameters()) == expected


def test_rope_offset_matches_sliced_full_application() -> None:
    torch.manual_seed(10)
    rope = RopeModule(8)
    x = torch.randn(2, 7, 8)
    full = rope(x, offset=4)
    chunks = torch.cat((rope(x[:, :3], offset=4), rope(x[:, 3:], offset=7)), dim=1)
    torch.testing.assert_close(full, chunks)


def test_rope_pins_adjacent_pair_rotation_convention() -> None:
    # Fixed-value pin: position 1, dim 4, default theta. The adjacent pairs
    # (0, 1) and (2, 3) rotate by 1.0 and 10000^(-2/4) radians respectively,
    # as (x0 cos - x1 sin, x1 cos + x0 sin). A sign flip or a split-half pair
    # convention cannot reproduce these numbers.
    rope = RopeModule(4)
    x = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
    fast, slow = 1.0, 10_000.0 ** -0.5
    expected = torch.tensor(
        [[[math.cos(fast), math.sin(fast), -math.sin(slow), math.cos(slow)]]]
    )
    torch.testing.assert_close(rope(x, offset=1), expected)


def test_rope_survives_module_dtype_conversion() -> None:
    # `.to(torch.bfloat16)` downcasts registered buffers; the rotation must
    # keep computing its angles in float32 or late positions lose precision.
    torch.manual_seed(12)
    x = torch.randn(1, 64, 8)
    reference = RopeModule(8)(x, offset=960)
    converted = RopeModule(8).to(torch.bfloat16)
    low_precision = converted(x.to(torch.bfloat16), offset=960)
    torch.testing.assert_close(
        low_precision.float(), reference, rtol=0.05, atol=0.05
    )


def test_triglu_validates_supplied_cache_tuple() -> None:
    torch.manual_seed(11)
    config = tiny_triglu_config(bias=False)
    mixer = TriGLU(config, RopeModule(config.d_model)).eval()
    x = torch.randn(1, 2, config.d_model)

    _, cache = mixer(x, cache=(None, None, 3), cache_position=3, use_cache=True)
    assert cache == (None, None, 5)

    with pytest.raises(ValueError, match="position mismatch"):
        mixer(x, cache=(None, None, 2))
    with pytest.raises(ValueError, match="stores no key/value"):
        mixer(x, cache=(x, x, 0))
