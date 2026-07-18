from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from triglu import MBMLP, ModelConfig, RopeModule, SwiGLUMixer, TriGLU


def tiny_config(layer_type: str, *, bias: bool = True) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        n_layers=1,
        d_model=8,
        n_heads=1,
        ffn_hidden_size=16,
        context_length=16,
        bias=bias,
        swiglu_mixer_hidden_size=11 if layer_type == "swiglu_mixer" else None,
        layer_types=[layer_type],
    )


def test_triglu_no_rope_matches_exact_equation_and_ignores_position() -> None:
    torch.manual_seed(201)
    config = tiny_config("triglu_no_rope")
    mixer = TriGLU(config, rope=None)
    x = torch.randn(2, 3, config.d_model)

    at_zero, _ = mixer(x, cache_position=0)
    at_later_position, _ = mixer(x, cache_position=9)
    k, g, v = F.linear(x, mixer.c_proj.weight, mixer.c_proj.bias).chunk(3, dim=-1)
    expected = F.linear(
        k * F.silu(g) * v,
        mixer.proj_out.weight,
        mixer.proj_out.bias,
    )

    torch.testing.assert_close(at_zero, expected)
    torch.testing.assert_close(at_later_position, expected)


def test_no_rope_control_differs_from_triglu_only_after_position_zero() -> None:
    torch.manual_seed(206)
    config = tiny_config("triglu_no_rope")
    no_rope = TriGLU(config, rope=None).eval()
    with_rope = TriGLU(config, RopeModule(config.d_model)).eval()
    with_rope.load_state_dict(no_rope.state_dict())
    x = torch.randn(2, 3, config.d_model)

    at_zero, _ = with_rope(x[:, :1], cache_position=0)
    no_rope_at_zero, _ = no_rope(x[:, :1], cache_position=0)
    at_five, _ = with_rope(x, cache_position=5)
    no_rope_at_five, _ = no_rope(x, cache_position=5)

    torch.testing.assert_close(at_zero, no_rope_at_zero)
    assert not torch.allclose(at_five, no_rope_at_five)


def test_mb_mlp_matches_documented_equation() -> None:
    torch.manual_seed(202)
    config = tiny_config("mb_mlp")
    mixer = MBMLP(config)
    x = torch.randn(2, 3, config.d_model)

    actual, cache = mixer(x, cache_position=4)
    k, g, v = F.linear(x, mixer.c_proj.weight, mixer.c_proj.bias).chunk(3, dim=-1)
    expected = F.linear(
        F.gelu(k * g) * v,
        mixer.proj_out.weight,
        mixer.proj_out.bias,
    )

    torch.testing.assert_close(actual, expected)
    assert cache is None


def test_swiglu_mixer_matches_documented_equation() -> None:
    torch.manual_seed(203)
    config = tiny_config("swiglu_mixer")
    mixer = SwiGLUMixer(config)
    x = torch.randn(2, 3, config.d_model)

    actual, cache = mixer(x, cache_position=4)
    g, v = F.linear(x, mixer.c_proj.weight, mixer.c_proj.bias).chunk(2, dim=-1)
    expected = F.linear(
        F.silu(g) * v,
        mixer.proj_out.weight,
        mixer.proj_out.bias,
    )

    torch.testing.assert_close(actual, expected)
    assert cache is None


@pytest.mark.parametrize(
    ("layer_type", "factory"),
    [
        ("triglu_no_rope", lambda config: TriGLU(config, rope=None)),
        ("mb_mlp", MBMLP),
        ("swiglu_mixer", SwiGLUMixer),
    ],
)
def test_ablation_mixers_are_token_local(
    layer_type: str,
    factory: Callable[[ModelConfig], torch.nn.Module],
) -> None:
    torch.manual_seed(204)
    config = tiny_config(layer_type)
    mixer = factory(config).eval()
    x = torch.randn(1, 5, config.d_model)
    changed = x.clone()
    changed[:, 2] += 100.0

    before, _ = mixer(x)
    after, _ = mixer(changed)

    torch.testing.assert_close(before[:, :2], after[:, :2])
    torch.testing.assert_close(before[:, 3:], after[:, 3:])
    assert not torch.equal(before[:, 2], after[:, 2])


@pytest.mark.parametrize(
    ("layer_type", "factory"),
    [
        ("triglu_no_rope", lambda config: TriGLU(config, rope=None)),
        ("mb_mlp", MBMLP),
        ("swiglu_mixer", SwiGLUMixer),
    ],
)
def test_ablation_mixers_chunk_exactly_and_store_no_kv(
    layer_type: str,
    factory: Callable[[ModelConfig], torch.nn.Module],
) -> None:
    torch.manual_seed(205)
    config = tiny_config(layer_type, bias=False)
    mixer = factory(config).eval()
    x = torch.randn(2, 6, config.d_model)

    full, full_cache = mixer(x, cache_position=3, use_cache=True)
    first, first_cache = mixer(x[:, :2], cache_position=3, use_cache=True)
    second, second_cache = mixer(
        x[:, 2:],
        cache=first_cache,
        cache_position=5,
        use_cache=True,
    )

    torch.testing.assert_close(full, torch.cat((first, second), dim=1))
    assert first_cache == (None, None, 5)
    assert second_cache == (None, None, 9)
    assert full_cache == (None, None, 9)


def test_mb_mlp_has_exact_attention_projection_parameter_count() -> None:
    for bias in (False, True):
        config = tiny_config("mb_mlp", bias=bias)
        mixer = MBMLP(config)
        expected = 4 * config.d_model**2 + (4 * config.d_model if bias else 0)
        assert sum(parameter.numel() for parameter in mixer.parameters()) == expected


def test_512_wide_swiglu_control_is_nearest_integer_parameter_match() -> None:
    config = ModelConfig(
        vocab_size=32,
        n_layers=1,
        d_model=512,
        n_heads=8,
        ffn_hidden_size=1376,
        context_length=16,
        bias=False,
        swiglu_mixer_hidden_size=683,
        layer_types=["swiglu_mixer"],
    )
    mixer = SwiGLUMixer(config)
    actual = sum(parameter.numel() for parameter in mixer.parameters())
    attention_or_triglu = 4 * config.d_model**2

    assert actual == 3 * config.d_model * 683
    assert actual - attention_or_triglu == 512
