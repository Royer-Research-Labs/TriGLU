from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from triglu import (
    CausalSelfAttention,
    DecoderLM,
    MBMLP,
    SwiGLUMixer,
    TriGLU,
    ModelConfig,
)


def tiny_config(layer_types: list[str] | None = None) -> ModelConfig:
    layer_types = layer_types or ["attention", "triglu"]
    return ModelConfig(
        vocab_size=64,
        n_layers=len(layer_types),
        d_model=16,
        n_heads=2,
        ffn_hidden_size=32,
        context_length=16,
        dropout=0.0,
        bias=False,
        swiglu_mixer_hidden_size=21 if "swiglu_mixer" in layer_types else None,
        layer_types=layer_types,
    )


def test_static_cache_rejects_decode_past_capacity() -> None:
    torch.manual_seed(13)
    config = tiny_config(["attention"])
    attention = CausalSelfAttention(config).eval()
    head_dim = config.d_model // config.n_heads
    shape = (1, config.n_heads, config.context_length, head_dim)
    static = (torch.zeros(shape), torch.zeros(shape), 0)

    with torch.inference_mode():
        x = torch.randn(1, config.context_length, config.d_model)
        _, cache = attention(x, cache=static, cache_position=0, use_cache=True)
        assert cache[2] == config.context_length
        # A full static cache must fail loudly, not silently grow past the
        # model's context by falling into the dynamic concatenation path.
        with pytest.raises(ValueError, match="capacity exceeded"):
            attention(
                torch.randn(1, 1, config.d_model),
                cache=cache,
                cache_position=config.context_length,
                use_cache=True,
            )


def test_model_uses_standard_rmsnorm_and_ties_embeddings() -> None:
    model = DecoderLM(tiny_config())
    assert isinstance(model.blocks[0].norm_1, nn.RMSNorm)
    assert isinstance(model.final_norm, nn.RMSNorm)
    assert model.lm_head.weight is model.token_embedding.weight


def test_explicit_layer_plan_selects_only_the_mixer() -> None:
    config = tiny_config(
        [
            "attention",
            "triglu",
            "triglu_no_rope",
            "mb_mlp",
            "swiglu_mixer",
        ]
    )
    model = DecoderLM(config)
    assert isinstance(model.blocks[0].mixer, CausalSelfAttention)
    assert isinstance(model.blocks[1].mixer, TriGLU)
    assert isinstance(model.blocks[2].mixer, TriGLU)
    assert model.blocks[2].mixer.rope is None
    assert isinstance(model.blocks[3].mixer, MBMLP)
    assert isinstance(model.blocks[4].mixer, SwiGLUMixer)
    for block in model.blocks[1:]:
        assert type(model.blocks[0].norm_1) is type(block.norm_1)
        assert type(model.blocks[0].ffn) is type(block.ffn)


def test_attention_and_triglu_plans_have_parameter_parity() -> None:
    attention_model = DecoderLM(tiny_config(["attention"]))
    triglu_model = DecoderLM(tiny_config(["triglu"]))
    assert attention_model.num_parameters() == triglu_model.num_parameters()


def test_attention_and_triglu_plans_start_from_identical_parameter_tensors() -> None:
    torch.manual_seed(101)
    attention_model = DecoderLM(tiny_config(["attention"]))
    torch.manual_seed(101)
    triglu_model = DecoderLM(tiny_config(["triglu"]))

    attention_parameters = dict(attention_model.named_parameters())
    triglu_parameters = dict(triglu_model.named_parameters())
    assert attention_parameters.keys() == triglu_parameters.keys()
    for name, value in attention_parameters.items():
        torch.testing.assert_close(value, triglu_parameters[name], rtol=0, atol=0)


@pytest.mark.parametrize("layer_type", ["triglu_no_rope", "mb_mlp"])
def test_exact_shape_controls_start_from_attention_parameter_tensors(
    layer_type: str,
) -> None:
    torch.manual_seed(102)
    attention_model = DecoderLM(tiny_config(["attention"]))
    torch.manual_seed(102)
    control_model = DecoderLM(tiny_config([layer_type]))

    attention_parameters = dict(attention_model.named_parameters())
    control_parameters = dict(control_model.named_parameters())
    assert attention_parameters.keys() == control_parameters.keys()
    for name, value in attention_parameters.items():
        torch.testing.assert_close(value, control_parameters[name], rtol=0, atol=0)


def test_model_is_causal() -> None:
    torch.manual_seed(11)
    model = DecoderLM(tiny_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % model.config.vocab_size

    original = model(tokens).logits
    modified = model(changed).logits

    torch.testing.assert_close(original[:, :5], modified[:, :5], rtol=0, atol=1e-6)


def test_cached_tokenwise_logits_match_full_forward() -> None:
    torch.manual_seed(12)
    model = DecoderLM(tiny_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    full = model(tokens).logits

    caches = None
    pieces: list[torch.Tensor] = []
    for position in range(tokens.size(1)):
        output = model(tokens[:, position : position + 1], caches=caches, use_cache=True)
        pieces.append(output.logits)
        caches = output.caches

    torch.testing.assert_close(full, torch.cat(pieces, dim=1), rtol=1e-4, atol=1e-5)
    assert caches is not None
    assert caches[0][0] is not None and caches[0][1] is not None
    assert caches[1] == (None, None, tokens.size(1))


@pytest.mark.parametrize(
    "layer_type",
    ["triglu_no_rope", "mb_mlp", "swiglu_mixer"],
)
def test_ablation_model_cached_logits_match_full_forward(layer_type: str) -> None:
    torch.manual_seed(123)
    model = DecoderLM(tiny_config([layer_type])).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    full = model(tokens).logits

    caches = None
    pieces: list[torch.Tensor] = []
    for _position in range(tokens.size(1)):
        position = len(pieces)
        output = model(
            tokens[:, position : position + 1],
            caches=caches,
            use_cache=True,
        )
        pieces.append(output.logits)
        caches = output.caches

    torch.testing.assert_close(full, torch.cat(pieces, dim=1), rtol=1e-4, atol=1e-5)
    assert caches == ((None, None, tokens.size(1)),)


def test_static_cached_logits_match_and_storage_stays_preallocated() -> None:
    torch.manual_seed(121)
    model = DecoderLM(tiny_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))

    with torch.inference_mode():
        full = model(tokens).logits
        caches = model.allocate_static_cache(
            tokens.size(0), device=tokens.device, dtype=model.token_embedding.weight.dtype
        )
        key, value, position = caches[0]
        assert key is not None and value is not None and position == 0
        assert key.shape == (
            tokens.size(0),
            model.config.n_heads,
            model.config.context_length,
            model.config.head_dim,
        )
        key_pointer = key.untyped_storage().data_ptr()
        value_pointer = value.untyped_storage().data_ptr()

        pieces: list[torch.Tensor] = []
        for token_position in range(tokens.size(1)):
            output = model(
                tokens[:, token_position : token_position + 1],
                caches=caches,
                use_cache=True,
            )
            pieces.append(output.logits)
            assert output.caches is not None
            caches = output.caches
            next_key, next_value, next_position = caches[0]
            assert next_key is not None and next_value is not None
            assert next_key.untyped_storage().data_ptr() == key_pointer
            assert next_value.untyped_storage().data_ptr() == value_pointer
            assert next_key.size(-2) == model.config.context_length
            assert next_position == token_position + 1
            assert caches[1] == (None, None, token_position + 1)

    torch.testing.assert_close(full, torch.cat(pieces, dim=1), rtol=1e-4, atol=1e-5)


def test_static_cache_allocates_kv_only_for_attention() -> None:
    model = DecoderLM(
        tiny_config(
            [
                "attention",
                "triglu",
                "triglu_no_rope",
                "mb_mlp",
                "swiglu_mixer",
            ]
        )
    )
    caches = model.allocate_static_cache(batch_size=2)

    assert caches[0][0] is not None and caches[0][1] is not None
    assert caches[0][2] == 0
    assert caches[1:] == (
        (None, None, 0),
        (None, None, 0),
        (None, None, 0),
        (None, None, 0),
    )


def test_cached_multitoken_suffix_uses_absolute_causal_mask() -> None:
    torch.manual_seed(120)
    model = DecoderLM(tiny_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    full = model(tokens).logits

    prefix = model(tokens[:, :3], use_cache=True)
    assert prefix.caches is not None
    suffix = model(tokens[:, 3:], caches=prefix.caches, use_cache=True)
    combined = torch.cat((prefix.logits, suffix.logits), dim=1)
    torch.testing.assert_close(full, combined, rtol=1e-4, atol=1e-5)


def test_labels_are_already_shifted_and_loss_is_finite() -> None:
    model = DecoderLM(tiny_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    labels = torch.roll(tokens, shifts=-1, dims=1)
    output = model(tokens, labels=labels)
    assert output.loss is not None and torch.isfinite(output.loss)
    manual = torch.nn.functional.cross_entropy(
        output.logits.flatten(0, 1), labels.flatten()
    )
    torch.testing.assert_close(output.loss, manual)


def test_logits_to_keep_matches_full_sequence_tail() -> None:
    torch.manual_seed(122)
    model = DecoderLM(tiny_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))

    full = model(tokens).logits
    last = model(tokens, logits_to_keep=1).logits

    assert last.shape == (tokens.size(0), 1, model.config.vocab_size)
    torch.testing.assert_close(last, full[:, -1:])


def test_residual_projection_initialization_is_depth_scaled() -> None:
    torch.manual_seed(13)
    config = ModelConfig(
        vocab_size=64,
        n_layers=4,
        d_model=128,
        n_heads=2,
        ffn_hidden_size=256,
        context_length=8,
        init_std=0.02,
        layer_types=["attention", "triglu", "attention", "triglu"],
    )
    model = DecoderLM(config)
    expected = config.init_std / math.sqrt(2 * config.n_layers)
    for block in model.blocks:
        mixer_std = float(block.mixer.proj_out.weight.detach().std())
        ffn_std = float(block.ffn.down_proj.weight.detach().std())
        assert mixer_std == pytest.approx(expected, rel=0.08)
        assert ffn_std == pytest.approx(expected, rel=0.08)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"layer_types": ["attention"]}, "one entry per layer"),
        ({"layer_types": ["attention", "convolution"]}, "unsupported layer"),
        (
            {"layer_types": ["attention", "swiglu_mixer"]},
            "swiglu_mixer_hidden_size",
        ),
        (
            {"swiglu_mixer_hidden_size": 21},
            "must be null",
        ),
        ({"d_model": 15}, "divisible"),
        ({"tie_embeddings": False}, "must remain true"),
    ],
)
def test_invalid_model_configs_fail_loudly(overrides: dict[str, object], message: str) -> None:
    values = tiny_config().to_dict()
    values.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        ModelConfig.from_dict(values)


def test_config_round_trip_is_exact() -> None:
    config = tiny_config()
    assert ModelConfig.from_dict(config.to_dict()) == config
