from __future__ import annotations

import json
import math

import pytest
import torch

from triglu.analyze_rank import _CovarianceAccumulator, run_rank_analysis
from triglu.config import ModelConfig
from triglu.model import DecoderLM
from triglu.runtime import ConfigurationError


def test_covariance_effective_rank_detects_one_dimensional_data() -> None:
    accumulator = _CovarianceAccumulator(feature_dimension=3)
    accumulator.update(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        ),
        maximum_rows=3,
    )
    result = accumulator.finalize(tolerance=1e-6)
    assert result["numerical_rank"] == 1
    assert result["stable_rank"] == pytest.approx(1.0)
    assert result["participation_ratio"] == pytest.approx(1.0)
    assert result["entropy_effective_rank"] == pytest.approx(1.0)


def test_rank_analysis_rejects_parallel_block_checkpoints(tmp_path) -> None:
    # The stage hooks assume the sequential two-norm block; a parallel-block
    # checkpoint must be rejected loudly, not crash on the absent second norm.
    config = ModelConfig(
        vocab_size=16,
        n_layers=2,
        d_model=8,
        n_heads=2,
        ffn_hidden_size=16,
        context_length=8,
        bias=False,
        block_mode="parallel",
        layer_types=["attention", "triglu_no_rope"],
    )
    model = DecoderLM(config)
    checkpoint_path = tmp_path / "parallel.pt"
    torch.save(
        {
            "format_version": 1,
            "step": 1,
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "resolved_config": {
                "data": {
                    "synthetic": True,
                    "vocab_size": 16,
                    "val_tokens": 64,
                    "seed": 7,
                    "pattern_length": 8,
                }
            },
        },
        checkpoint_path,
    )

    with pytest.raises(ConfigurationError, match="sequential two-norm"):
        run_rank_analysis(
            checkpoint_path,
            synthetic=True,
            synthetic_tokens=64,
            batch_size=1,
            sequence_length=8,
            rank_batches=1,
            rank_samples_per_batch=8,
            head_samples_per_batch=4,
            skip_sensitivity=True,
            device="cpu",
            dtype="float32",
            output=tmp_path / "unused.json",
        )


def test_rank_analysis_writes_complete_offline_diagnostics(tmp_path) -> None:
    config = ModelConfig(
        vocab_size=16,
        n_layers=2,
        d_model=8,
        n_heads=2,
        ffn_hidden_size=16,
        context_length=8,
        dropout=0.0,
        bias=False,
        layer_types=["attention", "triglu"],
    )
    model = DecoderLM(config)
    checkpoint_path = tmp_path / "tiny.pt"
    torch.save(
        {
            "format_version": 1,
            "step": 1,
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "resolved_config": {
                "data": {
                    "synthetic": True,
                    "vocab_size": 16,
                    "val_tokens": 64,
                    "seed": 7,
                    "pattern_length": 8,
                }
            },
        },
        checkpoint_path,
    )
    output_path = tmp_path / "rank-analysis.json"
    result = run_rank_analysis(
        checkpoint_path,
        synthetic=True,
        synthetic_tokens=64,
        batch_size=1,
        sequence_length=8,
        rank_batches=2,
        rank_samples_per_batch=8,
        head_samples_per_batch=4,
        sensitivity_batches=1,
        include_ffn_sensitivity=True,
        device="cpu",
        dtype="float32",
        output=output_path,
    )

    assert result["event"] == "rank_analysis"
    assert result["settings"]["rank_batches_observed"] == 2
    ranks = result["rank_metrics"]
    assert len(ranks["hidden_states"]) == 3
    assert len(ranks["mixer_updates"]) == 2
    assert len(ranks["residual_updates"]) == 2
    assert len(ranks["ffn_updates"]) == 2
    assert len(ranks["ffn_residual_updates"]) == 2
    assert len(ranks["layer_stages"]) == 14
    assert len(ranks["stage_transitions"]) == 2
    assert len(ranks["attention_head_contributions"]) == 1
    assert ranks["attention_head_contributions"][0]["layer"] == 0

    for group in ("hidden_states", "mixer_updates"):
        for layer in ranks[group]:
            assert 0 <= layer["entropy_effective_rank"] <= config.d_model
            assert all(math.isfinite(value) for value in layer["eigenvalues_descending"])

    sensitivity = result["layer_sensitivity"]
    assert sensitivity is not None
    assert sensitivity["baseline"]["tokens"] == 8
    assert len(sensitivity["mixer_ablations"]) == 2
    assert len(sensitivity["ffn_ablations"]) == 2
    assert len(sensitivity["attention_layer_ablations"]) == 1
    assert sensitivity["attention_layer_ablations"][0]["layer"] == 0

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["model"]["layer_types"] == ["attention", "triglu"]
    assert saved["model"]["ffn_type"] == "swiglu"
    assert saved["model"]["ffn_hidden_size"] == 16
    assert saved["model"]["ffn_hidden_sizes"] == [16, 16]
    assert saved["model"]["residual_init_depth"] == 2
    assert saved["rank_metrics"] == result["rank_metrics"]


def test_stage_transitions_separate_mixer_and_ffn_rank_changes(tmp_path) -> None:
    config = ModelConfig(
        vocab_size=16,
        n_layers=1,
        d_model=8,
        n_heads=2,
        ffn_hidden_size=16,
        context_length=8,
        layer_types=["triglu"],
    )
    model = DecoderLM(config)
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {
            "step": 0,
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "resolved_config": {"data": {"synthetic": True, "vocab_size": 16, "val_tokens": 32}},
        },
        checkpoint,
    )
    result = run_rank_analysis(
        checkpoint,
        synthetic=True,
        synthetic_tokens=32,
        sequence_length=8,
        rank_batches=1,
        rank_samples_per_batch=8,
        skip_sensitivity=True,
        device="cpu",
        dtype="float32",
    )
    stages = result["rank_metrics"]["layer_stages"]
    assert [stage["stage"] for stage in stages] == [
        "block_input",
        "mixer_norm_input",
        "mixer_update",
        "post_mixer_residual",
        "ffn_norm_input",
        "ffn_update",
        "block_output",
    ]
    transition = result["rank_metrics"]["stage_transitions"][0]
    assert transition["block_rank_delta"] == pytest.approx(
        transition["mixer_rank_delta"] + transition["ffn_rank_delta"]
    )


def test_rank_analysis_omits_nonexistent_ffn_only_mixer_stages(tmp_path) -> None:
    config = ModelConfig(
        vocab_size=16,
        n_layers=2,
        d_model=8,
        n_heads=2,
        ffn_hidden_size=16,
        ffn_hidden_sizes=[16, 25],
        residual_init_depth=4,
        context_length=8,
        layer_types=["attention", "ffn_only"],
    )
    model = DecoderLM(config)
    checkpoint = tmp_path / "ffn-only.pt"
    torch.save(
        {
            "step": 0,
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "resolved_config": {
                "data": {
                    "synthetic": True,
                    "vocab_size": 16,
                    "val_tokens": 32,
                }
            },
        },
        checkpoint,
    )

    result = run_rank_analysis(
        checkpoint,
        synthetic=True,
        synthetic_tokens=32,
        sequence_length=8,
        rank_batches=1,
        rank_samples_per_batch=8,
        sensitivity_batches=1,
        include_ffn_sensitivity=True,
        device="cpu",
        dtype="float32",
    )
    ranks = result["rank_metrics"]
    assert [row["layer"] for row in ranks["mixer_updates"]] == [0]
    assert [row["layer"] for row in ranks["residual_updates"]] == [0]
    assert [row["layer"] for row in ranks["ffn_updates"]] == [0, 1]
    assert [
        row["stage"] for row in ranks["layer_stages"] if row["layer"] == 1
    ] == [
        "block_input",
        "ffn_norm_input",
        "ffn_update",
        "block_output",
    ]
    transition = ranks["stage_transitions"][1]
    assert transition["post_mixer_entropy_effective_rank"] is None
    assert transition["mixer_rank_delta"] is None
    assert transition["mixer_rank_ratio"] is None
    assert transition["ffn_rank_delta"] == pytest.approx(
        transition["block_rank_delta"]
    )

    sensitivity = result["layer_sensitivity"]
    assert sensitivity is not None
    assert [row["layer"] for row in sensitivity["mixer_ablations"]] == [0]
    assert [row["layer"] for row in sensitivity["ffn_ablations"]] == [0, 1]
    assert result["schema_version"] == 3
    assert result["model"]["ffn_hidden_sizes"] == [16, 25]
    assert result["model"]["residual_init_depth"] == 4
