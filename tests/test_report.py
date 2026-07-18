from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from triglu.report import generate_report


def _write_run(
    root: Path,
    plan: str,
    layers: list[str],
    loss: float,
    speed: float,
    *,
    complete: bool = True,
    seed: int = 0,
) -> None:
    run = root / "suite" / plan
    run.mkdir(parents=True)
    config = {
        "model": {"n_layers": len(layers), "layer_types": layers, "context_length": 8},
        "training": {"batch_size": 1, "gradient_accumulation_steps": 1, "seed": seed},
        "runtime": {"parameter_count": 123},
    }
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    events = [
        {"event": "train", "step": 1, "tokens_seen": 8, "tokens_per_second": speed},
        {"event": "evaluation", "step": 1, "tokens_seen": 8, "loss": loss, "perplexity": 2.0, "accuracy": 0.5},
    ]
    if complete:
        events.append({"event": "complete", "step": 1, "tokens_seen": 8, "best_val_loss": loss})
    (run / "metrics.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_generate_report_writes_tables_and_dependency_free_charts(tmp_path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(runs, "2a0t", ["attention", "attention"], 2.0, 100.0)
    _write_run(runs, "1a1t", ["attention", "triglu"], 2.1, 120.0)
    _write_run(runs, "1a1t_seed9", ["attention", "triglu"], 9.0, 1.0, complete=False, seed=9)
    analysis = {
        "event": "rank_analysis",
        "checkpoint": str(runs / "suite" / "1a1t" / "latest.pt"),
        "rank_metrics": {
            "layer_stages": [
                {"layer": 0, "layer_type": "attention", "stage": "block_output", "entropy_effective_rank": 3.0, "participation_ratio": 2.5, "stable_rank": 2.0}
            ],
            "stage_transitions": [
                {"layer": 0, "layer_type": "attention", "block_input_entropy_effective_rank": 2.0, "post_mixer_entropy_effective_rank": 2.5, "block_output_entropy_effective_rank": 3.0, "mixer_rank_delta": 0.5, "ffn_rank_delta": 0.5, "block_rank_delta": 1.0, "mixer_rank_ratio": 1.25, "ffn_rank_ratio": 1.2}
            ],
            "residual_updates": [
                {"layer": 0, "layer_type": "attention", "update_to_residual_rms_ratio": 0.2, "mean_token_residual_update_cosine": 0.1, "mean_token_orthogonal_fraction": 0.99}
            ],
            "ffn_residual_updates": [
                {"layer": 0, "layer_type": "attention", "update_to_residual_rms_ratio": 0.3, "mean_token_residual_update_cosine": 0.2, "mean_token_orthogonal_fraction": 0.96}
            ],
        },
        "layer_sensitivity": {
            "mixer_ablations": [{"layer": 0, "layer_type": "attention", "loss_delta": 0.2, "accuracy_delta": -0.1}],
            "ffn_ablations": [],
        },
    }
    (results / "rank.json").write_text(json.dumps(analysis), encoding="utf-8")
    for plan in ("2a0t", "1a1t"):
        for context in (8, 16):
            benchmark = {
                "event": "benchmark",
                "benchmark_label": (
                    "context-scaling-validation" if context == 16 else "context-scaling"
                ),
                "source": f"configs/suite/{plan}.yaml",
                "model": {
                    "configured_context_length": 8,
                    "benchmark_context_length": context,
                },
                "settings": {
                    "training_sequence_length": context,
                    "prompt_length": context - 2,
                    "decode_tokens": 2,
                    "warmup": 1,
                    "iterations": 2,
                },
                "training": {
                    "median_tokens_per_second": 1000.0,
                    "peak_memory_allocated_bytes": 1024**3,
                },
                "prefill": {
                    "median_tokens_per_second": 2000.0,
                    "peak_memory_allocated_bytes": 512 * 1024**2,
                },
                "cached_decode": {
                    "median_tokens_per_second": 100.0,
                    "peak_memory_allocated_bytes": 256 * 1024**2,
                    "cache_capacity_bytes": context * 1024,
                    "final_cache_bytes": context * 1024,
                },
            }
            (results / f"context-{context}-{plan}.json").write_text(
                json.dumps(benchmark), encoding="utf-8"
            )

    report = generate_report(runs_root=runs, results_root=results, suite="suite")
    output = results / "suite-report"
    assert report["baseline_architecture"] == "2a0t"
    assert len(report["runs"]) == 2
    assert [run["plan"] for run in report["incomplete_runs_excluded"]] == ["1a1t_seed9"]
    assert (output / "summary.csv").exists()
    assert (output / "summary_by_architecture.csv").exists()
    assert (output / "incomplete_runs.csv").exists()
    assert (output / "training_curves.csv").exists()
    assert (output / "layer_diagnostics.csv").exists()
    assert (output / "layer_transitions.csv").exists()
    assert (output / "layer_sensitivity.csv").exists()
    assert (output / "context_benchmarks.csv").exists()
    assert "<svg" in (output / "quality_vs_throughput.svg").read_text(encoding="utf-8")
    assert "<svg" in (output / "context_prefill_throughput.svg").read_text(encoding="utf-8")
    assert "<svg" in (output / "context_kv_cache.svg").read_text(encoding="utf-8")
    assert "final 20%" in (output / "validation_loss_curves_zoomed.svg").read_text(encoding="utf-8")
    assert "<svg" in (output / "rank_delta-1a1t.svg").read_text(encoding="utf-8")
    assert "1a1t" in (output / "README.md").read_text(encoding="utf-8")
    assert len(report["context_benchmarks"]) == 4


def _write_minimal_analysis(path: Path, checkpoint: Path) -> None:
    analysis = {
        "event": "rank_analysis",
        "checkpoint": str(checkpoint),
        "rank_metrics": {
            "layer_stages": [
                {
                    "layer": 0,
                    "layer_type": "attention",
                    "stage": "block_output",
                    "entropy_effective_rank": 3.0,
                    "participation_ratio": 2.5,
                    "stable_rank": 2.0,
                }
            ],
            "stage_transitions": [
                {
                    "layer": 0,
                    "layer_type": "attention",
                    "block_input_entropy_effective_rank": 2.0,
                    "post_mixer_entropy_effective_rank": 2.5,
                    "block_output_entropy_effective_rank": 3.0,
                    "mixer_rank_delta": 0.5,
                    "ffn_rank_delta": 0.5,
                    "block_rank_delta": 1.0,
                    "mixer_rank_ratio": 1.25,
                    "ffn_rank_ratio": 1.2,
                }
            ],
            "residual_updates": [],
            "ffn_residual_updates": [],
        },
    }
    path.write_text(json.dumps(analysis), encoding="utf-8")


def test_duplicate_architecture_seed_runs_are_rejected(tmp_path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(runs, "2a0t", ["attention", "attention"], 2.0, 100.0, seed=0)
    _write_run(runs, "2a0t_seed0", ["attention", "attention"], 2.5, 90.0, seed=0)

    with pytest.raises(ValueError, match="both report architecture"):
        generate_report(runs_root=runs, results_root=results, suite="suite")


def test_regeneration_removes_stale_derived_outputs(tmp_path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(runs, "2a0t", ["attention", "attention"], 2.0, 100.0)
    _write_run(runs, "1a1t", ["attention", "triglu"], 2.1, 120.0)
    analysis_path = results / "rank.json"
    _write_minimal_analysis(analysis_path, runs / "suite" / "1a1t" / "latest.pt")

    output = results / "suite-report"
    generate_report(runs_root=runs, results_root=results, suite="suite")
    assert (output / "layer_diagnostics.csv").exists()
    assert (output / "rank_delta-1a1t.svg").exists()

    analysis_path.unlink()
    generate_report(runs_root=runs, results_root=results, suite="suite")
    assert not (output / "layer_diagnostics.csv").exists()
    assert not (output / "layer_transitions.csv").exists()
    assert not (output / "rank_delta-1a1t.svg").exists()
    assert not (output / "layer_stages-1a1t.svg").exists()
    assert (output / "summary.csv").exists()


def test_analyses_of_incomplete_runs_are_excluded(tmp_path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(runs, "2a0t", ["attention", "attention"], 2.0, 100.0)
    _write_run(runs, "1a1t", ["attention", "triglu"], 9.0, 1.0, complete=False)
    _write_minimal_analysis(
        results / "rank.json", runs / "suite" / "1a1t" / "latest.pt"
    )

    report = generate_report(runs_root=runs, results_root=results, suite="suite")
    output = results / "suite-report"
    assert report["rank_analyses"] == {}
    assert not (output / "layer_diagnostics.csv").exists()


def test_architecture_delta_uses_only_seed_matched_baselines(tmp_path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(runs, "2a0t", ["attention", "attention"], 2.0, 100.0, seed=0)
    _write_run(runs, "2a0t_seed9", ["attention", "attention"], 10.0, 100.0, seed=9)
    _write_run(runs, "1a1t", ["attention", "triglu"], 2.1, 120.0, seed=0)

    report = generate_report(runs_root=runs, results_root=results, suite="suite")
    hybrid = next(
        row for row in report["architectures"] if row["architecture"] == "1a1t"
    )
    assert hybrid["matched_baseline_runs"] == 1
    assert hybrid["matched_baseline_seeds"] == "0"
    assert hybrid["loss_delta_vs_baseline_mean"] == pytest.approx(0.1)


def test_report_labels_every_token_local_ablation_layer(tmp_path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(runs, "2a0t", ["attention", "attention"], 2.0, 100.0)
    variants = ("triglu", "triglu_no_rope", "mb_mlp", "swiglu_mixer")
    for index, layer_type in enumerate(variants, 1):
        _write_run(
            runs,
            f"1a1_{layer_type}",
            ["attention", layer_type],
            2.0 + index / 10,
            100.0 + index,
        )

    report = generate_report(runs_root=runs, results_root=results, suite="suite")
    by_architecture = {
        row["architecture"]: row for row in report["architectures"]
    }
    for layer_type in variants:
        row = by_architecture[f"1a1_{layer_type}"]
        assert row["attention_layers"] == 1
        assert row["token_local_layers"] == 1
        assert row["replacement_mixers"] == f"{layer_type}:1"
        assert row[f"{layer_type}_layers"] == 1

    summary_header = (
        results / "suite-report" / "summary.csv"
    ).read_text(encoding="utf-8").splitlines()[0]
    assert "token_local_layers" in summary_header
    assert "replacement_mixers" in summary_header
