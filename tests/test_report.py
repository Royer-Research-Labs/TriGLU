from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from triglu.report import (
    _effective_ffn_hidden_sizes,
    _experiment_family,
    _pareto_front,
    generate_report,
)


def _write_run(
    root: Path,
    plan: str,
    layers: list[str],
    loss: float,
    speed: float,
    *,
    complete: bool = True,
    seed: int = 0,
    ffn_type: str = "swiglu",
    ffn_hidden_size: int = 16,
    ffn_hidden_sizes: list[int] | None = None,
    residual_init_depth: int | None = None,
    parameter_count: int = 123,
) -> None:
    run = root / "suite" / plan
    run.mkdir(parents=True)
    config = {
        "model": {
            "n_layers": len(layers),
            "layer_types": layers,
            "ffn_type": ffn_type,
            "ffn_hidden_size": ffn_hidden_size,
            "context_length": 8,
        },
        "training": {"batch_size": 1, "gradient_accumulation_steps": 1, "seed": seed},
        "runtime": {"parameter_count": parameter_count},
    }
    if ffn_hidden_sizes is not None:
        config["model"]["ffn_hidden_sizes"] = ffn_hidden_sizes
    if residual_init_depth is not None:
        config["model"]["residual_init_depth"] = residual_init_depth
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    events = [
        {"event": "train", "step": 1, "tokens_seen": 8, "tokens_per_second": speed},
        {"event": "evaluation", "step": 1, "tokens_seen": 8, "loss": loss, "perplexity": 2.0, "accuracy": 0.5},
    ]
    if complete:
        events.append({"event": "complete", "step": 1, "tokens_seen": 8, "best_val_loss": loss})
    (run / "metrics.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


@pytest.mark.parametrize(
    "schedule",
    ([16, 20.5], [16, True], [16, 0]),
)
def test_report_rejects_malformed_explicit_ffn_width_metadata(
    schedule: list[object],
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _effective_ffn_hidden_sizes(
            {
                "n_layers": 2,
                "layer_types": ["attention", "attention"],
                "ffn_hidden_size": 16,
                "ffn_hidden_sizes": schedule,
            }
        )


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
                    "layer_types": (
                        ["attention", "attention"]
                        if plan == "2a0t"
                        else ["attention", "triglu"]
                    ),
                    "ffn_type": "swiglu",
                    "ffn_hidden_size": 16,
                    "parameters": 123,
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
    assert {
        row["benchmark_source"] for row in report["context_benchmarks"]
    } == {
        "configs/suite/2a0t.yaml",
        "configs/suite/1a1t.yaml",
    }


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


def test_experiment_family_classification_precedence() -> None:
    def family(**overrides) -> str:
        values = {
            "layer_types": ["attention"] * 4,
            "ffn_type": "swiglu",
            "ffn_hidden_sizes": [16] * 4,
            "residual_init_depth": 4,
            "block_mode": "sequential",
        }
        values.update(overrides)
        return _experiment_family(**values)

    assert family() == "primary"
    assert family(layer_types=["attention", "triglu", "attention", "triglu"]) == "primary"
    assert (
        family(layer_types=["attention", "triglu_no_rope", "attention", "attention"])
        == "attention_slot_control"
    )
    assert family(ffn_type="triglu_no_rope") == "ffn_form_control"
    # The combination run (replacement mixers AND a triple-product FFN) is
    # grouped with the FFN-form controls: topology checks win, then FFN form.
    assert (
        family(
            layer_types=["attention", "triglu_no_rope", "attention", "triglu_no_rope"],
            ffn_type="triglu_no_rope",
        )
        == "ffn_form_control"
    )
    # Any structural relaxation dominates: parallel block, ffn_only slots,
    # nonuniform widths, or a foreign residual-init depth.
    assert family(block_mode="parallel") == "residual_topology_control"
    assert (
        family(layer_types=["attention", "ffn_only", "attention", "attention"])
        == "residual_topology_control"
    )
    assert family(ffn_hidden_sizes=[16, 16, 32, 16]) == "residual_topology_control"
    assert family(residual_init_depth=20) == "residual_topology_control"


def test_pareto_front_maximizes_both_axes() -> None:
    points = [
        ("slow_good", 1.0, 10.0),
        ("fast_bad", 10.0, 1.0),
        ("dominated", 0.5, 9.0),
        ("dominated_mid", 5.0, 5.0),
        ("duplicate_a", 7.0, 7.0),
        ("duplicate_b", 7.0, 7.0),
    ]
    front = _pareto_front(points)
    # Exact duplicates are mutually non-dominating and both stay on the front;
    # (5, 5) is strictly dominated by them.
    assert front == {"slow_good", "fast_bad", "duplicate_a", "duplicate_b"}


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
        expected_family = (
            "primary" if layer_type == "triglu" else "attention_slot_control"
        )
        assert row["experiment_family"] == expected_family
        assert row["attention_layers"] == 1
        assert row["token_local_layers"] == 1
        assert row["replacement_mixers"] == f"{layer_type}:1"
        assert row[f"{layer_type}_layers"] == 1

    summary_header = (
        results / "suite-report" / "summary.csv"
    ).read_text(encoding="utf-8").splitlines()[0]
    assert "token_local_layers" in summary_header
    assert "replacement_mixers" in summary_header
    assert "ffn_type" in summary_header
    assert "ffn_hidden_size" in summary_header


def test_report_separates_ffn_only_blocks_and_nonuniform_widths(tmp_path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(
        runs,
        "2a0t",
        ["attention", "attention"],
        2.0,
        100.0,
    )
    _write_run(
        runs,
        "1a1f_single_residual",
        ["attention", "ffn_only"],
        2.1,
        110.0,
        ffn_hidden_sizes=[16, 25],
        residual_init_depth=4,
        parameter_count=124,
    )
    _write_run(
        runs,
        "grouped_width_residual_stress",
        ["attention"],
        1.9,
        120.0,
        ffn_hidden_sizes=[57],
        residual_init_depth=4,
        parameter_count=125,
    )

    report = generate_report(runs_root=runs, results_root=results, suite="suite")
    rows = {row["architecture"]: row for row in report["architectures"]}

    assert report["schema_version"] == 6
    assert report["baseline_architecture"] == "2a0t"
    assert rows["2a0t"]["experiment_family"] == "primary"
    ffn_only = rows["1a1f_single_residual"]
    assert ffn_only["experiment_family"] == "residual_topology_control"
    assert ffn_only["n_layers"] == 2
    assert ffn_only["attention_layers"] == 1
    assert ffn_only["token_local_layers"] == 0
    assert ffn_only["ffn_only_layers"] == 1
    assert ffn_only["residual_updates_per_forward"] == 3
    assert ffn_only["replacement_mixers"] == "none"
    assert ffn_only["structural_controls"] == "ffn_only:1"
    assert ffn_only["ffn_hidden_sizes"] == [16, 25]
    assert ffn_only["ffn_width_schedule"] == "16,25"
    assert ffn_only["ffn_total_hidden_size"] == 41
    assert ffn_only["residual_init_depth"] == 4

    grouped = rows["grouped_width_residual_stress"]
    assert grouped["n_layers"] == 1
    assert grouped["experiment_family"] == "residual_topology_control"
    assert grouped["attention_layers"] == 1
    assert grouped["ffn_only_layers"] == 0
    assert grouped["residual_updates_per_forward"] == 2
    assert grouped["structural_controls"] == "grouped_width_or_depth"
    assert grouped["ffn_hidden_sizes"] == [57]
    assert grouped["residual_init_depth"] == 4

    header = (
        results / "suite-report" / "summary_by_architecture.csv"
    ).read_text(encoding="utf-8").splitlines()[0]
    for field in (
        "n_layers",
        "experiment_family",
        "ffn_only_layers",
        "ffn_width_schedule",
        "ffn_total_hidden_size",
        "residual_init_depth",
    ):
        assert field in header
    generated_readme = (
        results / "suite-report" / "README.md"
    ).read_text(encoding="utf-8")
    assert "Model range: 1–2 physical blocks" in generated_readme
    assert "### Residual-topology controls" in generated_readme


def test_report_rejects_replica_parameter_or_budget_drift(tmp_path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(
        runs,
        "2a0t",
        ["attention", "attention"],
        2.0,
        100.0,
        seed=0,
        parameter_count=123,
    )
    _write_run(
        runs,
        "2a0t_seed9",
        ["attention", "attention"],
        2.1,
        101.0,
        seed=9,
        parameter_count=124,
    )

    with pytest.raises(ValueError, match="inconsistent parameter_count"):
        generate_report(runs_root=runs, results_root=results, suite="suite")


def test_ffn_control_is_reported_separately_without_replacing_baseline(
    tmp_path,
) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(
        runs,
        "2a0t",
        ["attention", "attention"],
        2.0,
        100.0,
        ffn_type="swiglu",
        ffn_hidden_size=16,
    )
    # Give the experimental FFN a better loss to exercise the baseline guard.
    _write_run(
        runs,
        "2a0t_triglu_no_rope_ffn",
        ["attention", "attention"],
        1.9,
        105.0,
        ffn_type="triglu_no_rope",
        ffn_hidden_size=12,
    )

    report = generate_report(runs_root=runs, results_root=results, suite="suite")

    assert report["baseline_architecture"] == "2a0t"
    rows = {row["architecture"]: row for row in report["architectures"]}
    assert rows["2a0t"]["ffn_type"] == "swiglu"
    assert rows["2a0t"]["experiment_family"] == "primary"
    assert rows["2a0t"]["ffn_hidden_size"] == 16
    control = rows["2a0t_triglu_no_rope_ffn"]
    assert control["experiment_family"] == "ffn_form_control"
    assert control["ffn_type"] == "triglu_no_rope"
    assert control["ffn_hidden_size"] == 12
    assert control["loss_delta_vs_baseline_mean"] == pytest.approx(-0.1)

    generated = (
        results / "suite-report" / "summary_by_architecture.csv"
    ).read_text(encoding="utf-8")
    assert "ffn_type" in generated.splitlines()[0]
    assert "2a0t_triglu_no_rope_ffn" in generated


def test_context_benchmark_with_mislabeled_layer_plan_is_rejected(tmp_path) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(runs, "2a0t", ["attention", "attention"], 2.0, 100.0)
    artifact = {
        "event": "benchmark",
        "benchmark_label": "context-scaling",
        "source": "configs/suite/2a0t.yaml",
        "model": {
            "parameters": 123,
            "configured_context_length": 8,
            "benchmark_context_length": 8,
            "layer_types": ["attention", "triglu"],
            "ffn_type": "swiglu",
            "ffn_hidden_size": 16,
        },
        "settings": {
            "training_sequence_length": 8,
            "prompt_length": 6,
            "decode_tokens": 2,
            "warmup": 1,
            "iterations": 2,
        },
        "training": {},
        "prefill": {},
        "cached_decode": {},
    }
    (results / "mislabeled.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="layer plan"):
        generate_report(runs_root=runs, results_root=results, suite="suite")


def test_context_benchmark_with_wrong_structural_metadata_is_rejected(
    tmp_path,
) -> None:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    results.mkdir()
    _write_run(
        runs,
        "1a1f_single_residual",
        ["attention", "ffn_only"],
        2.0,
        100.0,
        ffn_hidden_sizes=[16, 25],
        residual_init_depth=4,
    )
    artifact = {
        "event": "benchmark",
        "benchmark_label": "context-scaling",
        "source": "configs/suite/1a1f_single_residual.yaml",
        "model": {
            "parameters": 123,
            "n_layers": 2,
            "configured_context_length": 8,
            "benchmark_context_length": 8,
            "layer_types": ["attention", "ffn_only"],
            "ffn_type": "swiglu",
            "ffn_hidden_size": 16,
            "ffn_hidden_sizes": [16, 26],
            "residual_init_depth": 4,
        },
        "settings": {
            "training_sequence_length": 8,
            "prompt_length": 6,
            "decode_tokens": 2,
            "warmup": 1,
            "iterations": 2,
        },
        "training": {},
        "prefill": {},
        "cached_decode": {},
    }
    (results / "wrong-schedule.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="width schedule"):
        generate_report(runs_root=runs, results_root=results, suite="suite")

    artifact["model"]["ffn_hidden_sizes"] = [16, 25]
    artifact["model"]["ffn_total_hidden_size"] = 42
    (results / "wrong-schedule.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="declares FFN total"):
        generate_report(runs_root=runs, results_root=results, suite="suite")

    artifact["model"]["ffn_total_hidden_size"] = 41
    artifact["model"]["residual_init_depth"] = 5
    (results / "wrong-schedule.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="residual init depth"):
        generate_report(runs_root=runs, results_root=results, suite="suite")
