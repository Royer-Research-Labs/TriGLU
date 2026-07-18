"""Aggregate experiment artifacts into reproducible tables and SVG figures.

The reporter is deliberately read-only with respect to runs. It discovers completed
training logs, optional benchmark/evaluation JSON, and optional rank analyses, then
writes small derived artifacts suitable for review or publication.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

import yaml


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#4f46e5", "#be123c")
TOKEN_LOCAL_LAYER_TYPES = (
    "triglu",
    "triglu_no_rope",
    "mb_mlp",
    "swiglu_mixer",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _plan_from_source(value: str | Path) -> str:
    path = Path(value)
    if path.name == "latest.pt":
        return path.parent.name
    return path.stem


def _architecture_name(run_name: str) -> str:
    """Strip the replication suffix used by the bundled seed launcher."""

    return re.sub(r"_seed\d+$", "", run_name)


def _replacement_counts(layer_types: Sequence[str]) -> dict[str, Any]:
    counts = {
        layer_type: layer_types.count(layer_type)
        for layer_type in TOKEN_LOCAL_LAYER_TYPES
    }
    active = [
        f"{layer_type}:{count}"
        for layer_type, count in counts.items()
        if count
    ]
    return {
        "attention_layers": layer_types.count("attention"),
        "token_local_layers": sum(counts.values()),
        "replacement_mixers": ",".join(active) if active else "none",
        **{f"{layer_type}_layers": count for layer_type, count in counts.items()},
    }


_REPORT_OUTPUT_NAMES = (
    "README.md",
    "report.json",
    "summary.csv",
    "summary_by_architecture.csv",
    "summary_confirmatory.csv",
    "incomplete_runs.csv",
    "training_curves.csv",
    "layer_diagnostics.csv",
    "layer_transitions.csv",
    "layer_sensitivity.csv",
    "context_benchmarks.csv",
    "quality_vs_throughput.svg",
    "validation_loss_curves.svg",
    "validation_loss_curves_zoomed.svg",
    "layer_effective_rank.svg",
    "layer_sensitivity.svg",
    "context_training_throughput.svg",
    "context_prefill_throughput.svg",
    "context_decode_throughput.svg",
    "context_kv_cache.svg",
)
_REPORT_OUTPUT_PATTERNS = (
    "layer_stages-*.svg",
    "rank_delta-*.svg",
    "update_scale-*.svg",
)


def _clear_previous_outputs(output: Path) -> None:
    """Remove everything a previous generation may have written.

    The report directory is fully derived: a chart or table whose inputs
    disappeared must not survive a regeneration and silently misrepresent the
    current raw artifacts.
    """

    for name in _REPORT_OUTPUT_NAMES:
        (output / name).unlink(missing_ok=True)
    for pattern in _REPORT_OUTPUT_PATTERNS:
        for stale in output.glob(pattern):
            stale.unlink()


def _last_evaluation(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    evaluations = [row for row in rows if row.get("event") == "evaluation"]
    return max(evaluations, key=lambda row: (int(row.get("tokens_seen", 0)), int(row.get("step", 0)))) if evaluations else None


def _steady_throughput(rows: Sequence[Mapping[str, Any]]) -> float | None:
    values = [
        float(row["tokens_per_second"])
        for row in rows
        if row.get("event") == "train" and row.get("tokens_per_second") is not None
    ]
    if not values:
        return None
    # Ignore startup/warmup behavior while retaining enough samples for short smokes.
    tail = values[len(values) // 2 :]
    return statistics.median(tail)


def _load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = run_dir / "resolved_config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"expected a mapping in {config_path}")
    rows = _read_jsonl(metrics_path)
    model = config.get("model", {})
    training = config.get("training", {})
    runtime = config.get("runtime", {})
    environment_path = run_dir / "environment.json"
    environment = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_path.exists()
        else {}
    )
    provenance_path = run_dir / "data_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else config.get("data_provenance", {})
    )
    layer_types = list(model.get("layer_types", []))
    replacement_counts = _replacement_counts(layer_types)
    evaluation = _last_evaluation(rows)
    complete = next((row for row in reversed(rows) if row.get("event") == "complete"), None)
    summary: dict[str, Any] = {
        "plan": run_dir.name,
        "architecture": _architecture_name(run_dir.name),
        "seed": int(training.get("seed", 0)),
        "run_dir": str(run_dir),
        "complete": complete is not None,
        "step": int((evaluation or complete or {}).get("step", 0)),
        "tokens_seen": int((evaluation or complete or {}).get("tokens_seen", 0)),
        "n_layers": int(model.get("n_layers", len(layer_types))),
        **replacement_counts,
        "layer_types": layer_types,
        "parameter_count": runtime.get("parameter_count"),
        "context_length": model.get("context_length"),
        "batch_size": training.get("batch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "steady_train_tokens_per_second": _steady_throughput(rows),
        "validation_loss": float(evaluation["loss"]) if evaluation else None,
        "validation_perplexity": float(evaluation["perplexity"]) if evaluation else None,
        "validation_accuracy": float(evaluation["accuracy"]) if evaluation else None,
        "validation_tokens": (
            int(evaluation["tokens"])
            if evaluation and evaluation.get("tokens") is not None
            else None
        ),
        "peak_training_memory_bytes": max(
            (
                int(row["max_memory_allocated_bytes"])
                for row in rows
                if row.get("max_memory_allocated_bytes") is not None
            ),
            default=None,
        ),
        "device_name": environment.get(
            "cuda_device_name", environment.get("device")
        ),
        "dtype": environment.get("dtype", runtime.get("dtype")),
        "torch_version": environment.get("torch"),
        "data_manifest_sha256": (
            provenance.get("splits", {}).get("val", {}).get("manifest_sha256")
            if isinstance(provenance, Mapping)
            else None
        ),
    }
    return summary, rows


def _discover_json(results_root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if not results_root.exists():
        return values
    for path in sorted(results_root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(value, dict):
            value["_path"] = str(path)
            values.append(value)
    return values


def _baseline(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [run for run in runs if run.get("validation_loss") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda run: (int(run.get("attention_layers", 0)), -float(run["validation_loss"])))


def _aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        groups.setdefault(str(run["architecture"]), []).append(run)
    metrics = (
        "validation_loss",
        "validation_perplexity",
        "validation_accuracy",
        "steady_train_tokens_per_second",
    )
    aggregates: list[dict[str, Any]] = []
    for architecture, members in sorted(groups.items()):
        row: dict[str, Any] = {
            "architecture": architecture,
            "runs": len(members),
            "seeds": ",".join(str(run["seed"]) for run in sorted(members, key=lambda run: int(run["seed"]))),
            "attention_layers": members[0]["attention_layers"],
            "token_local_layers": members[0]["token_local_layers"],
            "replacement_mixers": members[0]["replacement_mixers"],
            "triglu_layers": members[0]["triglu_layers"],
            "triglu_no_rope_layers": members[0]["triglu_no_rope_layers"],
            "mb_mlp_layers": members[0]["mb_mlp_layers"],
            "swiglu_mixer_layers": members[0]["swiglu_mixer_layers"],
        }
        for metric in metrics:
            values = [float(run[metric]) for run in members if run.get(metric) is not None]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        matched = [
            run for run in members if run.get("loss_delta_vs_baseline") is not None
        ]
        row["matched_baseline_runs"] = len(matched)
        row["matched_baseline_seeds"] = ",".join(
            str(run["seed"])
            for run in sorted(matched, key=lambda run: int(run["seed"]))
        )
        for metric in (
            "loss_delta_vs_baseline",
            "perplexity_ratio_vs_baseline",
            "throughput_ratio_vs_baseline",
        ):
            values = [float(run[metric]) for run in matched if run.get(metric) is not None]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        aggregates.append(row)
    return aggregates


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label in fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values: list[str] = []
        for key, _label in fields:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _svg_document(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">\n'
        f"<title>{html.escape(title)}</title>\n"
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#111827}.axis{stroke:#6b7280;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.series{fill:none;stroke-width:2}.point{stroke:white;stroke-width:1.5}</style>\n'
        f"{body}\n</svg>\n"
    )


def _bounds(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    low, high = min(values), max(values)
    if low == high:
        padding = abs(low) * 0.05 or 1.0
    else:
        padding = (high - low) * 0.08
    return low - padding, high + padding


def _svg_xy_chart(path: Path, series: Sequence[tuple[str, Sequence[tuple[float, float]]]], *, title: str, x_label: str, y_label: str, labels: Sequence[tuple[str, float, float]] = ()) -> None:
    points = [point for _name, values in series for point in values]
    if not points:
        return
    # Reserve a dedicated legend gutter. Keeping legends outside the plotting
    # rectangle prevents them from colliding with x-axis ticks and titles.
    width, height = 1120, max(540, 95 + 21 * len(series))
    left, right, top, bottom = 88, 300, 55, 70
    x0, x1 = _bounds(point[0] for point in points)
    y0, y1 = _bounds(point[1] for point in points)
    px = lambda value: left + (value - x0) / (x1 - x0) * (width - left - right)
    py = lambda value: height - bottom - (value - y0) / (y1 - y0) * (height - top - bottom)
    body = [f'<text x="{width/2}" y="28" text-anchor="middle" font-size="19" font-weight="600">{html.escape(title)}</text>']
    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * (width - left - right)
        y = top + fraction * (height - top - bottom)
        xv = x0 + fraction * (x1 - x0)
        yv = y1 - fraction * (y1 - y0)
        body.extend((f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>', f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-size="12">{xv:.4g}</text>', f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>', f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12">{yv:.4g}</text>'))
    body.extend((f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>', f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>', f'<text x="{(left+width-right)/2}" y="{height-18}" text-anchor="middle" font-size="14">{html.escape(x_label)}</text>', f'<text transform="translate(20 {(top+height-bottom)/2}) rotate(-90)" text-anchor="middle" font-size="14">{html.escape(y_label)}</text>'))
    for index, (name, values) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        ordered = sorted(values)
        if len(ordered) > 1:
            body.append(f'<polyline class="series" stroke="{color}" points="' + " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in ordered) + '"/>')
        for x, y in ordered:
            body.append(f'<circle class="point" fill="{color}" cx="{px(x):.2f}" cy="{py(y):.2f}" r="4"/>')
        legend_x = width - right + 24
        legend_y = top + 18 + index * 21
        body.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+20}" y2="{legend_y}" stroke="{color}" stroke-width="3"/><text x="{legend_x+26}" y="{legend_y+4}" font-size="12">{html.escape(name)}</text>')
    for label, x, y in labels:
        body.append(f'<text x="{px(x)+7:.2f}" y="{py(y)-7:.2f}" font-size="11">{html.escape(label)}</text>')
    path.write_text(_svg_document(width, height, "\n".join(body), title), encoding="utf-8")


def _rank_rows(analysis: Mapping[str, Any], plan: str) -> list[dict[str, Any]]:
    metrics = analysis.get("rank_metrics", {})
    rows: list[dict[str, Any]] = []
    stages = metrics.get("layer_stages", [])
    if stages:
        for item in stages:
            rows.append({"plan": plan, "layer": item["layer"], "layer_type": item["layer_type"], "stage": item["stage"], "entropy_effective_rank": item["entropy_effective_rank"], "participation_ratio": item["participation_ratio"], "stable_rank": item["stable_rank"]})
    else:
        for item in metrics.get("hidden_states", []):
            if item.get("after_layer") is not None:
                rows.append({"plan": plan, "layer": item["after_layer"], "layer_type": item.get("layer_type"), "stage": "block_output", "entropy_effective_rank": item["entropy_effective_rank"], "participation_ratio": item["participation_ratio"], "stable_rank": item["stable_rank"]})
        for item in metrics.get("mixer_updates", []):
            rows.append({"plan": plan, "layer": item["layer"], "layer_type": item["layer_type"], "stage": "mixer_update", "entropy_effective_rank": item["entropy_effective_rank"], "participation_ratio": item["participation_ratio"], "stable_rank": item["stable_rank"]})
    return rows


def _mechanism_rows(analysis: Mapping[str, Any], plan: str) -> list[dict[str, Any]]:
    metrics = analysis.get("rank_metrics", {})
    transitions = {item["layer"]: item for item in metrics.get("stage_transitions", [])}
    mixer_updates = {item["layer"]: item for item in metrics.get("residual_updates", [])}
    ffn_updates = {item["layer"]: item for item in metrics.get("ffn_residual_updates", [])}
    rows: list[dict[str, Any]] = []
    for layer in sorted(set(transitions) | set(mixer_updates) | set(ffn_updates)):
        transition = transitions.get(layer, {})
        mixer = mixer_updates.get(layer, {})
        ffn = ffn_updates.get(layer, {})
        rows.append(
            {
                "plan": plan,
                "layer": layer,
                "layer_type": transition.get("layer_type", mixer.get("layer_type", ffn.get("layer_type"))),
                "block_input_entropy_effective_rank": transition.get("block_input_entropy_effective_rank"),
                "post_mixer_entropy_effective_rank": transition.get("post_mixer_entropy_effective_rank"),
                "block_output_entropy_effective_rank": transition.get("block_output_entropy_effective_rank"),
                "mixer_rank_delta": transition.get("mixer_rank_delta"),
                "ffn_rank_delta": transition.get("ffn_rank_delta"),
                "block_rank_delta": transition.get("block_rank_delta"),
                "mixer_rank_ratio": transition.get("mixer_rank_ratio"),
                "ffn_rank_ratio": transition.get("ffn_rank_ratio"),
                "mixer_update_to_residual_rms_ratio": mixer.get("update_to_residual_rms_ratio"),
                "mixer_residual_update_cosine": mixer.get("mean_token_residual_update_cosine"),
                "mixer_orthogonal_fraction": mixer.get("mean_token_orthogonal_fraction"),
                "ffn_update_to_residual_rms_ratio": ffn.get("update_to_residual_rms_ratio"),
                "ffn_residual_update_cosine": ffn.get("mean_token_residual_update_cosine"),
                "ffn_orthogonal_fraction": ffn.get("mean_token_orthogonal_fraction"),
            }
        )
    return rows


def generate_report(*, runs_root: str | Path = "runs", results_root: str | Path = "results/generated", suite: str, output_dir: str | Path | None = None) -> dict[str, Any]:
    runs_root, results_root = Path(runs_root), Path(results_root)
    output = Path(output_dir) if output_dir else results_root / f"{suite}-report"
    output.mkdir(parents=True, exist_ok=True)
    suite_dir = runs_root / suite
    discovered: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    if suite_dir.exists():
        for run_dir in sorted(path.parent for path in suite_dir.glob("*/metrics.jsonl") if (path.parent / "resolved_config.yaml").exists()):
            discovered.append(_load_run(run_dir))
    if not discovered:
        raise ValueError(f"no runs with metrics.jsonl and resolved_config.yaml found under {suite_dir}")
    all_runs = [item[0] for item in discovered]
    incomplete_runs = [run for run in all_runs if not run["complete"]]
    runs = [run for run in all_runs if run["complete"]]
    if not runs:
        raise ValueError(f"no completed runs found under {suite_dir}")
    seen_architecture_seeds: dict[tuple[str, int], str] = {}
    for run in runs:
        key = (str(run["architecture"]), int(run["seed"]))
        other = seen_architecture_seeds.get(key)
        if other is not None:
            raise ValueError(
                f"runs {other!r} and {run['plan']!r} both report architecture "
                f"{key[0]!r} with seed {key[1]}; aggregates cannot distinguish "
                "them — remove or rename one run directory"
            )
        seen_architecture_seeds[key] = str(run["plan"])
    # All validation that can reject the report has passed; a regeneration now
    # owns the output directory, so stale derived files must not survive it.
    _clear_previous_outputs(output)
    baseline_example = _baseline(runs)
    baseline_architecture = baseline_example["architecture"] if baseline_example else None
    baseline_by_seed = {
        int(run["seed"]): run
        for run in runs
        if run["architecture"] == baseline_architecture
    }
    if baseline_architecture:
        for run in runs:
            baseline = baseline_by_seed.get(int(run["seed"]))
            if (
                baseline
                and run["validation_loss"] is not None
                and baseline["validation_loss"] is not None
            ):
                run["loss_delta_vs_baseline"] = run["validation_loss"] - baseline["validation_loss"]
                run["perplexity_ratio_vs_baseline"] = run["validation_perplexity"] / baseline["validation_perplexity"]
            if baseline and run["steady_train_tokens_per_second"] and baseline["steady_train_tokens_per_second"]:
                run["throughput_ratio_vs_baseline"] = run["steady_train_tokens_per_second"] / baseline["steady_train_tokens_per_second"]

    artifacts = _discover_json(results_root)
    benchmark_by_architecture: dict[str, dict[str, Any]] = {}
    context_benchmark_rows: list[dict[str, Any]] = []
    evaluation_by_plan: dict[str, dict[str, Any]] = {}
    analyses: dict[str, dict[str, Any]] = {}
    completed_architectures = {str(run["architecture"]) for run in runs}
    completed_plans = {str(run["plan"]) for run in runs}
    for artifact in artifacts:
        event = artifact.get("event")
        if event == "benchmark":
            architecture = _architecture_name(
                _plan_from_source(str(artifact.get("source", "unknown")))
            )
            if architecture not in completed_architectures:
                continue
            benchmark_label = str(artifact.get("benchmark_label") or "")
            if benchmark_label.startswith("context-scaling"):
                settings = artifact.get("settings", {})
                model = artifact.get("model", {})
                training = artifact.get("training", {})
                prefill = artifact.get("prefill", {})
                decode = artifact.get("cached_decode", {})
                context_benchmark_rows.append(
                    {
                        "architecture": architecture,
                        "benchmark_label": benchmark_label,
                        "context_length": int(
                            model.get(
                                "benchmark_context_length",
                                settings.get("training_sequence_length", 0),
                            )
                        ),
                        "configured_context_length": model.get(
                            "configured_context_length"
                        ),
                        "prompt_length": settings.get("prompt_length"),
                        "decode_tokens": settings.get("decode_tokens"),
                        "warmup": settings.get("warmup"),
                        "iterations": settings.get("iterations"),
                        "training_tokens_per_second": training.get(
                            "median_tokens_per_second"
                        ),
                        "prefill_tokens_per_second": prefill.get(
                            "median_tokens_per_second"
                        ),
                        "decode_tokens_per_second": decode.get(
                            "median_tokens_per_second"
                        ),
                        "training_peak_memory_bytes": training.get(
                            "peak_memory_allocated_bytes"
                        ),
                        "prefill_peak_memory_bytes": prefill.get(
                            "peak_memory_allocated_bytes"
                        ),
                        "decode_peak_memory_bytes": decode.get(
                            "peak_memory_allocated_bytes"
                        ),
                        "kv_cache_capacity_bytes": decode.get(
                            "cache_capacity_bytes"
                        ),
                        "kv_cache_final_bytes": decode.get("final_cache_bytes"),
                        "kv_cache_capacity_gib": (
                            float(decode["cache_capacity_bytes"]) / (1024**3)
                            if decode.get("cache_capacity_bytes") is not None
                            else None
                        ),
                        "training_peak_memory_gib": (
                            float(training["peak_memory_allocated_bytes"])
                            / (1024**3)
                            if training.get("peak_memory_allocated_bytes") is not None
                            else None
                        ),
                        "source": artifact.get("_path"),
                    }
                )
            elif architecture not in benchmark_by_architecture:
                benchmark_by_architecture[architecture] = artifact
            else:
                previous = benchmark_by_architecture[architecture]
                previous_iterations = int(
                    previous.get("settings", {}).get("iterations", 0)
                )
                candidate_iterations = int(
                    artifact.get("settings", {}).get("iterations", 0)
                )
                if candidate_iterations > previous_iterations:
                    benchmark_by_architecture[architecture] = artifact
        elif event == "evaluation" and artifact.get("checkpoint"):
            plan = _plan_from_source(str(artifact["checkpoint"]))
            # Incomplete runs are excluded from every derived table, including
            # the standalone-evaluation and rank-diagnostic joins.
            if plan in completed_plans:
                evaluation_by_plan[plan] = artifact
        elif event == "rank_analysis" and artifact.get("checkpoint"):
            plan = _plan_from_source(str(artifact["checkpoint"]))
            if plan in completed_plans:
                analyses[plan] = artifact
    for run in runs:
        benchmark = benchmark_by_architecture.get(str(run["architecture"]))
        external_eval = evaluation_by_plan.get(run["plan"])
        if benchmark:
            run["benchmark_training_tokens_per_second"] = benchmark.get("training", {}).get("median_tokens_per_second")
            run["prefill_tokens_per_second"] = benchmark.get("prefill", {}).get("median_tokens_per_second")
            run["decode_tokens_per_second"] = benchmark.get("cached_decode", {}).get("median_tokens_per_second")
            run["kv_cache_bytes_per_token"] = benchmark.get("cached_decode", {}).get("cache_bytes_per_batch_token_at_final_length")
        if external_eval:
            run["standalone_evaluation_loss"] = external_eval.get("loss")

    summary_fields = ("plan", "architecture", "seed", "complete", "tokens_seen", "attention_layers", "token_local_layers", "replacement_mixers", "triglu_layers", "triglu_no_rope_layers", "mb_mlp_layers", "swiglu_mixer_layers", "parameter_count", "context_length", "validation_tokens", "validation_loss", "validation_perplexity", "validation_accuracy", "steady_train_tokens_per_second", "peak_training_memory_bytes", "device_name", "dtype", "torch_version", "data_manifest_sha256", "loss_delta_vs_baseline", "perplexity_ratio_vs_baseline", "throughput_ratio_vs_baseline", "prefill_tokens_per_second", "decode_tokens_per_second", "kv_cache_bytes_per_token")
    _write_csv(output / "summary.csv", runs, summary_fields)
    aggregates = _aggregate_runs(runs)
    aggregate_fields = (
        "architecture", "runs", "seeds", "attention_layers", "token_local_layers",
        "replacement_mixers", "triglu_layers", "triglu_no_rope_layers",
        "mb_mlp_layers", "swiglu_mixer_layers",
        "validation_loss_mean", "validation_loss_std", "validation_perplexity_mean", "validation_perplexity_std",
        "validation_accuracy_mean", "validation_accuracy_std", "steady_train_tokens_per_second_mean", "steady_train_tokens_per_second_std",
        "matched_baseline_runs", "matched_baseline_seeds",
        "loss_delta_vs_baseline_mean", "loss_delta_vs_baseline_std",
        "perplexity_ratio_vs_baseline_mean", "perplexity_ratio_vs_baseline_std",
        "throughput_ratio_vs_baseline_mean", "throughput_ratio_vs_baseline_std",
    )
    _write_csv(output / "summary_by_architecture.csv", aggregates, aggregate_fields)
    table_fields = (("architecture", "Architecture"), ("runs", "N"), ("seeds", "Seeds"), ("attention_layers", "Attention"), ("token_local_layers", "Token-local"), ("replacement_mixers", "Replacement mixer"), ("validation_loss_mean", "Mean val loss"), ("validation_loss_std", "Loss SD"), ("loss_delta_vs_baseline_mean", "Matched Δ loss"), ("validation_perplexity_mean", "Mean PPL"), ("steady_train_tokens_per_second_mean", "Observed train tok/s"))
    table = _markdown_table(sorted(aggregates, key=lambda row: float(row.get("validation_loss_mean") or math.inf)), table_fields)
    confirmatory_runs = [run for run in runs if int(run["seed"]) != 1337]
    confirmatory_aggregates = _aggregate_runs(confirmatory_runs) if confirmatory_runs else []
    confirmatory_table = ""
    if confirmatory_aggregates:
        _write_csv(
            output / "summary_confirmatory.csv",
            confirmatory_aggregates,
            aggregate_fields,
        )
        confirmatory_table = _markdown_table(
            sorted(
                confirmatory_aggregates,
                key=lambda row: float(row.get("validation_loss_mean") or math.inf),
            ),
            table_fields,
        )
    curve_rows: list[dict[str, Any]] = []
    for summary, metric_rows in discovered:
        if summary["plan"] not in completed_plans:
            continue
        for row in metric_rows:
            if row.get("event") == "evaluation":
                curve_rows.append({"plan": summary["plan"], "step": row.get("step"), "tokens_seen": row.get("tokens_seen"), "loss": row.get("loss"), "perplexity": row.get("perplexity"), "accuracy": row.get("accuracy")})
    _write_csv(output / "training_curves.csv", curve_rows, ("plan", "step", "tokens_seen", "loss", "perplexity", "accuracy"))

    # A validation rerun can preserve the original artifact while superseding it
    # in derived charts. Prefer the largest measured sample for each plan/context.
    selected_context_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in context_benchmark_rows:
        key = (str(row["architecture"]), int(row["context_length"]))
        previous = selected_context_rows.get(key)
        if previous is None or int(row.get("iterations") or 0) > int(
            previous.get("iterations") or 0
        ):
            selected_context_rows[key] = row
    context_benchmark_rows = list(selected_context_rows.values())
    context_benchmark_rows.sort(
        key=lambda row: (int(row["context_length"]), str(row["architecture"]))
    )
    context_benchmark_fields = (
        "architecture",
        "benchmark_label",
        "context_length",
        "configured_context_length",
        "prompt_length",
        "decode_tokens",
        "warmup",
        "iterations",
        "training_tokens_per_second",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "training_peak_memory_bytes",
        "prefill_peak_memory_bytes",
        "decode_peak_memory_bytes",
        "kv_cache_capacity_bytes",
        "kv_cache_final_bytes",
        "kv_cache_capacity_gib",
        "training_peak_memory_gib",
        "source",
    )
    if context_benchmark_rows:
        _write_csv(
            output / "context_benchmarks.csv",
            context_benchmark_rows,
            context_benchmark_fields,
        )

    rank_rows = [row for plan, analysis in analyses.items() for row in _rank_rows(analysis, plan)]
    if rank_rows:
        _write_csv(output / "layer_diagnostics.csv", rank_rows, ("plan", "layer", "layer_type", "stage", "entropy_effective_rank", "participation_ratio", "stable_rank"))
    mechanism_rows = [row for plan, analysis in analyses.items() for row in _mechanism_rows(analysis, plan)]
    if mechanism_rows:
        _write_csv(
            output / "layer_transitions.csv",
            mechanism_rows,
            (
                "plan", "layer", "layer_type",
                "block_input_entropy_effective_rank", "post_mixer_entropy_effective_rank", "block_output_entropy_effective_rank",
                "mixer_rank_delta", "ffn_rank_delta", "block_rank_delta", "mixer_rank_ratio", "ffn_rank_ratio",
                "mixer_update_to_residual_rms_ratio", "mixer_residual_update_cosine", "mixer_orthogonal_fraction",
                "ffn_update_to_residual_rms_ratio", "ffn_residual_update_cosine", "ffn_orthogonal_fraction",
            ),
        )

    quality_points = [(run["plan"], float(run["steady_train_tokens_per_second"]), float(run["validation_loss"])) for run in runs if run.get("steady_train_tokens_per_second") and run.get("validation_loss") is not None]
    _svg_xy_chart(output / "quality_vs_throughput.svg", [("runs", [(x, y) for _name, x, y in quality_points])], title=f"{suite}: quality versus observed run throughput", x_label="observed steady-state training tokens/s", y_label="validation loss (lower is better)", labels=quality_points)
    if context_benchmark_rows:
        context_metrics = (
            (
                "training_tokens_per_second",
                "context_training_throughput.svg",
                "training throughput by context length",
                "median training tokens/s",
            ),
            (
                "prefill_tokens_per_second",
                "context_prefill_throughput.svg",
                "prefill throughput by context length",
                "median prefill tokens/s",
            ),
            (
                "decode_tokens_per_second",
                "context_decode_throughput.svg",
                "cached-decode throughput by context length",
                "median decode tokens/s",
            ),
        )
        for metric, filename, title, y_label in context_metrics:
            series = []
            for architecture in sorted(
                {str(row["architecture"]) for row in context_benchmark_rows}
            ):
                values = [
                    (float(row["context_length"]), float(row[metric]))
                    for row in context_benchmark_rows
                    if row["architecture"] == architecture
                    and row.get(metric) is not None
                ]
                if values:
                    series.append((architecture, values))
            _svg_xy_chart(
                output / filename,
                series,
                title=f"{suite}: {title}",
                x_label="benchmark context length",
                y_label=y_label,
            )

        cache_series = []
        for architecture in sorted(
            {str(row["architecture"]) for row in context_benchmark_rows}
        ):
            values = [
                (
                    float(row["context_length"]),
                    float(row["kv_cache_capacity_bytes"]) / (1024**3),
                )
                for row in context_benchmark_rows
                if row["architecture"] == architecture
                and row.get("kv_cache_capacity_bytes") is not None
            ]
            if values:
                cache_series.append((architecture, values))
        _svg_xy_chart(
            output / "context_kv_cache.svg",
            cache_series,
            title=f"{suite}: allocated KV cache by context length",
            x_label="benchmark context length",
            y_label="KV-cache capacity (GiB)",
        )
    curve_series = []
    for run in runs:
        values = [(float(row["tokens_seen"]), float(row["loss"])) for row in curve_rows if row["plan"] == run["plan"]]
        if values:
            curve_series.append((run["plan"], values))
    _svg_xy_chart(output / "validation_loss_curves.svg", curve_series, title=f"{suite}: validation loss", x_label="training tokens", y_label="validation loss")
    if curve_rows:
        maximum_tokens = max(float(row["tokens_seen"]) for row in curve_rows)
        zoom_start = 0.8 * maximum_tokens
        zoom_series = []
        for run in runs:
            values = [
                (float(row["tokens_seen"]), float(row["loss"]))
                for row in curve_rows
                if row["plan"] == run["plan"]
                and float(row["tokens_seen"]) >= zoom_start
            ]
            if values:
                zoom_series.append((run["plan"], values))
        _svg_xy_chart(
            output / "validation_loss_curves_zoomed.svg",
            zoom_series,
            title=f"{suite}: validation loss, final 20% of training",
            x_label="training tokens",
            y_label="validation loss",
        )
    if rank_rows:
        stage_order = ("block_input", "mixer_update", "post_mixer_residual", "ffn_update", "block_output")
        rank_series = []
        for plan in sorted(analyses):
            values = [(float(row["layer"]), float(row["entropy_effective_rank"])) for row in rank_rows if row["plan"] == plan and row["stage"] == "block_output"]
            if values:
                rank_series.append((plan, values))
            plan_series = []
            for stage in stage_order:
                values = [(float(row["layer"]), float(row["entropy_effective_rank"])) for row in rank_rows if row["plan"] == plan and row["stage"] == stage]
                if values:
                    plan_series.append((stage, values))
            _svg_xy_chart(output / f"layer_stages-{plan}.svg", plan_series, title=f"{plan}: sublayer effective rank", x_label="layer (zero indexed)", y_label="entropy effective rank")
        _svg_xy_chart(output / "layer_effective_rank.svg", rank_series, title=f"{suite}: block-output effective rank", x_label="layer (zero indexed)", y_label="entropy effective rank")
    if mechanism_rows and any(row.get("mixer_rank_delta") is not None for row in mechanism_rows):
        for plan in sorted(analyses):
            mixer_delta = [(float(row["layer"]), float(row["mixer_rank_delta"])) for row in mechanism_rows if row["plan"] == plan and row.get("mixer_rank_delta") is not None]
            ffn_delta = [(float(row["layer"]), float(row["ffn_rank_delta"])) for row in mechanism_rows if row["plan"] == plan and row.get("ffn_rank_delta") is not None]
            _svg_xy_chart(output / f"rank_delta-{plan}.svg", [("mixer residual addition", mixer_delta), ("FFN residual addition", ffn_delta)], title=f"{plan}: sublayer rank change", x_label="layer (zero indexed)", y_label="change in entropy effective rank")
            mixer_rms = [(float(row["layer"]), float(row["mixer_update_to_residual_rms_ratio"])) for row in mechanism_rows if row["plan"] == plan and row.get("mixer_update_to_residual_rms_ratio") is not None]
            ffn_rms = [(float(row["layer"]), float(row["ffn_update_to_residual_rms_ratio"])) for row in mechanism_rows if row["plan"] == plan and row.get("ffn_update_to_residual_rms_ratio") is not None]
            _svg_xy_chart(output / f"update_scale-{plan}.svg", [("mixer update", mixer_rms), ("FFN update", ffn_rms)], title=f"{plan}: residual-update scale", x_label="layer (zero indexed)", y_label="update/residual RMS ratio")

    sensitivity_series = []
    sensitivity_rows: list[dict[str, Any]] = []
    for plan, analysis in analyses.items():
        sensitivity = analysis.get("layer_sensitivity") or {}
        mixer_rows = sensitivity.get("mixer_ablations", sensitivity.get("attention_layer_ablations", []))
        for item in mixer_rows:
            sensitivity_rows.append({"plan": plan, "component": "mixer", "layer": item["layer"], "layer_type": item.get("layer_type"), "loss_delta": item["loss_delta"], "accuracy_delta": item.get("accuracy_delta")})
        for item in sensitivity.get("ffn_ablations", []):
            sensitivity_rows.append({"plan": plan, "component": "ffn", "layer": item["layer"], "layer_type": item.get("layer_type"), "loss_delta": item["loss_delta"], "accuracy_delta": item.get("accuracy_delta")})
    if sensitivity_rows:
        _write_csv(output / "layer_sensitivity.csv", sensitivity_rows, ("plan", "component", "layer", "layer_type", "loss_delta", "accuracy_delta"))
        for plan in sorted(analyses):
            for component in ("mixer", "ffn"):
                values = [(float(row["layer"]), float(row["loss_delta"])) for row in sensitivity_rows if row["plan"] == plan and row["component"] == component]
                if values:
                    sensitivity_series.append((f"{plan}: {component}", values))
        _svg_xy_chart(output / "layer_sensitivity.svg", sensitivity_series, title=f"{suite}: residual-update ablation sensitivity", x_label="layer (zero indexed)", y_label="validation loss increase")

    if incomplete_runs:
        _write_csv(
            output / "incomplete_runs.csv",
            incomplete_runs,
            ("plan", "architecture", "seed", "step", "tokens_seen", "run_dir"),
        )
    context_section = ""
    if context_benchmark_rows:
        context_table = _markdown_table(
            context_benchmark_rows,
            (
                ("architecture", "Architecture"),
                ("context_length", "Context"),
                ("training_tokens_per_second", "Train tok/s"),
                ("prefill_tokens_per_second", "Prefill tok/s"),
                ("decode_tokens_per_second", "Decode tok/s"),
                ("training_peak_memory_gib", "Train peak GiB"),
                ("kv_cache_capacity_gib", "KV capacity GiB"),
            ),
        )
        context_section = (
            "\n## Context-scaling benchmark\n\n"
            + context_table
            + "\n\nContexts above the training configuration measure architecture-level "
            "efficiency extrapolation only; they do not establish language-modeling "
            "quality beyond the trained context window.\n"
        )

    example = runs[0]
    report = {
        "schema_version": 4,
        "suite": suite,
        "baseline_architecture": baseline_architecture,
        "runs": runs,
        "incomplete_runs_excluded": incomplete_runs,
        "architectures": aggregates,
        "confirmatory_architectures": confirmatory_aggregates,
        "context_benchmarks": context_benchmark_rows,
        "rank_analyses": {plan: value.get("_path") for plan, value in analyses.items()},
        "output_dir": str(output),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = (
        f"- Completed runs included: {len(runs)}\n"
        f"- Model: {example.get('n_layers')} layers, {int(example.get('parameter_count') or 0):,} parameters, context {example.get('context_length')}\n"
        f"- Training tokens per completed run: {int(example.get('tokens_seen') or 0):,}\n"
        f"- Validation targets per reported evaluation: {int(example.get('validation_tokens') or 0):,}\n"
        f"- Runtime: {example.get('device_name')} / {example.get('dtype')} / PyTorch {example.get('torch_version')}\n"
        f"- Data manifest SHA256: `{example.get('data_manifest_sha256')}`\n"
    )
    incomplete_note = (
        f"\n**Excluded incomplete runs:** {', '.join(run['plan'] for run in incomplete_runs)}. See `incomplete_runs.csv`.\n"
        if incomplete_runs
        else ""
    )
    confirmatory_section = (
        f"\n## Confirmatory seeds only\n\n{confirmatory_table}\n"
        if confirmatory_table
        else "\n## Confirmatory seeds only\n\nNo non-screening seed has completed yet.\n"
    )
    context_figures = (
        "- [Training throughput by context](context_training_throughput.svg)\n"
        "- [Prefill throughput by context](context_prefill_throughput.svg)\n"
        "- [Cached-decode throughput by context](context_decode_throughput.svg)\n"
        "- [Allocated KV cache by context](context_kv_cache.svg)\n"
        if context_benchmark_rows
        else ""
    )
    markdown = f"# {suite} results\n\n## Comparison metadata\n\n{metadata}{incomplete_note}\nSeed 1337 is the exploratory screening seed. Where focused replications are present, the pooled table reports all completed seeds and their sample standard deviation. `Matched Δ loss` averages only per-run differences against `{report['baseline_architecture']}` checkpoints with the same seed.\n\n## Pooled descriptive summary\n\n{table}\n{confirmatory_section}{context_section}\nThe throughput column and quality/throughput figure use observed training-log throughput, including batch sampling. They are useful operational measurements but are not substitutes for the controlled synthetic benchmark when making architecture-efficiency claims.\n\n## Figures\n\n- [Quality versus observed run throughput](quality_vs_throughput.svg)\n- [Validation-loss curves](validation_loss_curves.svg)\n- [Validation-loss curves, final 20%](validation_loss_curves_zoomed.svg)\n" + context_figures + ("- [Block-output effective rank comparison](layer_effective_rank.svg)\n- Per-plan `layer_stages-*.svg` sublayer diagnostics\n" if rank_rows else "") + ("- Per-plan `rank_delta-*.svg` and `update_scale-*.svg` causal-path diagnostics\n" if mechanism_rows and any(row.get("mixer_rank_delta") is not None for row in mechanism_rows) else "") + ("- [Layer sensitivity](layer_sensitivity.svg)\n" if sensitivity_rows else "") + "\nDerived files are reproducible from the raw run and result artifacts; do not edit them by hand.\n"
    (output / "README.md").write_text(markdown, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="run subdirectory, e.g. 20l_4k_1b")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--results-root", default="results/generated")
    parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = generate_report(runs_root=args.runs_root, results_root=args.results_root, suite=args.suite, output_dir=args.output_dir)
    print(f"Generated report for {len(report['runs'])} runs in {report['output_dir']}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "generate_report", "main"]
