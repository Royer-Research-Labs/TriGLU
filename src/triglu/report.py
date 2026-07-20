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


# Validated categorical palette (data-viz reference instance, light surface).
# Ordering is the CVD-safety mechanism — do not reorder. Slots 1–4 additionally
# clear the all-pairs gate used by scatter/marker charts.
COLORS = ("#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948")
# Composite second channel: past eight series, vary dash rather than cycling hue.
_DASHES = ("", "7 3", "2 3", "9 3 2 3")
# Fixed color+marker per experiment family (slots 1–4, all-pairs safe). Marker
# shape is a redundant channel so families stay separable in grayscale/CVD.
_FAMILY_STYLE: dict[str, tuple[str, str, str]] = {
    "primary": ("#2a78d6", "circle", "Primary (baseline & RoPE placements)"),
    "attention_slot_control": ("#008300", "square", "Attention-slot control"),
    "ffn_form_control": ("#e87ba4", "triangle", "FFN-form control"),
    "residual_topology_control": ("#eda100", "diamond", "Residual-topology control"),
}
_FAMILY_ORDER = tuple(_FAMILY_STYLE)
# The study's recommended architectures, emphasized on the scatter charts.
# Keyed by suite so a different suite is not annotated with these names.
_RECOMMENDED_BY_SUITE: dict[str, dict[str, str]] = {
    "20l_4k_1b": {
        "9a11_triglu_no_rope_nested": "efficiency",
        "20a0t_triglu_no_rope_ffn": "max quality",
    },
}
# Short, human-readable labels for the scatter charts. The full architecture
# names are machine identifiers; these keep the README figures legible. Any
# architecture without an entry falls back to its full name.
_SHORT_LABELS: dict[str, str] = {
    "20a0t": "20A (baseline)",
    "20a0t_triglu_no_rope_ffn": "20A + TriGLU FFN",
    "9a11_triglu_no_rope_nested": "9A / 11 (55%)",
    "9a11_triglu_no_rope_nested_parallel_block": "9A / 11 parallel",
    "9a11_triglu_no_rope_nested_triglu_ffn": "9A / 11 + TriGLU FFN",
    "12a8_triglu_no_rope_nested": "12A / 8",
    "15a5_triglu_no_rope_nested": "15A / 5",
    "15a5_triglu_no_rope_front_blend": "15A / 5 front",
    "15a5_swiglu_front_blend": "15A / 5 SwiGLU",
    "15a5_mb_mlp_front_blend": "15A / 5 MB-MLP",
    "15a5_wide_swiglu_single_residual_front_blend": "15A / 5 FFN-only",
    "6a14_triglu_no_rope_nested": "6A / 14",
    "5a15t": "5A / 15 (RoPE)",
    "15a5t_late_alternating": "15A / 5 (RoPE)",
    "15a5t_front_blend": "15A / 5 front (RoPE)",
    "9l_9a0t_standard_swiglu": "9-layer (54M)",
    "9l_9a0t_grouped_swiglu_9a11_nested_collapse": "9-layer collapse",
}
TOKEN_LOCAL_LAYER_TYPES = (
    "triglu",
    "triglu_no_rope",
    "mb_mlp",
    "swiglu_mixer",
)


def _positive_metadata_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def _effective_ffn_hidden_sizes(model: Mapping[str, Any]) -> list[int]:
    """Normalize legacy scalar FFN metadata into a per-layer schedule."""

    n_layers = _positive_metadata_int(
        model.get("n_layers", len(model.get("layer_types", []))),
        "model.n_layers",
    )
    schedule = model.get("ffn_hidden_sizes")
    if schedule is None:
        hidden_size = model.get("ffn_hidden_size")
        if hidden_size is None:
            return []
        return [
            _positive_metadata_int(hidden_size, "model.ffn_hidden_size")
        ] * n_layers
    if not isinstance(schedule, list):
        raise ValueError("model.ffn_hidden_sizes must be a list or null")
    if len(schedule) != n_layers:
        raise ValueError(
            "model.ffn_hidden_sizes length does not match model.n_layers: "
            f"{len(schedule)} != {n_layers}"
        )
    return [
        _positive_metadata_int(
            width,
            f"model.ffn_hidden_sizes[{index}]",
        )
        for index, width in enumerate(schedule)
    ]


def _effective_residual_init_depth(model: Mapping[str, Any]) -> int:
    """Normalize legacy residual initialization metadata."""

    value = model.get("residual_init_depth")
    return _positive_metadata_int(
        (
            model.get("n_layers", len(model.get("layer_types", [])))
            if value is None
            else value
        ),
        "model.residual_init_depth",
    )


def _width_summary(widths: Sequence[int]) -> str:
    if not widths:
        return "unknown"
    if len(set(widths)) == 1:
        return str(widths[0])
    return f"{min(widths)}–{max(widths)} (Σ{sum(widths)})"


def _experiment_family(
    *,
    layer_types: Sequence[str],
    ffn_type: str,
    ffn_hidden_sizes: Sequence[int],
    residual_init_depth: int,
    block_mode: str = "sequential",
) -> str:
    """Classify scientific scope without relying on opaque run names."""

    if (
        "ffn_only" in layer_types
        or len(set(ffn_hidden_sizes)) > 1
        or residual_init_depth != len(layer_types)
        or block_mode != "sequential"
    ):
        return "residual_topology_control"
    if ffn_type != "swiglu":
        return "ffn_form_control"
    if any(
        layer_type in {"triglu_no_rope", "mb_mlp", "swiglu_mixer"}
        for layer_type in layer_types
    ):
        return "attention_slot_control"
    return "primary"


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
    ffn_only_layers = layer_types.count("ffn_only")
    return {
        "attention_layers": layer_types.count("attention"),
        "token_local_layers": sum(counts.values()),
        "residual_updates_per_forward": 2 * len(layer_types) - ffn_only_layers,
        "replacement_mixers": ",".join(active) if active else "none",
        "ffn_only_layers": ffn_only_layers,
        "structural_controls": (
            f"ffn_only:{ffn_only_layers}" if ffn_only_layers else "none"
        ),
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
    "accuracy_vs_throughput_pareto.svg",
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
    n_layers = _positive_metadata_int(
        model.get("n_layers", len(layer_types)),
        "model.n_layers",
    )
    if len(layer_types) != n_layers:
        raise ValueError(
            f"model.layer_types length does not match model.n_layers: "
            f"{len(layer_types)} != {n_layers}"
        )
    replacement_counts = _replacement_counts(layer_types)
    ffn_hidden_sizes = _effective_ffn_hidden_sizes(model)
    residual_init_depth = _effective_residual_init_depth(model)
    ffn_type = str(model.get("ffn_type", "swiglu"))
    experiment_family = _experiment_family(
        layer_types=layer_types,
        ffn_type=ffn_type,
        ffn_hidden_sizes=ffn_hidden_sizes,
        residual_init_depth=residual_init_depth,
        block_mode=str(model.get("block_mode", "sequential")),
    )
    if (
        experiment_family == "residual_topology_control"
        and replacement_counts["structural_controls"] == "none"
    ):
        # Label the actual structural cause: a single-norm parallel block is
        # not a grouped-width or collapsed-depth control.
        if str(model.get("block_mode", "sequential")) != "sequential":
            replacement_counts["structural_controls"] = "parallel_block"
        else:
            replacement_counts["structural_controls"] = "grouped_width_or_depth"
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
        "n_layers": n_layers,
        "experiment_family": experiment_family,
        **replacement_counts,
        "layer_types": layer_types,
        # Older resolved configs predate the explicit fields and used SwiGLU
        # in the sequential wrapper.
        "block_mode": str(model.get("block_mode", "sequential")),
        "ffn_type": ffn_type,
        "ffn_hidden_size": model.get("ffn_hidden_size"),
        "ffn_hidden_sizes": ffn_hidden_sizes,
        "ffn_width_schedule": ",".join(str(width) for width in ffn_hidden_sizes),
        "ffn_width_summary": _width_summary(ffn_hidden_sizes),
        "ffn_total_hidden_size": sum(ffn_hidden_sizes),
        "residual_init_depth": residual_init_depth,
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
    # An all-attention FFN control must never silently replace the conventional
    # Transformer baseline merely because it obtains a lower loss.
    conventional = [run for run in candidates if run.get("ffn_type", "swiglu") == "swiglu"]
    if conventional:
        candidates = conventional
    canonical = [
        run
        for run in candidates
        if re.fullmatch(r"\d+a0t", str(run.get("architecture", "")))
    ]
    if canonical:
        candidates = canonical
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
        for field in (
            "n_layers",
            "experiment_family",
            "attention_layers",
            "token_local_layers",
            "residual_updates_per_forward",
            "replacement_mixers",
            "ffn_only_layers",
            "structural_controls",
            "ffn_type",
            "ffn_hidden_size",
            "residual_init_depth",
            "parameter_count",
            "context_length",
            "tokens_seen",
            "validation_tokens",
            "data_manifest_sha256",
        ):
            values = {run.get(field) for run in members}
            if len(values) != 1:
                raise ValueError(
                    f"architecture {architecture!r} has inconsistent {field}: "
                    f"{sorted(str(value) for value in values)}"
                )
        for field in ("layer_types", "ffn_hidden_sizes"):
            values = {tuple(run.get(field, [])) for run in members}
            if len(values) != 1:
                raise ValueError(
                    f"architecture {architecture!r} has inconsistent {field}: "
                    f"{sorted(str(value) for value in values)}"
                )
        row: dict[str, Any] = {
            "architecture": architecture,
            "runs": len(members),
            "seeds": ",".join(str(run["seed"]) for run in sorted(members, key=lambda run: int(run["seed"]))),
            "n_layers": members[0]["n_layers"],
            "experiment_family": members[0]["experiment_family"],
            "attention_layers": members[0]["attention_layers"],
            "token_local_layers": members[0]["token_local_layers"],
            "residual_updates_per_forward": members[0][
                "residual_updates_per_forward"
            ],
            "replacement_mixers": members[0]["replacement_mixers"],
            "ffn_only_layers": members[0]["ffn_only_layers"],
            "structural_controls": members[0]["structural_controls"],
            "ffn_type": members[0]["ffn_type"],
            "ffn_hidden_size": members[0]["ffn_hidden_size"],
            "ffn_hidden_sizes": members[0]["ffn_hidden_sizes"],
            "ffn_width_schedule": members[0]["ffn_width_schedule"],
            "ffn_width_summary": members[0]["ffn_width_summary"],
            "ffn_total_hidden_size": members[0]["ffn_total_hidden_size"],
            "residual_init_depth": members[0]["residual_init_depth"],
            "parameter_count": members[0]["parameter_count"],
            "context_length": members[0]["context_length"],
            "tokens_seen": members[0]["tokens_seen"],
            "validation_tokens": members[0]["validation_tokens"],
            "data_manifest_sha256": members[0]["data_manifest_sha256"],
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
        # Past eight series, disambiguate by dash pattern rather than reusing a
        # near-identical hue, so many curves stay individually readable.
        dash = _DASHES[(index // len(COLORS)) % len(_DASHES)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        ordered = sorted(values)
        if len(ordered) > 1:
            body.append(f'<polyline class="series" stroke="{color}"{dash_attr} points="' + " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in ordered) + '"/>')
        for x, y in ordered:
            body.append(f'<circle class="point" fill="{color}" cx="{px(x):.2f}" cy="{py(y):.2f}" r="4"/>')
        legend_x = width - right + 24
        legend_y = top + 18 + index * 21
        body.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+20}" y2="{legend_y}" stroke="{color}" stroke-width="3"{dash_attr}/><text x="{legend_x+26}" y="{legend_y+4}" font-size="12">{html.escape(name)}</text>')
    for label, x, y in labels:
        body.append(f'<text x="{px(x)+7:.2f}" y="{py(y)-7:.2f}" font-size="11">{html.escape(label)}</text>')
    path.write_text(_svg_document(width, height, "\n".join(body), title), encoding="utf-8")


def _pareto_front(items: Sequence[tuple[str, float, float]]) -> set[str]:
    """Names on the Pareto frontier when maximizing both coordinates."""

    front: set[str] = set()
    for name, x, y in items:
        if not any(
            ox >= x and oy >= y and (ox > x or oy > y)
            for other, ox, oy in items
            if other != name
        ):
            front.add(name)
    return front


def _marker_svg(shape: str, cx: float, cy: float, color: str, r: float = 5.0) -> str:
    """A filled data marker; shape is a redundant channel alongside color."""

    if shape == "square":
        return f'<rect class="point" fill="{color}" x="{cx-r:.2f}" y="{cy-r:.2f}" width="{2*r:.2f}" height="{2*r:.2f}"/>'
    if shape == "triangle":
        return f'<polygon class="point" fill="{color}" points="{cx:.2f},{cy-r-1:.2f} {cx-r-1:.2f},{cy+r:.2f} {cx+r+1:.2f},{cy+r:.2f}"/>'
    if shape == "diamond":
        return f'<polygon class="point" fill="{color}" points="{cx:.2f},{cy-r-1.5:.2f} {cx+r+1.5:.2f},{cy:.2f} {cx:.2f},{cy+r+1.5:.2f} {cx-r-1.5:.2f},{cy:.2f}"/>'
    return f'<circle class="point" fill="{color}" cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"/>'


def _svg_family_scatter(
    path: Path,
    points: Sequence[tuple[str, float, float, str]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    highlight: set[str],
    x_better: str = "",
    y_better: str = "",
    frontier: set[str] = frozenset(),
    recommended: Mapping[str, str] | None = None,
    display: Mapping[str, str] | None = None,
) -> None:
    """Scatter of (family, x, y, name) points: color+marker by experiment family,
    a family legend, and direct labels only on the highlighted names. When
    ``frontier`` is given, a light dashed line connects those points in x order.
    ``recommended`` maps a name to a role and gives it a bold ``★`` emphasis.
    ``display`` supplies short label text per name (default: the name itself)."""

    if not points:
        return
    recommended = dict(recommended or {})
    display = dict(display or {})
    families = [f for f in _FAMILY_ORDER if any(p[0] == f for p in points)]
    width, height = 1120, 560
    left, right, top, bottom = 92, 320, 58, 74
    x0, x1 = _bounds(p[1] for p in points)
    y0, y1 = _bounds(p[2] for p in points)
    px = lambda v: left + (v - x0) / (x1 - x0) * (width - left - right)
    py = lambda v: height - bottom - (v - y0) / (y1 - y0) * (height - top - bottom)
    body = [f'<text x="{width/2}" y="30" text-anchor="middle" font-size="19" font-weight="600">{html.escape(title)}</text>']
    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * (width - left - right)
        y = top + fraction * (height - top - bottom)
        xv = x0 + fraction * (x1 - x0)
        yv = y1 - fraction * (y1 - y0)
        body.extend((
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>',
            f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-size="12" fill="#52514e">{xv:.4g}</text>',
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>',
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#52514e">{yv:.4g}</text>',
        ))
    body.extend((
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
        f'<text x="{(left+width-right)/2}" y="{height-16}" text-anchor="middle" font-size="14">{html.escape(x_label)}</text>',
        f'<text transform="translate(22 {(top+height-bottom)/2}) rotate(-90)" text-anchor="middle" font-size="14">{html.escape(y_label)}</text>',
    ))
    if x_better:
        body.append(f'<text x="{width-right-4:.2f}" y="{top-6}" text-anchor="end" font-size="11" fill="#898781">{html.escape(x_better)} →</text>')
    if y_better:
        body.append(f'<text x="{left+4:.2f}" y="{top-6}" font-size="11" fill="#898781">↑ {html.escape(y_better)}</text>')
    if frontier:
        line_points = sorted((px(x), py(y)) for f, x, y, name in points if name in frontier)
        if len(line_points) > 1:
            body.append('<polyline fill="none" stroke="#898781" stroke-width="1.5" stroke-dasharray="5 3" points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in line_points) + '"/>')
    # Number the labeled points left-to-right (by throughput). Identities live in
    # a key box, so nothing collides even where markers overlap. Enlarged markers
    # carry a white-haloed number legible on any family hue.
    labeled = sorted(
        ((px(x), py(y), name) for f, x, y, name in points if name in highlight or name in recommended),
        key=lambda t: t[0],
    )
    number_of = {name: index + 1 for index, (_px, _py, name) in enumerate(labeled)}
    for family in families:
        color, shape, _ = _FAMILY_STYLE[family]
        for f, x, y, name in points:
            if f != family:
                continue
            number = number_of.get(name)
            body.append(_marker_svg(shape, px(x), py(y), color, r=9.0 if number else 5.0))
            if name in recommended:
                # A bold dark ring reads as "emphasized" independent of family hue.
                body.append(f'<circle fill="none" stroke="#0b0b0b" stroke-width="2.4" cx="{px(x):.2f}" cy="{py(y):.2f}" r="13"/>')
            if number:
                body.append(f'<text x="{px(x):.2f}" y="{py(y)+3.6:.2f}" text-anchor="middle" font-size="10" font-weight="700" fill="#0b0b0b" stroke="#ffffff" stroke-width="2.6" style="paint-order:stroke">{number}</text>')
    # Right gutter: family legend, then a numbered key.
    key_x = width - right + 18
    for index, family in enumerate(families):
        color, shape, label = _FAMILY_STYLE[family]
        ly = top + 18 + index * 22
        body.append(_marker_svg(shape, key_x + 8, ly - 4, color))
        body.append(f'<text x="{key_x+26}" y="{ly}" font-size="12">{html.escape(label)}</text>')
    key_top = top + 18 + len(families) * 22 + 16
    body.append(f'<text x="{key_x}" y="{key_top}" font-size="12" font-weight="700">Key (numbered by throughput)</text>')
    for index, (_px, _py, name) in enumerate(labeled):
        ky = key_top + 20 + index * 18
        text = display.get(name, name)
        role = recommended.get(name, "")
        number = number_of[name]
        if role:
            body.append(f'<text x="{key_x}" y="{ky}" font-size="11" font-weight="700">★ {number}. {html.escape(text)} — {html.escape(role)}</text>')
        else:
            body.append(f'<text x="{key_x}" y="{ky}" font-size="11">{number}. {html.escape(text)}</text>')
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
    baseline_example = _baseline(runs)
    baseline_architecture = baseline_example["architecture"] if baseline_example else None
    # A run at a different physical depth than the baseline (e.g. the shallow
    # 9-layer control) is a structural control regardless of its per-run
    # classification; the per-run classifier cannot see the suite's reference
    # depth.
    if baseline_example is not None:
        reference_depth = baseline_example.get("n_layers")
        for run in all_runs:
            if (
                run["experiment_family"] != "residual_topology_control"
                and run.get("n_layers") != reference_depth
            ):
                run["experiment_family"] = "residual_topology_control"
                if run.get("structural_controls") == "none":
                    run["structural_controls"] = "shallow_depth"
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
    architecture_metadata = {
        str(run["architecture"]): run
        for run in runs
    }
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
                run_metadata = architecture_metadata[architecture]
                artifact_path = str(artifact.get("_path", "<unknown>"))
                artifact_layer_types = model.get("layer_types")
                if artifact_layer_types != run_metadata.get("layer_types"):
                    raise ValueError(
                        f"context benchmark {artifact_path} has a layer plan that "
                        f"does not match completed architecture {architecture!r}"
                    )
                artifact_n_layers = _positive_metadata_int(
                    model.get("n_layers", len(artifact_layer_types or [])),
                    "benchmark.model.n_layers",
                )
                if artifact_n_layers != run_metadata.get("n_layers"):
                    raise ValueError(
                        f"context benchmark {artifact_path} has "
                        f"{artifact_n_layers} layers, expected "
                        f"{run_metadata.get('n_layers')}"
                    )
                artifact_ffn_type = str(model.get("ffn_type", "swiglu"))
                if artifact_ffn_type != run_metadata.get("ffn_type", "swiglu"):
                    raise ValueError(
                        f"context benchmark {artifact_path} has FFN type "
                        f"{artifact_ffn_type!r}, expected "
                        f"{run_metadata.get('ffn_type')!r}"
                    )
                if "block_mode" in model:
                    # Artifacts written before the explicit field carry no
                    # block_mode; only compare when the artifact declares one.
                    artifact_block_mode = str(model["block_mode"])
                    if artifact_block_mode != run_metadata.get("block_mode", "sequential"):
                        raise ValueError(
                            f"context benchmark {artifact_path} has block_mode "
                            f"{artifact_block_mode!r}, expected "
                            f"{run_metadata.get('block_mode')!r}"
                        )
                if "ffn_hidden_size" in model:
                    artifact_ffn_hidden_size = model["ffn_hidden_size"]
                elif artifact_ffn_type == "swiglu":
                    # Context artifacts from v0.1 predate the explicit field.
                    artifact_ffn_hidden_size = run_metadata.get("ffn_hidden_size")
                else:
                    raise ValueError(
                        f"context benchmark {artifact_path} is missing "
                        "model.ffn_hidden_size"
                    )
                if artifact_ffn_hidden_size != run_metadata.get("ffn_hidden_size"):
                    raise ValueError(
                        f"context benchmark {artifact_path} has FFN width "
                        f"{artifact_ffn_hidden_size!r}, expected "
                        f"{run_metadata.get('ffn_hidden_size')!r}"
                    )
                artifact_ffn_hidden_sizes = _effective_ffn_hidden_sizes(
                    {
                        "n_layers": artifact_n_layers,
                        "layer_types": artifact_layer_types,
                        "ffn_hidden_size": artifact_ffn_hidden_size,
                        "ffn_hidden_sizes": model.get("ffn_hidden_sizes"),
                    }
                )
                if artifact_ffn_hidden_sizes != run_metadata.get(
                    "ffn_hidden_sizes"
                ):
                    raise ValueError(
                        f"context benchmark {artifact_path} has FFN width "
                        f"schedule {artifact_ffn_hidden_sizes!r}, expected "
                        f"{run_metadata.get('ffn_hidden_sizes')!r}"
                    )
                declared_ffn_total = model.get("ffn_total_hidden_size")
                if declared_ffn_total is not None:
                    declared_ffn_total = _positive_metadata_int(
                        declared_ffn_total,
                        "benchmark.model.ffn_total_hidden_size",
                    )
                    if declared_ffn_total != sum(artifact_ffn_hidden_sizes):
                        raise ValueError(
                            f"context benchmark {artifact_path} declares FFN "
                            f"total {declared_ffn_total}, but its schedule sums "
                            f"to {sum(artifact_ffn_hidden_sizes)}"
                        )
                artifact_residual_init_depth = _effective_residual_init_depth(
                    {
                        "n_layers": artifact_n_layers,
                        "layer_types": artifact_layer_types,
                        "residual_init_depth": model.get("residual_init_depth"),
                    }
                )
                if artifact_residual_init_depth != run_metadata.get(
                    "residual_init_depth"
                ):
                    raise ValueError(
                        f"context benchmark {artifact_path} has residual init "
                        f"depth {artifact_residual_init_depth!r}, expected "
                        f"{run_metadata.get('residual_init_depth')!r}"
                    )
                artifact_parameters = model.get("parameters")
                run_parameters = run_metadata.get("parameter_count")
                if artifact_parameters is None:
                    raise ValueError(
                        f"context benchmark {artifact_path} is missing "
                        "model.parameters"
                    )
                artifact_parameters = _positive_metadata_int(
                    artifact_parameters,
                    "benchmark.model.parameters",
                )
                if (
                    run_parameters is not None
                    and artifact_parameters != int(run_parameters)
                ):
                    raise ValueError(
                        f"context benchmark {artifact_path} has "
                        f"{artifact_parameters} parameters, expected {run_parameters}"
                    )
                training = artifact.get("training", {})
                prefill = artifact.get("prefill", {})
                decode = artifact.get("cached_decode", {})
                device = artifact.get("device", {})
                artifact_replacement_counts = _replacement_counts(
                    artifact_layer_types
                )
                context_benchmark_rows.append(
                    {
                        "architecture": architecture,
                        "experiment_family": run_metadata.get(
                            "experiment_family"
                        ),
                        "benchmark_label": benchmark_label,
                        "n_layers": artifact_n_layers,
                        "attention_layers": artifact_replacement_counts[
                            "attention_layers"
                        ],
                        "token_local_layers": artifact_replacement_counts[
                            "token_local_layers"
                        ],
                        "residual_updates_per_forward": (
                            artifact_replacement_counts[
                                "residual_updates_per_forward"
                            ]
                        ),
                        "ffn_only_layers": artifact_replacement_counts[
                            "ffn_only_layers"
                        ],
                        "context_length": int(
                            model.get(
                                "benchmark_context_length",
                                settings.get("training_sequence_length", 0),
                            )
                        ),
                        "configured_context_length": model.get(
                            "configured_context_length"
                        ),
                        "ffn_type": model.get(
                            "ffn_type", run_metadata.get("ffn_type", "swiglu")
                        ),
                        "ffn_hidden_size": artifact_ffn_hidden_size,
                        "ffn_width_schedule": ",".join(
                            str(width) for width in artifact_ffn_hidden_sizes
                        ),
                        "ffn_width_summary": _width_summary(
                            artifact_ffn_hidden_sizes
                        ),
                        "ffn_total_hidden_size": sum(
                            artifact_ffn_hidden_sizes
                        ),
                        "residual_init_depth": artifact_residual_init_depth,
                        "parameter_count": model.get(
                            "parameters", run_metadata.get("parameter_count")
                        ),
                        "device_name": device.get(
                            "cuda_device_name", device.get("device")
                        ),
                        "dtype": device.get("dtype"),
                        "torch_version": device.get("torch"),
                        "cuda_version": device.get("cuda_version"),
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
                        "benchmark_source": artifact.get("source"),
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

    summary_fields = ("plan", "architecture", "experiment_family", "seed", "complete", "tokens_seen", "n_layers", "attention_layers", "token_local_layers", "ffn_only_layers", "residual_updates_per_forward", "replacement_mixers", "structural_controls", "block_mode", "ffn_type", "ffn_hidden_size", "ffn_width_schedule", "ffn_width_summary", "ffn_total_hidden_size", "residual_init_depth", "triglu_layers", "triglu_no_rope_layers", "mb_mlp_layers", "swiglu_mixer_layers", "parameter_count", "context_length", "validation_tokens", "validation_loss", "validation_perplexity", "validation_accuracy", "steady_train_tokens_per_second", "peak_training_memory_bytes", "device_name", "dtype", "torch_version", "data_manifest_sha256", "loss_delta_vs_baseline", "perplexity_ratio_vs_baseline", "throughput_ratio_vs_baseline", "prefill_tokens_per_second", "decode_tokens_per_second", "kv_cache_bytes_per_token")
    # Aggregate BEFORE any write: _aggregate_runs can still reject the report
    # (replica-consistency checks), and artifact validation above can too. Only
    # once every rejection path has passed does the regeneration own the output
    # directory — then stale derived files must not survive it.
    aggregates = _aggregate_runs(runs)
    _clear_previous_outputs(output)
    _write_csv(output / "summary.csv", runs, summary_fields)
    aggregate_fields = (
        "architecture", "experiment_family", "runs", "seeds", "n_layers", "attention_layers", "token_local_layers",
        "ffn_only_layers", "residual_updates_per_forward",
        "replacement_mixers", "structural_controls",
        "ffn_type", "ffn_hidden_size", "ffn_width_schedule",
        "ffn_width_summary", "ffn_total_hidden_size", "residual_init_depth",
        "parameter_count", "context_length", "tokens_seen", "validation_tokens",
        "data_manifest_sha256",
        "triglu_layers", "triglu_no_rope_layers",
        "mb_mlp_layers", "swiglu_mixer_layers",
        "validation_loss_mean", "validation_loss_std", "validation_perplexity_mean", "validation_perplexity_std",
        "validation_accuracy_mean", "validation_accuracy_std", "steady_train_tokens_per_second_mean", "steady_train_tokens_per_second_std",
        "matched_baseline_runs", "matched_baseline_seeds",
        "loss_delta_vs_baseline_mean", "loss_delta_vs_baseline_std",
        "perplexity_ratio_vs_baseline_mean", "perplexity_ratio_vs_baseline_std",
        "throughput_ratio_vs_baseline_mean", "throughput_ratio_vs_baseline_std",
    )
    _write_csv(output / "summary_by_architecture.csv", aggregates, aggregate_fields)
    table_fields = (("architecture", "Architecture"), ("experiment_family", "Family"), ("runs", "N"), ("seeds", "Seeds"), ("n_layers", "Blocks"), ("residual_updates_per_forward", "Residual updates"), ("attention_layers", "Attention"), ("token_local_layers", "Token-local"), ("ffn_only_layers", "FFN-only"), ("replacement_mixers", "Replacement mixer"), ("ffn_type", "FFN"), ("ffn_width_summary", "FFN widths"), ("parameter_count", "Parameters"), ("validation_loss_mean", "Mean val loss"), ("validation_loss_std", "Loss SD"), ("loss_delta_vs_baseline_mean", "Matched Δ loss"), ("validation_perplexity_mean", "Mean PPL"), ("steady_train_tokens_per_second_mean", "Observed train tok/s"))
    family_labels = {
        "primary": "Primary attention/TriGLU plans",
        "attention_slot_control": "Attention-slot controls",
        "ffn_form_control": "FFN-form controls",
        "residual_topology_control": "Residual-topology controls",
    }
    family_tables: list[str] = []
    for family, label in family_labels.items():
        family_rows = [
            row for row in aggregates if row["experiment_family"] == family
        ]
        if family_rows:
            family_tables.append(
                f"### {label}\n\n"
                + _markdown_table(
                    sorted(
                        family_rows,
                        key=lambda row: float(
                            row.get("validation_loss_mean") or math.inf
                        ),
                    ),
                    table_fields,
                )
            )
    table = "\n\n".join(family_tables)
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
        "experiment_family",
        "benchmark_label",
        "n_layers",
        "attention_layers",
        "token_local_layers",
        "ffn_only_layers",
        "residual_updates_per_forward",
        "context_length",
        "configured_context_length",
        "ffn_type",
        "ffn_hidden_size",
        "ffn_width_schedule",
        "ffn_width_summary",
        "ffn_total_hidden_size",
        "residual_init_depth",
        "parameter_count",
        "device_name",
        "dtype",
        "torch_version",
        "cuda_version",
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
        "benchmark_source",
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

    # One point per architecture (seed means), colored/marked by experiment
    # family, with only the baseline and best-per-family architectures labeled —
    # 55 labeled run points were unreadable.
    quality_points = [
        (
            str(agg["experiment_family"]),
            float(agg["steady_train_tokens_per_second_mean"]),
            float(agg["validation_loss_mean"]),
            str(agg["architecture"]),
        )
        for agg in aggregates
        if agg.get("steady_train_tokens_per_second_mean")
        and agg.get("validation_loss_mean") is not None
    ]
    present_architectures = {str(agg["architecture"]) for agg in aggregates}
    recommended = {
        name: role
        for name, role in _RECOMMENDED_BY_SUITE.get(suite, {}).items()
        if name in present_architectures
    }
    quality_highlight: set[str] = set(recommended)
    if baseline_architecture:
        quality_highlight.add(str(baseline_architecture))
    best_per_family: dict[str, tuple[float, str]] = {}
    for family, _tput, loss, name in quality_points:
        if family not in best_per_family or loss < best_per_family[family][0]:
            best_per_family[family] = (loss, name)
    quality_highlight.update(name for _loss, name in best_per_family.values())
    _svg_family_scatter(
        output / "quality_vs_throughput.svg",
        quality_points,
        title=f"{suite}: quality versus observed run throughput",
        x_label="observed steady-state training tokens/s (higher is better)",
        y_label="validation loss (lower is better)",
        highlight=quality_highlight,
        x_better="faster",
        recommended=recommended,
        display=_SHORT_LABELS,
    )

    # Accuracy-vs-throughput Pareto view: only the non-dominated architectures
    # (best accuracy for their throughput, or vice versa) plus the baseline for
    # reference — a small, legible subset. Both axes maximize, so the good corner
    # is upper-right and the frontier line traces the quality/speed trade.
    accuracy_points = [
        (
            str(agg["architecture"]),
            float(agg["steady_train_tokens_per_second_mean"]),
            float(agg["validation_accuracy_mean"]),
            str(agg["experiment_family"]),
        )
        for agg in aggregates
        if agg.get("steady_train_tokens_per_second_mean")
        and agg.get("validation_accuracy_mean") is not None
    ]
    pareto = _pareto_front([(name, tput, acc) for name, tput, acc, _fam in accuracy_points])
    shown = set(pareto)
    if baseline_architecture:
        shown.add(str(baseline_architecture))
    shown.update(recommended)  # always show the recommended picks, even if off-frontier
    pareto_scatter = [
        (fam, tput, acc, name)
        for name, tput, acc, fam in accuracy_points
        if name in shown
    ]
    if pareto_scatter:
        _svg_family_scatter(
            output / "accuracy_vs_throughput_pareto.svg",
            pareto_scatter,
            title=f"{suite}: accuracy versus throughput (Pareto frontier)",
            x_label="observed steady-state training tokens/s (higher is better)",
            y_label="validation accuracy (higher is better)",
            highlight={name for _f, _t, _a, name in pareto_scatter},
            x_better="faster",
            frontier=pareto,
            recommended=recommended,
            display=_SHORT_LABELS,
        )

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
    # Plotting all ~30 architectures' curves is unreadable; show only the
    # headline set (baseline and best per family, the same points labeled in the
    # quality scatter). Every run's curve remains in training_curves.csv.
    plan_for_arch: dict[str, str] = {}
    for run in runs:
        arch = str(run["architecture"])
        if arch not in quality_highlight:
            continue
        if arch not in plan_for_arch or run["plan"] == arch:
            plan_for_arch[arch] = run["plan"]
    curated_curves = sorted(plan_for_arch.items())
    curve_series = []
    for arch, plan in curated_curves:
        values = [(float(row["tokens_seen"]), float(row["loss"])) for row in curve_rows if row["plan"] == plan]
        if values:
            curve_series.append((arch, values))
    _svg_xy_chart(output / "validation_loss_curves.svg", curve_series, title=f"{suite}: validation loss (baseline & best per family)", x_label="training tokens", y_label="validation loss (lower is better)")
    if curve_rows:
        maximum_tokens = max(float(row["tokens_seen"]) for row in curve_rows)
        zoom_start = 0.8 * maximum_tokens
        zoom_series = []
        for arch, plan in curated_curves:
            values = [
                (float(row["tokens_seen"]), float(row["loss"]))
                for row in curve_rows
                if row["plan"] == plan
                and float(row["tokens_seen"]) >= zoom_start
            ]
            if values:
                zoom_series.append((arch, values))
        _svg_xy_chart(
            output / "validation_loss_curves_zoomed.svg",
            zoom_series,
            title=f"{suite}: validation loss, final 20% (baseline & best per family)",
            x_label="training tokens",
            y_label="validation loss (lower is better)",
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
                ("experiment_family", "Family"),
                ("n_layers", "Blocks"),
                ("residual_updates_per_forward", "Residual updates"),
                ("attention_layers", "Attention"),
                ("ffn_only_layers", "FFN-only"),
                ("context_length", "Context"),
                ("ffn_type", "FFN"),
                ("ffn_width_summary", "FFN widths"),
                ("parameter_count", "Parameters"),
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
        "schema_version": 6,
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
    depths = sorted({int(run["n_layers"]) for run in runs})
    parameter_counts = sorted(
        {
            int(run["parameter_count"])
            for run in runs
            if run.get("parameter_count") is not None
        }
    )
    contexts = sorted(
        {
            int(run["context_length"])
            for run in runs
            if run.get("context_length") is not None
        }
    )
    depth_text = str(depths[0]) if len(depths) == 1 else f"{depths[0]}–{depths[-1]}"
    parameter_text = (
        "unknown"
        if not parameter_counts
        else (
            f"{parameter_counts[0]:,}"
            if len(parameter_counts) == 1
            else f"{parameter_counts[0]:,}–{parameter_counts[-1]:,}"
        )
    )
    context_text = (
        "unknown"
        if not contexts
        else (
            str(contexts[0])
            if len(contexts) == 1
            else f"{contexts[0]}–{contexts[-1]}"
        )
    )
    metadata = (
        f"- Completed runs included: {len(runs)}\n"
        f"- Model range: {depth_text} physical blocks, {parameter_text} parameters, context {context_text}\n"
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
    markdown = f"# {suite} results\n\n## Comparison metadata\n\n{metadata}{incomplete_note}\nSeed 1337 is the exploratory screening seed. Where focused replications are present, the pooled table reports all completed seeds and their sample standard deviation. `Matched Δ loss` averages only per-run differences against `{report['baseline_architecture']}` checkpoints with the same seed. Across experiment families it is a baseline-relative quality descriptor, not a controlled estimate of one component's causal effect.\n\n## Pooled descriptive summary by experiment family\n\n{table}\n{confirmatory_section}{context_section}\nThe throughput column and quality/throughput figure use observed training-log throughput, including batch sampling. They are useful operational measurements but are not substitutes for the controlled synthetic benchmark when making architecture-efficiency claims.\n\n## Figures\n\nThe quality/throughput scatter plots one point per architecture (seed means), colored and marked by experiment family, labeling only the baseline and the best architecture in each family. The loss-curve figures show that same headline set; every run's full curve is in `training_curves.csv`.\n\n- [Quality versus observed run throughput](quality_vs_throughput.svg)\n- [Accuracy versus throughput, Pareto frontier](accuracy_vs_throughput_pareto.svg)\n- [Validation-loss curves, baseline & best per family](validation_loss_curves.svg)\n- [Validation-loss curves, final 20%](validation_loss_curves_zoomed.svg)\n" + context_figures + ("- [Block-output effective rank comparison](layer_effective_rank.svg)\n- Per-plan `layer_stages-*.svg` sublayer diagnostics\n" if rank_rows else "") + ("- Per-plan `rank_delta-*.svg` and `update_scale-*.svg` causal-path diagnostics\n" if mechanism_rows and any(row.get("mixer_rank_delta") is not None for row in mechanism_rows) else "") + ("- [Layer sensitivity](layer_sensitivity.svg)\n" if sensitivity_rows else "") + "\nDerived files are reproducible from the raw run and result artifacts; do not edit them by hand.\n"
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
