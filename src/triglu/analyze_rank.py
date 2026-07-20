"""Analyze layerwise sublayer geometry and component sensitivity in a saved model.

This is a post-training diagnostic.  It does not alter checkpoints or training
behavior.  Numerical effective rank is measured from centered token-by-channel
covariance spectra; this is more informative than the algebraic rank of a causal
attention matrix, which is commonly full because of its triangular structure.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch

from .config import ModelConfig
from .model import DecoderLM
from .runtime import (
    ConfigurationError,
    autocast_context,
    create_token_stream,
    environment_info,
    evaluate_language_model,
    load_checkpoint,
    parameter_count,
    resolve_runtime,
    save_json,
    utc_now,
    verify_token_manifest,
)


def _sample_rows(tensor: torch.Tensor, maximum: int) -> torch.Tensor:
    """Flatten batch/token dimensions and choose deterministic, even samples."""

    rows = tensor.detach().reshape(-1, tensor.size(-1))
    if rows.size(0) <= maximum:
        return rows
    indices = torch.linspace(
        0,
        rows.size(0) - 1,
        steps=maximum,
        device=rows.device,
        dtype=torch.float64,
    ).round().long()
    return rows.index_select(0, indices)


def _spectrum_metrics(eigenvalues: torch.Tensor, tolerance: float) -> dict[str, Any]:
    """Summarize a non-negative covariance/Gram spectrum."""

    values = eigenvalues.detach().double().clamp_min(0).sort(descending=True).values
    dimension = int(values.numel())
    total = float(values.sum())
    maximum = float(values[0]) if dimension else 0.0
    if total <= 0.0 or maximum <= 0.0:
        return {
            "numerical_rank": 0,
            "stable_rank": 0.0,
            "participation_ratio": 0.0,
            "entropy_effective_rank": 0.0,
            "top_1_explained_fraction": 0.0,
            "top_10_explained_fraction": 0.0,
            "eigenvalues_descending": [float(value) for value in values],
        }

    probabilities = values / total
    positive = probabilities > 0
    entropy = -float(
        (probabilities[positive] * probabilities[positive].log()).sum()
    )
    squared_sum = float(values.square().sum())
    threshold = maximum * tolerance
    return {
        "numerical_rank": int((values > threshold).sum()),
        "stable_rank": total / maximum,
        "participation_ratio": total * total / squared_sum,
        "entropy_effective_rank": math.exp(entropy),
        "top_1_explained_fraction": float(values[:1].sum()) / total,
        "top_10_explained_fraction": float(values[: min(10, dimension)].sum()) / total,
        "eigenvalues_descending": [float(value) for value in values],
    }


class _CovarianceAccumulator:
    """Streaming centered feature covariance without retaining activations."""

    def __init__(self, feature_dimension: int) -> None:
        self.feature_dimension = feature_dimension
        self.count = 0
        self.sum = torch.zeros(feature_dimension, dtype=torch.float64)
        self.gram = torch.zeros(
            feature_dimension, feature_dimension, dtype=torch.float64
        )

    def update(self, tensor: torch.Tensor, maximum_rows: int) -> None:
        # Hooks fire inside the caller's autocast region, where the Gram matmul
        # would silently run in reduced precision despite the float() cast.
        with torch.autocast(device_type=tensor.device.type, enabled=False):
            rows = _sample_rows(tensor, maximum_rows).float()
            self.count += rows.size(0)
            self.sum += rows.sum(dim=0).double().cpu()
            self.gram += (rows.transpose(0, 1) @ rows).double().cpu()

    def finalize(self, tolerance: float) -> dict[str, Any]:
        if self.count < 2:
            covariance = torch.zeros_like(self.gram)
        else:
            centered = self.gram - torch.outer(self.sum, self.sum) / self.count
            covariance = centered / (self.count - 1)
            covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        return {
            "sample_count": self.count,
            "feature_dimension": self.feature_dimension,
            "rank_tolerance_relative_to_max": tolerance,
            **_spectrum_metrics(eigenvalues, tolerance),
        }


class _ResidualUpdateAccumulator:
    """Measure the scale and directional novelty of a mixer residual update."""

    def __init__(self) -> None:
        self.sample_count = 0
        self.residual_energy = 0.0
        self.update_energy = 0.0
        self.dot = 0.0
        self.token_cosine_sum = 0.0
        self.token_orthogonal_fraction_sum = 0.0
        self.valid_cosines = 0

    def update(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
        maximum_rows: int,
    ) -> None:
        residual_rows = _sample_rows(residual, maximum_rows).float()
        update_rows = _sample_rows(update, maximum_rows).float()
        self.sample_count += residual_rows.size(0)
        self.residual_energy += float(residual_rows.square().sum())
        self.update_energy += float(update_rows.square().sum())
        self.dot += float((residual_rows * update_rows).sum())

        denominator = residual_rows.norm(dim=-1) * update_rows.norm(dim=-1)
        valid = denominator > 0
        if valid.any():
            cosine = (
                (residual_rows[valid] * update_rows[valid]).sum(dim=-1)
                / denominator[valid]
            ).clamp(-1, 1)
            self.token_cosine_sum += float(cosine.sum())
            self.token_orthogonal_fraction_sum += float((1 - cosine.square()).sum())
            self.valid_cosines += int(valid.sum())

    def finalize(self) -> dict[str, Any]:
        energy_product = self.residual_energy * self.update_energy
        return {
            "sample_count": self.sample_count,
            "update_to_residual_rms_ratio": (
                math.sqrt(self.update_energy / self.residual_energy)
                if self.residual_energy > 0
                else 0.0
            ),
            "global_residual_update_cosine": (
                self.dot / math.sqrt(energy_product) if energy_product > 0 else 0.0
            ),
            "mean_token_residual_update_cosine": (
                self.token_cosine_sum / self.valid_cosines
                if self.valid_cosines
                else 0.0
            ),
            "mean_token_orthogonal_fraction": (
                self.token_orthogonal_fraction_sum / self.valid_cosines
                if self.valid_cosines
                else 0.0
            ),
        }


class _AttentionHeadAccumulator:
    """Measure redundancy between projected per-head residual contributions."""

    def __init__(self, n_heads: int, output_dimension: int) -> None:
        self.n_heads = n_heads
        self.output_dimension = output_dimension
        self.count = 0
        self.sum = torch.zeros(n_heads, output_dimension, dtype=torch.float64)
        self.gram = torch.zeros(n_heads, n_heads, dtype=torch.float64)

    def update(
        self,
        merged_heads: torch.Tensor,
        projection_weight: torch.Tensor,
        maximum_rows: int,
    ) -> None:
        # Hooks fire inside the caller's autocast region, where these einsums
        # would silently run in reduced precision despite the float() casts.
        with torch.autocast(device_type=merged_heads.device.type, enabled=False):
            rows = _sample_rows(merged_heads, maximum_rows).float()
            head_dimension = rows.size(-1) // self.n_heads
            head_values = rows.view(rows.size(0), self.n_heads, head_dimension)
            weights = projection_weight.detach().float().view(
                self.output_dimension, self.n_heads, head_dimension
            )
            # Each head is projected separately into the shared residual space.  This
            # makes cross-head cosine meaningful despite independent head coordinates.
            contributions = torch.einsum("nhd,ohd->nho", head_values, weights)
            self.count += contributions.size(0)
            self.sum += contributions.sum(dim=0).double().cpu()
            self.gram += torch.einsum(
                "nho,njo->hj", contributions, contributions
            ).double().cpu()

    def finalize(self, tolerance: float) -> dict[str, Any]:
        if self.count:
            centered = self.gram - torch.einsum(
                "ho,jo->hj", self.sum, self.sum
            ) / self.count
        else:
            centered = self.gram
        centered = 0.5 * (centered + centered.transpose(0, 1))
        eigenvalues = torch.linalg.eigvalsh(centered).clamp_min(0)
        diagonal = centered.diag().clamp_min(0)
        denominator = torch.sqrt(torch.outer(diagonal, diagonal))
        correlation = torch.where(
            denominator > 0, centered / denominator.clamp_min(1e-30), 0
        )
        off_diagonal = ~torch.eye(self.n_heads, dtype=torch.bool)
        energy_total = float(diagonal.sum())
        return {
            "sample_count": self.count,
            "head_count": self.n_heads,
            "mean_absolute_off_diagonal_cosine": (
                float(correlation[off_diagonal].abs().mean())
                if self.n_heads > 1
                else 0.0
            ),
            "head_energy_fractions": (
                [float(value) for value in diagonal / energy_total]
                if energy_total > 0
                else [0.0] * self.n_heads
            ),
            **_spectrum_metrics(eigenvalues, tolerance),
        }


class _LayerDiagnostics:
    """Forward-hook collector for one uncompiled model pass stream."""

    def __init__(
        self,
        model: DecoderLM,
        *,
        maximum_rank_rows: int,
        maximum_head_rows: int,
        rank_tolerance: float,
    ) -> None:
        self.model = model
        self.maximum_rank_rows = maximum_rank_rows
        self.maximum_head_rows = maximum_head_rows
        self.rank_tolerance = rank_tolerance
        channels = model.config.d_model
        self.hidden = [
            _CovarianceAccumulator(channels)
            for _ in range(model.config.n_layers + 1)
        ]
        self.updates = [
            _CovarianceAccumulator(channels) for _ in range(model.config.n_layers)
        ]
        # These stages isolate the two residual sublayers in DecoderBlock:
        #   r0 -> norm1 -> mixer -> r1 -> norm2 -> FFN -> r2.
        # Block inputs/outputs are already represented by ``hidden`` above.
        self.mixer_inputs = [
            _CovarianceAccumulator(channels) for _ in range(model.config.n_layers)
        ]
        self.post_mixer = [
            _CovarianceAccumulator(channels) for _ in range(model.config.n_layers)
        ]
        self.ffn_inputs = [
            _CovarianceAccumulator(channels) for _ in range(model.config.n_layers)
        ]
        self.ffn_updates = [
            _CovarianceAccumulator(channels) for _ in range(model.config.n_layers)
        ]
        self.residual_updates = [
            _ResidualUpdateAccumulator() for _ in range(model.config.n_layers)
        ]
        self.ffn_residual_updates = [
            _ResidualUpdateAccumulator() for _ in range(model.config.n_layers)
        ]
        self.heads: dict[int, _AttentionHeadAccumulator] = {
            index: _AttentionHeadAccumulator(model.config.n_heads, channels)
            for index, block in enumerate(model.blocks)
            if block.layer_type == "attention"
        }
        self.pending_residuals: dict[int, torch.Tensor] = {}
        self.pending_post_mixer: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def install(self) -> None:
        # The stage hooks are defined against the sequential two-norm block
        # (mixer norm / post-mixer residual / FFN norm are distinct points).
        # A single-norm parallel block has no such stages; reject it loudly
        # instead of crashing on the absent second norm.
        if getattr(self.model.config, "block_mode", "sequential") != "sequential":
            raise ConfigurationError(
                "rank analysis requires the sequential two-norm decoder block; "
                f"this checkpoint uses block_mode="
                f"{self.model.config.block_mode!r}, whose merged block has no "
                "separable mixer/FFN stages"
            )

        def embedding_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            self.hidden[0].update(output, self.maximum_rank_rows)

        self.handles.append(self.model.token_embedding.register_forward_hook(embedding_hook))

        for index, block in enumerate(self.model.blocks):
            def block_pre_hook(
                _module: Any,
                inputs: tuple[torch.Tensor, ...],
                *,
                layer_index: int = index,
            ) -> None:
                self.pending_residuals[layer_index] = inputs[0].detach()

            def mixer_hook(
                _module: Any,
                _inputs: Any,
                output: tuple[torch.Tensor, Any],
                *,
                layer_index: int = index,
            ) -> None:
                mixed = output[0]
                self.updates[layer_index].update(mixed, self.maximum_rank_rows)
                residual = self.pending_residuals[layer_index]
                self.residual_updates[layer_index].update(
                    residual, mixed, self.maximum_rank_rows
                )

            def mixer_input_hook(
                _module: Any,
                _inputs: Any,
                output: torch.Tensor,
                *,
                layer_index: int = index,
            ) -> None:
                self.mixer_inputs[layer_index].update(
                    output, self.maximum_rank_rows
                )

            def post_mixer_hook(
                _module: Any,
                inputs: tuple[torch.Tensor, ...],
                *,
                layer_index: int = index,
            ) -> None:
                value = inputs[0]
                self.pending_post_mixer[layer_index] = value.detach()
                self.post_mixer[layer_index].update(value, self.maximum_rank_rows)

            def ffn_input_hook(
                _module: Any,
                _inputs: Any,
                output: torch.Tensor,
                *,
                layer_index: int = index,
            ) -> None:
                self.ffn_inputs[layer_index].update(output, self.maximum_rank_rows)

            def ffn_hook(
                _module: Any,
                _inputs: Any,
                output: torch.Tensor,
                *,
                layer_index: int = index,
            ) -> None:
                self.ffn_updates[layer_index].update(output, self.maximum_rank_rows)
                residual = self.pending_post_mixer.get(
                    layer_index,
                    self.pending_residuals[layer_index],
                )
                self.ffn_residual_updates[layer_index].update(
                    residual,
                    output,
                    self.maximum_rank_rows,
                )

            def block_hook(
                _module: Any,
                _inputs: Any,
                output: tuple[torch.Tensor, Any],
                *,
                layer_index: int = index,
            ) -> None:
                self.hidden[layer_index + 1].update(
                    output[0], self.maximum_rank_rows
                )
                self.pending_residuals.pop(layer_index, None)
                self.pending_post_mixer.pop(layer_index, None)

            self.handles.append(block.register_forward_pre_hook(block_pre_hook))
            if block.layer_type != "ffn_only":
                self.handles.append(
                    block.norm_1.register_forward_hook(mixer_input_hook)
                )
                self.handles.append(block.mixer.register_forward_hook(mixer_hook))
                # norm_2 receives the residual immediately after the mixer addition.
                self.handles.append(
                    block.norm_2.register_forward_pre_hook(post_mixer_hook)
                )
            self.handles.append(block.norm_2.register_forward_hook(ffn_input_hook))
            self.handles.append(block.ffn.register_forward_hook(ffn_hook))
            self.handles.append(block.register_forward_hook(block_hook))

            if block.layer_type == "attention":
                mixer = block.mixer

                def projection_pre_hook(
                    module: torch.nn.Linear,
                    inputs: tuple[torch.Tensor, ...],
                    *,
                    layer_index: int = index,
                ) -> None:
                    self.heads[layer_index].update(
                        inputs[0], module.weight, self.maximum_head_rows
                    )

                self.handles.append(
                    mixer.proj_out.register_forward_pre_hook(projection_pre_hook)
                )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending_residuals.clear()
        self.pending_post_mixer.clear()

    def finalize(self) -> dict[str, Any]:
        layer_types = self.model.config.layer_types
        # Finalize each accumulator exactly once: several output sections view
        # the same spectrum, and every finalize runs a full eigendecomposition.
        hidden_metrics = [
            accumulator.finalize(self.rank_tolerance) for accumulator in self.hidden
        ]
        update_metrics = [
            accumulator.finalize(self.rank_tolerance) for accumulator in self.updates
        ]
        ffn_update_metrics = [
            accumulator.finalize(self.rank_tolerance)
            for accumulator in self.ffn_updates
        ]

        hidden_states: list[dict[str, Any]] = [
            {
                "position": "embedding_output",
                "after_layer": None,
                **hidden_metrics[0],
            }
        ]
        for index, metrics in enumerate(hidden_metrics[1:]):
            hidden_states.append(
                {
                    "position": "block_output",
                    "after_layer": index,
                    "layer_type": layer_types[index],
                    **metrics,
                }
            )

        mixer_updates = [
            {
                "layer": index,
                "layer_type": layer_types[index],
                **metrics,
            }
            for index, metrics in enumerate(update_metrics)
            if layer_types[index] != "ffn_only"
        ]
        residual_updates = [
            {
                "layer": index,
                "layer_type": layer_types[index],
                **accumulator.finalize(),
            }
            for index, accumulator in enumerate(self.residual_updates)
            if layer_types[index] != "ffn_only"
        ]
        ffn_residual_updates = [
            {
                "layer": index,
                "layer_type": layer_types[index],
                **accumulator.finalize(),
            }
            for index, accumulator in enumerate(self.ffn_residual_updates)
        ]
        layer_stages: list[dict[str, Any]] = []
        for index in range(self.model.config.n_layers):
            if layer_types[index] == "ffn_only":
                stage_accumulators = (
                    ("block_input", hidden_metrics[index]),
                    (
                        "ffn_norm_input",
                        self.ffn_inputs[index].finalize(self.rank_tolerance),
                    ),
                    ("ffn_update", ffn_update_metrics[index]),
                    ("block_output", hidden_metrics[index + 1]),
                )
            else:
                stage_accumulators = (
                    ("block_input", hidden_metrics[index]),
                    (
                        "mixer_norm_input",
                        self.mixer_inputs[index].finalize(self.rank_tolerance),
                    ),
                    ("mixer_update", update_metrics[index]),
                    (
                        "post_mixer_residual",
                        self.post_mixer[index].finalize(self.rank_tolerance),
                    ),
                    (
                        "ffn_norm_input",
                        self.ffn_inputs[index].finalize(self.rank_tolerance),
                    ),
                    ("ffn_update", ffn_update_metrics[index]),
                    ("block_output", hidden_metrics[index + 1]),
                )
            layer_stages.extend(
                {
                    "layer": index,
                    "layer_type": layer_types[index],
                    "stage": stage,
                    **metrics,
                }
                for stage, metrics in stage_accumulators
            )

        stage_transitions: list[dict[str, Any]] = []
        for index in range(self.model.config.n_layers):
            stages = {
                item["stage"]: item
                for item in layer_stages
                if item["layer"] == index
            }
            input_rank = stages["block_input"]["entropy_effective_rank"]
            output_rank = stages["block_output"]["entropy_effective_rank"]
            if layer_types[index] == "ffn_only":
                post_mixer_rank = None
                mixer_rank_delta = None
                ffn_rank_delta = output_rank - input_rank
                mixer_rank_ratio = None
                ffn_rank_ratio = output_rank / input_rank if input_rank else 0.0
            else:
                post_mixer_rank = stages["post_mixer_residual"][
                    "entropy_effective_rank"
                ]
                mixer_rank_delta = post_mixer_rank - input_rank
                ffn_rank_delta = output_rank - post_mixer_rank
                mixer_rank_ratio = (
                    post_mixer_rank / input_rank if input_rank else 0.0
                )
                ffn_rank_ratio = (
                    output_rank / post_mixer_rank if post_mixer_rank else 0.0
                )
            stage_transitions.append(
                {
                    "layer": index,
                    "layer_type": layer_types[index],
                    "block_input_entropy_effective_rank": input_rank,
                    "post_mixer_entropy_effective_rank": post_mixer_rank,
                    "block_output_entropy_effective_rank": output_rank,
                    "mixer_rank_delta": mixer_rank_delta,
                    "ffn_rank_delta": ffn_rank_delta,
                    "block_rank_delta": output_rank - input_rank,
                    "mixer_rank_ratio": mixer_rank_ratio,
                    "ffn_rank_ratio": ffn_rank_ratio,
                }
            )
        attention_heads = [
            {
                "layer": index,
                "layer_type": "attention",
                **self.heads[index].finalize(self.rank_tolerance),
            }
            for index in sorted(self.heads)
        ]
        return {
            "hidden_states": hidden_states,
            "mixer_updates": mixer_updates,
            "residual_updates": residual_updates,
            "ffn_updates": [
                {
                    "layer": index,
                    "layer_type": layer_types[index],
                    **metrics,
                }
                for index, metrics in enumerate(ffn_update_metrics)
            ],
            "ffn_residual_updates": ffn_residual_updates,
            "layer_stages": layer_stages,
            "stage_transitions": stage_transitions,
            "attention_head_contributions": attention_heads,
        }


@contextmanager
def _bypass_residual_update(module: torch.nn.Module) -> Iterator[None]:
    """Temporarily zero a mixer or FFN residual update."""

    def hook(
        _module: Any,
        _inputs: Any,
        output: torch.Tensor | tuple[torch.Tensor, Any],
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        if isinstance(output, tuple):
            update, cache = output
            return torch.zeros_like(update), cache
        return torch.zeros_like(output)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _resolved_data(
    checkpoint: dict[str, Any],
    model_config: ModelConfig,
    *,
    data_path: str | Path | None,
    synthetic: bool,
    synthetic_tokens: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = checkpoint.get("resolved_config", {})
    saved_data = resolved.get("data", {}) if isinstance(resolved, dict) else {}
    data_config = dict(saved_data) if isinstance(saved_data, dict) else {}
    data_config.setdefault("vocab_size", model_config.vocab_size)
    data_config.setdefault("seed", 1337)
    data_config.setdefault("pattern_length", 256)
    data_config.setdefault("val_tokens", 100_000)

    if data_path is not None:
        data_config["synthetic"] = False
        data_config["val_path"] = str(data_path)
    elif synthetic:
        data_config["synthetic"] = True
        if synthetic_tokens is not None:
            data_config["val_tokens"] = synthetic_tokens
    elif "synthetic" not in data_config:
        raise ConfigurationError(
            "checkpoint has no data configuration; pass --data or --synthetic"
        )

    if bool(data_config.get("synthetic", False)):
        provenance: dict[str, Any] = {
            "kind": "synthetic",
            "num_tokens": int(data_config["val_tokens"]),
            "vocab_size": int(data_config["vocab_size"]),
            "seed": int(data_config.get("seed", 0)) + 1,
            "pattern_length": int(data_config.get("pattern_length", 256)),
        }
    else:
        expected = None if data_path is not None else int(data_config["val_tokens"])
        provenance = verify_token_manifest(
            data_config["val_path"], "val", expected_num_tokens=expected
        )
    return data_config, provenance


@torch.no_grad()
def run_rank_analysis(
    checkpoint_path: str | Path,
    *,
    data_path: str | Path | None = None,
    synthetic: bool = False,
    synthetic_tokens: int | None = None,
    batch_size: int = 1,
    sequence_length: int | None = None,
    rank_batches: int = 2,
    rank_samples_per_batch: int = 512,
    head_samples_per_batch: int = 128,
    rank_tolerance: float = 1e-6,
    sensitivity_batches: int = 2,
    skip_sensitivity: bool = False,
    include_ffn_sensitivity: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run stage-rank, residual, head-diversity, and sensitivity diagnostics."""

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = ModelConfig.from_dict(dict(checkpoint["model_config"]))
    if sequence_length is None:
        sequence_length = model_config.context_length
    positive_values = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "rank_batches": rank_batches,
        "rank_samples_per_batch": rank_samples_per_batch,
        "head_samples_per_batch": head_samples_per_batch,
        "sensitivity_batches": sensitivity_batches,
    }
    for name, value in positive_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive integer")
    if sequence_length > model_config.context_length:
        raise ConfigurationError(
            f"sequence_length exceeds model context ({sequence_length} > "
            f"{model_config.context_length})"
        )
    if not 0 < rank_tolerance < 1:
        raise ConfigurationError("rank_tolerance must be in (0, 1)")

    runtime = resolve_runtime(device, dtype)
    model = DecoderLM(model_config).to(runtime.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    data_config, data_provenance = _resolved_data(
        checkpoint,
        model_config,
        data_path=data_path,
        synthetic=synthetic,
        synthetic_tokens=synthetic_tokens,
    )
    stream = create_token_stream(data_config, "val")

    collector = _LayerDiagnostics(
        model,
        maximum_rank_rows=rank_samples_per_batch,
        maximum_head_rows=head_samples_per_batch,
        rank_tolerance=rank_tolerance,
    )
    collector.install()
    observed_rank_batches = 0
    try:
        iterator = stream.sequential_batches(
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_batches=rank_batches,
            device=runtime.device,
        )
        for input_ids, _labels in iterator:
            with autocast_context(runtime):
                model(input_ids, logits_to_keep=1)
            observed_rank_batches += 1
    finally:
        collector.remove()
    if observed_rank_batches == 0:
        raise ValueError("analysis stream yielded no rank-analysis batches")
    rank_metrics = collector.finalize()

    sensitivity: dict[str, Any] | None = None
    if not skip_sensitivity:
        baseline = evaluate_language_model(
            model,
            stream,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_batches=sensitivity_batches,
            runtime=runtime,
        )
        mixer_ablations: list[dict[str, Any]] = []
        for index, block in enumerate(model.blocks):
            if block.layer_type == "ffn_only":
                continue
            with _bypass_residual_update(block.mixer):
                metrics = evaluate_language_model(
                    model,
                    stream,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    max_batches=sensitivity_batches,
                    runtime=runtime,
                )
            mixer_ablations.append(
                {
                    "layer": index,
                    "layer_type": block.layer_type,
                    "component": "mixer",
                    **metrics,
                    "loss_delta": metrics["loss"] - baseline["loss"],
                    "perplexity_ratio": metrics["perplexity"] / baseline["perplexity"],
                    "accuracy_delta": metrics["accuracy"] - baseline["accuracy"],
                }
            )
        ffn_ablations: list[dict[str, Any]] = []
        if include_ffn_sensitivity:
            for index, block in enumerate(model.blocks):
                with _bypass_residual_update(block.ffn):
                    metrics = evaluate_language_model(
                        model,
                        stream,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        max_batches=sensitivity_batches,
                        runtime=runtime,
                    )
                ffn_ablations.append(
                    {
                        "layer": index,
                        "layer_type": block.layer_type,
                        "component": "ffn",
                        **metrics,
                        "loss_delta": metrics["loss"] - baseline["loss"],
                        "perplexity_ratio": metrics["perplexity"] / baseline["perplexity"],
                        "accuracy_delta": metrics["accuracy"] - baseline["accuracy"],
                    }
                )
        sensitivity = {
            "intervention": "zero one residual update while retaining all other sublayers",
            "baseline": baseline,
            "mixer_ablations": mixer_ablations,
            "ffn_ablations": ffn_ablations,
            # Retain the original field so existing notebooks remain readable.
            "attention_layer_ablations": [
                item for item in mixer_ablations if item["layer_type"] == "attention"
            ],
        }

    result: dict[str, Any] = {
        "event": "rank_analysis",
        "schema_version": 3,
        "time": utc_now(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "model": {
            "parameter_count": parameter_count(model),
            "n_layers": model_config.n_layers,
            "d_model": model_config.d_model,
            "n_heads": model_config.n_heads,
            "context_length": model_config.context_length,
            "layer_types": list(model_config.layer_types),
            "ffn_type": model_config.ffn_type,
            "ffn_hidden_size": model_config.ffn_hidden_size,
            "ffn_hidden_sizes": model_config.effective_ffn_hidden_sizes,
            "ffn_total_hidden_size": sum(
                model_config.effective_ffn_hidden_sizes
            ),
            "residual_init_depth": model_config.effective_residual_init_depth,
        },
        "settings": {
            "device": str(runtime.device),
            "dtype": runtime.dtype_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "rank_batches_requested": rank_batches,
            "rank_batches_observed": observed_rank_batches,
            "rank_samples_per_batch": rank_samples_per_batch,
            "head_samples_per_batch": head_samples_per_batch,
            "rank_tolerance": rank_tolerance,
            "sensitivity_batches": 0 if skip_sensitivity else sensitivity_batches,
            "include_ffn_sensitivity": include_ffn_sensitivity,
        },
        "definitions": {
            "entropy_effective_rank": "exp(entropy(normalized covariance eigenvalues))",
            "participation_ratio": "sum(eigenvalues)^2 / sum(eigenvalues^2)",
            "stable_rank": "sum(eigenvalues) / max(eigenvalue)",
            "attention_head_contributions": (
                "per-head pre-output-projection values projected separately into "
                "the shared residual space"
            ),
            "layer_stages": (
                "centered channel-covariance spectra before and after each residual "
                "addition; mixer stages are omitted for FFN-only blocks"
            ),
        },
        "environment": environment_info(runtime),
        "data_provenance": data_provenance,
        "rank_metrics": rank_metrics,
        "layer_sensitivity": sensitivity,
    }
    if output is not None:
        save_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--data", help="headerless uint16 validation tokens")
    data_group.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--rank-batches", type=int, default=2)
    parser.add_argument("--rank-samples-per-batch", type=int, default=512)
    parser.add_argument("--head-samples-per-batch", type=int, default=128)
    parser.add_argument("--rank-tolerance", type=float, default=1e-6)
    parser.add_argument("--sensitivity-batches", type=int, default=2)
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="omit one-mixer-at-a-time loss sensitivity",
    )
    parser.add_argument(
        "--include-ffn-sensitivity",
        action="store_true",
        help="also zero each FFN update individually (adds one evaluation per layer)",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "bf16", "float32", "fp32"),
    )
    parser.add_argument("--output", help="optional JSON output path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_rank_analysis(
        args.checkpoint,
        data_path=args.data,
        synthetic=args.synthetic,
        synthetic_tokens=args.synthetic_tokens,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        rank_batches=args.rank_batches,
        rank_samples_per_batch=args.rank_samples_per_batch,
        head_samples_per_batch=args.head_samples_per_batch,
        rank_tolerance=args.rank_tolerance,
        sensitivity_batches=args.sensitivity_batches,
        skip_sensitivity=args.skip_sensitivity,
        include_ffn_sensitivity=args.include_ffn_sensitivity,
        device=args.device,
        dtype=args.dtype,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main", "run_rank_analysis"]
