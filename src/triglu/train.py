"""Single-GPU training entry point for controlled TriGLU ablations."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .model import DecoderLM, ModelConfig
from .runtime import (
    CHECKPOINT_FORMAT_VERSION,
    ConfigurationError,
    append_jsonl,
    atomic_torch_save,
    autocast_context,
    capture_rng_state,
    create_token_stream,
    environment_info,
    evaluate_language_model,
    jsonable,
    learning_rate_for_step,
    load_checkpoint,
    load_experiment_config,
    make_adamw,
    parameter_count,
    record_data_provenance,
    resolve_runtime,
    restore_rng_state,
    verify_data_provenance,
    save_json,
    save_yaml,
    seed_everything,
    set_optimizer_learning_rate,
    synchronize,
    utc_now,
)


def _model_config_dict(config: ModelConfig) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    raise TypeError("ModelConfig must provide to_dict() for self-contained checkpoints")


def _save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    model: DecoderLM,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    resolved_config: Mapping[str, Any],
    train_generator: torch.Generator,
    tokens_seen: int,
    best_val_loss: float | None,
    save_numbered: bool = True,
) -> Path:
    values = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at": utc_now(),
        # `step` is the number of completed optimizer steps and therefore the
        # next zero-based step index after resume.
        "step": step,
        "tokens_seen": tokens_seen,
        "model_config": _model_config_dict(model_config),
        "resolved_config": jsonable(resolved_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": capture_rng_state(train_generator),
        "best_val_loss": best_val_loss,
    }
    latest_path = output_dir / "latest.pt"
    if save_numbered:
        numbered_path = output_dir / f"checkpoint_step_{step:08d}.pt"
        atomic_torch_save(values, numbered_path)
    atomic_torch_save(values, latest_path)
    return numbered_path if save_numbered else latest_path


def _configs_match(left: ModelConfig, right_values: Mapping[str, Any]) -> bool:
    right = ModelConfig.from_dict(dict(right_values))
    return _model_config_dict(left) == _model_config_dict(right)


def _resume_config_differences(
    checkpoint: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """Return scientific config changes that would invalidate continuation."""

    saved = checkpoint.get("resolved_config")
    if not isinstance(saved, Mapping):
        return ["checkpoint.resolved_config is missing"]
    differences: list[str] = []
    for section in ("data", "training"):
        saved_section = saved.get(section)
        current_section = current.get(section)
        if not isinstance(saved_section, Mapping):
            differences.append(f"checkpoint.resolved_config.{section} is missing")
            continue
        if not isinstance(current_section, Mapping):
            differences.append(f"current resolved config {section} is missing")
            continue
        # log_interval only changes console/metric cadence, never the science.
        ignored = (
            {"output_dir", "resume", "log_interval"}
            if section == "training"
            else set()
        )
        keys = (set(saved_section) | set(current_section)) - ignored
        for key in sorted(keys):
            saved_value = saved_section.get(key, "<missing>")
            current_value = current_section.get(key, "<missing>")
            if saved_value != current_value:
                differences.append(
                    f"{section}.{key}: checkpoint={saved_value!r}, "
                    f"current={current_value!r}"
                )
    return differences


def _resume_provenance_differences(
    checkpoint: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """Compare content identities, not run-local manifest-copy paths."""

    resolved = checkpoint.get("resolved_config")
    saved = resolved.get("data_provenance") if isinstance(resolved, Mapping) else None
    if not isinstance(saved, Mapping):
        return ["checkpoint has no verified data_provenance"]
    if saved.get("kind") != current.get("kind"):
        return [
            f"data kind: checkpoint={saved.get('kind')!r}, "
            f"current={current.get('kind')!r}"
        ]
    if current.get("kind") == "synthetic":
        return [] if saved == current else ["synthetic data provenance changed"]

    saved_splits = saved.get("splits")
    current_splits = current.get("splits")
    if not isinstance(saved_splits, Mapping) or not isinstance(current_splits, Mapping):
        return ["prepared-data split provenance is missing"]
    differences: list[str] = []
    if set(saved_splits) != set(current_splits):
        differences.append(
            "data_provenance split set: "
            f"checkpoint={sorted(saved_splits)!r}, current={sorted(current_splits)!r}"
        )
    stable_fields = ("sha256", "manifest_sha256", "num_tokens", "num_bytes")
    for split, current_split in current_splits.items():
        saved_split = saved_splits.get(split)
        if not isinstance(saved_split, Mapping) or not isinstance(current_split, Mapping):
            differences.append(f"data_provenance.splits.{split} is missing")
            continue
        for field in stable_fields:
            if saved_split.get(field) != current_split.get(field):
                differences.append(
                    f"data_provenance.splits.{split}.{field}: "
                    f"checkpoint={saved_split.get(field)!r}, "
                    f"current={current_split.get(field)!r}"
                )
    return differences


def _format_compact(value: int | float) -> str:
    magnitude = abs(float(value))
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= scale:
            return f"{float(value) / scale:.2f}{suffix}"
    return f"{float(value):.0f}"


def _format_bytes(value: int | float) -> str:
    amount = float(value)
    for scale, suffix in (
        (1024**3, "GiB"),
        (1024**2, "MiB"),
        (1024, "KiB"),
    ):
        if abs(amount) >= scale:
            return f"{amount / scale:.2f} {suffix}"
    return f"{amount:.0f} B"


def _format_progress(values: Mapping[str, Any], max_steps: int | None) -> str:
    step = int(values.get("step", 0))
    if max_steps is None or max_steps <= 0:
        return f"step {step}"
    width = len(str(max_steps))
    percent = 100.0 * step / max_steps
    return f"step {step:>{width}}/{max_steps} ({percent:5.1f}%)"


def _format_console_event(
    values: Mapping[str, Any], *, max_steps: int | None = None
) -> str:
    """Format one concise terminal line without changing metrics.jsonl."""

    event = str(values.get("event", "event"))
    label = {
        "evaluation": "eval",
        "checkpoint": "checkpoint",
    }.get(event, event)
    progress = _format_progress(values, max_steps)

    if event in {"start", "resume"}:
        compile_mode = "compile on" if values.get("compile") else "compile off"
        fused_mode = "fused AdamW" if values.get("fused_optimizer") else "AdamW"
        return " | ".join(
            (
                f"{label:<10} {progress}",
                str(values.get("device", "unknown")),
                str(values.get("dtype", "unknown")),
                f"{_format_compact(int(values.get('parameters', 0)))} params",
                compile_mode,
                fused_mode,
            )
        )

    if event == "train":
        fields = [
            f"{label:<10} {progress}",
            f"tokens {_format_compact(int(values['tokens_seen']))}",
            f"loss {float(values['loss']):.4f}",
            f"ppl {float(values['perplexity']):.2f}",
            f"lr {float(values['learning_rate']):.3e}",
        ]
        if values.get("grad_norm") is not None:
            fields.append(f"grad {float(values['grad_norm']):.3f}")
        fields.append(f"{_format_compact(float(values['tokens_per_second']))} tok/s")
        if values.get("max_memory_allocated_bytes") is not None:
            fields.append(f"mem {_format_bytes(int(values['max_memory_allocated_bytes']))}")
        return " | ".join(fields)

    if event == "evaluation":
        return " | ".join(
            (
                f"{label:<10} {progress}",
                f"tokens {_format_compact(int(values['tokens_seen']))}",
                f"loss {float(values['loss']):.4f}",
                f"ppl {float(values['perplexity']):.2f}",
                f"acc {100.0 * float(values['accuracy']):.2f}%",
                f"eval {_format_compact(int(values['tokens']))} tokens",
            )
        )

    if event == "checkpoint":
        return f"{label:<10} {progress} | {values.get('path', '<unknown path>')}"

    if event == "complete":
        best = values.get("best_val_loss")
        best_text = "n/a" if best is None else f"{float(best):.4f}"
        return " | ".join(
            (
                f"{label:<10} {progress}",
                f"tokens {_format_compact(int(values['tokens_seen']))}",
                f"best val {best_text}",
                str(values.get("checkpoint", "<unknown path>")),
            )
        )

    return json.dumps(jsonable(values), sort_keys=True)


def _print_event(values: Mapping[str, Any], *, max_steps: int | None = None) -> None:
    print(_format_console_event(values, max_steps=max_steps), flush=True)


_RUN_ARTIFACT_PATTERNS = (
    "metrics.jsonl",
    "latest.pt",
    "checkpoint_step_*.pt",
    "resolved_config.yaml",
    "environment.json",
    "data_provenance.json",
    "data_manifest*.json",
)


def _existing_run_artifacts(output_dir: Path) -> list[Path]:
    """Return run records already present in a directory."""

    if not output_dir.is_dir():
        return []
    found: list[Path] = []
    for pattern in _RUN_ARTIFACT_PATTERNS:
        found.extend(sorted(output_dir.glob(pattern)))
    return found


def run_training(
    config_path: str | Path,
    *,
    resume_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    device_override: str | None = None,
    dtype_override: str | None = None,
    compile_override: bool | None = None,
    seed_override: int | None = None,
    overwrite_run: bool = False,
) -> dict[str, Any]:
    """Train an experiment and return a compact summary for tests/callers."""

    resolved_config = load_experiment_config(config_path)
    training = resolved_config["training"]
    data_config = resolved_config["data"]
    if resume_override is not None:
        training["resume"] = str(resume_override)
    if output_dir_override is not None:
        training["output_dir"] = str(output_dir_override)
    if device_override is not None:
        training["device"] = device_override
    if dtype_override is not None:
        training["dtype"] = dtype_override
    if compile_override is not None:
        training["compile"] = compile_override
    if seed_override is not None:
        if isinstance(seed_override, bool) or seed_override < 0:
            raise ConfigurationError("seed override must be a non-negative integer")
        training["seed"] = seed_override

    runtime = resolve_runtime(str(training["device"]), str(training["dtype"]))
    output_dir = Path(training["output_dir"])
    metrics_path = output_dir / "metrics.jsonl"
    resume_path = training.get("resume")

    # A completed or in-progress run's records are scientific evidence; a fresh
    # invocation must never clobber them implicitly.
    if overwrite_run and resume_path:
        raise ConfigurationError("overwrite_run cannot be combined with resume")
    existing_artifacts = _existing_run_artifacts(output_dir)
    if not resume_path and existing_artifacts:
        if not overwrite_run:
            names = ", ".join(path.name for path in existing_artifacts[:3])
            if len(existing_artifacts) > 3:
                names += ", ..."
            raise ConfigurationError(
                f"output directory {output_dir} already contains run records "
                f"({names}); pass --resume to continue that run or "
                "--overwrite-run to discard it and start fresh"
            )
        for artifact in existing_artifacts:
            artifact.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig.from_dict(dict(resolved_config["model"]))
    start_step = 0
    tokens_seen = 0
    best_val_loss: float | None = None
    checkpoint: dict[str, Any] | None = None
    if resume_path:
        checkpoint = load_checkpoint(resume_path, map_location="cpu")
        if not _configs_match(model_config, checkpoint["model_config"]):
            raise ConfigurationError(
                "checkpoint model_config does not match the requested experiment model"
            )
        resume_differences = _resume_config_differences(checkpoint, resolved_config)
        if resume_differences:
            formatted = "\n  - ".join(resume_differences)
            raise ConfigurationError(
                "resume would change scientific data/training settings:\n  - " + formatted
            )
        if "optimizer" not in checkpoint:
            raise ValueError("training resume checkpoint is missing optimizer state")
        start_step = int(checkpoint.get("step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        saved_best = checkpoint.get("best_val_loss")
        best_val_loss = float(saved_best) if saved_best is not None else None
        if start_step < 0 or start_step > int(training["max_steps"]):
            raise ConfigurationError(
                f"checkpoint step {start_step} is outside configured "
                f"max_steps={training['max_steps']}"
            )
        if start_step == int(training["max_steps"]):
            raise ConfigurationError(
                f"checkpoint already completed max_steps={training['max_steps']}; "
                "resuming would only append duplicate events to metrics.jsonl"
            )

    provenance_splits = ["train"]
    if int(training["eval_interval"]) > 0:
        provenance_splits.append("val")
    # Verify data identity before writing anything into the run directory so a
    # rejected resume cannot alter an existing run's recorded provenance.
    data_provenance = verify_data_provenance(data_config, splits=provenance_splits)
    if checkpoint is not None:
        provenance_differences = _resume_provenance_differences(
            checkpoint, data_provenance
        )
        if provenance_differences:
            formatted = "\n  - ".join(provenance_differences)
            raise ConfigurationError(
                "resume data content identity changed:\n  - " + formatted
            )
    data_provenance = record_data_provenance(
        data_config, output_dir, splits=provenance_splits, verified=data_provenance
    )

    seed = int(training["seed"])
    seed_everything(seed)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(seed + 1)

    model = DecoderLM(model_config).to(runtime.device)
    optimizer, fused_optimizer = make_adamw(model.parameters(), training, runtime)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])

    train_stream = create_token_stream(data_config, "train")
    validation_stream = None
    if int(training["eval_interval"]) > 0:
        validation_stream = create_token_stream(data_config, "val")

    train_model: torch.nn.Module = model
    compile_enabled = bool(training.get("compile", False))
    if compile_enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("training.compile requires a PyTorch build with torch.compile")
        train_model = torch.compile(model)

    # Restore only after model/optimizer construction, which consume RNG state.
    if checkpoint is not None and "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"], train_generator)

    run_config = {
        **resolved_config,
        "data_provenance": data_provenance,
        "runtime": {
            "device": str(runtime.device),
            "dtype": runtime.dtype_name,
            "compile": compile_enabled,
            "fused_optimizer": fused_optimizer,
            "parameter_count": parameter_count(model),
        },
    }
    save_yaml(output_dir / "resolved_config.yaml", run_config)
    save_json(output_dir / "environment.json", environment_info(runtime))

    batch_size = int(training["batch_size"])
    sequence_length = int(training["sequence_length"])
    accumulation_steps = int(training["gradient_accumulation_steps"])
    max_steps = int(training["max_steps"])
    tokens_per_step = batch_size * sequence_length * accumulation_steps

    start_event = {
        "event": "start" if start_step == 0 else "resume",
        "time": utc_now(),
        "step": start_step,
        "tokens_seen": tokens_seen,
        "parameters": parameter_count(model),
        "device": str(runtime.device),
        "dtype": runtime.dtype_name,
        "compile": compile_enabled,
        "fused_optimizer": fused_optimizer,
    }
    append_jsonl(metrics_path, start_event)
    _print_event(start_event, max_steps=max_steps)

    train_model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_steps = 0
    window_tokens = 0
    last_eval_step = -1
    last_saved_step = -1
    synchronize(runtime.device)
    window_start = time.perf_counter()

    for step_index in range(start_step, max_steps):
        learning_rate = learning_rate_for_step(step_index, training)
        set_optimizer_learning_rate(optimizer, learning_rate)
        step_loss = 0.0

        for _ in range(accumulation_steps):
            input_ids, labels = train_stream.sample_batch(
                batch_size=batch_size,
                sequence_length=sequence_length,
                generator=train_generator,
                device=runtime.device,
            )
            with autocast_context(runtime):
                output = train_model(input_ids, labels=labels)
                if output.loss is None:
                    raise RuntimeError("model did not return a loss when labels were supplied")
                micro_loss = output.loss
                scaled_loss = micro_loss / accumulation_steps
            scaled_loss.backward()
            step_loss += float(micro_loss.detach().float()) / accumulation_steps

        grad_clip = float(training["grad_clip"])
        grad_norm: float | None = None
        if grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm = float(norm.detach().float())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        completed_step = step_index + 1
        tokens_seen += tokens_per_step
        running_loss += step_loss
        running_steps += 1
        window_tokens += tokens_per_step

        if completed_step % int(training["log_interval"]) == 0 or completed_step == max_steps:
            synchronize(runtime.device)
            elapsed = max(time.perf_counter() - window_start, 1.0e-12)
            train_event: dict[str, Any] = {
                "event": "train",
                "time": utc_now(),
                "step": completed_step,
                "tokens_seen": tokens_seen,
                "loss": running_loss / running_steps,
                "perplexity": math.exp(min(running_loss / running_steps, 80.0)),
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
                "tokens_per_second": window_tokens / elapsed,
            }
            if runtime.device.type == "cuda":
                train_event["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    runtime.device
                )
            append_jsonl(metrics_path, train_event)
            _print_event(train_event, max_steps=max_steps)
            running_loss = 0.0
            running_steps = 0
            window_tokens = 0
            window_start = time.perf_counter()

        eval_interval = int(training["eval_interval"])
        if validation_stream is not None and completed_step % eval_interval == 0:
            synchronize(runtime.device)
            overhead_started = time.perf_counter()
            metrics = evaluate_language_model(
                model,
                validation_stream,
                batch_size=batch_size,
                sequence_length=sequence_length,
                max_batches=int(training["eval_batches"]),
                runtime=runtime,
            )
            eval_event = {
                "event": "evaluation",
                "time": utc_now(),
                "step": completed_step,
                "tokens_seen": tokens_seen,
                **metrics,
            }
            append_jsonl(metrics_path, eval_event)
            _print_event(eval_event, max_steps=max_steps)
            best_val_loss = (
                metrics["loss"]
                if best_val_loss is None
                else min(best_val_loss, float(metrics["loss"]))
            )
            last_eval_step = completed_step
            train_model.train()
            # Do not charge deterministic evaluation to training throughput.
            window_start += time.perf_counter() - overhead_started

        checkpoint_interval = int(training["checkpoint_interval"])
        if (
            checkpoint_interval > 0
            and completed_step % checkpoint_interval == 0
            and completed_step < max_steps
        ):
            synchronize(runtime.device)
            overhead_started = time.perf_counter()
            saved_path = _save_checkpoint(
                output_dir=output_dir,
                step=completed_step,
                model=model,
                model_config=model_config,
                optimizer=optimizer,
                resolved_config=run_config,
                train_generator=train_generator,
                tokens_seen=tokens_seen,
                best_val_loss=best_val_loss,
            )
            checkpoint_event = {
                "event": "checkpoint",
                "time": utc_now(),
                "step": completed_step,
                "path": str(saved_path),
            }
            append_jsonl(metrics_path, checkpoint_event)
            _print_event(checkpoint_event, max_steps=max_steps)
            last_saved_step = completed_step
            # Checkpoint serialization may synchronize CUDA and is not model work.
            window_start += time.perf_counter() - overhead_started

    # Short smoke runs still get a validation result when evaluation is enabled.
    if validation_stream is not None and last_eval_step != max_steps:
        metrics = evaluate_language_model(
            model,
            validation_stream,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_batches=int(training["eval_batches"]),
            runtime=runtime,
        )
        eval_event = {
            "event": "evaluation",
            "time": utc_now(),
            "step": max_steps,
            "tokens_seen": tokens_seen,
            **metrics,
        }
        append_jsonl(metrics_path, eval_event)
        _print_event(eval_event, max_steps=max_steps)
        best_val_loss = (
            metrics["loss"]
            if best_val_loss is None
            else min(best_val_loss, float(metrics["loss"]))
        )

    if last_saved_step != max_steps:
        saved_path = _save_checkpoint(
            output_dir=output_dir,
            step=max_steps,
            model=model,
            model_config=model_config,
            optimizer=optimizer,
            resolved_config=run_config,
            train_generator=train_generator,
            tokens_seen=tokens_seen,
            best_val_loss=best_val_loss,
            save_numbered=False,
        )
        checkpoint_event = {
            "event": "checkpoint",
            "time": utc_now(),
            "step": max_steps,
            "path": str(saved_path),
        }
        append_jsonl(metrics_path, checkpoint_event)
        _print_event(checkpoint_event, max_steps=max_steps)

    complete_event = {
        "event": "complete",
        "time": utc_now(),
        "step": max_steps,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "checkpoint": str(output_dir / "latest.pt"),
    }
    append_jsonl(metrics_path, complete_event)
    _print_event(complete_event, max_steps=max_steps)
    return complete_event


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--resume", help="checkpoint to resume (overrides training.resume)")
    parser.add_argument("--output-dir", help="run output directory override")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), help="device override")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "bf16", "float32", "fp32"),
        help="numeric precision override",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable torch.compile",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="training/model/data-sampling seed override",
    )
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help=(
            "discard run records already present in the output directory and "
            "start fresh (fresh runs refuse to clobber them otherwise)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(
        args.config,
        resume_override=args.resume,
        output_dir_override=args.output_dir,
        device_override=args.device,
        dtype_override=args.dtype,
        compile_override=args.compile,
        seed_override=args.seed,
        overwrite_run=args.overwrite_run,
    )


if __name__ == "__main__":
    main()
