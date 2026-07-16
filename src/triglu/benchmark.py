"""Benchmark training, prompt prefill, and cached autoregressive decode."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .model import DecoderLM, ModelConfig
from .runtime import (
    ConfigurationError,
    DEFAULT_TRAINING_CONFIG,
    autocast_context,
    environment_info,
    load_checkpoint,
    load_experiment_config,
    make_adamw,
    parameter_count,
    resolve_runtime,
    save_json,
    seed_everything,
    synchronize,
    utc_now,
)


def _measure(
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[list[float], int | None]:
    for _ in range(warmup):
        operation()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        synchronize(device)
        samples.append(time.perf_counter() - started)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return samples, peak_memory


def _throughput_result(
    *,
    samples: list[float],
    iterations: int,
    tokens_per_iteration: int,
    peak_memory: int | None,
) -> dict[str, Any]:
    elapsed = sum(samples)
    total_tokens = iterations * tokens_per_iteration
    result: dict[str, Any] = {
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "mean_iteration_ms": 1_000.0 * elapsed / iterations,
        "median_iteration_ms": 1_000.0 * statistics.median(samples),
        "iteration_ms": [1_000.0 * sample for sample in samples],
        "tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed,
        "median_tokens_per_second": tokens_per_iteration / statistics.median(samples),
    }
    if peak_memory is not None:
        result["peak_memory_allocated_bytes"] = peak_memory
    return result


def _cache_bytes(caches: Any) -> int:
    """Count allocated tensor payload bytes in unified per-layer caches."""

    if caches is None:
        return 0
    total = 0
    for layer_cache in caches:
        if layer_cache is None:
            continue
        for value in layer_cache[:2]:
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total


def _cache_used_bytes(caches: Any) -> int:
    """Count logically initialized KV bytes, excluding unused static capacity."""

    if caches is None:
        return 0
    total = 0
    for layer_cache in caches:
        if layer_cache is None:
            continue
        position = int(layer_cache[2])
        for value in layer_cache[:2]:
            if isinstance(value, torch.Tensor):
                capacity = value.size(-2)
                if position > capacity:
                    raise ValueError("cache position exceeds tensor capacity")
                values_per_position = value.numel() // capacity
                total += values_per_position * position * value.element_size()
    return total


def _load_model_inputs(
    *,
    config_path: str | Path | None,
    checkpoint_path: str | Path | None,
) -> tuple[ModelConfig, dict[str, Any], dict[str, Any] | None]:
    if config_path is not None:
        experiment = load_experiment_config(config_path)
        return ModelConfig.from_dict(dict(experiment["model"])), experiment, None
    if checkpoint_path is None:
        raise ConfigurationError("pass either --config or --checkpoint")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = ModelConfig.from_dict(dict(checkpoint["model_config"]))
    resolved = checkpoint.get("resolved_config")
    experiment = dict(resolved) if isinstance(resolved, Mapping) else {}
    return model_config, experiment, checkpoint


def run_benchmarks(
    *,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    batch_size: int = 4,
    gradient_accumulation_steps: int | None = None,
    context_length: int | None = None,
    sequence_length: int | None = None,
    prompt_length: int | None = None,
    decode_tokens: int | None = None,
    warmup: int = 3,
    iterations: int = 10,
    device: str | None = None,
    dtype: str | None = None,
    compile_model: bool = False,
    label: str | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run the three benchmarks and return a JSON-serializable result."""

    model_config, experiment, checkpoint = _load_model_inputs(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    configured_context_length = model_config.context_length
    if context_length is not None:
        if (
            isinstance(context_length, bool)
            or not isinstance(context_length, int)
            or context_length <= 0
        ):
            raise ConfigurationError("context_length must be a positive integer")
        model_values = model_config.to_dict()
        model_values["context_length"] = context_length
        model_config = ModelConfig.from_dict(model_values)
    saved_training = experiment.get("training", {})
    if not isinstance(saved_training, Mapping):
        saved_training = {}
    requested_device = device if device is not None else str(saved_training.get("device", "auto"))
    requested_dtype = dtype if dtype is not None else str(saved_training.get("dtype", "auto"))
    runtime = resolve_runtime(requested_device, requested_dtype)

    if batch_size <= 0:
        raise ConfigurationError("batch_size must be positive")
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = int(
            saved_training.get("gradient_accumulation_steps", 1)
        )
    if gradient_accumulation_steps <= 0:
        raise ConfigurationError("gradient_accumulation_steps must be positive")
    if warmup < 0 or iterations <= 0:
        raise ConfigurationError("warmup must be non-negative and iterations must be positive")
    if sequence_length is None:
        sequence_length = min(256, model_config.context_length)
    if sequence_length <= 0 or sequence_length > model_config.context_length:
        raise ConfigurationError(
            f"sequence_length must be in [1, {model_config.context_length}]"
        )
    if decode_tokens is None:
        decode_tokens = min(32, max(1, model_config.context_length // 2))
    if decode_tokens <= 0 or decode_tokens >= model_config.context_length:
        raise ConfigurationError(
            "decode_tokens must be positive and smaller than model.context_length"
        )
    if prompt_length is None:
        prompt_length = min(sequence_length, model_config.context_length - decode_tokens)
    if prompt_length <= 0 or prompt_length + decode_tokens > model_config.context_length:
        raise ConfigurationError(
            "prompt_length must be positive and prompt_length + decode_tokens must not "
            "exceed model.context_length"
        )

    seed = int(saved_training.get("seed", 1337))
    seed_everything(seed)
    model = DecoderLM(model_config).to(runtime.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)

    benchmark_model: torch.nn.Module = model
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires a PyTorch build with torch.compile")
        benchmark_model = torch.compile(model)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    prompt = torch.randint(
        model_config.vocab_size,
        (batch_size, prompt_length),
        generator=generator,
        dtype=torch.long,
    ).to(runtime.device)

    # Prefill uses the compiled model when requested; its shape is static.
    benchmark_model.eval()

    @torch.inference_mode()
    def prefill_operation() -> None:
        with autocast_context(runtime):
            output = benchmark_model(
                prompt,
                use_cache=True,
                cache_position=0,
                logits_to_keep=1,
            )
        if output.caches is None:
            raise RuntimeError("model did not return caches for use_cache=True")

    prefill_samples, prefill_memory = _measure(
        prefill_operation,
        warmup=warmup,
        iterations=iterations,
        device=runtime.device,
    )
    prefill_result = _throughput_result(
        samples=prefill_samples,
        iterations=iterations,
        tokens_per_iteration=batch_size * prompt_length,
        peak_memory=prefill_memory,
    )

    # Keep cached decode eager. The fixed-capacity cache avoids both per-token
    # recompilation and repeated torch.cat copies of the complete KV history.
    model.eval()

    decode_inputs = torch.randint(
        model_config.vocab_size,
        (decode_tokens, batch_size, 1),
        generator=generator,
        dtype=torch.long,
    ).to(runtime.device)

    static_caches = model.allocate_static_cache(
        batch_size,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    with torch.inference_mode(), autocast_context(runtime):
        initial_decode_output = model(
            prompt,
            caches=static_caches,
            use_cache=True,
            cache_position=0,
            logits_to_keep=1,
        )
    initial_caches = initial_decode_output.caches
    if initial_caches is None:
        raise RuntimeError("model did not return caches for use_cache=True")
    cache_capacity_bytes = _cache_bytes(initial_caches)
    initial_cache_bytes = _cache_used_bytes(initial_caches)
    final_caches = initial_caches
    # Prompt logits are not live during decode timing in a real generation loop.
    del initial_decode_output, static_caches

    @torch.inference_mode()
    def decode_operation() -> None:
        nonlocal final_caches
        # The seed tuple keeps the prompt's logical position. Each iteration
        # reuses the same capacity and overwrites only the decoded suffix.
        caches = initial_caches
        with autocast_context(runtime):
            for token_index in range(decode_tokens):
                output = model(
                    decode_inputs[token_index],
                    caches=caches,
                    use_cache=True,
                    cache_position=prompt_length + token_index,
                    logits_to_keep=1,
                )
                caches = output.caches
                if caches is None:
                    raise RuntimeError("model stopped returning caches during decode")
        final_caches = caches

    decode_samples, decode_memory = _measure(
        decode_operation,
        warmup=warmup,
        iterations=iterations,
        device=runtime.device,
    )
    # Byte accounting is measurement bookkeeping; keep it outside the timed loop.
    final_cache_bytes = _cache_used_bytes(final_caches)
    decode_result = _throughput_result(
        samples=decode_samples,
        iterations=iterations,
        tokens_per_iteration=batch_size * decode_tokens,
        peak_memory=decode_memory,
    )
    decode_elapsed = sum(decode_samples)
    decode_result["mean_token_step_ms"] = (
        1_000.0 * decode_elapsed / (iterations * decode_tokens)
    )
    decode_result["initial_cache_bytes"] = initial_cache_bytes
    decode_result["final_cache_bytes"] = final_cache_bytes
    decode_result["cache_capacity_bytes"] = cache_capacity_bytes
    decode_result["cache_bytes_per_batch_token_at_final_length"] = (
        final_cache_bytes / (batch_size * (prompt_length + decode_tokens))
    )

    # Release decode-only tensors before resetting peak memory for training.
    del decode_operation, initial_caches, final_caches, prompt, decode_inputs
    if runtime.device.type == "cuda":
        torch.cuda.empty_cache()

    # Allocate training-only batches after inference peak-memory measurements so
    # those measurements contain only the model, inference inputs, and caches.
    training_inputs = torch.randint(
        model_config.vocab_size,
        (gradient_accumulation_steps, batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to(runtime.device)
    training_labels = torch.randint(
        model_config.vocab_size,
        (gradient_accumulation_steps, batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to(runtime.device)

    optimizer_config = dict(DEFAULT_TRAINING_CONFIG)
    optimizer_config.update(saved_training)
    optimizer, fused_optimizer = make_adamw(model.parameters(), optimizer_config, runtime)
    benchmark_model.train()

    def training_operation() -> None:
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(gradient_accumulation_steps):
            with autocast_context(runtime):
                output = benchmark_model(
                    training_inputs[micro_step], labels=training_labels[micro_step]
                )
                if output.loss is None:
                    raise RuntimeError("model did not return a training loss")
                scaled_loss = output.loss / gradient_accumulation_steps
            scaled_loss.backward()
        grad_clip = float(optimizer_config.get("grad_clip", 1.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    training_samples, training_memory = _measure(
        training_operation,
        warmup=warmup,
        iterations=iterations,
        device=runtime.device,
    )
    training_result = _throughput_result(
        samples=training_samples,
        iterations=iterations,
        tokens_per_iteration=(
            batch_size * sequence_length * gradient_accumulation_steps
        ),
        peak_memory=training_memory,
    )

    result: dict[str, Any] = {
        "event": "benchmark",
        "benchmark_label": label,
        "time": utc_now(),
        "source": str(checkpoint_path if checkpoint_path is not None else config_path),
        "checkpoint_step": int(checkpoint.get("step", 0)) if checkpoint is not None else None,
        "device": environment_info(runtime),
        "model": {
            "parameters": parameter_count(model),
            "n_layers": model_config.n_layers,
            "d_model": model_config.d_model,
            "configured_context_length": configured_context_length,
            "benchmark_context_length": model_config.context_length,
            "layer_types": list(model_config.layer_types),
        },
        "settings": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "training_sequence_length": sequence_length,
            "prompt_length": prompt_length,
            "decode_tokens": decode_tokens,
            "warmup": warmup,
            "iterations": iterations,
            "compile": compile_model,
            "cached_decode_compile": False,
            "cached_decode_kv_cache": "preallocated",
            "fused_optimizer": fused_optimizer,
        },
        "training": training_result,
        "prefill": prefill_result,
        "cached_decode": decode_result,
    }
    if output is not None:
        save_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        help="YAML experiment configuration (randomly initialized model)",
    )
    source.add_argument("--checkpoint", help="trained checkpoint")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument(
        "--context-length",
        type=int,
        help=(
            "benchmark-only context capacity override; changes RoPE position range and "
            "KV-cache capacity without changing learned parameter shapes"
        ),
    )
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--decode-tokens", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "bf16", "float32", "fp32"),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--label",
        help="optional benchmark-family label used when aggregating related sweeps",
    )
    parser.add_argument("--output", help="optional JSON result path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_benchmarks(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        context_length=args.context_length,
        sequence_length=args.sequence_length,
        prompt_length=args.prompt_length,
        decode_tokens=args.decode_tokens,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile,
        label=args.label,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
