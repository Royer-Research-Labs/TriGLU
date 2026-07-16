"""Evaluate a TriGLU experiment checkpoint on a sequential token stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .model import DecoderLM, ModelConfig
from .runtime import (
    ConfigurationError,
    create_token_stream,
    evaluate_language_model,
    load_checkpoint,
    resolve_runtime,
    save_json,
    utc_now,
    verify_token_manifest,
)


def run_evaluation(
    checkpoint_path: str | Path,
    *,
    data_path: str | Path | None = None,
    synthetic: bool = False,
    synthetic_tokens: int | None = None,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    batches: int | None = None,
    device: str = "auto",
    dtype: str = "auto",
    compile_model: bool = False,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and return aggregate validation metrics."""

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = ModelConfig.from_dict(dict(checkpoint["model_config"]))
    runtime = resolve_runtime(device, dtype)
    model: torch.nn.Module = DecoderLM(model_config).to(runtime.device)
    model.load_state_dict(checkpoint["model"], strict=True)

    resolved = checkpoint.get("resolved_config", {})
    if not isinstance(resolved, dict):
        resolved = {}
    saved_data = resolved.get("data", {})
    saved_training = resolved.get("training", {})
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

    if batch_size is None:
        batch_size = int(saved_training.get("batch_size", 8))
    if sequence_length is None:
        sequence_length = int(
            saved_training.get("sequence_length", model_config.context_length)
        )
    if batches is None:
        saved_batches = int(saved_training.get("eval_batches", 0))
        batches = saved_batches if saved_batches > 0 else None
    if batch_size <= 0:
        raise ConfigurationError("batch size must be positive")
    if sequence_length <= 0 or sequence_length > model_config.context_length:
        raise ConfigurationError(
            f"sequence length must be in [1, {model_config.context_length}]"
        )
    if batches is not None and batches <= 0:
        batches = None

    if bool(data_config.get("synthetic", False)):
        data_provenance: dict[str, Any] = {
            "kind": "synthetic",
            "num_tokens": int(data_config["val_tokens"]),
            "vocab_size": int(data_config["vocab_size"]),
            "seed": int(data_config.get("seed", 0)) + 1,
            "pattern_length": int(data_config.get("pattern_length", 256)),
        }
    else:
        expected_tokens = None if data_path is not None else int(data_config["val_tokens"])
        data_provenance = verify_token_manifest(
            data_config["val_path"], "val", expected_num_tokens=expected_tokens
        )

    stream = create_token_stream(data_config, "val")
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires a PyTorch build with torch.compile")
        model = torch.compile(model)

    metrics = evaluate_language_model(
        model,
        stream,
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_batches=batches,
        runtime=runtime,
    )
    result: dict[str, Any] = {
        "event": "evaluation",
        "time": utc_now(),
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint.get("step", 0)),
        "device": str(runtime.device),
        "dtype": runtime.dtype_name,
        "compile": compile_model,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "data_provenance": data_provenance,
        **metrics,
    }
    if output is not None:
        save_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="checkpoint produced by triglu.train"
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--data", help="headerless little-endian uint16 evaluation tokens")
    data_group.add_argument(
        "--synthetic",
        action="store_true",
        help="evaluate on the deterministic synthetic stream",
    )
    parser.add_argument("--synthetic-tokens", type=int, help="synthetic evaluation stream size")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument(
        "--batches",
        type=int,
        help="maximum evaluation batches; 0 consumes the entire stream",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "bf16", "float32", "fp32"),
    )
    parser.add_argument("--compile", action="store_true", help="evaluate through torch.compile")
    parser.add_argument("--output", help="optional JSON result path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_evaluation(
        args.checkpoint,
        data_path=args.data,
        synthetic=args.synthetic,
        synthetic_tokens=args.synthetic_tokens,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        batches=args.batches,
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
