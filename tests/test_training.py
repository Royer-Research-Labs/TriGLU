from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from triglu.benchmark import run_benchmarks
from triglu.evaluate import run_evaluation
from triglu.runtime import (
    ConfigurationError,
    RuntimeSpec,
    learning_rate_for_step,
    make_adamw,
    resolve_runtime,
)
from triglu.train import _format_console_event, run_training


def write_tiny_config(path: Path, output_dir: Path) -> None:
    values = {
        "model": {
            "vocab_size": 16,
            "n_layers": 2,
            "d_model": 8,
            "n_heads": 1,
            "ffn_hidden_size": 16,
            "context_length": 4,
            "dropout": 0.0,
            "bias": False,
            "tie_embeddings": True,
            "layer_types": ["attention", "triglu"],
        },
        "data": {
            "synthetic": True,
            "vocab_size": 16,
            "train_tokens": 64,
            "val_tokens": 32,
            "pattern_length": 8,
            "seed": 21,
        },
        "training": {
            "output_dir": str(output_dir),
            "device": "cpu",
            "dtype": "float32",
            "compile": False,
            "batch_size": 1,
            "sequence_length": 4,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
            "learning_rate": 0.001,
            "min_lr_ratio": 0.1,
            "warmup_steps": 1,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1.0e-8,
            "grad_clip": 1.0,
            "fused_optimizer": False,
            "log_interval": 1,
            "eval_interval": 0,
            "eval_batches": 0,
            "checkpoint_interval": 1,
            "seed": 21,
        },
    }
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def test_learning_rate_warmup_and_cosine_endpoints() -> None:
    config = {
        "learning_rate": 1.0,
        "warmup_steps": 2,
        "max_steps": 6,
        "min_lr_ratio": 0.1,
    }
    assert learning_rate_for_step(0, config) == pytest.approx(0.5)
    assert learning_rate_for_step(1, config) == pytest.approx(1.0)
    assert learning_rate_for_step(2, config) == pytest.approx(1.0)
    assert learning_rate_for_step(5, config) == pytest.approx(0.1)


def test_cpu_runtime_uses_fp32_fallback() -> None:
    assert resolve_runtime("cpu", "auto").dtype == torch.float32
    assert resolve_runtime("cpu", "bfloat16").dtype == torch.float32


def test_training_console_events_are_compact_and_readable() -> None:
    train_line = _format_console_event(
        {
            "event": "train",
            "step": 10,
            "tokens_seen": 655_360,
            "loss": 4.56344,
            "perplexity": 95.9128,
            "learning_rate": 3.1e-5,
            "grad_norm": 0.6353,
            "tokens_per_second": 266_508,
            "max_memory_allocated_bytes": 3_552_829_440,
        },
        max_steps=1_526,
    )
    assert train_line == (
        "train      step   10/1526 (  0.7%) | tokens 655.36K | loss 4.5634 | "
        "ppl 95.91 | lr 3.100e-05 | grad 0.635 | 266.51K tok/s | mem 3.31 GiB"
    )

    eval_line = _format_console_event(
        {
            "event": "evaluation",
            "step": 1_526,
            "tokens_seen": 100_007_936,
            "loss": 4.608687,
            "perplexity": 100.3523,
            "accuracy": 0.2714258,
            "tokens": 204_800,
        },
        max_steps=1_526,
    )
    assert eval_line == (
        "eval       step 1526/1526 (100.0%) | tokens 100.01M | loss 4.6087 | "
        "ppl 100.35 | acc 27.14% | eval 204.80K tokens"
    )


def test_train_checkpoint_resume_and_standalone_evaluation(tmp_path) -> None:
    config_path = tmp_path / "tiny.yaml"
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    write_tiny_config(config_path, uninterrupted_dir)

    summary = run_training(config_path)
    assert summary["step"] == 2
    assert summary["tokens_seen"] == 8
    checkpoint_step_one = uninterrupted_dir / "checkpoint_step_00000001.pt"
    assert checkpoint_step_one.is_file()
    assert not (uninterrupted_dir / "checkpoint_step_00000002.pt").exists()
    assert (uninterrupted_dir / "latest.pt").is_file()
    assert (uninterrupted_dir / "resolved_config.yaml").is_file()
    assert (uninterrupted_dir / "environment.json").is_file()
    assert (uninterrupted_dir / "data_provenance.json").is_file()

    resumed = run_training(
        config_path,
        resume_override=checkpoint_step_one,
        output_dir_override=resumed_dir,
    )
    assert resumed["step"] == 2
    assert resumed["tokens_seen"] == 8

    uninterrupted_checkpoint = torch.load(
        uninterrupted_dir / "latest.pt", map_location="cpu", weights_only=True
    )
    resumed_checkpoint = torch.load(
        resumed_dir / "latest.pt", map_location="cpu", weights_only=True
    )
    for name, expected in uninterrupted_checkpoint["model"].items():
        torch.testing.assert_close(expected, resumed_checkpoint["model"][name], rtol=0, atol=0)

    evaluation = run_evaluation(
        resumed_dir / "latest.pt",
        synthetic=True,
        synthetic_tokens=32,
        batch_size=1,
        sequence_length=4,
        batches=2,
        device="cpu",
        dtype="float32",
        output=resumed_dir / "evaluation.json",
    )
    assert evaluation["tokens"] == 8
    assert evaluation["loss"] > 0
    saved_evaluation = json.loads(
        (resumed_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    assert saved_evaluation["loss"] == evaluation["loss"]
    assert saved_evaluation["compile"] is False

    benchmark = run_benchmarks(
        checkpoint_path=resumed_dir / "latest.pt",
        batch_size=1,
        gradient_accumulation_steps=1,
        context_length=8,
        sequence_length=4,
        prompt_length=6,
        decode_tokens=2,
        warmup=0,
        iterations=1,
        device="cpu",
        dtype="float32",
        label="context-scaling",
        output=resumed_dir / "benchmark.json",
    )
    cache_metrics = benchmark["cached_decode"]
    assert benchmark["schema_version"] == 2
    assert cache_metrics["initial_cache_bytes"] == 384
    assert cache_metrics["final_cache_bytes"] == 512
    assert cache_metrics["cache_capacity_bytes"] == 512
    assert benchmark["benchmark_label"] == "context-scaling"
    assert benchmark["model"]["configured_context_length"] == 4
    assert benchmark["model"]["benchmark_context_length"] == 8
    assert benchmark["model"]["ffn_type"] == "swiglu"
    assert benchmark["model"]["block_mode"] == "sequential"
    assert benchmark["model"]["ffn_hidden_size"] == 16
    assert benchmark["model"]["ffn_hidden_sizes"] == [16, 16]
    assert benchmark["model"]["ffn_total_hidden_size"] == 32
    assert benchmark["model"]["residual_init_depth"] == 2
    assert benchmark["settings"]["cached_decode_kv_cache"] == "preallocated"


def test_fresh_run_refuses_to_clobber_existing_run_records(tmp_path) -> None:
    config_path = tmp_path / "tiny.yaml"
    output_dir = tmp_path / "run"
    write_tiny_config(config_path, output_dir)
    run_training(config_path)
    original_metrics = (output_dir / "metrics.jsonl").read_bytes()

    with pytest.raises(ConfigurationError, match="already contains run records"):
        run_training(config_path)
    assert (output_dir / "metrics.jsonl").read_bytes() == original_metrics
    assert (output_dir / "latest.pt").is_file()

    overwritten = run_training(config_path, overwrite_run=True)
    assert overwritten["step"] == 2
    events = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "complete"


def test_resuming_a_completed_checkpoint_is_rejected(tmp_path) -> None:
    config_path = tmp_path / "tiny.yaml"
    output_dir = tmp_path / "run"
    write_tiny_config(config_path, output_dir)
    run_training(config_path)

    with pytest.raises(ConfigurationError, match="already completed"):
        run_training(config_path, resume_override=output_dir / "latest.pt")


def test_resume_permits_log_interval_change_only(tmp_path) -> None:
    config_path = tmp_path / "tiny.yaml"
    output_dir = tmp_path / "run"
    write_tiny_config(config_path, output_dir)
    run_training(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    values["training"]["log_interval"] = 5
    relogged = tmp_path / "relogged.yaml"
    relogged.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    resumed = run_training(
        relogged,
        resume_override=output_dir / "checkpoint_step_00000001.pt",
        output_dir_override=tmp_path / "resumed",
    )
    assert resumed["step"] == 2

    values["training"]["grad_clip"] = 2.0
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="training.grad_clip"):
        run_training(
            drifted,
            resume_override=output_dir / "checkpoint_step_00000001.pt",
            output_dir_override=tmp_path / "drifted",
        )


def test_rejected_resume_preserves_existing_run_records(tmp_path) -> None:
    config_path = tmp_path / "tiny.yaml"
    output_dir = tmp_path / "run"
    write_tiny_config(config_path, output_dir)
    run_training(config_path)
    original_provenance = (output_dir / "data_provenance.json").read_bytes()
    original_metrics = (output_dir / "metrics.jsonl").read_bytes()

    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["data"]["pattern_length"] = 4
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        run_training(
            drifted, resume_override=output_dir / "checkpoint_step_00000001.pt"
        )

    assert (output_dir / "data_provenance.json").read_bytes() == original_provenance
    assert (output_dir / "metrics.jsonl").read_bytes() == original_metrics


def test_make_adamw_falls_back_when_fused_construction_fails(monkeypatch) -> None:
    model = torch.nn.Linear(4, 4)
    training = {
        "learning_rate": 1.0e-3,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1.0e-8,
        "weight_decay": 0.1,
        "fused_optimizer": True,
    }
    real_adamw = torch.optim.AdamW

    class FusedRejectingAdamW(real_adamw):
        def __init__(self, params, *args, fused=None, **kwargs):
            if fused:
                # Torch drains the parameter iterable before rejecting fused
                # construction; the fallback must survive that.
                list(params)
                raise RuntimeError("fused AdamW is unsupported in this test")
            super().__init__(params, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", FusedRejectingAdamW)
    runtime = RuntimeSpec(
        device=torch.device("cuda"), dtype=torch.float32, dtype_name="float32"
    )

    optimizer, used_fused = make_adamw(model.parameters(), training, runtime)
    assert used_fused is False
    optimizer_parameters = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    assert optimizer_parameters == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_training_seed_and_output_overrides_are_recorded(tmp_path) -> None:
    config_path = tmp_path / "tiny.yaml"
    configured_dir = tmp_path / "configured"
    overridden_dir = tmp_path / "seeded"
    write_tiny_config(config_path, configured_dir)

    run_training(
        config_path,
        output_dir_override=overridden_dir,
        seed_override=2357,
    )

    resolved = yaml.safe_load(
        (overridden_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert resolved["training"]["seed"] == 2357
    assert resolved["training"]["output_dir"] == str(overridden_dir)
    assert not configured_dir.exists()
