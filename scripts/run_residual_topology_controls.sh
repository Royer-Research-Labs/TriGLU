#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly python_command="${PYTHON_BIN:-python}"
python_bin="$(command -v "${python_command}" 2>/dev/null || true)"
readonly python_bin
readonly suite="20l_4k_1b"
readonly config_dir="configs/${suite}/ablations"
readonly decode_tokens=128
readonly warmup=10
readonly iterations=100
readonly benchmark_label="context-scaling-final-validation"
readonly lock_dir="runs/.residual-topology-controls.lock"

readonly single_plan="15a5_wide_swiglu_single_residual_front_blend"
readonly grouped_plan="9l_9a0t_grouped_swiglu_9a11_nested_collapse"
readonly single_config="${config_dir}/${single_plan}.yaml"
readonly grouped_config="${config_dir}/${grouped_plan}.yaml"

train_specs=(
  "${grouped_config}|1337|${grouped_plan}"
  "${single_config}|1337|${single_plan}"
)
benchmark_specs=(
  "${grouped_plan}|${grouped_config}"
  "${single_plan}|${single_config}"
)
contexts=(4096 8192 16384)

mkdir -p runs/logs results/generated
readonly log_file="runs/logs/residual-topology-controls-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${log_file}") 2>&1

if ! mkdir "${lock_dir}" 2>/dev/null; then
  echo "error: another residual-topology queue owns ${lock_dir}" >&2
  exit 1
fi
trap 'rmdir "${lock_dir}" 2>/dev/null || true' EXIT
trap 'status=$?; echo "error: residual-topology queue stopped with status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2; exit "${status}"' ERR

echo "queue      residual-topology exploratory controls"
echo "log        ${log_file}"
echo "python     ${python_bin}"

if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "error: required interpreter is not executable: ${python_command}" >&2
  exit 1
fi

upstream_log="${UPSTREAM_LOG:-}"
if [[ -z "${upstream_log}" ]]; then
  shopt -s nullglob
  upstream_logs=(runs/logs/v0.2.0-ffn-control-final-benchmarks-*.log)
  shopt -u nullglob
  if (( ${#upstream_logs[@]} == 0 )); then
    echo "error: no upstream FFN-control/final-benchmark log was found; set UPSTREAM_LOG" >&2
    exit 1
  fi
  upstream_log="${upstream_logs[$((${#upstream_logs[@]} - 1))]}"
fi

if [[ ! -f "${upstream_log}" ]]; then
  echo "error: upstream log does not exist: ${upstream_log}" >&2
  exit 1
fi

if [[ "${WAIT_FOR_UPSTREAM:-1}" == "1" ]]; then
  readonly maximum_idle_seconds="${MAX_UPSTREAM_IDLE_SECONDS:-3600}"
  echo "wait       ${upstream_log}"
  while true; do
    last_line="$(tail -n 1 "${upstream_log}" 2>/dev/null || true)"
    if [[ "${last_line}" == "complete   FFN-control/final-benchmark queue" ]]; then
      echo "validated  upstream queue completed"
      break
    fi
    if grep -Fq "error: FFN-control/final-benchmark queue stopped" "${upstream_log}"; then
      echo "error: upstream queue reported failure: ${upstream_log}" >&2
      exit 1
    fi
    if [[ "${maximum_idle_seconds}" =~ ^[0-9]+$ ]] \
      && (( maximum_idle_seconds > 0 )); then
      modified="$(stat -c %Y "${upstream_log}")"
      now="$(date +%s)"
      if (( now - modified > maximum_idle_seconds )); then
        echo "error: upstream log has not changed for $((now - modified)) seconds: ${upstream_log}" >&2
        exit 1
      fi
    fi
    sleep 30
  done
else
  if [[ "$(tail -n 1 "${upstream_log}")" != "complete   FFN-control/final-benchmark queue" ]]; then
    echo "error: upstream queue is not complete: ${upstream_log}" >&2
    exit 1
  fi
fi

validate_complete_run() {
  local config_path="$1"
  local seed="$2"
  local output_dir="$3"

  "${python_bin}" - "${config_path}" "${seed}" "${output_dir}" <<'PY'
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from triglu.config import ModelConfig
from triglu.runtime import load_experiment_config, load_yaml

config_path = Path(sys.argv[1])
seed = int(sys.argv[2])
output_dir = Path(sys.argv[3])
metrics_path = output_dir / "metrics.jsonl"
resolved_path = output_dir / "resolved_config.yaml"
manifest_path = output_dir / "data_manifest.json"
baseline_manifest_path = Path("runs/20l_4k_1b/20a0t/data_manifest.json")

for required in (metrics_path, resolved_path, manifest_path, baseline_manifest_path):
    if not required.is_file():
        raise SystemExit(f"error: missing completed-run artifact: {required}")

events = []
with metrics_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"error: invalid JSON in {metrics_path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise SystemExit(f"error: non-object metric in {metrics_path}:{line_number}")
        events.append(event)
if not events or events[-1].get("event") != "complete":
    raise SystemExit(f"error: {metrics_path} does not end with a complete event")

expected = load_experiment_config(config_path)
expected["training"]["seed"] = seed
expected["training"]["output_dir"] = str(output_dir)
actual = load_yaml(resolved_path)
actual_model = actual.get("model")
if not isinstance(actual_model, Mapping):
    raise SystemExit(f"error: {resolved_path} has no model mapping")
actual["model"] = ModelConfig.from_dict(dict(actual_model)).to_dict()
for section in ("model", "data", "training"):
    if actual.get(section) != expected[section]:
        raise SystemExit(
            f"error: {resolved_path} differs from {config_path} after seed/output "
            f"overrides (section: {section})"
        )

terminal = events[-1]
steps = int(expected["training"]["max_steps"])
tokens_per_step = (
    int(expected["training"]["batch_size"])
    * int(expected["training"]["sequence_length"])
    * int(expected["training"]["gradient_accumulation_steps"])
)
if terminal.get("step") != steps or terminal.get("tokens_seen") != steps * tokens_per_step:
    raise SystemExit(f"error: terminal step/token budget is wrong for {output_dir}")

digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if digest(manifest_path) != digest(baseline_manifest_path):
    raise SystemExit(f"error: {manifest_path} differs from the suite baseline manifest")
print(f"validated  {output_dir} | seed {seed} | complete")
PY
}

validate_benchmark() {
  local output="$1"
  local config_path="$2"
  local context_length="$3"
  local prompt_length="$4"

  "${python_bin}" - \
    "${output}" "${config_path}" "${context_length}" "${prompt_length}" \
    "${decode_tokens}" "${warmup}" "${iterations}" "${benchmark_label}" <<'PY'
import json
import sys
from pathlib import Path

from triglu.config import ModelConfig
from triglu.model import DecoderLM
from triglu.runtime import load_experiment_config

output = Path(sys.argv[1])
config_path = Path(sys.argv[2])
context, prompt, decode, warmup, iterations = map(int, sys.argv[3:8])
label = sys.argv[8]
config = load_experiment_config(config_path)
model_config = ModelConfig.from_dict(dict(config["model"]))
if output.exists():
    expected_parameters = DecoderLM(model_config).num_parameters()
    try:
        artifact = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"error: cannot read benchmark artifact {output}: {exc}"
        ) from exc
    if not isinstance(artifact, dict):
        raise SystemExit(f"error: benchmark artifact is not an object: {output}")

    problems = []
    if artifact.get("event") != "benchmark":
        problems.append("event")
    if artifact.get("schema_version") != 2:
        problems.append("schema_version")
    if artifact.get("benchmark_label") != label:
        problems.append("benchmark_label")
    if artifact.get("source") != str(config_path):
        problems.append("source")
    if artifact.get("checkpoint_step") is not None:
        problems.append("checkpoint_step")

    model = artifact.get("model", {})
    expected_model = {
        "parameters": expected_parameters,
        "n_layers": model_config.n_layers,
        "configured_context_length": model_config.context_length,
        "benchmark_context_length": context,
        "layer_types": model_config.layer_types,
        "ffn_type": model_config.ffn_type,
        "ffn_hidden_size": model_config.ffn_hidden_size,
        "ffn_hidden_sizes": model_config.effective_ffn_hidden_sizes,
        "ffn_total_hidden_size": sum(model_config.effective_ffn_hidden_sizes),
        "residual_init_depth": model_config.effective_residual_init_depth,
    }
    for key, value in expected_model.items():
        if model.get(key) != value:
            problems.append(f"model.{key}")

    settings = artifact.get("settings", {})
    expected_settings = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "training_sequence_length": context,
        "prompt_length": prompt,
        "decode_tokens": decode,
        "warmup": warmup,
        "iterations": iterations,
        "compile": True,
        "cached_decode_compile": False,
        "cached_decode_kv_cache": "preallocated",
    }
    for key, value in expected_settings.items():
        if settings.get(key) != value:
            problems.append(f"settings.{key}")

    device = artifact.get("device", {})
    if device.get("device") != "cuda":
        problems.append("device.device")
    if device.get("dtype") != "bfloat16":
        problems.append("device.dtype")
    for section in ("training", "prefill", "cached_decode"):
        values = artifact.get(section, {})
        if values.get("iterations") != iterations:
            problems.append(f"{section}.iterations")
        samples = values.get("iteration_ms")
        if not isinstance(samples, list) or len(samples) != iterations:
            problems.append(f"{section}.iteration_ms")
    if problems:
        raise SystemExit(
            f"error: benchmark {output} does not match: " + ", ".join(problems)
        )
    print(f"validated  {output}")

# Reject a competing artifact that the reporter could select instead of this
# destination. Lower-sample exploratory measurements remain valid raw evidence.
import re

plan = config_path.stem
for candidate in sorted(Path("results/generated").glob("*.json")):
    if candidate == output:
        continue
    try:
        competing = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if not isinstance(competing, dict) or competing.get("event") != "benchmark":
        continue
    if not str(competing.get("benchmark_label") or "").startswith(
        "context-scaling"
    ):
        continue
    source_path = Path(str(competing.get("source") or ""))
    source_plan = (
        source_path.parent.name
        if source_path.name == "latest.pt"
        else source_path.stem
    )
    source_plan = re.sub(r"_seed\d+$", "", source_plan)
    competing_context = competing.get("model", {}).get(
        "benchmark_context_length"
    )
    competing_iterations = int(
        competing.get("settings", {}).get("iterations") or 0
    )
    if (
        source_plan == plan
        and competing_context == context
        and competing_iterations >= iterations
    ):
        raise SystemExit(
            f"error: competing {competing_iterations}-iteration artifact would "
            f"conflict with {output}: {candidate}"
        )
PY
}

# The upstream terminal marker is written only after its 21 benchmark rows have
# been validated. Recheck every suite metric stream before reserving the GPU so
# an unrelated partial run cannot be silently ignored.
"${python_bin}" - "${suite}" <<'PY'
import json
import sys
from pathlib import Path

suite_dir = Path("runs") / sys.argv[1]
for metrics_path in sorted(suite_dir.glob("*/metrics.jsonl")):
    events = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"error: invalid JSON in {metrics_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise SystemExit(
                    f"error: non-object metric in {metrics_path}:{line_number}"
                )
            events.append(event)
    if not events or events[-1].get("event") != "complete":
        raise SystemExit(f"error: {metrics_path} does not end with a complete event")
print(f"validated  all existing metric streams under {suite_dir}")
PY

for spec in "${train_specs[@]}"; do
  IFS="|" read -r config_path seed run_name <<<"${spec}"
  output_dir="runs/${suite}/${run_name}"
  if [[ ! -f "${config_path}" ]]; then
    echo "error: missing training config ${config_path}" >&2
    exit 1
  fi
  if [[ -e "${output_dir}" ]]; then
    validate_complete_run "${config_path}" "${seed}" "${output_dir}"
  fi
done

# Preflight all six benchmark destinations before the first long training run.
for context_length in "${contexts[@]}"; do
  prompt_length=$((context_length - decode_tokens))
  for spec in "${benchmark_specs[@]}"; do
    IFS="|" read -r plan config_path <<<"${spec}"
    output="results/generated/${suite}-context-${context_length}-${plan}-final-validation-benchmark.json"
    validate_benchmark \
      "${output}" "${config_path}" "${context_length}" "${prompt_length}"
  done
done

"${python_bin}" - "${single_config}" "${grouped_config}" <<'PY'
import sys

from triglu.config import ModelConfig
from triglu.model import DecoderLM
from triglu.runtime import load_experiment_config

expected = (89_018_880, 89_019_392)
for path, count in zip(sys.argv[1:], expected, strict=True):
    config = load_experiment_config(path)
    model_config = ModelConfig.from_dict(dict(config["model"]))
    actual = DecoderLM(model_config).num_parameters()
    if actual != count:
        raise SystemExit(f"error: {path} has {actual:,} parameters, expected {count:,}")
    print(f"validated  {path} | {actual:,} parameters")
PY

"${python_bin}" - <<'PY'
import torch
import triglu

if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is not available")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("error: CUDA BF16 is not supported")
print(f"runtime    Python/Torch ready | {torch.__version__} | {triglu.__file__}")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi is required for fail-closed GPU-idle validation" >&2
  exit 1
fi
if ! gpu_query="$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null
)"; then
  echo "error: nvidia-smi compute-process query failed" >&2
  exit 1
fi
compute_pids="$(printf '%s\n' "${gpu_query}" | sed '/^[[:space:]]*$/d')"
if [[ -n "${compute_pids}" ]]; then
  echo "error: GPU already has compute processes: ${compute_pids}" >&2
  exit 1
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight  complete"
  exit 0
fi

for spec in "${train_specs[@]}"; do
  IFS="|" read -r config_path seed run_name <<<"${spec}"
  output_dir="runs/${suite}/${run_name}"
  if [[ -e "${output_dir}" ]]; then
    echo "skip       ${run_name} | exact completed run already exists"
    continue
  fi

  echo "train      ${run_name} | seed ${seed}"
  "${python_bin}" -m triglu.train \
    --config "${config_path}" \
    --seed "${seed}" \
    --output-dir "${output_dir}"
  validate_complete_run "${config_path}" "${seed}" "${output_dir}"
done

"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated
echo "complete   residual-topology training phase"

for context_length in "${contexts[@]}"; do
  prompt_length=$((context_length - decode_tokens))
  for spec in "${benchmark_specs[@]}"; do
    IFS="|" read -r plan config_path <<<"${spec}"
    output="results/generated/${suite}-context-${context_length}-${plan}-final-validation-benchmark.json"

    if [[ -f "${output}" ]]; then
      validate_benchmark \
        "${output}" "${config_path}" "${context_length}" "${prompt_length}"
      echo "skip       ${plan} @ ${context_length} | exact artifact exists"
      continue
    fi

    echo "benchmark  ${plan} @ ${context_length} | warmup ${warmup}, iterations ${iterations}"
    "${python_bin}" -m triglu.benchmark \
      --config "${config_path}" \
      --context-length "${context_length}" \
      --batch-size 1 \
      --sequence-length "${context_length}" \
      --prompt-length "${prompt_length}" \
      --decode-tokens "${decode_tokens}" \
      --warmup "${warmup}" \
      --iterations "${iterations}" \
      --device cuda \
      --dtype bfloat16 \
      --compile \
      --label "${benchmark_label}" \
      --output "${output}"
    validate_benchmark \
      "${output}" "${config_path}" "${context_length}" "${prompt_length}"
  done
done

"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated

"${python_bin}" - "${suite}" "${benchmark_label}" "${warmup}" "${iterations}" \
  <<'PY'
import csv
import sys
from pathlib import Path

suite, label = sys.argv[1:3]
warmup, iterations = map(int, sys.argv[3:5])
contexts = {4096, 8192, 16384}
sources = {
    "20a0t": "configs/20l_4k_1b/20a0t.yaml",
    "15a5t_front_blend": "configs/20l_4k_1b/15a5t_front_blend.yaml",
    "15a5_triglu_no_rope_front_blend": (
        "configs/20l_4k_1b/ablations/"
        "15a5_triglu_no_rope_front_blend.yaml"
    ),
    "9a11_triglu_no_rope_nested": (
        "configs/20l_4k_1b/placement_amount/"
        "9a11_triglu_no_rope_nested.yaml"
    ),
    "15a5_swiglu_front_blend": (
        "configs/20l_4k_1b/ablations/15a5_swiglu_front_blend.yaml"
    ),
    "9a11_swiglu_nested": (
        "configs/20l_4k_1b/placement_amount/9a11_swiglu_nested.yaml"
    ),
    "20a0t_triglu_no_rope_ffn": (
        "configs/20l_4k_1b/ablations/20a0t_triglu_no_rope_ffn.yaml"
    ),
    "15a5_wide_swiglu_single_residual_front_blend": (
        "configs/20l_4k_1b/ablations/"
        "15a5_wide_swiglu_single_residual_front_blend.yaml"
    ),
    "9l_9a0t_grouped_swiglu_9a11_nested_collapse": (
        "configs/20l_4k_1b/ablations/"
        "9l_9a0t_grouped_swiglu_9a11_nested_collapse.yaml"
    ),
}
table = Path("results/generated") / f"{suite}-report" / "context_benchmarks.csv"
with table.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

selected = {
    (row["architecture"], int(row["context_length"])): row
    for row in rows
    if row["architecture"] in sources and int(row["context_length"]) in contexts
}
expected = {(plan, context) for plan in sources for context in contexts}
if selected.keys() != expected:
    raise SystemExit(
        f"error: regenerated report has wrong final-validation cohort; "
        f"missing={sorted(expected - selected.keys())}, "
        f"extra={sorted(selected.keys() - expected)}"
    )
for (plan, context), row in selected.items():
    expected_values = {
        "benchmark_label": label,
        "benchmark_source": sources[plan],
        "configured_context_length": "4096",
        "prompt_length": str(context - 128),
        "decode_tokens": "128",
        "warmup": str(warmup),
        "iterations": str(iterations),
        "dtype": "bfloat16",
    }
    for field, value in expected_values.items():
        if row[field] != value:
            raise SystemExit(
                f"error: report selected wrong {field} for {(plan, context)}: "
                f"{row[field]!r}, expected {value!r}"
            )
    if not row["device_name"]:
        raise SystemExit(
            f"error: report selected benchmark without device name for "
            f"{(plan, context)}"
        )
print(f"validated  report selected the complete {len(expected)}-row final cohort")
PY

echo "complete   residual-topology control queue"
