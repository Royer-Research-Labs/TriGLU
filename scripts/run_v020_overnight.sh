#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly python_command="${PYTHON_BIN:-python}"
python_bin="$(command -v "${python_command}" 2>/dev/null || true)"
readonly python_bin
readonly suite="20l_4k_1b"
readonly config_dir="configs/${suite}/placement_amount"
readonly decode_tokens=128
readonly warmup=3
readonly iterations=10

mkdir -p runs/logs results/generated
readonly log_file="runs/logs/v0.2.0-overnight-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${log_file}") 2>&1

trap 'status=$?; echo "error: overnight queue stopped with status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2; exit "${status}"' ERR

echo "overnight  log ${log_file}"
echo "python     ${python_bin}"

if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "error: required interpreter is not executable: ${python_command}" >&2
  exit 1
fi

"${python_bin}" - <<'PY'
import torch
import triglu

if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is not available")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("error: CUDA BF16 is not supported")
print(f"runtime    Python/Torch ready | {torch.__version__} | {triglu.__file__}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  compute_pids="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
      sed '/^[[:space:]]*$/d'
  )"
  if [[ -n "${compute_pids}" ]]; then
    echo "error: GPU already has compute processes: ${compute_pids}" >&2
    exit 1
  fi
fi

for reference in \
  20a0t_seed2357 \
  20a0t_seed7331 \
  15a5_triglu_no_rope_front_blend \
  15a5_swiglu_front_blend
do
  metrics="runs/${suite}/${reference}/metrics.jsonl"
  if [[ ! -f "${metrics}" ]] ||
    ! tail -n 1 "${metrics}" | grep -q '"event": "complete"'; then
    echo "error: complete reference run ${reference} is required" >&2
    exit 1
  fi
done

for metrics in runs/${suite}/*/metrics.jsonl; do
  [[ -f "${metrics}" ]] || continue
  if ! tail -n 1 "${metrics}" | grep -q '"event": "complete"'; then
    echo "error: ${metrics} is incomplete; stop or archive it before launching" >&2
    exit 1
  fi
done

train_specs=(
  "9a11_triglu_no_rope_nested.yaml|2357|9a11_triglu_no_rope_nested_seed2357"
  "9a11_triglu_no_rope_nested.yaml|7331|9a11_triglu_no_rope_nested_seed7331"
  "15a5_swiglu_repeating.yaml|1337|15a5_swiglu_repeating"
  "9a11_swiglu_nested.yaml|1337|9a11_swiglu_nested"
)

for spec in "${train_specs[@]}"; do
  IFS="|" read -r config seed run_name <<<"${spec}"
  config_path="${config_dir}/${config}"
  output_dir="runs/${suite}/${run_name}"
  metrics="${output_dir}/metrics.jsonl"

  if [[ ! -f "${config_path}" ]]; then
    echo "error: missing training config ${config_path}" >&2
    exit 1
  fi
  if [[ -f "${metrics}" ]] &&
    tail -n 1 "${metrics}" | grep -q '"event": "complete"'; then
    continue
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "error: ${output_dir} exists but is not complete; archive it before retrying" >&2
    exit 1
  fi
done

benchmark_plans=(
  "20a0t|configs/${suite}/20a0t.yaml"
  "15a5_triglu_no_rope_front_blend|configs/${suite}/ablations/15a5_triglu_no_rope_front_blend.yaml"
  "9a11_triglu_no_rope_nested|${config_dir}/9a11_triglu_no_rope_nested.yaml"
  "15a5_swiglu_front_blend|configs/${suite}/ablations/15a5_swiglu_front_blend.yaml"
  "9a11_swiglu_nested|${config_dir}/9a11_swiglu_nested.yaml"
)
contexts=(4096 8192 16384)

for context_length in "${contexts[@]}"; do
  prompt_length=$((context_length - decode_tokens))

  for entry in "${benchmark_plans[@]}"; do
    IFS="|" read -r plan config <<<"${entry}"
    output="results/generated/${suite}-context-${context_length}-${plan}-benchmark.json"

    if [[ ! -f "${config}" ]]; then
      echo "error: missing benchmark config ${config}" >&2
      exit 1
    fi
    if [[ ! -f "${output}" ]]; then
      continue
    fi

    "${python_bin}" - \
      "${output}" "${config}" "${context_length}" "${prompt_length}" \
      "${decode_tokens}" "${warmup}" "${iterations}" <<'PY'
import json
import sys
from pathlib import Path

from triglu.runtime import load_experiment_config

output, config_path = map(Path, sys.argv[1:3])
context, prompt, decode, warmup, iterations = map(int, sys.argv[3:])
artifact = json.loads(output.read_text(encoding="utf-8"))
config = load_experiment_config(config_path)

expected_settings = {
    "batch_size": 1,
    "training_sequence_length": context,
    "prompt_length": prompt,
    "decode_tokens": decode,
    "warmup": warmup,
    "iterations": iterations,
    "compile": True,
    "cached_decode_compile": False,
}
problems = []
if artifact.get("event") != "benchmark":
    problems.append("event")
if artifact.get("benchmark_label") != "context-scaling":
    problems.append("benchmark_label")
if artifact.get("source") != str(config_path):
    problems.append("source")
model = artifact.get("model", {})
if model.get("configured_context_length") != config["model"]["context_length"]:
    problems.append("configured_context_length")
if model.get("benchmark_context_length") != context:
    problems.append("benchmark_context_length")
if model.get("layer_types") != config["model"]["layer_types"]:
    problems.append("layer_types")
settings = artifact.get("settings", {})
for key, value in expected_settings.items():
    if settings.get(key) != value:
        problems.append(f"settings.{key}")
device = artifact.get("device", {})
if device.get("device") != "cuda":
    problems.append("device")
if device.get("dtype") != "bfloat16":
    problems.append("dtype")
if problems:
    raise SystemExit(
        f"error: existing benchmark {output} does not match: {', '.join(problems)}"
    )
print(f"validated  {output}")
PY
  done
done

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight  complete"
  exit 0
fi

for spec in "${train_specs[@]}"; do
  IFS="|" read -r config seed run_name <<<"${spec}"
  output_dir="runs/${suite}/${run_name}"
  metrics="${output_dir}/metrics.jsonl"

  if [[ -f "${metrics}" ]] &&
    tail -n 1 "${metrics}" | grep -q '"event": "complete"'; then
    echo "skip       ${run_name} | already complete"
    continue
  fi

  echo "train      ${run_name} | seed ${seed}"
  "${python_bin}" -m triglu.train \
    --config "${config_dir}/${config}" \
    --seed "${seed}" \
    --output-dir "${output_dir}"

  if [[ ! -f "${metrics}" ]] ||
    ! tail -n 1 "${metrics}" | grep -q '"event": "complete"'; then
    echo "error: ${run_name} returned without a complete event" >&2
    exit 1
  fi
done

for context_length in "${contexts[@]}"; do
  prompt_length=$((context_length - decode_tokens))

  for entry in "${benchmark_plans[@]}"; do
    IFS="|" read -r plan config <<<"${entry}"
    output="results/generated/${suite}-context-${context_length}-${plan}-benchmark.json"

    if [[ -f "${output}" ]]; then
      echo "skip       ${plan} @ ${context_length} | ${output} exists"
      continue
    fi

    echo "benchmark  ${plan} @ ${context_length} | prompt ${prompt_length}, decode ${decode_tokens}"
    "${python_bin}" -m triglu.benchmark \
      --config "${config}" \
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
      --label context-scaling \
      --output "${output}"
  done
done

"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated

echo "complete   v0.2.0 overnight queue"
