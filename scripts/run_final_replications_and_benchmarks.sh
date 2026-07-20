#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly python_command="${PYTHON_BIN:-python}"
python_bin="$(command -v "${python_command}" 2>/dev/null || true)"
readonly python_bin
readonly suite="20l_4k_1b"
readonly config_dir="configs/${suite}"
readonly placement_dir="${config_dir}/placement_amount"
readonly decode_tokens=128
readonly warmup=10
readonly iterations=100

readonly early_plan="15a5_triglu_no_rope_early_intrusion"
readonly swiglu_plan="9a11_swiglu_nested"
readonly ffn_plan="20a0t_triglu_no_rope_ffn"
readonly early_config="${placement_dir}/${early_plan}.yaml"
readonly swiglu_config="${placement_dir}/${swiglu_plan}.yaml"
readonly ffn_config="${config_dir}/ablations/${ffn_plan}.yaml"

train_specs=(
  "${early_config}|2357|${early_plan}_seed2357"
  "${early_config}|7331|${early_plan}_seed7331"
  "${swiglu_config}|2357|${swiglu_plan}_seed2357"
  "${swiglu_config}|7331|${swiglu_plan}_seed7331"
  "${ffn_config}|1337|${ffn_plan}"
  "${ffn_config}|2357|${ffn_plan}_seed2357"
  "${ffn_config}|7331|${ffn_plan}_seed7331"
)

reference_specs=(
  "${config_dir}/20a0t.yaml|1337|20a0t"
  "${config_dir}/20a0t.yaml|2357|20a0t_seed2357"
  "${config_dir}/20a0t.yaml|7331|20a0t_seed7331"
  "${config_dir}/15a5t_front_blend.yaml|1337|15a5t_front_blend"
  "${config_dir}/15a5t_front_blend.yaml|2357|15a5t_front_blend_seed2357"
  "${config_dir}/15a5t_front_blend.yaml|7331|15a5t_front_blend_seed7331"
  "${early_config}|1337|${early_plan}"
  "${swiglu_config}|1337|${swiglu_plan}"
  "${config_dir}/ablations/15a5_triglu_no_rope_front_blend.yaml|1337|15a5_triglu_no_rope_front_blend"
  "${config_dir}/ablations/15a5_triglu_no_rope_front_blend.yaml|2357|15a5_triglu_no_rope_front_blend_seed2357"
  "${config_dir}/ablations/15a5_triglu_no_rope_front_blend.yaml|7331|15a5_triglu_no_rope_front_blend_seed7331"
  "${placement_dir}/9a11_triglu_no_rope_nested.yaml|1337|9a11_triglu_no_rope_nested"
  "${placement_dir}/9a11_triglu_no_rope_nested.yaml|2357|9a11_triglu_no_rope_nested_seed2357"
  "${placement_dir}/9a11_triglu_no_rope_nested.yaml|7331|9a11_triglu_no_rope_nested_seed7331"
  "${config_dir}/ablations/15a5_swiglu_front_blend.yaml|1337|15a5_swiglu_front_blend"
  "${config_dir}/ablations/15a5_swiglu_front_blend.yaml|2357|15a5_swiglu_front_blend_seed2357"
  "${config_dir}/ablations/15a5_swiglu_front_blend.yaml|7331|15a5_swiglu_front_blend_seed7331"
)

benchmark_specs=(
  "20a0t|${config_dir}/20a0t.yaml"
  "15a5t_front_blend|${config_dir}/15a5t_front_blend.yaml"
  "15a5_triglu_no_rope_front_blend|${config_dir}/ablations/15a5_triglu_no_rope_front_blend.yaml"
  "9a11_triglu_no_rope_nested|${placement_dir}/9a11_triglu_no_rope_nested.yaml"
  "15a5_swiglu_front_blend|${config_dir}/ablations/15a5_swiglu_front_blend.yaml"
  "9a11_swiglu_nested|${swiglu_config}"
  "${ffn_plan}|${ffn_config}"
)
contexts=(4096 8192 16384)

mkdir -p runs/logs results/generated
readonly log_file="runs/logs/v0.2.0-ffn-control-final-benchmarks-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${log_file}") 2>&1

trap 'status=$?; echo "error: FFN-control/final-benchmark queue stopped with status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2; exit "${status}"' ERR

echo "queue      final replications, FFN control, and context validation"
echo "log        ${log_file}"
echo "python     ${python_bin}"

if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "error: required interpreter is not executable: ${python_command}" >&2
  exit 1
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

if not metrics_path.is_file():
    raise SystemExit(f"error: missing metrics: {metrics_path}")

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

expected = load_experiment_config(config_path)
expected["training"]["seed"] = seed
expected["training"]["output_dir"] = str(output_dir)
actual = load_yaml(resolved_path)
actual_model = actual.get("model")
if not isinstance(actual_model, Mapping):
    raise SystemExit(f"error: {resolved_path} has no model mapping")
try:
    # Older completed runs predate explicit model fields such as ffn_type.
    # Normalize them through the current public defaults before comparing.
    actual["model"] = ModelConfig.from_dict(dict(actual_model)).to_dict()
except (TypeError, ValueError) as exc:
    raise SystemExit(
        f"error: cannot normalize model section in {resolved_path}: {exc}"
    ) from exc
for section in ("model", "data", "training"):
    if actual.get(section) != expected[section]:
        raise SystemExit(
            f"error: {resolved_path} does not match {config_path} after the "
            f"seed/output overrides (section: {section})"
        )

terminal = events[-1]
expected_steps = int(expected["training"]["max_steps"])
tokens_per_step = (
    int(expected["training"]["batch_size"])
    * int(expected["training"]["sequence_length"])
    * int(expected["training"]["gradient_accumulation_steps"])
)
expected_tokens = expected_steps * tokens_per_step
if terminal.get("step") != expected_steps:
    raise SystemExit(
        f"error: terminal step for {output_dir} is {terminal.get('step')}, "
        f"expected {expected_steps}"
    )
if terminal.get("tokens_seen") != expected_tokens:
    raise SystemExit(
        f"error: terminal tokens for {output_dir} are "
        f"{terminal.get('tokens_seen')}, expected {expected_tokens}"
    )

for required in (manifest_path, baseline_manifest_path):
    if not required.is_file():
        raise SystemExit(f"error: missing data manifest: {required}")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if digest(manifest_path) != digest(baseline_manifest_path):
    raise SystemExit(
        f"error: {manifest_path} does not match the suite baseline manifest"
    )

print(f"validated  {output_dir} | seed {seed} | complete")
PY
}

validate_benchmark_target() {
  local output="$1"
  local config_path="$2"
  local plan="$3"
  local context_length="$4"
  local prompt_length="$5"

  "${python_bin}" - \
    "${output}" "${config_path}" "${plan}" "${context_length}" \
    "${prompt_length}" "${decode_tokens}" "${warmup}" "${iterations}" <<'PY'
import json
import re
import sys
from pathlib import Path

from triglu.config import ModelConfig
from triglu.runtime import load_experiment_config

output = Path(sys.argv[1])
config_path = Path(sys.argv[2])
plan = sys.argv[3]
context, prompt, decode, warmup, iterations = map(int, sys.argv[4:])
config = load_experiment_config(config_path)
model_config = ModelConfig.from_dict(dict(config["model"]))

expected_settings = {
    "batch_size": 1,
    "training_sequence_length": context,
    "prompt_length": prompt,
    "decode_tokens": decode,
    "warmup": warmup,
    "iterations": iterations,
    "gradient_accumulation_steps": 1,
    "compile": True,
    "cached_decode_compile": False,
    "cached_decode_kv_cache": "preallocated",
}

def load_artifact(path: Path) -> dict:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: cannot read benchmark artifact {path}: {exc}") from exc
    if not isinstance(artifact, dict):
        raise SystemExit(f"error: benchmark artifact is not an object: {path}")
    return artifact

def validate(artifact: dict, path: Path) -> None:
    problems = []
    if artifact.get("event") != "benchmark":
        problems.append("event")
    if artifact.get("benchmark_label") != "context-scaling-final-validation":
        problems.append("benchmark_label")
    if artifact.get("source") != str(config_path):
        problems.append("source")
    if artifact.get("checkpoint_step") is not None:
        problems.append("checkpoint_step")

    model = artifact.get("model", {})
    if model.get("configured_context_length") != model_config.context_length:
        problems.append("model.configured_context_length")
    if model.get("benchmark_context_length") != context:
        problems.append("model.benchmark_context_length")
    if model.get("layer_types") != model_config.layer_types:
        problems.append("model.layer_types")
    # Benchmarks written before the FFN control used these implicit defaults.
    if model.get("ffn_type", "swiglu") != model_config.ffn_type:
        problems.append("model.ffn_type")
    if model.get("ffn_hidden_size", 1_376) != model_config.ffn_hidden_size:
        problems.append("model.ffn_hidden_size")

    settings = artifact.get("settings", {})
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
            f"error: existing benchmark {path} does not match: "
            + ", ".join(problems)
        )

if output.exists():
    validate(load_artifact(output), output)
    print(f"validated  {output}")

# The reporter selects the greatest iteration count for each architecture/context.
# Reject a competing artifact with at least as many samples so filename ordering
# cannot silently select it. Earlier 10- and 50-iteration evidence is preserved.
for candidate in sorted(Path("results/generated").glob("*.json")):
    if candidate == output:
        continue
    try:
        artifact = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if not isinstance(artifact, dict):
        continue
    if artifact.get("event") != "benchmark":
        continue
    if not str(artifact.get("benchmark_label") or "").startswith("context-scaling"):
        continue
    source_path = Path(str(artifact.get("source") or ""))
    source_plan = (
        source_path.parent.name
        if source_path.name == "latest.pt"
        else source_path.stem
    )
    source_plan = re.sub(r"_seed\d+$", "", source_plan)
    artifact_context = artifact.get("model", {}).get("benchmark_context_length")
    artifact_iterations = int(artifact.get("settings", {}).get("iterations") or 0)
    if (
        source_plan == plan
        and artifact_context == context
        and artifact_iterations >= iterations
    ):
        raise SystemExit(
            f"error: competing {artifact_iterations}-iteration artifact would "
            f"conflict with {output}: {candidate}"
        )
PY
}

# Every existing suite metric stream must parse completely and end in a terminal
# completion event. This prevents a second process from contending with an
# interrupted or active training run.
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
        raise SystemExit(
            f"error: {metrics_path} does not end with a complete event"
        )
print(f"validated  all existing metric streams under {suite_dir}")
PY

# Validate every reused reference against its config, seed, terminal token
# budget, and data manifest—not merely the presence of a metrics file.
for spec in "${reference_specs[@]}"; do
  IFS="|" read -r config_path seed run_name <<<"${spec}"
  output_dir="runs/${suite}/${run_name}"
  if [[ ! -f "${config_path}" ]]; then
    echo "error: missing reference config ${config_path}" >&2
    exit 1
  fi
  validate_complete_run "${config_path}" "${seed}" "${output_dir}"
done

# Preflight all seven destinations before starting the first several-hour run.
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

# Preflight all benchmark destinations before training. Existing files are
# skippable only when their full schema and execution settings match.
for context_length in "${contexts[@]}"; do
  prompt_length=$((context_length - decode_tokens))
  for spec in "${benchmark_specs[@]}"; do
    IFS="|" read -r plan config_path <<<"${spec}"
    output="results/generated/${suite}-context-${context_length}-${plan}-final-validation-benchmark.json"
    if [[ ! -f "${config_path}" ]]; then
      echo "error: missing benchmark config ${config_path}" >&2
      exit 1
    fi
    validate_benchmark_target \
      "${output}" "${config_path}" "${plan}" "${context_length}" "${prompt_length}"
  done
done

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
echo "complete   replication and FFN-control training phase"

for context_length in "${contexts[@]}"; do
  prompt_length=$((context_length - decode_tokens))
  for spec in "${benchmark_specs[@]}"; do
    IFS="|" read -r plan config_path <<<"${spec}"
    output="results/generated/${suite}-context-${context_length}-${plan}-final-validation-benchmark.json"

    if [[ -f "${output}" ]]; then
      echo "skip       ${plan} @ ${context_length} | exact final-validation artifact exists"
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
      --label context-scaling-final-validation \
      --output "${output}"
    validate_benchmark_target \
      "${output}" "${config_path}" "${plan}" "${context_length}" "${prompt_length}"
  done
done

"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated

"${python_bin}" - "${suite}" "${warmup}" "${iterations}" <<'PY'
import csv
import sys
from pathlib import Path

suite, warmup, iterations = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
expected_plans = {
    "20a0t",
    "15a5t_front_blend",
    "15a5_triglu_no_rope_front_blend",
    "9a11_triglu_no_rope_nested",
    "15a5_swiglu_front_blend",
    "9a11_swiglu_nested",
    "20a0t_triglu_no_rope_ffn",
}
expected_sources = {
    "20a0t": "configs/20l_4k_1b/20a0t.yaml",
    "15a5t_front_blend": "configs/20l_4k_1b/15a5t_front_blend.yaml",
    "15a5_triglu_no_rope_front_blend": "configs/20l_4k_1b/ablations/15a5_triglu_no_rope_front_blend.yaml",
    "9a11_triglu_no_rope_nested": "configs/20l_4k_1b/placement_amount/9a11_triglu_no_rope_nested.yaml",
    "15a5_swiglu_front_blend": "configs/20l_4k_1b/ablations/15a5_swiglu_front_blend.yaml",
    "9a11_swiglu_nested": "configs/20l_4k_1b/placement_amount/9a11_swiglu_nested.yaml",
    "20a0t_triglu_no_rope_ffn": "configs/20l_4k_1b/ablations/20a0t_triglu_no_rope_ffn.yaml",
}
expected_contexts = {4096, 8192, 16384}
table = Path("results/generated") / f"{suite}-report" / "context_benchmarks.csv"
with table.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

selected = {
    (row["architecture"], int(row["context_length"])): row
    for row in rows
    if row["architecture"] in expected_plans
    and int(row["context_length"]) in expected_contexts
}
expected = {
    (plan, context)
    for plan in expected_plans
    for context in expected_contexts
}
missing = sorted(expected - selected.keys())
if missing:
    raise SystemExit(f"error: regenerated report is missing benchmark rows: {missing}")
for key, row in selected.items():
    if int(row["warmup"]) != warmup or int(row["iterations"]) != iterations:
        raise SystemExit(
            f"error: regenerated report selected stale benchmark settings for "
            f"{key}: warmup={row['warmup']}, iterations={row['iterations']}"
        )
    if row["benchmark_label"] != "context-scaling-final-validation":
        raise SystemExit(
            f"error: regenerated report selected stale benchmark label for "
            f"{key}: {row['benchmark_label']!r}"
        )
    expected_source = expected_sources[key[0]]
    if row["benchmark_source"] != expected_source:
        raise SystemExit(
            f"error: regenerated report selected the wrong benchmark source for "
            f"{key}: {row['benchmark_source']!r}, expected {expected_source!r}"
        )
print(f"validated  report selected all {len(expected)} final-validation context rows")
PY

echo "complete   FFN-control/final-benchmark queue"
