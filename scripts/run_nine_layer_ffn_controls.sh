#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly python_command="${PYTHON_BIN:-python}"
python_bin="$(command -v "${python_command}" 2>/dev/null || true)"
readonly python_bin
readonly suite="20l_4k_1b"
readonly config_dir="configs/${suite}/ablations"
readonly grouped_plan="9l_9a0t_grouped_triglu_no_rope_ffn"
readonly standard_plan="9l_9a0t_standard_swiglu"
readonly grouped_config="${config_dir}/${grouped_plan}.yaml"
readonly standard_config="${config_dir}/${standard_plan}.yaml"
readonly upstream_log="${UPSTREAM_LOG:-}"
readonly upstream_lock="runs/.residual-topology-controls.lock"
readonly lock_dir="runs/.nine-layer-ffn-controls.lock"
readonly decode_tokens=128
readonly warmup=10
readonly iterations=100
readonly benchmark_label="context-scaling-final-validation"

train_specs=(
  "${grouped_config}|1337|${grouped_plan}|89019904|20"
  "${standard_config}|1337|${standard_plan}|54224384|9"
)
contexts=(4096 8192 16384)

mkdir -p runs/logs results/generated
readonly log_file="runs/logs/nine-layer-ffn-controls-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${log_file}") 2>&1

if ! mkdir "${lock_dir}" 2>/dev/null; then
  echo "error: another nine-layer FFN queue owns ${lock_dir}" >&2
  exit 1
fi
trap 'rmdir "${lock_dir}" 2>/dev/null || true' EXIT
trap 'status=$?; echo "error: nine-layer FFN queue stopped with status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2; exit "${status}"' ERR

echo "queue      nine-layer FFN and shallow controls"
echo "log        ${log_file}"
echo "python     ${python_bin}"
echo "upstream   ${upstream_log:-<unset>}"

if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "error: required interpreter is not executable: ${python_command}" >&2
  exit 1
fi
if [[ -z "${upstream_log}" ]]; then
  echo "error: UPSTREAM_LOG must identify the exact residual-topology queue log" >&2
  exit 1
fi
if [[ ! -f "${upstream_log}" ]]; then
  echo "error: upstream log does not exist: ${upstream_log}" >&2
  exit 1
fi

declare -A config_sha
for config_path in "${grouped_config}" "${standard_config}"; do
  if [[ ! -f "${config_path}" ]]; then
    echo "error: missing training config ${config_path}" >&2
    exit 1
  fi
  config_sha["${config_path}"]="$(sha256sum "${config_path}" | cut -d' ' -f1)"
  echo "config     ${config_path} | sha256 ${config_sha[${config_path}]}"
done

# Static checks run before waiting so configuration errors cannot reserve an
# overnight queue slot. This does not query or otherwise contend for the GPU.
"${python_bin}" - "${grouped_config}" "${standard_config}" <<'PY'
import sys
from pathlib import Path

from triglu.config import ModelConfig
from triglu.model import DecoderLM
from triglu.runtime import load_experiment_config

expected = {
    Path(sys.argv[1]).stem: {
        "path": sys.argv[1],
        "ffn_type": "triglu_no_rope",
        "widths": [1032, 1032, 1032, 1032, 1032, 4121, 5666, 5666, 5665],
        "init_depth": 20,
        "parameters": 89_019_904,
    },
    Path(sys.argv[2]).stem: {
        "path": sys.argv[2],
        "ffn_type": "swiglu",
        "widths": [1376] * 9,
        "init_depth": 9,
        "parameters": 54_224_384,
    },
}
for plan, specification in expected.items():
    config = load_experiment_config(specification["path"])
    model_config = ModelConfig.from_dict(dict(config["model"]))
    training = config["training"]
    data = config["data"]
    problems = []
    checks = {
        "model.n_layers": (model_config.n_layers, 9),
        "model.layer_types": (model_config.layer_types, ["attention"] * 9),
        "model.ffn_type": (model_config.ffn_type, specification["ffn_type"]),
        "model.ffn_hidden_sizes": (
            model_config.effective_ffn_hidden_sizes,
            specification["widths"],
        ),
        "model.residual_init_depth": (
            model_config.effective_residual_init_depth,
            specification["init_depth"],
        ),
        "model.context_length": (model_config.context_length, 4096),
        "model.parameters": (
            DecoderLM(model_config).num_parameters(),
            specification["parameters"],
        ),
        "training.output_dir": (
            training.get("output_dir"),
            f"runs/20l_4k_1b/{plan}",
        ),
        "training.batch_size": (training.get("batch_size"), 16),
        "training.sequence_length": (training.get("sequence_length"), 4096),
        "training.gradient_accumulation_steps": (
            training.get("gradient_accumulation_steps"),
            1,
        ),
        "training.max_steps": (training.get("max_steps"), 15_259),
        "training.seed": (training.get("seed"), 1337),
        "data.train_tokens": (data.get("train_tokens"), 1_000_000_000),
        "data.val_tokens": (data.get("val_tokens"), 5_000_000),
        "data.seed": (data.get("seed"), 1337),
        "data.train_path": (
            data.get("train_path"),
            "data/fineweb_edu_10bt_1b/train.bin",
        ),
        "data.val_path": (
            data.get("val_path"),
            "data/fineweb_edu_10bt_1b/val.bin",
        ),
    }
    for field, (actual, wanted) in checks.items():
        if actual != wanted:
            problems.append(f"{field}={actual!r}, expected {wanted!r}")
    if problems:
        raise SystemExit(
            f"error: static config validation failed for {plan}: "
            + "; ".join(problems)
        )
    print(
        f"validated  {specification['path']} | "
        f"{specification['parameters']:,} parameters | "
        f"init depth {specification['init_depth']}"
    )
PY

validate_complete_run() {
  local config_path="$1"
  local seed="$2"
  local output_dir="$3"

  "${python_bin}" - "${config_path}" "${seed}" "${output_dir}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from triglu.config import ModelConfig
from triglu.runtime import load_experiment_config, load_yaml

config_path = Path(sys.argv[1])
seed = int(sys.argv[2])
output_dir = Path(sys.argv[3])
required = {
    "metrics": output_dir / "metrics.jsonl",
    "resolved config": output_dir / "resolved_config.yaml",
    "checkpoint": output_dir / "latest.pt",
    "environment": output_dir / "environment.json",
    "data provenance": output_dir / "data_provenance.json",
    "data manifest": output_dir / "data_manifest.json",
}
baseline_manifest = Path("runs/20l_4k_1b/20a0t/data_manifest.json")
for label, path in required.items():
    if not path.is_file():
        raise SystemExit(f"error: missing completed-run {label}: {path}")
if not baseline_manifest.is_file():
    raise SystemExit(f"error: missing suite baseline manifest: {baseline_manifest}")

events = []
with required["metrics"].open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"error: invalid JSON in {required['metrics']}:{line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise SystemExit(
                f"error: non-object metric in {required['metrics']}:{line_number}"
            )
        events.append(event)
if not events or events[-1].get("event") != "complete":
    raise SystemExit(
        f"error: {required['metrics']} does not end with a complete event"
    )

expected = load_experiment_config(config_path)
expected["model"] = ModelConfig.from_dict(dict(expected["model"])).to_dict()
expected["training"]["seed"] = seed
expected["training"]["output_dir"] = str(output_dir)
actual = load_yaml(required["resolved config"])
actual["model"] = ModelConfig.from_dict(dict(actual["model"])).to_dict()
for section in ("model", "data", "training"):
    if actual.get(section) != expected.get(section):
        raise SystemExit(
            f"error: {required['resolved config']} differs from {config_path} "
            f"after seed/output overrides (section: {section})"
        )

terminal = events[-1]
steps = int(expected["training"]["max_steps"])
tokens_per_step = (
    int(expected["training"]["batch_size"])
    * int(expected["training"]["sequence_length"])
    * int(expected["training"]["gradient_accumulation_steps"])
)
if terminal.get("step") != steps:
    raise SystemExit(f"error: terminal step is wrong for {output_dir}")
if terminal.get("tokens_seen") != steps * tokens_per_step:
    raise SystemExit(f"error: terminal token budget is wrong for {output_dir}")

digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if digest(required["data manifest"]) != digest(baseline_manifest):
    raise SystemExit(
        f"error: {required['data manifest']} differs from the suite baseline manifest"
    )
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
import re
import sys
from pathlib import Path

from triglu.config import ModelConfig
from triglu.model import DecoderLM
from triglu.runtime import load_experiment_config

output = Path(sys.argv[1])
config_path = Path(sys.argv[2])
context, prompt, decode, warmup, iterations = map(int, sys.argv[3:8])
label = sys.argv[8]
model_config = ModelConfig.from_dict(
    dict(load_experiment_config(config_path)["model"])
)

if output.exists():
    try:
        artifact = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: cannot read benchmark {output}: {exc}") from exc
    if not isinstance(artifact, dict):
        raise SystemExit(f"error: benchmark artifact is not an object: {output}")
    problems = []
    expected_top = {
        "event": "benchmark",
        "schema_version": 2,
        "benchmark_label": label,
        "source": str(config_path),
        "checkpoint_step": None,
    }
    for key, value in expected_top.items():
        if artifact.get(key) != value:
            problems.append(key)
    model = artifact.get("model", {})
    expected_model = {
        "parameters": DecoderLM(model_config).num_parameters(),
        "n_layers": model_config.n_layers,
        "configured_context_length": 4096,
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

# Refuse a competing final-validation artifact that the report could select.
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
    if (
        source_plan == plan
        and competing.get("model", {}).get("benchmark_context_length") == context
        and int(competing.get("settings", {}).get("iterations") or 0) >= iterations
    ):
        raise SystemExit(
            f"error: competing benchmark conflicts with {output}: {candidate}"
        )
PY
}

# Refuse partial targets now. A matching complete run may be reused.
for spec in "${train_specs[@]}"; do
  IFS="|" read -r config_path seed run_name _ _ <<<"${spec}"
  output_dir="runs/${suite}/${run_name}"
  if [[ -e "${output_dir}" ]]; then
    validate_complete_run "${config_path}" "${seed}" "${output_dir}"
  fi
done
for context_length in "${contexts[@]}"; do
  prompt_length=$((context_length - decode_tokens))
  for config_path in "${grouped_config}" "${standard_config}"; do
    plan="$(basename "${config_path}" .yaml)"
    output="results/generated/${suite}-context-${context_length}-${plan}-final-validation-benchmark.json"
    validate_benchmark \
      "${output}" "${config_path}" "${context_length}" "${prompt_length}"
  done
done

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight  complete (static checks only; upstream and GPU not queried)"
  exit 0
fi

readonly maximum_idle_seconds="${MAX_UPSTREAM_IDLE_SECONDS:-3600}"
echo "wait       ${upstream_log}"
while true; do
  last_line="$(tail -n 1 "${upstream_log}" 2>/dev/null || true)"
  if [[ "${last_line}" == "complete   residual-topology control queue" ]]; then
    echo "validated  upstream terminal success marker"
    break
  fi
  if grep -Fq "error: residual-topology queue stopped" "${upstream_log}"; then
    echo "error: upstream residual-topology queue reported failure" >&2
    exit 1
  fi
  if [[ "${maximum_idle_seconds}" =~ ^[0-9]+$ ]] \
    && (( maximum_idle_seconds > 0 )); then
    modified="$(stat -c %Y "${upstream_log}")"
    now="$(date +%s)"
    if (( now - modified > maximum_idle_seconds )); then
      echo "error: upstream log has been idle for $((now - modified)) seconds" >&2
      exit 1
    fi
  fi
  sleep 30
done

# The marker is printed immediately before the upstream EXIT trap removes its
# lock. Wait for that lock release to eliminate the short hand-off race.
for _ in $(seq 1 60); do
  if [[ ! -e "${upstream_lock}" ]]; then
    break
  fi
  sleep 2
done
if [[ -e "${upstream_lock}" ]]; then
  echo "error: upstream lock remained after its success marker" >&2
  exit 1
fi

for config_path in "${grouped_config}" "${standard_config}"; do
  current_sha="$(sha256sum "${config_path}" | cut -d' ' -f1)"
  if [[ "${current_sha}" != "${config_sha[${config_path}]}" ]]; then
    echo "error: config changed while waiting: ${config_path}" >&2
    exit 1
  fi
done

# Scan directories rather than only metric paths so an artifact-only partial
# run cannot be overlooked.
"${python_bin}" - "${suite}" <<'PY'
import json
import sys
from pathlib import Path

suite_dir = Path("runs") / sys.argv[1]
artifact_names = {
    "metrics.jsonl",
    "resolved_config.yaml",
    "latest.pt",
    "environment.json",
    "data_provenance.json",
    "data_manifest.json",
}
for run_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
    present = artifact_names.intersection(path.name for path in run_dir.iterdir())
    if not present:
        continue
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise SystemExit(
            f"error: artifact-bearing run directory has no metrics: {run_dir}"
        )
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
        raise SystemExit(f"error: {metrics_path} is not complete")
print(f"validated  all existing run directories under {suite_dir}")
PY

wait_for_idle_gpu() {
  local consecutive=0
  local query
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi is required for fail-closed GPU validation" >&2
    return 1
  fi
  for _ in $(seq 1 120); do
    if ! query="$(
      nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null
    )"; then
      echo "error: nvidia-smi compute-process query failed" >&2
      return 1
    fi
    query="$(printf '%s\n' "${query}" | sed '/^[[:space:]]*$/d')"
    if [[ -z "${query}" ]]; then
      consecutive=$((consecutive + 1))
      if (( consecutive >= 2 )); then
        echo "validated  GPU idle for two consecutive samples"
        return 0
      fi
    else
      consecutive=0
    fi
    sleep 5
  done
  echo "error: GPU did not become stably idle within 10 minutes" >&2
  return 1
}

"${python_bin}" - <<'PY'
import torch
import triglu

if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is not available")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("error: CUDA BF16 is not supported")
print(f"runtime    Python/Torch ready | {torch.__version__} | {triglu.__file__}")
PY

for spec in "${train_specs[@]}"; do
  IFS="|" read -r config_path seed run_name _ _ <<<"${spec}"
  output_dir="runs/${suite}/${run_name}"
  if [[ -e "${output_dir}" ]]; then
    validate_complete_run "${config_path}" "${seed}" "${output_dir}"
    echo "skip       ${run_name} | exact completed run already exists"
    continue
  fi
  current_sha="$(sha256sum "${config_path}" | cut -d' ' -f1)"
  if [[ "${current_sha}" != "${config_sha[${config_path}]}" ]]; then
    echo "error: config changed before launch: ${config_path}" >&2
    exit 1
  fi
  wait_for_idle_gpu
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
echo "complete   nine-layer FFN training phase"

for context_length in "${contexts[@]}"; do
  prompt_length=$((context_length - decode_tokens))
  for config_path in "${grouped_config}" "${standard_config}"; do
    plan="$(basename "${config_path}" .yaml)"
    output="results/generated/${suite}-context-${context_length}-${plan}-final-validation-benchmark.json"
    if [[ -f "${output}" ]]; then
      validate_benchmark \
        "${output}" "${config_path}" "${context_length}" "${prompt_length}"
      echo "skip       ${plan} @ ${context_length} | exact artifact exists"
      continue
    fi
    wait_for_idle_gpu
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

"${python_bin}" - "${suite}" "${benchmark_label}" "${warmup}" "${iterations}" <<'PY'
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
        "configs/20l_4k_1b/ablations/15a5_triglu_no_rope_front_blend.yaml"
    ),
    "9a11_triglu_no_rope_nested": (
        "configs/20l_4k_1b/placement_amount/9a11_triglu_no_rope_nested.yaml"
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
    "9l_9a0t_grouped_triglu_no_rope_ffn": (
        "configs/20l_4k_1b/ablations/"
        "9l_9a0t_grouped_triglu_no_rope_ffn.yaml"
    ),
    "9l_9a0t_standard_swiglu": (
        "configs/20l_4k_1b/ablations/9l_9a0t_standard_swiglu.yaml"
    ),
}
report_dir = Path("results/generated") / f"{suite}-report"
with (report_dir / "context_benchmarks.csv").open(
    "r", encoding="utf-8", newline=""
) as handle:
    context_rows = list(csv.DictReader(handle))
selected = {
    (row["architecture"], int(row["context_length"])): row
    for row in context_rows
    if row["architecture"] in sources and int(row["context_length"]) in contexts
}
expected = {(plan, context) for plan in sources for context in contexts}
if selected.keys() != expected:
    raise SystemExit(
        "error: regenerated report has wrong final-validation cohort; "
        f"missing={sorted(expected - selected.keys())}, "
        f"extra={sorted(selected.keys() - expected)}"
    )
for (plan, context), row in selected.items():
    wanted = {
        "benchmark_label": label,
        "benchmark_source": sources[plan],
        "configured_context_length": "4096",
        "prompt_length": str(context - 128),
        "decode_tokens": "128",
        "warmup": str(warmup),
        "iterations": str(iterations),
        "dtype": "bfloat16",
    }
    for field, value in wanted.items():
        if row[field] != value:
            raise SystemExit(
                f"error: report selected wrong {field} for {(plan, context)}: "
                f"{row[field]!r}, expected {value!r}"
            )

with (report_dir / "summary_by_architecture.csv").open(
    "r", encoding="utf-8", newline=""
) as handle:
    summaries = {row["architecture"]: row for row in csv.DictReader(handle)}
expected_summaries = {
    # Seed counts reflect the collapse-seed replications; the shallow control
    # is reclassified to residual_topology_control by the reporter's
    # reference-depth rule.
    "9l_9a0t_grouped_triglu_no_rope_ffn": {
        "experiment_family": "residual_topology_control",
        "runs": "3",
        "seeds": "1337,2357,7331",
        "n_layers": "9",
        "ffn_type": "triglu_no_rope",
        "residual_init_depth": "20",
        "parameter_count": "89019904",
    },
    "9l_9a0t_standard_swiglu": {
        "experiment_family": "residual_topology_control",
        "runs": "1",
        "seeds": "1337",
        "n_layers": "9",
        "ffn_type": "swiglu",
        "residual_init_depth": "9",
        "parameter_count": "54224384",
    },
}
for architecture, wanted in expected_summaries.items():
    row = summaries.get(architecture)
    if row is None:
        raise SystemExit(f"error: report is missing architecture {architecture}")
    for field, value in wanted.items():
        if row[field] != value:
            raise SystemExit(
                f"error: wrong {field} for {architecture}: "
                f"{row[field]!r}, expected {value!r}"
            )
    if not row["validation_loss_mean"]:
        raise SystemExit(f"error: report has no validation loss for {architecture}")

print(f"validated  report selected the complete {len(expected)}-row final cohort")
print("validated  both nine-layer architecture summaries")
PY

echo "complete   nine-layer FFN control queue"
