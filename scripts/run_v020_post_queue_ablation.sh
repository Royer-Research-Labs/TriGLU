#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly python_command="${PYTHON_BIN:-python}"
python_bin="$(command -v "${python_command}" 2>/dev/null || true)"
readonly python_bin
readonly suite="20l_4k_1b"
readonly plan="15a5_triglu_no_rope_early_intrusion"
readonly config="configs/${suite}/placement_amount/${plan}.yaml"
readonly output_dir="runs/${suite}/${plan}"
readonly metrics="${output_dir}/metrics.jsonl"
# Default to the newest overnight-queue log; a clean checkout has none, in
# which case UPSTREAM_LOG must name the log to gate on.
default_upstream_log="$(ls -t runs/logs/v0.2.0-overnight-*.log 2>/dev/null | head -n 1 || true)"
readonly upstream_log="${UPSTREAM_LOG:-${default_upstream_log}}"
if [[ -z "${upstream_log}" || ! -f "${upstream_log}" ]]; then
  echo "error: no overnight queue log found under runs/logs/; set UPSTREAM_LOG to the completed queue log" >&2
  exit 1
fi

mkdir -p runs/logs
readonly log_file="runs/logs/v0.2.0-post-queue-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${log_file}") 2>&1

trap 'status=$?; echo "error: post-queue ablation stopped with status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2; exit "${status}"' ERR

echo "post-queue log ${log_file}"

if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "error: required interpreter is not executable: ${python_command}" >&2
  exit 1
fi
if [[ ! -f "${upstream_log}" ]] ||
  ! tail -n 1 "${upstream_log}" | grep -q '^complete   v0.2.0 overnight queue$'; then
  echo "error: the primary overnight queue did not finish cleanly" >&2
  exit 1
fi

for required_run in \
  9a11_triglu_no_rope_nested_seed2357 \
  9a11_triglu_no_rope_nested_seed7331 \
  15a5_swiglu_repeating \
  9a11_swiglu_nested
do
  required_metrics="runs/${suite}/${required_run}/metrics.jsonl"
  if [[ ! -f "${required_metrics}" ]] ||
    ! tail -n 1 "${required_metrics}" | grep -q '"event": "complete"'; then
    echo "error: complete upstream run ${required_run} is required" >&2
    exit 1
  fi
done

benchmark_plans=(
  20a0t
  15a5_triglu_no_rope_front_blend
  9a11_triglu_no_rope_nested
  15a5_swiglu_front_blend
  9a11_swiglu_nested
)
for context_length in 4096 8192 16384; do
  for benchmark_plan in "${benchmark_plans[@]}"; do
    artifact="results/generated/${suite}-context-${context_length}-${benchmark_plan}-benchmark.json"
    if [[ ! -f "${artifact}" ]]; then
      echo "error: required benchmark artifact is missing: ${artifact}" >&2
      exit 1
    fi
  done
done

for existing_metrics in runs/${suite}/*/metrics.jsonl; do
  [[ -f "${existing_metrics}" ]] || continue
  if ! tail -n 1 "${existing_metrics}" | grep -q '"event": "complete"'; then
    echo "error: ${existing_metrics} is incomplete" >&2
    exit 1
  fi
done

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

if [[ -f "${metrics}" ]] &&
  tail -n 1 "${metrics}" | grep -q '"event": "complete"'; then
  echo "skip       ${plan} | already complete"
elif [[ -e "${output_dir}" ]]; then
  echo "error: ${output_dir} exists but is not complete" >&2
  exit 1
else
  echo "train      ${plan} | seed 1337"
  "${python_bin}" -m triglu.train \
    --config "${config}" \
    --seed 1337 \
    --output-dir "${output_dir}"
fi

if [[ ! -f "${metrics}" ]] ||
  ! tail -n 1 "${metrics}" | grep -q '"event": "complete"'; then
  echo "error: ${plan} did not finish with a complete event" >&2
  exit 1
fi

"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated

echo "complete   ${plan}"
