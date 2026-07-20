#!/usr/bin/env bash
# Exploratory seed-1337 runs for two follow-up questions raised after the
# v0.2.0 control sweep:
#
#   1. Combination: does using the no-RoPE triple product in BOTH roles at once
#      (9a11 nested replacement + triple-product FFN, parameter-matched) stack
#      the independently measured replacement and FFN wins?
#   2. Merged/parallel block: the single-norm x + mixer(norm(x)) + ffn(norm(x))
#      speed/quality lever, run all-attention (isolates the merge) and on the
#      9a11 hybrid (tests whether merging and replacement compose).
#
# All three are single-seed probes; replicate whichever earns it afterward.
# Runtime: 3 runs at roughly 75 min each on an RTX PRO 6000 (~3.75 hours).
# Set PYTHON_BIN if the active interpreter is not named `python`.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON_BIN:-python}"
suite="20l_4k_1b"
plans=(
  9a11_triglu_no_rope_nested_triglu_ffn
  20a0t_parallel_block
  9a11_triglu_no_rope_nested_parallel_block
)

log_dir="runs/logs"
mkdir -p "${log_dir}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${log_dir}/combination-and-parallel-controls-${stamp}.log"
exec > >(tee -a "${log_file}") 2>&1
echo "combination-and-parallel-controls run ${stamp}"

# Complete means the FINAL metrics line is the complete event, matching the
# other launchers; a stray complete followed by resumed lines does not count.
is_complete() { [[ -f "$1" ]] && tail -n 1 "$1" | grep -q '"event": "complete"'; }

command -v "${python_bin}" >/dev/null 2>&1 || {
  echo "error: PYTHON_BIN '${python_bin}' is not executable" >&2
  exit 1
}

# Preflight: configs present, the pinned 1B token stream prepared, the seed-1337
# comparison references complete, and no incomplete output dir in the way —
# all before the first GPU-hour is spent.
for plan in "${plans[@]}"; do
  config="configs/${suite}/ablations/${plan}.yaml"
  [[ -f "${config}" ]] || { echo "error: missing config ${config}" >&2; exit 1; }
done
for split in train val; do
  bin="data/fineweb_edu_10bt_1b/${split}.bin"
  [[ -f "${bin}" ]] || { echo "error: missing prepared data ${bin}" >&2; exit 1; }
done
for reference in 20a0t 9a11_triglu_no_rope_nested; do
  if ! is_complete "runs/${suite}/${reference}/metrics.jsonl"; then
    echo "error: comparison reference ${reference} is not complete" >&2
    exit 1
  fi
done
for plan in "${plans[@]}"; do
  output_dir="runs/${suite}/${plan}"
  if [[ -e "${output_dir}" ]] && ! is_complete "${output_dir}/metrics.jsonl"; then
    echo "error: ${output_dir} exists but is not complete; move it aside before launching" >&2
    exit 1
  fi
done
echo "preflight ok: interpreter, configs, data, references, and output dirs verified"

for plan in "${plans[@]}"; do
  output_dir="runs/${suite}/${plan}"
  metrics="${output_dir}/metrics.jsonl"

  if is_complete "${metrics}"; then
    echo "skip       ${plan} | already complete"
    continue
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "error: ${output_dir} exists but is not complete; move it aside before retrying" >&2
    exit 1
  fi

  echo "train      ${plan} | start $(date -u +%H:%M:%SZ)"
  "${python_bin}" -m triglu.train \
    --config "configs/${suite}/ablations/${plan}.yaml"
  echo "done       ${plan} | end   $(date -u +%H:%M:%SZ)"
done

echo "regenerating suite report"
"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated

echo "combination-and-parallel-controls complete; log at ${log_file}"
