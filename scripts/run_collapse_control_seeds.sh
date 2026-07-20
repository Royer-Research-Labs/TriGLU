#!/usr/bin/env bash
# Replicate the parameter-matched collapse controls at seeds 2357 and 7331.
#
# The seed-1337 collapse controls show that a 20-layer hybrid beats a
# parameter-matched 9-layer "collapsed" model whose freed budget is grouped
# into fat FFNs at the same depths. This launcher adds the two confirmatory
# seeds so the structure-beats-concentrated-capacity result is replicated
# rather than exploratory. The matched 9a11 hybrids already exist at all three
# seeds; only the collapses are missing at 2357/7331.
#
# Runtime: 4 runs at roughly 75 min each on an RTX PRO 6000 (~5 hours total).
# Set PYTHON_BIN if the active interpreter is not named `python`.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON_BIN:-python}"
suite="20l_4k_1b"
seeds=(2357 7331)

# collapse plan -> matched hybrid whose per-seed run must already be complete,
# so the structure gap is computable at every replicated seed.
plans=(
  9l_9a0t_grouped_triglu_no_rope_ffn
  9l_9a0t_grouped_swiglu_9a11_nested_collapse
)
matched_hybrid=(
  9a11_triglu_no_rope_nested
  9a11_swiglu_nested
)

log_dir="runs/logs"
mkdir -p "${log_dir}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${log_dir}/collapse-control-seeds-${stamp}.log"
exec > >(tee -a "${log_file}") 2>&1
echo "collapse-control-seeds run ${stamp}"

# Complete means the FINAL metrics line is the complete event, matching the
# other launchers; a stray complete followed by resumed lines does not count.
is_complete() { [[ -f "$1" ]] && tail -n 1 "$1" | grep -q '"event": "complete"'; }

command -v "${python_bin}" >/dev/null 2>&1 || {
  echo "error: PYTHON_BIN '${python_bin}' is not executable" >&2
  exit 1
}

# Preflight: fail before launching anything if a config or a matched hybrid is
# missing, so we never burn GPU time on a comparison that cannot be computed.
for i in "${!plans[@]}"; do
  plan="${plans[$i]}"
  config="configs/${suite}/ablations/${plan}.yaml"
  [[ -f "${config}" ]] || { echo "error: missing config ${config}" >&2; exit 1; }
  for seed in "${seeds[@]}"; do
    hybrid_metrics="runs/${suite}/${matched_hybrid[$i]}_seed${seed}/metrics.jsonl"
    if ! is_complete "${hybrid_metrics}"; then
      echo "error: matched hybrid ${matched_hybrid[$i]}_seed${seed} is not complete; needed for the structure comparison" >&2
      exit 1
    fi
  done
done
# Refuse any pre-existing incomplete output dir BEFORE the first training run,
# so a conflict on a later plan cannot waste hours on the earlier ones.
for seed in "${seeds[@]}"; do
  for plan in "${plans[@]}"; do
    output_dir="runs/${suite}/${plan}_seed${seed}"
    if [[ -e "${output_dir}" ]] && ! is_complete "${output_dir}/metrics.jsonl"; then
      echo "error: ${output_dir} exists but is not complete; move it aside before launching" >&2
      exit 1
    fi
  done
done
echo "preflight ok: interpreter, configs, matched hybrids, and output dirs verified"

for seed in "${seeds[@]}"; do
  for plan in "${plans[@]}"; do
    run_name="${plan}_seed${seed}"
    output_dir="runs/${suite}/${run_name}"
    metrics="${output_dir}/metrics.jsonl"

    if is_complete "${metrics}"; then
      echo "skip       ${run_name} | already complete"
      continue
    fi
    if [[ -e "${output_dir}" ]]; then
      echo "error: ${output_dir} exists but is not complete; move it aside before retrying" >&2
      exit 1
    fi

    echo "train      ${run_name} | start $(date -u +%H:%M:%SZ)"
    "${python_bin}" -m triglu.train \
      --config "configs/${suite}/ablations/${plan}.yaml" \
      --seed "${seed}" \
      --output-dir "${output_dir}"
    echo "done       ${run_name} | end   $(date -u +%H:%M:%SZ)"
  done
done

echo "regenerating suite report"
"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated

echo "collapse-control-seeds complete; log at ${log_file}"
