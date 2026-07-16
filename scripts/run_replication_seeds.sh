#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON_BIN:-python}"
suite="20l_4k_1b"
seeds=(2357 7331)
plans=(20a0t 15a5t_front_blend 15a5t_final_attention)

for seed in "${seeds[@]}"; do
  for plan in "${plans[@]}"; do
    run_name="${plan}_seed${seed}"
    output_dir="runs/${suite}/${run_name}"
    metrics="${output_dir}/metrics.jsonl"

    if [[ -f "${metrics}" ]] && grep -q '"event": "complete"' "${metrics}"; then
      echo "skip       ${run_name} | already complete"
      continue
    fi
    if [[ -e "${output_dir}" ]]; then
      echo "error: ${output_dir} exists but is not complete; move it aside before retrying" >&2
      exit 1
    fi

    "${python_bin}" -m triglu.train \
      --config "configs/${suite}/${plan}.yaml" \
      --seed "${seed}" \
      --output-dir "${output_dir}"
  done
done

"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated
