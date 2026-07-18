#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON_BIN:-python}"
suite="20l_4k_1b"
read -r -a seeds <<< "${ABLATION_SEEDS:-1337 2357 7331}"
plans=(
  15a5_triglu_no_rope_front_blend
  15a5_mb_mlp_front_blend
  15a5_swiglu_front_blend
)
references=(
  20a0t
  15a5t_front_blend
)

# Refuse before launching anything if the matched all-attention or TriGLU
# reference is absent. Otherwise the report could not compute seed-matched
# deltas, and a partial suite would waste substantial GPU time.
for seed in "${seeds[@]}"; do
  if [[ "${seed}" == "1337" ]]; then
    seed_suffix=""
  else
    seed_suffix="_seed${seed}"
  fi
  for reference in "${references[@]}"; do
    reference_name="${reference}${seed_suffix}"
    reference_metrics="runs/${suite}/${reference_name}/metrics.jsonl"
    if [[ ! -f "${reference_metrics}" ]] ||
      ! grep -q '"event": "complete"' "${reference_metrics}"; then
      echo "error: complete reference run ${reference_name} is required before launching the controls" >&2
      exit 1
    fi
  done
done

for seed in "${seeds[@]}"; do
  if [[ "${seed}" == "1337" ]]; then
    seed_suffix=""
  else
    seed_suffix="_seed${seed}"
  fi
  for plan in "${plans[@]}"; do
    run_name="${plan}${seed_suffix}"
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
      --config "configs/${suite}/ablations/${plan}.yaml" \
      --seed "${seed}" \
      --output-dir "${output_dir}"
  done
done

"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated
