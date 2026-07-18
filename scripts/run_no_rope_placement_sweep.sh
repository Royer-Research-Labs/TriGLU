#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON_BIN:-python}"
suite="20l_4k_1b"
seed=1337
config_dir="configs/${suite}/placement_amount"
plans=(
  18a2_triglu_no_rope_nested
  15a5_triglu_no_rope_nested
  12a8_triglu_no_rope_nested
  9a11_triglu_no_rope_nested
  6a14_triglu_no_rope_nested
  15a5_triglu_no_rope_repeating
  15a5_triglu_no_rope_tail_block
)

# The seed-1337 all-attention and no-RoPE front-blend runs are reused as
# comparison points and must be complete.
prerequisites=(
  20a0t
  15a5_triglu_no_rope_front_blend
)

for prerequisite in "${prerequisites[@]}"; do
  metrics="runs/${suite}/${prerequisite}/metrics.jsonl"
  if [[ ! -f "${metrics}" ]] ||
    ! grep -q '"event": "complete"' "${metrics}"; then
    echo "error: complete prerequisite run ${prerequisite} is required" >&2
    exit 1
  fi
done

# Refuse to contend with any incomplete run already recorded in this suite.
# In the intended workflow this catches the active prior-art launcher and makes
# the user wait for it to return before beginning the placement study.
for metrics in runs/${suite}/*/metrics.jsonl; do
  [[ -f "${metrics}" ]] || continue
  if ! grep -q '"event": "complete"' "${metrics}"; then
    echo "error: ${metrics} is incomplete; let the current training process finish first" >&2
    exit 1
  fi
done

# Validate every target before launching the first nine-hour sequence. This
# prevents a later stale directory from wasting earlier completed GPU work.
for plan in "${plans[@]}"; do
  output_dir="runs/${suite}/${plan}"
  metrics="${output_dir}/metrics.jsonl"
  if [[ -f "${metrics}" ]] && grep -q '"event": "complete"' "${metrics}"; then
    continue
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "error: ${output_dir} exists but is not complete; move it aside before retrying" >&2
    exit 1
  fi
done

for plan in "${plans[@]}"; do
  output_dir="runs/${suite}/${plan}"
  metrics="${output_dir}/metrics.jsonl"

  if [[ -f "${metrics}" ]] && grep -q '"event": "complete"' "${metrics}"; then
    echo "skip       ${plan} | already complete"
    continue
  fi
  "${python_bin}" -m triglu.train \
    --config "${config_dir}/${plan}.yaml" \
    --seed "${seed}" \
    --output-dir "${output_dir}"
done

"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated
