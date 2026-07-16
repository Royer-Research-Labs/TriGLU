#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON_BIN:-python}"
suite="20l_4k_1b"
plans=(20a0t 15a5t_front_blend)
contexts=(1024 2048 4096 8192 16384)
decode_tokens="${DECODE_TOKENS:-128}"
warmup="${WARMUP:-3}"
iterations="${ITERATIONS:-10}"

mkdir -p results/generated

for context_length in "${contexts[@]}"; do
  prompt_length=$((context_length - decode_tokens))
  if ((prompt_length <= 0)); then
    echo "error: DECODE_TOKENS must be smaller than every benchmark context" >&2
    exit 2
  fi

  for plan in "${plans[@]}"; do
    output="results/generated/${suite}-context-${context_length}-${plan}-benchmark.json"
    if [[ -f "${output}" ]]; then
      echo "skip       ${plan} @ ${context_length} | ${output} exists"
      continue
    fi

    echo "benchmark  ${plan} @ ${context_length} | prompt ${prompt_length}, decode ${decode_tokens}"
    "${python_bin}" -m triglu.benchmark \
      --config "configs/${suite}/${plan}.yaml" \
      --context-length "${context_length}" \
      --batch-size 1 \
      --sequence-length "${context_length}" \
      --prompt-length "${prompt_length}" \
      --decode-tokens "${decode_tokens}" \
      --warmup "${warmup}" \
      --iterations "${iterations}" \
      --device cuda --dtype bfloat16 --compile \
      --label context-scaling \
      --output "${output}"
  done
done

"${python_bin}" -m triglu.report \
  --suite "${suite}" \
  --runs-root runs \
  --results-root results/generated
