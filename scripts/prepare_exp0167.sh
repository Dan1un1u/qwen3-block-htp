#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exe="${QBH_PYTHON:-/home/daniuniu/.cache/qwen3-block-htp-py/bin/python}"
logical_source="${QBH_EXP0167_LOGICAL_SOURCE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0163/logical_capacity257_source}"
output_root="${QBH_EXP0167_OUTPUT_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0167}"

"${python_exe}" "${project_root}/scripts/prepare_exp0162_cache_packages.py" \
    --source "${logical_source}" \
    --output-root "${output_root}" \
    --capacity 80 --decode-steps 16 --experiment EXP-0167 \
    --mode candidate --candidate-name base_segmented_capacity80

"${python_exe}" \
    "${project_root}/scripts/prepare_exp0167_generation_package.py"

