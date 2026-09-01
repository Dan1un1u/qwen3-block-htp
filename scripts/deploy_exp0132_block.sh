#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fp16_package="${QBH_EXP0132_FP16_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/block_package_layer14_m64}"
remote_root="${EXP0132_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0132-w4f16}"

EXP0022_PACKAGE_DIR="${fp16_package}" REMOTE_ROOT="${remote_root}" \
    "${project_root}/scripts/deploy_exp0022_block.sh"
