#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fp16_package="${QBH_EXP0138_FP16_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/block_package_layer14_m64}"
remote_root="${EXP0138_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0138-w4f16}"

QBH_EXP0028_PACKAGE="${fp16_package}" EXP0028_REMOTE_ROOT="${remote_root}" \
    "${project_root}/scripts/deploy_exp0028_block.sh"
