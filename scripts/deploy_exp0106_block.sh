#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fp16_package="${QBH_EXP0106_FP16_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0022/block_package_layer14_m64}"
u8_package="${QBH_EXP0106_U8_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
fp16_remote="${EXP0106_FP16_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0106-fp16}"
u8_remote="${EXP0106_U8_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0106-u8}"

EXP0022_PACKAGE_DIR="${fp16_package}" REMOTE_ROOT="${fp16_remote}" \
    "${project_root}/scripts/deploy_exp0022_block.sh"
QBH_EXP0079_PACKAGE="${u8_package}" EXP0079_REMOTE_ROOT="${u8_remote}" \
    "${project_root}/scripts/deploy_exp0079_block.sh"
