#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
u8_package="${QBH_EXP0118_U8_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
remote_root="${EXP0118_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0118-u8}"

QBH_EXP0079_PACKAGE="${u8_package}" EXP0079_REMOTE_ROOT="${remote_root}" \
    "${project_root}/scripts/deploy_exp0079_block.sh"
