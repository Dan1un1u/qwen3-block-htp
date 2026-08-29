#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0044-block}"
export QBH_EXP0042_PACKAGE="${QBH_EXP0044_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
exec "${project_root}/scripts/deploy_exp0042_block.sh"
