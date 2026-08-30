#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package="${QBH_EXP0061_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel}"
remote_root="${EXP0061_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0061}"
QBH_EXP0060_PACKAGE="${package}" EXP0060_REMOTE_ROOT="${remote_root}" \
    exec "${project_root}/scripts/deploy_exp0060_block.sh"
