#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
past_length="${1:?past length required}"
package="${QBH_EXP0161_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0161/decode_l${past_length}_w4u8_hmx_delta_v2}"
remote_root="${EXP0161_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0161-l${past_length}}"

EXP0147_REMOTE_ROOT="${remote_root}" \
    bash "${project_root}/scripts/deploy_exp0147_package.sh" "${package}"
