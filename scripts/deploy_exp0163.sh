#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:?control or candidate required}"
case "${mode}" in
control)
    package="${QBH_EXP0163_CONTROL_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0163/control_delta_capacity257}"
    ;;
candidate)
    package="${QBH_EXP0163_CANDIDATE_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0163/candidate_segmented_capacity257}"
    ;;
*) exit 2 ;;
esac

EXP0147_REMOTE_ROOT="${EXP0163_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0163-${mode}}" \
    bash "${project_root}/scripts/deploy_exp0147_package.sh" "${package}"
