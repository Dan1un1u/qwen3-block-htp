#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP0173_REMOTE_ROOT=/data/local/tmp/qwen3-block-htp/exp0173-exp0183-w4u8 \
QBH_EXP0173_ADB_STAGE=/mnt/d/llm_exp/models/qwen3-block-htp/.adb-staging/exp0183 \
    "${project_root}/scripts/deploy_exp0173.sh"
