#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP0173_REMOTE_ROOT=/data/local/tmp/qwen3-block-htp/exp0215-prefill-direct-w4-mlp \
QBH_EXP0173_ADB_STAGE=/mnt/d/llm_exp/models/qwen3-block-htp/.adb-staging/exp0215 \
    "${project_root}/scripts/deploy_exp0173.sh"
