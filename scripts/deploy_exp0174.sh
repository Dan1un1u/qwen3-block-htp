#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXP0173_REMOTE_ROOT="${EXP0174_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0173-exp0174-w4u8}"
export QBH_EXP0173_ADB_STAGE="${QBH_EXP0174_ADB_STAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/.adb-staging/exp0174}"
exec "${project_root}/scripts/deploy_exp0173.sh"
