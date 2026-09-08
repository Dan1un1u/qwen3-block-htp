#!/usr/bin/env bash
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${root}/scripts/env_exp0001.sh"
cd "$root"
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py preflight --source-worktree "$root"
"${EXP0001_CMAKE_ROOT}/bin/cmake" -S . -B android_ReleaseG_aarch64 -DQBH_EXP0240_SINGLE_LAYER=OFF -DQBH_EXP0247_DENSE_R3=ON
"${EXP0001_CMAKE_ROOT}/bin/cmake" -S . -B hexagon_ReleaseG_toolv19_v79 -DQBH_EXP0240_SINGLE_LAYER=OFF -DQBH_EXP0247_DENSE_R3=ON
build_cmake android BUILD=ReleaseG
build_cmake hexagon BUILD=ReleaseG DSP_ARCH=v79
