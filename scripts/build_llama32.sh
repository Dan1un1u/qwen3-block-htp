#!/usr/bin/env bash
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 /home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py preflight --source-worktree "$root"
source "$root/scripts/env_exp0001.sh"
cd "$root"
build_cmake android BUILD=ReleaseG QBH_MODEL_LLAMA32=ON QBH_LLAMA_LAYER_COUNT=16 -j8
build_cmake hexagon BUILD=ReleaseG DSP_ARCH=v79 QBH_MODEL_LLAMA32=ON QBH_LLAMA_LAYER_COUNT=16 -j8
