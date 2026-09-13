#!/usr/bin/env bash
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py preflight --source-worktree "$root"
source "$root/scripts/env_exp0001.sh"
cd "$root"
build_cmake android BUILD=ReleaseG QBH_MODEL_LLAMA32=OFF QBH_NATIVE_SP2=ON QBH_EXP0247_DENSE_R3=OFF QBH_EXP0240_SINGLE_LAYER=OFF QBH_EXP0257_LAYER_COUNT=${1:-28} -j8
build_cmake hexagon BUILD=ReleaseG DSP_ARCH=v79 QBH_MODEL_LLAMA32=OFF QBH_NATIVE_SP2=ON QBH_EXP0247_DENSE_R3=OFF QBH_EXP0240_SINGLE_LAYER=OFF QBH_EXP0257_LAYER_COUNT=${1:-28} -j8

mkdir -p "$root/build"
python3 - "$root" <<'PYSEAL'
from pathlib import Path
import json,hashlib,subprocess,sys
r=Path(sys.argv[1]);files=[r/'android_ReleaseG_aarch64/ship/qwen3_block_cli',r/'android_ReleaseG_aarch64/ship/llama_sp2_cli',r/'android_ReleaseG_aarch64/ship/libqwen3_probe.so',r/'hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so']
if all(p.exists() for p in files):
 record=dict(source_head=subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],text=True).strip(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
 (r/'build/qwen3-sp2-build-seal.json').write_text(json.dumps(record,indent=2)+'\n')
PYSEAL
