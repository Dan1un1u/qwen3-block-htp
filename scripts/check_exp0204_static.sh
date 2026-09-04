#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(97)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(204)' include/block_protocol.h
grep -q 'QBH_BLOCK_HVX_POOL_U8_SWIGLU_STREAM' src/dsp/block_imp.c
grep -q 'qbh_w4u8_swiglu_stream_worker_run' src/dsp/block_imp.c
grep -q 'qbh_publish_w4u8_gate_up_swiglu_group' src/dsp/block_imp.c
grep -q 'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_SWIGLU_STREAM' \
    src/host/block_main.c scripts/run_exp0173.sh scripts/run_exp0204.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES=16' scripts/run_exp0204.sh
grep -q 'QBH_EXP0176_O_BATCH_TILES="${o_batch}"' scripts/run_exp0204.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0204.sh scripts/deploy_exp0204.sh \
    scripts/run_exp0204.sh scripts/run_exp0204_short_gate.sh \
    scripts/run_exp0204_formal.sh scripts/check_exp0204_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0204.py
printf '%s\n' '{"experiment":"EXP-0204","static_gate":"pass","candidate":"direct-W4-O-batch8","qnn_dependency":false}'
