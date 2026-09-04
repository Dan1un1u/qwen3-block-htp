#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(94)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(199)' include/block_protocol.h
grep -q 'state.tiles_per_batch != 16U' src/dsp/block_imp.c
grep -q 'state.tiles_per_batch == 16U' src/dsp/block_imp.c
grep -q 'state.slot_count = state.direct_n_weights' src/dsp/block_imp.c
grep -q 'batch_index % state.slot_count' src/dsp/block_imp.c
grep -q 'state.bias_slots\[slot\] = slot == 0U' src/dsp/block_imp.c
grep -q '4|8|16' scripts/run_exp0173.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES="${qkv_batch}"' \
    scripts/run_exp0199.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_CONTINUOUS=1' \
    scripts/run_exp0199.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0199.sh scripts/deploy_exp0199.sh \
    scripts/run_exp0199.sh scripts/run_exp0199_short_gate.sh \
    scripts/run_exp0199_formal.sh scripts/check_exp0199_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0199.py
printf '%s\n' '{"experiment":"EXP-0199","static_gate":"pass","candidate":"direct-n-QKV-batch16-two-slot","qnn_dependency":false}'
