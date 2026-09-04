#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(99)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(206)' include/block_protocol.h
grep -q 'state.tiles_per_batch != 32U' src/dsp/block_imp.c
grep -q 'state.tiles_per_batch >= 16U' src/dsp/block_imp.c
grep -q 'state.slot_count = state.direct_n_weights' src/dsp/block_imp.c
grep -q 'batch_index % state.slot_count' src/dsp/block_imp.c
grep -q 'state.bias_slots\[slot\] = slot == 0U' src/dsp/block_imp.c
grep -q 'w4u8_decode_direct_n_qkv_batch_n_tiles != 32U' src/host/block_main.c
grep -q 'state.tiles_per_batch == 32U' src/dsp/block_imp.c
grep -q '? buffers->gate' src/dsp/block_imp.c
grep -q '4|8|16|32' scripts/run_exp0173.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES="${qkv_batch}"' \
    scripts/run_exp0206.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_CONTINUOUS=1' \
    scripts/run_exp0206.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES=4' \
    scripts/run_exp0206.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0206.sh scripts/deploy_exp0206.sh \
    scripts/run_exp0206.sh scripts/run_exp0206_short_gate.sh \
    scripts/run_exp0206_formal.sh scripts/check_exp0206_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0206.py
printf '%s\n' '{"experiment":"EXP-0206","static_gate":"pass","candidate":"direct-W4-QKV-batch32-two-slot","qnn_dependency":false}'
