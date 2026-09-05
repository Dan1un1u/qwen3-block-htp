#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(104)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(210)' include/block_protocol.h
grep -q '#define QBH_BLOCK_W4U8_O_MAX_BATCH_N_TILES UINT32_C(8)' \
    src/dsp/block_imp.c
grep -q 'desc == &header->projections\[QBH_BLOCK_PROJ_O\].*' \
    src/dsp/block_imp.c
grep -q 'QBH_BLOCK_HVX_POOL_U8_SWIGLU_STREAM' src/dsp/block_imp.c
grep -q 'qbh_w4u8_swiglu_stream_worker_run' src/dsp/block_imp.c
grep -q 'qbh_publish_w4u8_gate_up_swiglu_group' src/dsp/block_imp.c
grep -q 'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_SWIGLU_STREAM' \
    src/host/block_main.c scripts/run_exp0173.sh scripts/run_exp0210.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES=16' scripts/run_exp0210.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_Q_BATCH_N_TILES=32' scripts/run_exp0210.sh
grep -q 'QBH_EXP0176_O_BATCH_TILES="${o_batch}"' scripts/run_exp0210.sh
grep -q 'w4u8_decode_direct_n_down_batch_n_tiles' include/block_protocol.h \
    src/dsp/block_imp.c src/host/block_main.c
grep -q 'w4u8_decode_direct_n_down_single_dma' include/block_protocol.h \
    src/dsp/block_imp.c src/host/block_main.c
grep -q 'w4u8_decode_direct_n_q_batch_n_tiles' include/block_protocol.h \
    src/dsp/block_imp.c src/host/block_main.c
grep -q 'w4u8_decode_direct_n_o_single_dma' include/block_protocol.h \
    src/dsp/block_imp.c src/host/block_main.c
grep -q 'QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES=4' \
    scripts/run_exp0210.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_DOWN_SINGLE_DMA=1' \
    scripts/run_exp0210.sh
grep -q 'QBH_W4U8_DECODE_DIRECT_N_O_SINGLE_DMA="${o_single_dma}"' \
    scripts/run_exp0210.sh
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt | \
        grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0210.sh scripts/deploy_exp0210.sh \
    scripts/run_exp0210.sh scripts/run_exp0210_short_gate.sh \
    scripts/run_exp0210_formal.sh scripts/check_exp0210_static.sh
python3 -m py_compile scripts/summarize_exp0189.py scripts/summarize_exp0210.py
printf '%s\n' '{"experiment":"EXP-0210","static_gate":"pass","candidate":"direct-W4-O16-single-DMA","qnn_dependency":false}'
