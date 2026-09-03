#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(74)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(173)' include/block_protocol.h
grep -q 'w4u8_decode_lm_head_group_tiles' include/block_protocol.h
grep -q 'QBH_W4U8_DECODE_LM_HEAD_GROUP_TILES' src/host/block_main.c
grep -q 'decode_batch16' src/dsp/block_imp.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0173.sh scripts/deploy_exp0173.sh \
    scripts/run_exp0173.sh scripts/run_exp0173_short_gate.sh \
    scripts/run_exp0173_formal.sh scripts/check_exp0173_static.sh
python3 -m py_compile scripts/summarize_exp0173.py
printf '%s\n' '{"experiment":"EXP-0173","static_gate":"pass","variant":"W4U8","source_parent":"EXP-0170","control_batch":8,"candidate_batch":16,"qnn_dependency":false}'
