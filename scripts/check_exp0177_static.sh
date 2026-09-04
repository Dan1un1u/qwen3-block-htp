#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(76)' include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(177)' include/block_protocol.h
grep -q 'QBH_BLOCK_W4U8_AV_REQUANT_DECODE_ROWS UINT32_C(4)' include/block_protocol.h
grep -q 'w4u8_decode_av_requant_rows' include/block_protocol.h
grep -q 'w4u8_av_requant_vector_count' include/block_protocol.h
grep -q 'QBH_W4U8_DECODE_AV_REQUANT_ROWS' src/host/block_main.c
grep -q 'qbh_attention_u8_requant_av_rows' src/dsp/attention_u8_core.c
grep -q 'qbh_attention_u8_poison_av_padding' src/dsp/attention_u8_core.c
if grep -RIn -E 'Qnn|QAIRT|qti\.aisw' src include CMakeLists.txt \
        | grep -v 'qnn":"none' >/dev/null; then
    printf 'unexpected QNN dependency\n' >&2
    exit 1
fi
bash -n scripts/build_exp0177.sh scripts/deploy_exp0177.sh \
    scripts/run_exp0173.sh scripts/run_exp0177.sh \
    scripts/run_exp0177_short_gate.sh scripts/check_exp0177_static.sh
python3 -m py_compile scripts/summarize_exp0177_short.py
printf '%s\n' '{"experiment":"EXP-0177","static_gate":"pass","variant":"W4U8","control_rows":64,"candidate_rows":4,"padding_poison_audit":true,"qnn_dependency":false}'
