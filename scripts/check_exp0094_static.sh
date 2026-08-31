#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

grep -q '#define QBH_BLOCK_ABI_VERSION UINT32_C(43)' \
    include/block_protocol.h
grep -q '#define QBH_BLOCK_EXPERIMENT UINT32_C(94)' \
    include/block_protocol.h
grep -q 'DEPENDENCY_STREAM_V_CACHE = 16' include/block_protocol.h
grep -q 'QBH_ATTN_U8_PERSISTENT_V_CACHE_BYTES' \
    include/attention_u8_core.h
grep -q 'qbh_attention_u8_build_v_recenter_lut_and_av_bias' \
    src/dsp/attention_u8_core.c
grep -q 'qbh_attention_u8_pack_v_native_vgather_vdeal_prebuilt' \
    src/dsp/attention_u8_core.c
grep -q 'qbh_prepare_attention_v_cache' src/dsp/block_imp.c
grep -q 'attention_v_cache_session_build_group_count' \
    src/dsp/probe_imp.c include/block_protocol.h
if grep -RInE 'Qnn|qti\.aisw' src include CMakeLists.txt >/dev/null; then
    printf 'QNN dependency detected\n' >&2
    exit 1
fi

python3 - <<'PY'
import json

print(json.dumps({
    "experiment": "EXP-0094",
    "static_gate": "pass",
    "backend": "standalone_fastrpc_dsp",
    "qnn": "none",
    "persistent_v_cache_bytes": 12288,
    "persistent_groups": 8,
}, sort_keys=True))
PY
