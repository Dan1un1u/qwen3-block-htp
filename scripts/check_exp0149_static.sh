#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${QBH_EXP0149_MODEL_ROOT:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0149}"
host_cli="${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
host_stub="${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
dsp_skel="${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so"

for artifact in "${host_cli}" "${host_stub}" "${dsp_skel}"; do
    test -f "${artifact}"
done
grep -q 'QBH_BLOCK_ABI_VERSION UINT32_C(56)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_BLOCK_EXPERIMENT UINT32_C(149)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_DECODE_SESSION_ABI_VERSION UINT32_C(2)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_VERTICAL_SLICE_FIRST_LAYER UINT32_C(13)' \
    "${project_root}/include/block_protocol.h"
grep -q 'QBH_VERTICAL_SLICE_LAYER_COUNT UINT32_C(3)' \
    "${project_root}/include/block_protocol.h"
grep -q '#define QBH_REPLAY_FP16_ATOL (0.0625)' \
    "${project_root}/src/host/block_main.c"
grep -q '#define QBH_REPLAY_FP16_RTOL (0.002)' \
    "${project_root}/src/host/block_main.c"
grep -q '#define QBH_REPLAY_FP16_MIN_COSINE (0.99999)' \
    "${project_root}/src/host/block_main.c"
grep -q 'output_mixed_tolerance_violations' \
    "${project_root}/src/host/block_main.c"

if grep -RInE 'QnnGraph|QnnContext|QnnBackend|libQnn' \
        "${project_root}/src" "${project_root}/include" >/dev/null; then
    printf 'QNN dependency found in EXP-0149 source\n' >&2
    exit 1
fi
if readelf -d "${host_cli}" "${host_stub}" 2>/dev/null \
        | grep -qi 'libQnn'; then
    printf 'QNN dependency found in EXP-0149 Host artifacts\n' >&2
    exit 1
fi

python3 - "${model_root}" <<'PY'
import json
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
record = struct.Struct("<32sfi2f")
for recipe in ("f16f16", "w4f16", "w4u8"):
    package = root / recipe
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["experiment"] == "EXP-0149"
    assert manifest["contract"]["active_layers"] == [13, 14, 15]
    qparams = {}
    for layer in (13, 14, 15):
        path = package / f"layer{layer}" / "qparams_u8.bin"
        payload = path.read_bytes()
        assert len(payload) == 17 * record.size
        layer_qparams = {}
        for offset in range(0, len(payload), record.size):
            name, scale, zero, minimum, maximum = record.unpack_from(
                payload, offset
            )
            layer_qparams[name.split(b"\0", 1)[0].decode()] = (scale, zero)
        qparams[layer] = layer_qparams
        suffix = "u8" if recipe == "w4u8" else "f16"
        expected_cache_bytes = 8 * 72 * 128 * (1 if suffix == "u8" else 2)
        for kind in ("k", "v"):
            assert (
                package / f"layer{layer}" / f"kv_cache_{kind}_{suffix}.bin"
            ).stat().st_size == expected_cache_bytes
    assert qparams[13]["block_output"] == qparams[14]["block_input"]
    assert qparams[14]["block_output"] == qparams[15]["block_input"]
PY

printf '%s\n' '{"experiment":"EXP-0149","static_gate":"pass","execution_unit":"qwen3_real_layers13_14_15_one_dsp_invocation","decode_session_abi":2,"active_layers":[13,14,15],"declared_layer_count":28,"prefill_positions":"0-63","decode_positions":"64-71","cache_capacity_per_layer":72,"host_hidden_handoff":false,"one_fastrpc_per_three_layer_step":true,"one_hmx_owner":true,"exact_vtcm_request_bytes":8388608,"intermediate_ddr_allowed":false,"persistent_kv_cache_ddr_allowed":true,"fp16_composition_gate":"abs(actual-reference)<=0.0625+0.002*abs(reference);cosine>=0.99999;no_nonfinite","w4u8_gate":"exact_zero_lsb","qnn_dependency":false}'
