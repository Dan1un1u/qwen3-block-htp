#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${1:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0217/f16f16_greedy16}"
remote_root="${EXP0217_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0217-f16f16}"
source_remote="/data/local/tmp/qwen3-block-htp/exp0158-f16f16/block_package_layer14_m64"
stage_root="${QBH_EXP0217_ADB_STAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/.adb-staging/exp0217}"
overlay="${stage_root}/generation_overlay.tar"

for artifact in \
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${package}/manifest.json" \
    "${package}/generation_embedding_weight_f16.bin" \
    "${package}/generation_lm_head_weight_f16_hmx.bin"; do
    test -f "${artifact}"
done

mkdir -p "${stage_root}"
cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" "${stage_root}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" "${stage_root}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" "${stage_root}/"
mapfile -t cache_files < <(
    cd "${package}"
    find layer* -maxdepth 1 -type f -name '*hmx_f16*' -print | sort
)
mapfile -t decode_rope_files < <(
    cd "${package}"
    find . -maxdepth 1 -type f \
        -name 'generation_decode_rope_*_f16.bin' -printf '%f\n' | sort
)
tar -C "${package}" -cf "${overlay}" \
    manifest.json \
    generation_prompt_token_ids_u32.bin \
    generation_expected_token_ids_u32.bin \
    generation_final_norm_weight_f16.bin \
    rope_cos_f16.bin rope_sin_f16.bin \
    "${decode_rope_files[@]}" "${cache_files[@]}"

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "test -d ${source_remote}"
"${adb_exe}" shell "mkdir -p ${remote_root}"
if ! "${adb_exe}" shell \
    "test -d ${remote_root}/block_package_layer14_m64"; then
    "${adb_exe}" shell \
        "mkdir -p ${remote_root}/block_package_layer14_m64 && cp -as ${source_remote}/. ${remote_root}/block_package_layer14_m64/"
fi
"${adb_exe}" push "$(wslpath -w "${stage_root}/qwen3_block_cli")" \
    "${remote_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push "$(wslpath -w "${stage_root}/libqwen3_probe.so")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${stage_root}/libqwen3_probe_skel.so")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${overlay}")" \
    "${remote_root}/generation_overlay.tar" >/dev/null

# Break only the EXP-0217 hard links before replacing cache carriers.  This
# preserves the retained EXP-0158 package used as the immutable source.
"${adb_exe}" shell \
    "cd ${remote_root}/block_package_layer14_m64 && find layer* -maxdepth 1 -name '*hmx_f16*' -exec rm -f {} + && rm -f manifest.json generation_* rope_cos_f16.bin rope_sin_f16.bin && tar -xf ../generation_overlay.tar && cd .. && rm -f generation_overlay.tar && chmod 755 qwen3_block_cli"

for name in \
    generation_embedding_weight_f16.bin \
    generation_lm_head_weight_f16_hmx.bin; do
    "${adb_exe}" push "$(wslpath -w "${package}/${name}")" \
        "${remote_root}/block_package_layer14_m64/${name}"
done

printf 'REMOTE_ROOT=%s\n' "${remote_root}"
