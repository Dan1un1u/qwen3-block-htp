#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
package="${QBH_EXP0168_PACKAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0167/w4u8_greedy16}"
remote_root="${EXP0168_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0168-w4u8}"
stage_root="${QBH_EXP0168_ADB_STAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/.adb-staging/exp0168}"
archive="${stage_root}/package.tar"

for artifact in \
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${package}/manifest.json" \
    "${package}/generation_embedding_weight_u8.bin" \
    "${package}/generation_lm_head_weight_w4_hmx.bin" \
    "${package}/generation_lm_head_bias_u32.bin"; do
    test -f "${artifact}"
done

mkdir -p "${stage_root}"
cp "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" "${stage_root}/"
cp "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" "${stage_root}/"
cp "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" "${stage_root}/"
tar -C "${package}" -cf "${archive}" \
    --exclude='generation_embedding_weight_u8.bin' \
    --exclude='generation_lm_head_weight_w4_hmx.bin' \
    .

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "mkdir -p ${remote_root}/block_package_layer14_m64"
"${adb_exe}" push "$(wslpath -w "${stage_root}/qwen3_block_cli")" \
    "${remote_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push "$(wslpath -w "${stage_root}/libqwen3_probe.so")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${stage_root}/libqwen3_probe_skel.so")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${archive}")" \
    "${remote_root}/package.tar" >/dev/null
"${adb_exe}" shell \
    "cd ${remote_root}/block_package_layer14_m64 && tar -xf ../package.tar && cd .. && rm package.tar && chmod 755 qwen3_block_cli"

for name in \
    generation_embedding_weight_u8.bin \
    generation_lm_head_weight_w4_hmx.bin; do
    "${adb_exe}" push "$(wslpath -w "${package}/${name}")" \
        "${remote_root}/block_package_layer14_m64/${name}" >/dev/null
done
printf 'REMOTE_ROOT=%s\n' "${remote_root}"
