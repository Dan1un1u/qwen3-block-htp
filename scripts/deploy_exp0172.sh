#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
overlay="${QBH_EXP0172_OVERLAY:-/mnt/d/llm_exp/models/qwen3-block-htp/exp0169/w4u8_greedy193_overlay}"
remote_root="${EXP0172_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0172-w4u8}"
remote_transformer="${EXP0172_REMOTE_TRANSFORMER:-/data/local/tmp/qwen3-block-htp/exp0163-candidate/block_package_layer14_m64}"
remote_generation="${EXP0172_REMOTE_GENERATION:-/data/local/tmp/qwen3-block-htp/exp0168-w4u8/block_package_layer14_m64}"
stage_root="${QBH_EXP0172_ADB_STAGE:-/mnt/d/llm_exp/models/qwen3-block-htp/.adb-staging/exp0172}"
archive="${stage_root}/overlay.tar"

case "${remote_root}" in
/data/local/tmp/qwen3-block-htp/exp0172-*) ;;
*) printf 'unsafe EXP-0172 remote root: %s\n' "${remote_root}" >&2; exit 2 ;;
esac

for artifact in \
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli" \
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so" \
    "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so" \
    "${overlay}/manifest.json"; do
    test -f "${artifact}"
done

mkdir -p "${stage_root}"
(
    cd "${overlay}"
    tar -cf "${archive}" manifest.json generation_decode_rope_*_f16.bin
)

"${adb_exe}" get-state >/dev/null
"${adb_exe}" shell "test -f ${remote_transformer}/manifest.json"
"${adb_exe}" shell "rm -rf ${remote_root} && mkdir -p ${remote_root}/block_package_layer14_m64"
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli")" \
    "${remote_root}/qwen3_block_cli" >/dev/null
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so")" \
    "${remote_root}/libqwen3_probe.so" >/dev/null
"${adb_exe}" push \
    "$(wslpath -w "${project_root}/hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so")" \
    "${remote_root}/libqwen3_probe_skel.so" >/dev/null
"${adb_exe}" push "$(wslpath -w "${archive}")" \
    "${remote_root}/overlay.tar" >/dev/null
"${adb_exe}" shell \
    "cd ${remote_root}/block_package_layer14_m64 && tar -xf ../overlay.tar && rm ../overlay.tar"

for layer in $(seq 0 27); do
    "${adb_exe}" shell "ln -s ${remote_transformer}/layer${layer} ${remote_root}/block_package_layer14_m64/layer${layer}"
done
for name in \
    reference_w4u8_block_input_u8.bin \
    reference_w4u8_integer_attention_block_output_u8.bin \
    rope_cos_f16.bin rope_sin_f16.bin; do
    "${adb_exe}" shell "ln -s ${remote_transformer}/${name} ${remote_root}/block_package_layer14_m64/${name}"
done

for name in \
    generation_prompt_token_ids_u32.bin \
    generation_expected_token_ids_u32.bin \
    generation_embedding_weight_u8.bin \
    generation_final_norm_weight_f16.bin \
    generation_lm_head_weight_w4_hmx.bin \
    generation_lm_head_weight_w4_scale_f32.bin \
    generation_lm_head_bias_u32.bin \
    generation_qparams_u8.bin; do
    if "${adb_exe}" shell "test -f ${remote_generation}/${name}"; then
        "${adb_exe}" shell "ln -s ${remote_generation}/${name} ${remote_root}/block_package_layer14_m64/${name}"
    else
        "${adb_exe}" push "$(wslpath -w "${overlay}/${name}")" \
            "${remote_root}/block_package_layer14_m64/${name}" >/dev/null
    fi
done
"${adb_exe}" shell "chmod 755 ${remote_root}/qwen3_block_cli"
rm -f "${archive}"
printf 'REMOTE_ROOT=%s\n' "${remote_root}"
