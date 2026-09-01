#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env_exp0001.sh"

cd "${project_root}"
build_cmake android BUILD=ReleaseG
build_cmake hexagon BUILD=ReleaseG DSP_ARCH="${DSP_ARCH:-v79}"

printf 'HOST_EXECUTABLE=%s\n' \
    "${project_root}/android_ReleaseG_aarch64/ship/qwen3_block_cli"
printf 'HOST_STUB=%s\n' \
    "${project_root}/android_ReleaseG_aarch64/ship/libqwen3_probe.so"
printf 'DSP_SKEL=%s\n' \
    "${project_root}/hexagon_ReleaseG_toolv19_${DSP_ARCH:-v79}/ship/libqwen3_probe_skel.so"
