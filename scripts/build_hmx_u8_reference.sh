#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_tools="${HEXAGON_TOOLS_ROOT:-/opt/qcom/hexagon/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools}"
output="${1:-${project_root}/build/reference/qbh_hmx_u8_reference}"
shared_output="${output}.so"

mkdir -p "$(dirname "${output}")"
gcc -O2 -Wall -Wextra -Werror \
    "${project_root}/tools/hmx_u8_reference.c" \
    -I"${sdk_tools}/libnative/include" \
    -L"${sdk_tools}/libnative/lib" -lnative -lm \
    -o "${output}"
gcc -O2 -Wall -Wextra -Werror -fPIC -shared \
    "${project_root}/tools/hmx_u8_reference.c" \
    -I"${sdk_tools}/libnative/include" \
    -L"${sdk_tools}/libnative/lib" -lnative -lm \
    -o "${shared_output}"
printf 'EXECUTABLE=%s\nSHARED_LIBRARY=%s\n' \
    "${output}" "${shared_output}"
