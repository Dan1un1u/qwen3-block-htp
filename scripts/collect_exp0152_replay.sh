#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
recipe="${1:?recipe required}"
package="${2:?package directory required}"
remote_root="${3:?remote root required}"
output="${4:?output directory required}"
runs="${5:-10}"

mkdir -p "${output}"
for ((run_index = 1; run_index <= runs; ++run_index)); do
    printf -v label '%02d' "${run_index}"
    EXP0152_REMOTE_ROOT="${remote_root}" \
    QBH_EXP0152_PACKAGE="${package}" \
        "${project_root}/scripts/run_exp0152.sh" \
        "${recipe}" replay \
        > "${output}/device_replay_r${label}.log" 2>&1
    printf 'completed_%s\n' "${label}"
done
