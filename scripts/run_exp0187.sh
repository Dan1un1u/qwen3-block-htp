#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${QBH_EXP0187_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0187-real}"
cell="${1:?control or direct_n required}"
projection="${2:?gate_up_pair or down required}"
repeat_count="${3:-1}"

case "${projection}" in
    gate_up_pair)
        zero_point=127
        control_plan=stream32_gate_hvx6
        ;;
    down)
        zero_point=102
        control_plan=stream32_down_hvx6
        ;;
    *)
        printf 'invalid projection: %s\n' "${projection}" >&2
        exit 2
        ;;
esac

case "${cell}" in
    control)
        storage=packed_w4_hmx_postscale
        plan="${control_plan}"
        ;;
    direct_n)
        storage=packed_w4_direct_n
        plan=exp0005_full_bundle_control
        ;;
    *)
        printf 'invalid cell: %s\n' "${cell}" >&2
        exit 2
        ;;
esac

"${adb_exe}" shell \
    "cd ${remote_root} && LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_probe_cli ${storage} ${projection} identity ${repeat_count} ${plan} linked_2d_dma prepared_session single_invocation 1 ${remote_root}/package_hmxref ${zero_point}"
