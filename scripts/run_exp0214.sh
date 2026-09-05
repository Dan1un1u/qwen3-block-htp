#!/usr/bin/env bash
set -euo pipefail

adb_exe="${ADB_EXE:-/mnt/c/adb/adb.exe}"
remote_root="${QBH_EXP0214_REMOTE_ROOT:-/data/local/tmp/qwen3-block-htp/exp0214-m64-direct-w4}"
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

# The retained `real_layer14_m64` package predates the EXP-0187 native-HMX
# conversion reference used by the formal M1 gate.  Its external bytes use the
# older software postscale/rounding model and are therefore diagnostic for this
# M64 experiment rather than byte authoritative.  The probe reports a non-zero
# host exit status for those expected rounding differences even when every
# DSP/physical gate passes.  Preserve the complete diagnostic record here and
# let summarize_exp0214.py enforce the DSP gates plus complete-output FNV
# identity between the expanded-S8 control and direct-W4 candidate.
set +e
output="$("${adb_exe}" shell \
    "cd ${remote_root}
LD_LIBRARY_PATH=${remote_root} DSP_LIBRARY_PATH=${remote_root} ADSP_LIBRARY_PATH=${remote_root} ./qwen3_probe_cli ${storage} ${projection} identity ${repeat_count} ${plan} linked_2d_dma prepared_session single_invocation 64 ${remote_root}/package_hmxref ${zero_point} 2>${remote_root}/exp0214_reference_diagnostic.log
cat ${remote_root}/exp0214_reference_diagnostic.log
true")"
probe_status=$?
set -e
printf '%s\n' "${output}"

if ! printf '%s\n' "${output}" | grep -q '^{'; then
    printf 'probe produced no JSON record (status=%s)\n' "${probe_status}" >&2
    exit "${probe_status}"
fi
