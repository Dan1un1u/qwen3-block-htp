#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant="${1:-W4F16}"
repeat_count="${2:-10}"

case "${variant}" in
    F16F16)
        fixed_args=(F16F16 "${repeat_count}" on off fused batch2 hvx 2 32 control hvx)
        ;;
    W4F16)
        fixed_args=(W4F16 "${repeat_count}" on off fused serial hvx 3 32 adaptive_down96_cross hvx)
        ;;
    *)
        echo "unsupported variant: ${variant}" >&2
        exit 2
        ;;
esac

run_case() {
    local mode="$1"
    local contexts="$2"
    local chunk_vectors="$3"
    local output json

    output="$("${project_root}/scripts/run_exp0030_block.sh" \
        "${fixed_args[@]}" "${mode}" "${contexts}" "${chunk_vectors}")"
    json="$(printf '%s\n' "${output}" | tr -d '\r' | awk '/^\{/{line=$0} END{print line}')"
    if [[ -z "${json}" ]]; then
        echo "missing JSON output for ${mode}/${contexts}/${chunk_vectors}" >&2
        exit 3
    fi
    python3 -c '
import json, sys
d = json.load(sys.stdin)
keys = (
    "variant", "mlp_mode", "mlp_hvx_contexts", "mlp_chunk_vectors",
    "host_wall_ns_per_block", "total_ticks", "activation_ticks",
    "mlp_silu_main_work_ticks", "mlp_silu_worker_work_ticks",
    "mlp_silu_pool_wait_ticks", "output_hash", "intermediate_ddr_read_bytes",
    "intermediate_ddr_write_bytes", "numerical_status",
)
print("\t".join(str(d[key]) for key in keys))
' <<<"${json}"
}

printf 'variant\tmode\tcontexts\tchunk_vectors\thost_ns_per_block\ttotal_ticks\tactivation_ticks\tmain_work_ticks\tworker_work_ticks\tpool_wait_ticks\toutput_hash\tintermediate_ddr_read\tintermediate_ddr_write\tnumerical_status\n'
run_case control 1 64
for contexts in 2 3 4; do
    for chunk_vectors in 16 32 64 128 256; do
        run_case parallel_silu "${contexts}" "${chunk_vectors}"
    done
done
