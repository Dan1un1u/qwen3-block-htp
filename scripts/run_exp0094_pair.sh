#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cell="${1:-control}"
repeat_count="${2:-1}"
attribution_mode="${3:-on}"
audit_mode="${4:-off}"

case "${cell}" in
    control)
        exec "${project_root}/scripts/run_exp0084.sh" \
            W4U8 control "${repeat_count}" \
            "${attribution_mode}" "${audit_mode}"
        ;;
    candidate)
        exec "${project_root}/scripts/run_exp0094.sh" \
            "${repeat_count}" "${attribution_mode}" "${audit_mode}"
        ;;
    *)
        printf 'usage: %s control|candidate [repeat] [attribution] [audit]\n' \
            "$0" >&2
        exit 2
        ;;
esac
