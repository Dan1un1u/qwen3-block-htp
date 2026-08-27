#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exe="${EXP0022_PYTHON:-/mnt/d/llm_exp/mllm-v1/.venv-quant/bin/python}"
python_deps="${EXP0022_PYTHONPATH:-/home/daniuniu/.cache/qwen3-block-htp-exp0022-python}"

export PYTHONPATH="${python_deps}${PYTHONPATH:+:${PYTHONPATH}}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec "${python_exe}" "${project_root}/scripts/export_exp0022_block.py" "$@"
