# Llama 3.2 current context

Read PROJECT_CONTRACT.md, PROJECT_STATUS.yaml, this file, experiments/index.yaml
in that order after bootstrap. This is standalone HTP, not mllm. Qwen3 research
remains frozen at48eb1ea7f9db0eb197a7c7908ab954d5a6635fc5. Its historical selected
records, failed ideal-R3 whole-layer gate and unaccepted R4 quality remain intact.
Never inspect or reuse excluded legacy mllm work or Qwen model payloads.

L32-0001 is completed. Original BF16 Llama-3.2-1B-Instruct input is at
/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin; six hashes verified. Both source
branches have identical common adapter code; only config/branch.json differs.
No-rotation keeps W4A8 OFF default, rotation keeps R3 with optional R3+R4. Those
are future migration schedules, not validated Llama W4/A8 execution support.

W16A16 model adapter passes independent FP32/FP16 Transformers comparisons,
single device layer0/7/15 prefill/decode, continuous3 and16 layers, full tied-head
text, and small heldout PPL integration checks. Device text is Paris is the
capital of France. All16 fixed replay tokens match FP16 reference. Whole16-layer
output NRMSE0.00232789 prefill /0.00166363 decode; exact8MiB VTCM and no intermediate
DDR/spill. Long RoPE positions checked on host only; device uses M64 plus bounded
decode. Initial scalar DSP RoPE prioritizes correctness; no CPU fallback.

Authoritative result /mnt/d/llm_exp/results/llama32-htp/l32-0001/validation_summary.json.
Tiny eight-document EN/ZH integration diagnostic,128 target tokens, fresh Llama
token IDs. Device PPL25.1047703 vs original BF16 25.0220618 (+0.33054%), FP16 teacher
25.1078689. Each language gap<0.35%. The original BF16 teacher is loaded directly;
initial frontend result used BF16<-FP16 roundtrip and is superseded for that field.
No general quantization acceptance or baseline quality promotion. Functional
M64+15 speed395.4202/5.7742 tok/s is auxiliary, not formal profiling. Do not compare
this conservative initial adapter to frozen optimized Qwen timings as paired data.

Source entrypoints and limitations: docs/LLAMA32_W16A16.md. Original model hashes,
independent math, exporters, layer/stack/frontend runners live under tools/.
Old Qwen experiment launchers remain historical only. tools/recipe.py is read-only
and verifies immutable Qwen provenance at its frozen commit, not current adapter.

Next authorized order remains W4A16 then W4A8. Register L32-0002 on no-rotation
worktree for fresh per-output-channel W4 [-7,7] calibration/export. C64 GPTQ method
from Qwen is a starting algorithm; no old tensors/calibration/prefix reused.
Use the W16A16 device path and floating teacher as references, and a larger fixed
independent PPL set before judging quantization acceptance. The128-token port set
is too small to become that acceptance suite. W4A8/rotations wait for W4A16.
