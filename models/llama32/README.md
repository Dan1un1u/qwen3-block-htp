# Llama 3.2 model adapter

Original ModelScope Llama-3.2-1B-Instruct BF16 is pinned in model.json and
read-only under D:/llm_exp/models/llama3.2-1B-Instruct-origin. Each recipe starts
from those tensors. W16A16 and W4A16 functional records are retained separately;
W4A16 has not passed model-quality acceptance. W4A8 OFF is validated under
L32-0003 without a quality threshold. See recipe docs and validation records.
No Qwen weights, quantization parameters, token IDs or prefix are model inputs.
Rotated Llama recipes are unsupported pending a separate approved experiment.

L32-0004 adds formal paired speed validation for W4A16 and W4A8 OFF, recorded in pipeline-speed-validation.json and docs/LLAMA32_PIPELINE_SPEED.md. Quality status is unchanged.

L32-0005 matched A8 relative-speed results and current launch paths: docs/LLAMA32_A8_RELATIVE_SPEED.md. Prior timings remain historical evidence.
