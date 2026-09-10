# Llama 3.2 model adapter

Original ModelScope Llama-3.2-1B-Instruct BF16 is pinned in model.json and
read-only under D:/llm_exp/models/llama3.2-1B-Instruct-origin. Each recipe starts
from those tensors. W16A16 and W4A16 functional records are retained separately;
W4A16 has not passed model-quality acceptance. W4A8 OFF is validated under
L32-0003 without a quality threshold. See recipe docs and validation records.
No Qwen weights, quantization parameters, token IDs or prefix are model inputs.
Rotated Llama recipes are unsupported pending a separate approved experiment.
