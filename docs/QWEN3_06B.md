# Qwen3-0.6B runtime (EXP-0288)

Supported: original-derived per-output-channel W4A8-SP2 mode8, FP32 residual mode2, no online rotations, native Qwen Q/K Norm and RoPE, original tied embeddings.
Model identity: hidden1024, Q/attention2048, FFN3072,28layers,16Q/8KV,head_dim128,vocab151936. ABI135 rejects unsupported recipes. Default1.7B identity remains separate.

Original: /mnt/d/llm_exp/models/Qwen3-0.6B-origin
Final package: /mnt/d/llm_exp/models/qwen3-block-htp/exp0288/frontend64-a03
Evidence: /mnt/d/llm_exp/results/qwen3-block-htp/exp0288
Build after approved experiment preflight:
QBH_QWEN_MODEL_SIZE=0.6B bash scripts/build_qwen3_sp2.sh 28

Retained OPT1 reuses native-W4 LM head for the final prefill token. Primary M64+42 repeat10 formal:3020.4273/92.4021 token/s. M64+63 repeat10 supplemental:3021.5299/92.1258 token/s. Peak VTCM plan5744384B,grant8MiB,zero timed explicit intermediateDDR/spill. No model-quality/PPL claim or automatic promotion of another model baseline.

Read experiments/EXP-0288-RESULTS.md and EXP-0288-MODULES.md. Full64 independently exact after CPU QK reciprocal-sqrt reference double-rounding repair; native Q/K arithmetic unchanged. Scripts prepare/reference/frontend/device/full/audit/measure_exp0288 preserve failed attempts and frozen calibration provenance.
