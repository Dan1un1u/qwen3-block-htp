# EXP0225 learned offline R1/R2

User approved SpinQuant-based rotation learning on 2026-09-06. Runtime and other recipes stay frozen. Trainable global R1 and layer-specific R2 use the pinned official Cayley optimizer; no online R3/R4. Original BF16 tensors retain BF16 storage without early FP16 rounding; rotation and gamma products use FP32 with TF32 disabled. Weight STE uses absmax symmetric [-7,7], one FP32 scale/output row and FP16 reconstructed weights. Export uses fresh original FP32 gamma/rotation folds, true-sequential GPTQ and EXP0224 output-scale selection. The head is refolded and quantized with EXP0221 absmax GPTQ. This is a declared training/export surrogate difference.

Frozen protocol: 100 steps, batch8 (4 English/4 Chinese), 128 tokens/window, checkpoints0/50/100, fixed seed225, learning rate1.5 cosine decay. Training800 windows, validation32 windows from32 distinct documents; source documents split and 32-token overlap excluded against GPTQ calibration and qbh-lite evaluation/holdout. GPTQ calibration retains8192 tokens. Data SHA256 bcb67126349e8d086f29619807e9ab7ebba06253792b7fce4e5468f9d39e8280.

Implementation commands (run project-memory preflight before each stage):
- CPU venv: check_exp0225.py (algebra), check_exp0225_export.py (GPTQ/output oracle), check_exp0225_model.py (full-model and chunked CE).
- Separate CUDA venv: learned_rotation_exp0225.py smoke, then train.
- CPU venv: export_exp0225.py step000 / step050 / step100 / control; select by independent actual-export validation NLL, earliest tie; report languages separately.
- CPU venv: export_exp0225.py quality after selection; measure_exp0225.py deploy/quick/full/repeat/warmup/short/formal. Candidate versus EXP0224 A,5short/10formal.

All evidence is immutable under D:/llm_exp/results/qwen3-block-htp/exp0225; packages and rotations under D:/llm_exp/models/qwen3-block-htp/exp0225. Failed attempts retain their outputs. No automatic baseline promotion. Complete profiling and final quality are pending until execution passes the required gates.
