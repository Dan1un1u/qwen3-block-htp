# L32-0014 running

# L32-0014 temporary all-A8 precision experiment

User authorization: current request explicitly suspends SP2 and performance optimization, requests the supplied SmoothRot-style up-absorbed smoothing, full8192 R4, guided static A8 calibration, renewed per-channel W4 quantization, and dense matrix multiplication for R3/R4. Existing original BF16 checkpoint is read-only. Reference rotation-quant is read-only; snapshot exact local method files without modifying dirty user work. Qwen and floating controls remain frozen. No baseline promotion.

## Bounded execution

1. Rebuild from original BF16, fixed seeded R1/R2 offline; no old quantized/folded weight reuse. Positive smoothing exponent0.5, median normalization, clamp[1/32,32], absorbed into up rows/down columns. Full normalized8192 Hadamard after SwiGLU, offline inverse into down. R3 if enabled is normalized64 Hadamard after RoPE on both Q/K. All online rotations explicit dense matrix multiplication, no butterfly.
2. Train-only calibration: seed42,8 disjoint512-token windows, cover first32 plus64 sampled later positions, inverse inclusion weighting. Freeze selected windows before scoring. Squared CE-output-gradient importance,33 coarse+9 local A8 range candidates; W4 output-channel clipping ratios[.5,.6,.7,.8,.9,1,1.1], signed[-7,7], no group scales. Token-weighted PSD channel-shrinkage GPTQ with1% diagonal damping. Record RTN and GPTQ separately. At most10 channel affine CE/KL refinement updates, with a fixed disjoint train-only selection set; report if not needed/applied.
3. Fixed same-weight controls distinguish all112 Linear-input A8 from genuinely all native A8 carriers. Include embedding, projection outputs, RMS boundaries, integer log2 attention, KV, residuals, finalnorm/head/logits. Existing affine per-tensor minmax outputs remain the control; no prefix or dynamic scale. Dense R4 consumes gated values before final input quantization, and retains upstream Gate/Up U8 quantization in native contract. Scalar float temporaries/accumulators do not imply BF16 stored activation bypass.
4. Validate transform equivalence and integer component arithmetic independently. Run matched L32-0013 M64+16/2048-target bridge on teacher, method input-only, and all-A8 candidate; isolate residual/Down boundaries as diagnostic controls if full A8 fails. Full WT2 validation2048+tail software PPL for method and all-A8 feasible paths; no validation-based calibration/checkpoint selection. Device implementation/gates follow if required to resolve an arithmetic discrepancy or establish a viable native candidate; software failure is reported as such, never as a device result. No speed gate/profiling required by user.
5. Do not infer general impossibility from one failed recipe. Preserve source, parameter provenance, raw metrics and failures; report scope and next precision bottleneck.

No changes to production defaults or SP2 artifacts. Owned implementation repairs proceed under authorization.

Previous closure L32-0013 remains in docs/LLAMA32_C_RTN_SP2_ALIGNMENT.md.
