# Next direction after EXP0244 — discussion only

EXP0244 has valid completed software attribution evidence. No rotation, training, weight export, hardware execution or baseline promotion occurred. No next experiment is registered or authorized by this document. Read EXP0244-RESULTS and its independent final PPL/intervals.

Prioritize splitting early-layer Q-RoPE, K-cache, tied V projection/cache and Q/K joint effects, starting at layer0 then a bounded early-layer check. Fixed C64 per-channel W4 and EOS remain the control. Do not attribute the combined Q/K/V group's full benefit to K alone. Separate fixed-prefix cache quantization from body contribution where appropriate under a separately frozen protocol; do not use diagnostic FP16 restoration as a deployment escape.

If Q/K dominates, consider a paired online Q/K transform before broad residual R1 or late local R4. SpinQuant's QK wrapper uses a matched normalized Hadamard (R3); the unquantized orthogonal-pair identity preserves the dot product. Its dynamic token/head K quantizer is not this project's static-U8 carrier and must not be copied unchanged. Reference method only: https://github.com/facebookresearch/SpinQuant/blob/main/train_utils/apply_r3_r4.py and https://arxiv.org/html/2405.16406v4 . No R3 result is claimed here.

Residual restoration remains a useful secondary lead; global SwiGLU alone leaves substantial loss and late isolated restores were weak in screening. Restoration NLL reductions are conditional, not additive causal fractions or bounds on attainable rotation gains. Any later weight-changing rotation requires fresh original-FP folding and new per-channel quantization, with F16 equivalence, W4A16 cost and A8 benefit separated; never reuse prior LPBQ or folded rotation weights.

Bulk/step FP16 arithmetic differences materially change A8 ranking, so actual cached sequential scoring and an independently frozen final panel remain required. Preserve native HMX W4, existing5%/10% accuracy thresholds, and >10% single-layer complete Host-wall slowdown stop before full-device work. Other recipes stay frozen; E2E token/s for EXP0244 is N/A.
