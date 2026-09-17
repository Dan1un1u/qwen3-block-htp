# L32-0045 implementation audit
Scope: Llama-3.2-3B-Instruct W4A8-SP2mode8, no rotation, FP32 residual.
Frozen L32-0044 opt2c/head32/AVrows4 is the control. Frozen original-derived L32-0041 weights and L32-0044 long fixtures remain inputs.

## Bounded iterations
A: QKV batch4 to8, existing bias arena. 
B: QKV batch32; each weight slot holds 32*96*512 =1,572,864 bytes, exactly the existing1.5MiB capacity. Two bias slots use 2*32*256 =16,384 bytes at the start of phase-dead Gate storage. Live RoPE coefficients are not overwritten.
C: first Q DMA starts before input FP32 Norm and joins before audit/consumer. Last Q HMX overlaps first K DMA; last K HMX overlaps first V DMA. The next projection consumes an explicitly recorded slot without rereading weights.
D: final Up HMX overlaps first Down8 weight/bias DMA (1MiB weights). SwiGLU must still publish/join before Down executes; no speculative activation consumption.
Final native build ff213089c6f5cfc07c620c9509b22a5e00486e80 retains D and honors existing pipeline-disable bit16 for added cross-operator overlap. All-on uses mask0. One HMX owner, unchanged dot-product order/weight bytes. New fields are scalar ownership metadata, not tensor allocations.

## Buffer ownership
Input Norm uses residual, gamma, normalized/SP2 scratch and activation storage; Q weights use expanded_weight slot0 and Gate bias0. Norm does not write those ranges. DMA joins before any boundary-audit DMA or Q launch.
During Q/K HMX, current_slot stays immutable; only alternate weight/bias slot receives next projection. HMX joins before slot reuse. Transfer records are reset at each block and consumed once.
Gate/Up bias occupies rope_cos after attention. Down prefetch writes scale_or_bias, separately from live Up bias, plus the alternate weight slot. Down consumes the recorded slot after Gate/Up and SwiGLU join.
No extra tensor arena, no intermediate DDR buffer/cache, no weight expansion or quantization change. Per-token HMX tile products/weight and cache bytes must match control; QKV submits980 fewer HMX commands over28 layers. Prefetch changes time ownership, so interpret full Host wall and sums of adjacent stages, not only individual stage reductions.

## Validation and reporting
Each A/B/C/D and final: selected0/13/27, chain3, chain28, full64-step true-greedy hidden/selected token/logit/prefill-KV equality against independent frozen reference. Audit timing is excluded from speed claims. Final shared-code1B bounded layer7 regression.
Pilot A/B/C compare three paired repeat10 rounds with frozen control; D compares three paired rounds with C. Final performance uses fixed5short+10formal pairedAB/BA repeat10 M64+42/cache128. No performance stopping gate. Preserve every run and uncertainty; no model-quality/PPL or automatic promotion claim.
