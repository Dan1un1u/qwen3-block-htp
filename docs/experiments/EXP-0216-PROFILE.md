# EXP-0216 complete profiling report

Measured source: `codex/exp-0216-w4u8-prefill-direct-w4-qkvo` @ `25b2fdfead7546a981ccb457d9106e081bb6cb34`. Control mask31 is EXP-0215 direct-W4 MLP; candidate mask63 adds M64 Q/K/V/O direct-W4. Both use one M64 token-input pass through all 28 layers, final norm and LM head, followed by 192 continuous token-feedback decode calls. Ten rotated pairs; fresh short gate has five pairs. Repairs from f724a22 affect Host JSON telemetry, audit-only prefill cache exports after the timed RPC, and strict schema/audit validation. DSP kernels are unchanged. No automatic baseline promotion.

R1 below is the first formal rotated pair (diagnostic, no best-run selection); R10 is the median of all ten formal sessions. Every device invocation has repeat_count=1 because cache state advances once per token. R10 does not mean replaying a frozen cache ten times. Decode timings are normalized over all 192 actual decode calls in each session. Time deltas are candidate/control−1 (negative is faster). Independent module medians need not sum to the median Host wall; every individual raw ledger closes exactly.
Final model RMSNorm is nested inside generation_lm_head_ticks in the DSP ledger. To show both exclusive rows, LM-head-exclusive = generation_lm_head_ticks − generation_final_norm_ticks per record. The raw inclusive LM-head counter remains in the overlapping diagnostics. This prevents counting final norm twice.

## Direct full-stack throughput

| Scope | Tokens/session | Control complete Host us | Candidate complete Host us | Control tok/s | Candidate tok/s | Median paired speed |
|---|---|---|---|---|---|---|
| prefill R1 | 64 | 43433.385 | 39017.500 | 1473.520887 | 1640.289614 | +11.318% |
| prefill R10 | 64 | 43741.224 | 39296.484 | 1463.150659 | 1628.644435 | +11.171% |
| decode R1 | 192 | 4146030.201 | 4167681.189 | 46.309359 | 46.068783 | -0.519% |
| decode R10 | 192 | 4177184.944 | 4179219.555 | 45.963969 | 45.941592 | -0.110% |

Throughput above is computed from the exact displayed Host-wall statistic. The original summary retains medians of individual session throughputs, whose tiny numerical difference from the inverse median wall for even sample counts does not change paired ordering. Per-pair throughput/Host ordering is checked directly. F16F16 complete token-to-token throughput: N/A (EXP-0158 ends before final norm/LM head). W4F16 EXP-0166 uses 15 decode calls rather than 192 and is not an equivalent decode comparator.

## Three-variant M64 module overview (R10, non-paired historical context)

F16F16: EXP-0158 latest valid transformer-only formal evidence; W4F16: selected EXP-0166 complete token boundary; W4U8: this EXP-0216 candidate. F16F16 has no embedding/final norm/head timing, shown N/A; its complete Host scope differs, so its column is diagnostic context. W4F16/W4U8 module ratios are historical, not paired experiment gains. Each cell is us (% of its own complete Host).

| Module | F16F16 EXP-0158 | W4F16 EXP-0166 | W4U8 EXP-0216 | W4U8 speed vs W4F16 |
|---|---|---|---|---|
| I/O and metadata | 105.911 (0.16%) | 366.823 (0.58%) | 238.750 (0.61%) | +53.64% |
| Input RMSNorm | 492.292 (0.73%) | 493.906 (0.78%) | 559.167 (1.42%) | -11.67% |
| QKV + Q/K Norm-RoPE | 11374.844 (16.92%) | 11772.552 (18.69%) | 7032.969 (17.90%) | +67.39% |
| QK-Softmax-AV | 4005.833 (5.96%) | 3974.714 (6.31%) | 3233.333 (8.23%) | +22.93% |
| O projection | 5701.927 (8.48%) | 5044.896 (8.01%) | 1240.260 (3.16%) | +306.76% |
| Post-attention residual + RMSNorm | 471.562 (0.70%) | 471.823 (0.75%) | 658.438 (1.68%) | -28.34% |
| Gate/Up + SwiGLU | 28796.536 (42.84%) | 22459.479 (35.66%) | 14296.797 (36.38%) | +57.09% |
| Down projection | 13050.208 (19.42%) | 8516.198 (13.52%) | 3339.115 (8.50%) | +155.04% |
| Final residual | 140.234 (0.21%) | 140.286 (0.22%) | 184.714 (0.47%) | -24.05% |
| KV-cache carrier conversion | 166.224 (0.25%) | 147.995 (0.24%) | 205.443 (0.52%) | -27.96% |
| KV-cache append DMA | 311.016 (0.46%) | 338.880 (0.54%) | 455.859 (1.16%) | -25.66% |
| Block internal orchestration | 14.714 (0.02%) | 16.771 (0.03%) | 34.115 (0.09%) | -50.84% |
| Layer bookkeeping | 18.542 (0.03%) | 22.083 (0.04%) | 21.146 (0.05%) | +4.43% |
| Stage-boundary bookkeeping | 6.146 (0.01%) | 4.375 (0.01%) | 26.953 (0.07%) | -83.77% |
| DSP unattributed residual | 0.000 (0.00%) | 0.000 (0.00%) | 0.000 (0.00%) | N/A (zero/unavailable) |
| DSP runtime setup/teardown | 86.354 (0.13%) | 99.870 (0.16%) | 109.818 (0.28%) | -9.06% |
| Token embedding | N/A (not emitted/in scope) | 77.526 (0.12%) | 68.542 (0.17%) | +13.11% |
| Final model RMSNorm | N/A (not emitted/in scope) | 48.333 (0.08%) | 4.714 (0.01%) | +925.41% |
| LM head + greedy selection (excluding final norm) | N/A (not emitted/in scope) | 6701.953 (10.64%) | 5223.542 (13.29%) | +28.30% |
| True Host-DSP boundary | 2506.563 (3.73%) | 2288.463 (3.63%) | 2311.406 (5.88%) | -0.99% |
| Complete Host wall | 67212.812 (100.00%) | 62974.713 (100.00%) | 39296.484 (100.00%) | +60.26% |

Historical evidence: `/mnt/d/llm_exp/results/qwen3-block-htp/exp0158/20260902T071708Z_264c911a65a3_formal` and `/mnt/d/llm_exp/results/qwen3-block-htp/exp0166/20260903T_exp0166_8e0dcf8_formal`. Raw files are included by SHA256 in the closure manifest; no baseline values are inferred from partial scope.

## prefill: complete additive timing ledger (us)

| Interval | Control R1 | Candidate R1 | Time delta | Control R10 | Candidate R10 | Time delta |
|---|---|---|---|---|---|---|
| Token embedding | 70.208 | 63.073 | -10.16% | 70.469 | 68.542 | -2.73% |
| Input staging | 0.677 | 0.573 | -15.38% | 0.469 | 0.495 | +5.56% |
| Metadata | 241.458 | 239.583 | -0.78% | 238.438 | 238.203 | -0.10% |
| Input RMSNorm | 547.188 | 537.760 | -1.72% | 554.948 | 559.167 | +0.76% |
| QKV projection | 7754.792 | 7011.719 | -9.58% | 7752.760 | 7032.370 | -9.29% |
| Q/K Norm-RoPE (separate tail) | 0.573 | 0.312 | -45.45% | 0.521 | 0.495 | -5.00% |
| QK-Softmax-AV | 3207.969 | 3214.844 | +0.21% | 3229.635 | 3233.333 | +0.11% |
| O projection | 4816.562 | 1239.375 | -74.27% | 4835.182 | 1240.260 | -74.35% |
| Post-attention residual | 663.698 | 655.260 | -1.27% | 658.958 | 657.760 | -0.18% |
| Post-attention RMSNorm (fused) | 0.677 | 0.625 | -7.69% | 0.677 | 0.625 | -7.69% |
| Gate/Up | 6801.354 | 6650.208 | -2.22% | 6768.151 | 6686.328 | -1.21% |
| SwiGLU | 7614.740 | 7610.938 | -0.05% | 7615.625 | 7610.651 | -0.07% |
| Down | 3309.896 | 3335.938 | +0.79% | 3345.885 | 3339.115 | -0.20% |
| Final residual | 184.010 | 184.844 | +0.45% | 185.208 | 184.714 | -0.27% |
| Output staging | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| KV-cache carrier conversion | 209.062 | 204.115 | -2.37% | 208.776 | 205.443 | -1.60% |
| KV-cache append DMA | 455.521 | 454.792 | -0.16% | 455.469 | 455.859 | +0.09% |
| Block orchestration | 33.958 | 36.198 | +6.60% | 34.010 | 34.115 | +0.31% |
| Layer bookkeeping | 20.729 | 21.198 | +2.26% | 20.964 | 21.146 | +0.87% |
| Stage-boundary bookkeeping | 26.562 | 26.719 | +0.59% | 27.292 | 26.953 | -1.24% |
| Final model RMSNorm | 4.062 | 4.844 | +19.23% | 4.062 | 4.714 | +16.03% |
| LM head + greedy selection (excluding final norm) | 5118.073 | 5197.604 | +1.55% | 5225.651 | 5223.542 | -0.04% |
| Runtime setup | 58.958 | 55.573 | -5.74% | 58.568 | 57.630 | -1.60% |
| Runtime teardown | 52.396 | 51.198 | -2.29% | 52.604 | 52.578 | -0.05% |
| DSP unattributed | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| True Host-DSP boundary | 2240.260 | 2220.208 | -0.90% | 2297.448 | 2311.406 | +0.61% |
| Complete Host wall | 43433.385 | 39017.500 | -10.17% | 43741.224 | 39296.484 | -10.16% |

Envelope diagnostics below overlap the ledger; do not add them again.

| Envelope | Control R1 | Candidate R1 | Time delta | Control R10 | Candidate R10 | Time delta |
|---|---|---|---|---|---|---|
| total_ticks | 41134.167 | 36741.719 | -10.68% | 41301.406 | 36878.073 | -10.71% |
| invocation_ticks | 41193.125 | 36797.292 | -10.67% | 41360.312 | 36938.203 | -10.69% |
| ledger_named_ticks | 41193.125 | 36797.292 | -10.67% | 41360.312 | 36938.203 | -10.69% |
| ledger_unattributed_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |

## decode: complete additive timing ledger (us/token)

| Interval | Control R1 | Candidate R1 | Time delta | Control R10 | Candidate R10 | Time delta |
|---|---|---|---|---|---|---|
| Token embedding | 2.439 | 2.455 | +0.66% | 2.436 | 2.474 | +1.57% |
| Input staging | 0.394 | 0.404 | +2.54% | 0.398 | 0.399 | +0.27% |
| Metadata | 226.387 | 226.647 | +0.11% | 227.851 | 227.146 | -0.31% |
| Input RMSNorm | 149.827 | 149.828 | +0.00% | 149.873 | 149.849 | -0.02% |
| QKV projection | 2479.394 | 2484.110 | +0.19% | 2495.605 | 2498.578 | +0.12% |
| Q/K Norm-RoPE (separate tail) | 0.346 | 0.346 | +0.08% | 0.346 | 0.350 | +0.94% |
| QK-Softmax-AV | 3570.500 | 3572.800 | +0.06% | 3571.417 | 3573.056 | +0.05% |
| O projection | 1238.078 | 1243.960 | +0.48% | 1247.043 | 1248.234 | +0.10% |
| Post-attention residual | 199.854 | 200.529 | +0.34% | 199.578 | 199.661 | +0.04% |
| Post-attention RMSNorm (fused) | 0.515 | 0.504 | -2.21% | 0.506 | 0.500 | -1.21% |
| Gate/Up | 6458.329 | 6480.467 | +0.34% | 6524.192 | 6517.893 | -0.10% |
| SwiGLU | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| Down | 3319.643 | 3339.806 | +0.61% | 3351.363 | 3356.176 | +0.14% |
| Final residual | 43.046 | 43.083 | +0.09% | 43.050 | 43.085 | +0.08% |
| Output staging | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| KV-cache carrier conversion | 354.827 | 354.715 | -0.03% | 354.749 | 354.769 | +0.01% |
| KV-cache append DMA | 131.403 | 131.173 | -0.17% | 131.132 | 131.299 | +0.13% |
| Block orchestration | 27.367 | 27.220 | -0.53% | 27.328 | 27.334 | +0.02% |
| Layer bookkeeping | 16.168 | 16.048 | -0.74% | 16.064 | 16.047 | -0.11% |
| Stage-boundary bookkeeping | 1.669 | 1.674 | +0.29% | 1.684 | 1.691 | +0.36% |
| Final model RMSNorm | 2.805 | 2.827 | +0.76% | 2.785 | 2.813 | +1.00% |
| LM head + greedy selection (excluding final norm) | 2708.405 | 2720.987 | +0.46% | 2732.392 | 2735.950 | +0.13% |
| Runtime setup | 37.565 | 37.582 | +0.04% | 37.586 | 37.580 | -0.02% |
| Runtime teardown | 35.648 | 35.597 | -0.14% | 35.648 | 35.667 | +0.05% |
| DSP unattributed | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| True Host-DSP boundary | 589.297 | 633.911 | +7.57% | 584.401 | 611.141 | +4.58% |
| Complete Host wall | 21593.907 | 21706.673 | +0.52% | 21756.172 | 21766.769 | +0.05% |

Envelope diagnostics below overlap the ledger; do not add them again.

| Envelope | Control R1 | Candidate R1 | Time delta | Control R10 | Candidate R10 | Time delta |
|---|---|---|---|---|---|---|
| total_ticks | 20967.045 | 21035.180 | +0.32% | 21119.331 | 21117.225 | -0.01% |
| invocation_ticks | 21004.611 | 21072.762 | +0.32% | 21156.926 | 21154.788 | -0.01% |
| ledger_named_ticks | 21004.611 | 21072.762 | +0.32% | 21156.926 | 21154.788 | -0.01% |
| ledger_unattributed_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |

## prefill: overlapping engine/work/wait diagnostics (us)

All rows can overlap other engines or ledger intervals. Zero means the exported counter was zero, not that an uninstrumented path performed no work. In particular, legacy MLP pipeline counters are not populated by the direct-W4 path; HMX commands/tile pairs, aggregate HMX work and additive projection intervals carry that evidence.

| Counter | Control R1 | Candidate R1 | Time delta | Control R10 | Candidate R10 | Time delta |
|---|---|---|---|---|---|---|
| attention_av_hmx_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_av_pack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_av_unpack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_gqa_pipeline_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_qk_hmx_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_qk_pack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_qk_unpack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_setup_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_softmax_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_unattributed_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| f16_cache_native_append_update_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| generation_lm_head_argmax_ticks | 605.938 | 602.552 | -0.56% | 607.188 | 604.896 | -0.38% |
| generation_lm_head_expand_ticks | 3651.406 | 3720.000 | +1.88% | 3715.990 | 3715.208 | -0.02% |
| generation_lm_head_hmx_tail_wait_ticks | 150.156 | 152.917 | +1.84% | 173.099 | 178.828 | +3.31% |
| generation_lm_head_hmx_ticks | 4468.021 | 4550.365 | +1.84% | 4572.839 | 4568.411 | -0.10% |
| generation_lm_head_scale_dma_ticks | 21.927 | 22.552 | +2.85% | 21.693 | 22.318 | +2.88% |
| generation_lm_head_scale_init_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| generation_lm_head_ticks | 5122.135 | 5202.448 | +1.57% | 5229.766 | 5227.734 | -0.04% |
| generation_lm_head_weight_dma_ticks | 5080.833 | 5159.167 | +1.54% | 5187.578 | 5182.943 | -0.09% |
| generation_lm_head_weight_dma_wait_ticks | 272.031 | 285.677 | +5.02% | 287.214 | 288.411 | +0.42% |
| hmx_compute_ticks | 11067.188 | 9568.750 | -13.54% | 11100.573 | 9567.448 | -13.81% |
| hmx_ready_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| projection_hmx_wait_ticks | 819.740 | 748.542 | -8.69% | 832.786 | 755.651 | -9.26% |
| projection_pack_ticks | 4.896 | 5.000 | +2.13% | 4.974 | 5.026 | +1.05% |
| projection_unpack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| scan_cache_stage_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| scan_dynamic_attention_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_attention_av_hmx_ticks | 621.250 | 632.812 | +1.86% | 625.182 | 633.698 | +1.36% |
| u8_attention_av_requant_ticks | 1521.354 | 1521.562 | +0.01% | 1522.396 | 1519.896 | -0.16% |
| u8_attention_k_pack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_attention_pipeline_wait_ticks | 1779.062 | 1765.938 | -0.74% | 1837.578 | 1858.568 | +1.14% |
| u8_attention_qk_hmx_ticks | 604.115 | 595.781 | -1.38% | 602.240 | 606.953 | +0.78% |
| u8_attention_qk_norm_rope_ticks | 23251.562 | 24420.312 | +5.03% | 23278.698 | 24474.375 | +5.14% |
| u8_attention_qk_requant_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_attention_softmax_ticks | 8433.802 | 8431.823 | -0.02% | 8434.323 | 8439.375 | +0.06% |
| u8_attention_v_pack_ticks | 3959.062 | 3979.792 | +0.52% | 3969.219 | 3971.120 | +0.05% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_native_append_update_ticks | 662.865 | 657.240 | -0.85% | 661.953 | 659.427 | -0.38% |
| w4f16_cross_prefetch_lifetime_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_cross_prefetch_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_expand_pool_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_expand_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_expand_work_ticks | 218.750 | 0.000 | -100.00% | 219.193 | 0.000 | -100.00% |
| w4f16_gate_up_expand_pool_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_expand_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_expand_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_hmx_tail_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_hmx_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_stream_join_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_stream_ready_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_stream_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_weight_dma_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_hmx_tail_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_prefetch_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_final_residual_main_work_ticks | 106.250 | 110.104 | +3.63% | 106.849 | 109.974 | +2.92% |
| w4u8_final_residual_pool_wait_ticks | 27.969 | 24.948 | -10.80% | 27.292 | 25.026 | -8.30% |
| w4u8_final_residual_worker_work_ticks | 546.094 | 545.729 | -0.07% | 546.380 | 546.458 | +0.01% |
| w4u8_gate_up_swiglu_join_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_gate_up_swiglu_ready_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_gate_up_swiglu_worker_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_input_norm_main_work_ticks | 411.510 | 438.073 | +6.45% | 427.708 | 437.969 | +2.40% |
| w4u8_input_norm_pool_wait_ticks | 84.583 | 48.854 | -42.24% | 79.115 | 70.026 | -11.49% |
| w4u8_input_norm_worker_work_ticks | 2147.708 | 2103.542 | -2.06% | 2157.448 | 2153.646 | -0.18% |
| w4u8_mlp_activation_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_down_pipeline_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_hmx_compute_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_producer_slot_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_weight_expand_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_weight_stage_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_o_gate_prefetch_lifetime_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_o_gate_prefetch_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_post_residual_main_work_ticks | 466.562 | 504.479 | +8.13% | 495.547 | 508.620 | +2.64% |
| w4u8_post_residual_pool_wait_ticks | 147.031 | 101.979 | -30.64% | 115.807 | 100.078 | -13.58% |
| w4u8_post_residual_worker_work_ticks | 2688.125 | 2620.781 | -2.51% | 2641.536 | 2633.385 | -0.31% |
| w4u8_qkv_ring_dma_wait_ticks | 2955.156 | 2131.510 | -27.87% | 2960.911 | 2141.172 | -27.69% |
| w4u8_qkv_ring_expand_ticks | 6151.719 | 0.000 | -100.00% | 6152.630 | 0.000 | -100.00% |
| w4u8_qkv_ring_hmx_compute_ticks | 336.302 | 546.510 | +62.51% | 335.104 | 551.458 | +64.56% |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 2071.667 | 1439.219 | -30.53% | 2083.802 | 1453.411 | -30.25% |
| w4u8_qkv_ring_pipeline_ticks | 7720.208 | 6978.438 | -9.61% | 7717.839 | 6998.177 | -9.32% |
| w4u8_qkv_ring_pool_wait_ticks | 4015.833 | 4493.750 | +11.90% | 4014.271 | 4505.859 | +12.25% |
| w4u8_qkv_ring_producer_slot_wait_ticks | 127.396 | 20.260 | -84.10% | 105.625 | 12.344 | -88.31% |
| w4u8_qkvo_hmx_lifetime_ticks | 16660.000 | 11853.542 | -28.85% | 16685.964 | 11895.547 | -28.71% |
| w4u8_qkvo_prefetch_wait_ticks | 4446.875 | 2132.812 | -52.04% | 4467.214 | 2142.422 | -52.04% |
| w4u8_qkvo_weight_expand_ticks | 9210.833 | 0.000 | -100.00% | 9216.120 | 0.000 | -100.00% |
| weight_dma_ticks | 19506.198 | 18013.854 | -7.65% | 19634.766 | 18088.594 | -7.87% |

## decode: overlapping engine/work/wait diagnostics (us/token)

All rows can overlap other engines or ledger intervals. Zero means the exported counter was zero, not that an uninstrumented path performed no work. In particular, legacy MLP pipeline counters are not populated by the direct-W4 path; HMX commands/tile pairs, aggregate HMX work and additive projection intervals carry that evidence.

| Counter | Control R1 | Candidate R1 | Time delta | Control R10 | Candidate R10 | Time delta |
|---|---|---|---|---|---|---|
| attention_av_hmx_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_av_pack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_av_unpack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_gqa_pipeline_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_qk_hmx_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_qk_pack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_qk_unpack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_setup_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_softmax_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| attention_unattributed_ticks | 2235.690 | 2236.793 | +0.05% | 2236.201 | 2236.929 | +0.03% |
| f16_cache_native_append_update_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| generation_lm_head_argmax_ticks | 469.353 | 469.513 | +0.03% | 469.353 | 469.495 | +0.03% |
| generation_lm_head_expand_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| generation_lm_head_hmx_tail_wait_ticks | 22.221 | 30.869 | +38.92% | 24.096 | 27.323 | +13.39% |
| generation_lm_head_hmx_ticks | 2198.998 | 2210.375 | +0.52% | 2222.084 | 2224.887 | +0.13% |
| generation_lm_head_scale_dma_ticks | 20.219 | 21.160 | +4.65% | 20.219 | 20.918 | +3.46% |
| generation_lm_head_scale_init_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| generation_lm_head_ticks | 2711.210 | 2723.813 | +0.46% | 2735.175 | 2738.761 | +0.13% |
| generation_lm_head_weight_dma_ticks | 2674.075 | 2684.087 | +0.37% | 2696.036 | 2699.374 | +0.12% |
| generation_lm_head_weight_dma_wait_ticks | 2122.129 | 2123.714 | +0.07% | 2138.784 | 2142.656 | +0.18% |
| hmx_compute_ticks | 7574.521 | 7562.012 | -0.17% | 7563.284 | 7592.184 | +0.38% |
| hmx_ready_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| projection_hmx_wait_ticks | 553.386 | 550.432 | -0.53% | 550.162 | 551.735 | +0.29% |
| projection_pack_ticks | 4.181 | 4.215 | +0.82% | 4.189 | 4.201 | +0.29% |
| projection_unpack_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| scan_cache_stage_ticks | 696.370 | 698.690 | +0.33% | 698.755 | 698.914 | +0.02% |
| scan_dynamic_attention_ticks | 3565.730 | 3567.984 | +0.06% | 3566.628 | 3568.267 | +0.05% |
| u8_attention_av_hmx_ticks | 221.114 | 221.786 | +0.30% | 221.254 | 221.964 | +0.32% |
| u8_attention_av_requant_ticks | 154.041 | 154.381 | +0.22% | 153.969 | 154.428 | +0.30% |
| u8_attention_k_pack_ticks | 58.238 | 58.269 | +0.05% | 58.243 | 58.253 | +0.02% |
| u8_attention_pipeline_wait_ticks | 71.553 | 71.418 | -0.19% | 71.384 | 71.511 | +0.18% |
| u8_attention_qk_hmx_ticks | 182.093 | 182.412 | +0.17% | 182.180 | 182.464 | +0.16% |
| u8_attention_qk_norm_rope_ticks | 2070.912 | 2004.785 | -3.19% | 2043.097 | 2025.879 | -0.84% |
| u8_attention_qk_requant_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_attention_softmax_ticks | 538.848 | 538.710 | -0.03% | 538.706 | 538.701 | -0.00% |
| u8_attention_v_pack_ticks | 108.922 | 109.030 | +0.10% | 108.945 | 108.963 | +0.02% |
| u8_cache_k_vtcm_tail_hvx_row_update_ticks | 38.321 | 38.324 | +0.01% | 38.316 | 38.360 | +0.12% |
| u8_cache_native_append_update_ticks | 483.945 | 483.609 | -0.07% | 483.547 | 483.779 | +0.05% |
| w4f16_cross_prefetch_lifetime_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_cross_prefetch_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_expand_pool_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_expand_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_expand_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_expand_pool_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_expand_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_expand_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_hmx_tail_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_hmx_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_stream_join_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_stream_ready_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_stream_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_gate_up_weight_dma_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_hmx_tail_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4f16_prefetch_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_final_residual_main_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_final_residual_pool_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_final_residual_worker_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_gate_up_swiglu_join_wait_ticks | 134.432 | 134.119 | -0.23% | 133.932 | 134.120 | +0.14% |
| w4u8_gate_up_swiglu_ready_wait_ticks | 5116.765 | 5167.550 | +0.99% | 5212.403 | 5187.613 | -0.48% |
| w4u8_gate_up_swiglu_worker_ticks | 806.920 | 783.544 | -2.90% | 771.728 | 784.503 | +1.66% |
| w4u8_input_norm_main_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_input_norm_pool_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_input_norm_worker_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_activation_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_down_pipeline_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_expanded_slot_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_gate_up_pipeline_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_hmx_compute_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_hmx_ready_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_producer_slot_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_weight_expand_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_weight_stage_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_o_gate_prefetch_lifetime_ticks | 532.826 | 528.438 | -0.82% | 532.656 | 533.735 | +0.20% |
| w4u8_o_gate_prefetch_wait_ticks | 324.392 | 319.366 | -1.55% | 324.459 | 325.487 | +0.32% |
| w4u8_post_residual_main_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_post_residual_pool_wait_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_post_residual_worker_work_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_qkv_ring_dma_wait_ticks | 2124.445 | 2132.061 | +0.36% | 2144.420 | 2146.994 | +0.12% |
| w4u8_qkv_ring_expand_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_qkv_ring_hmx_compute_ticks | 567.945 | 562.061 | -1.04% | 557.843 | 557.926 | +0.01% |
| w4u8_qkv_ring_hmx_ready_wait_ticks | 1414.318 | 1425.665 | +0.80% | 1438.227 | 1442.579 | +0.30% |
| w4u8_qkv_ring_pipeline_ticks | 2449.330 | 2454.049 | +0.19% | 2465.591 | 2468.545 | +0.12% |
| w4u8_qkv_ring_pool_wait_ticks | 3.606 | 3.604 | -0.07% | 3.608 | 3.604 | -0.10% |
| w4u8_qkv_ring_producer_slot_wait_ticks | 14.215 | 12.488 | -12.15% | 12.166 | 11.950 | -1.77% |
| w4u8_qkvo_hmx_lifetime_ticks | 12187.872 | 12241.444 | +0.44% | 12309.513 | 12299.510 | -0.08% |
| w4u8_qkvo_prefetch_wait_ticks | 2125.623 | 2133.254 | +0.36% | 2145.639 | 2148.180 | +0.12% |
| w4u8_qkvo_weight_expand_ticks | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| weight_dma_ticks | 15467.144 | 15536.547 | +0.45% | 15629.351 | 15620.371 | -0.06% |

## prefill: physical/resource/task counters (per pass)

| Counter | Control R1 | Candidate R1 | Change | Control R10 | Candidate R10 | Change |
|---|---|---|---|---|---|---|
| block_invocation_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| boundary_ddr_read_bytes | 4976000.000 | 4976000.000 | +0.00% | 4976000.000 | 4976000.000 | +0.00% |
| boundary_ddr_write_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| boundary_dma_descriptor_count | 289.000 | 289.000 | +0.00% | 289.000 | 289.000 | +0.00% |
| cache_tensor_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| f16_cache_full_prefix_pack_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| f16_cache_native_incremental_append_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| f16_cache_native_prefill_reuse_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| generation_embedding_ddr_read_bytes | 131328.000 | 131328.000 | +0.00% | 131328.000 | 131328.000 | +0.00% |
| generation_lm_head_command_count | 594.000 | 594.000 | +0.00% | 594.000 | 594.000 | +0.00% |
| generation_lm_head_ddr_read_bytes | 156797952.000 | 156797952.000 | +0.00% | 156797952.000 | 156797952.000 | +0.00% |
| generation_lm_head_prefetch_count | 593.000 | 593.000 | +0.00% | 593.000 | 593.000 | +0.00% |
| generation_lm_head_scale_resident_bytes | 1215488.000 | 1215488.000 | +0.00% | 1215488.000 | 1215488.000 | +0.00% |
| hmx_command_count | 2946.000 | 1882.000 | -36.12% | 2946.000 | 1882.000 | -36.12% |
| hmx_fp16_tile_pair_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| hmx_u8s8_tile_pair_count | 1687296.000 | 1687296.000 | +0.00% | 1687296.000 | 1687296.000 | +0.00% |
| intermediate_ddr_read_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| intermediate_ddr_write_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| intermediate_dma_descriptor_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| intermediate_spill_fill_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| repeat_count | 1.000 | 1.000 | +0.00% | 1.000 | 1.000 | +0.00% |
| scan_attention_overlay_capacity_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| scan_attention_overlay_required_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| scan_cache_ddr_read_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| scan_cache_ddr_write_bytes | 15368192.000 | 15368192.000 | +0.00% | 15368192.000 | 15368192.000 | +0.00% |
| scan_cache_dma_descriptor_count | 448.000 | 448.000 | +0.00% | 448.000 | 448.000 | +0.00% |
| u8_attention_audit_ddr_write_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_full_prefix_pack_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_cached_head_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_fallback_head_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_init_bytes | 827904.000 | 827904.000 | +0.00% | 827904.000 | 827904.000 | +0.00% |
| u8_cache_k_vtcm_tail_init_count | 1.000 | 1.000 | +0.00% | 1.000 | 1.000 | +0.00% |
| u8_cache_k_vtcm_tail_native_load_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_row_update_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_seal_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_native_incremental_append_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_native_prefill_build_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_native_prefill_reuse_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| u8_cache_native_prefill_reused_carrier_bytes | 4014080.000 | 4014080.000 | +0.00% | 4014080.000 | 4014080.000 | +0.00% |
| u8_cache_segment_seal_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_segment_sealed_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_segment_tail_append_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_quartet_append_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_quartet_attention_publish_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_quartet_native_load_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_quartet_publish_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_vtcm_tail_init_bytes | 917504.000 | 917504.000 | +0.00% | 917504.000 | 917504.000 | +0.00% |
| u8_cache_v_vtcm_tail_init_count | 1.000 | 1.000 | +0.00% | 1.000 | 1.000 | +0.00% |
| u8_cache_v_vtcm_tail_native_load_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_vtcm_tail_publish_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_vtcm_tail_row_update_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_vtcm_tail_seal_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| vtcm_acquired_bytes | 8388608.000 | 8388608.000 | +0.00% | 8388608.000 | 8388608.000 | +0.00% |
| vtcm_peak_plan_bytes | 8365824.000 | 8365824.000 | +0.00% | 8365824.000 | 8365824.000 | +0.00% |
| vtcm_requested_bytes | 8388608.000 | 8388608.000 | +0.00% | 8388608.000 | 8388608.000 | +0.00% |
| w4u8_av_padding_poison_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_av_requant_call_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_av_requant_vector_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_common_padding_poison_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_decode_direct_n_hmx_command_count | 560.000 | 840.000 | +50.00% | 560.000 | 840.000 | +50.00% |
| w4u8_decode_direct_n_projection_count | 84.000 | 196.000 | +133.33% | 84.000 | 196.000 | +133.33% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 528482304.000 | 704643072.000 | +33.33% | 528482304.000 | 704643072.000 | +33.33% |
| w4u8_decode_k_pair_row4_call_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_decode_k_temp_carrier_skipped_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_decode_q_pair_row4_call_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_decode_softmax_hvx_tile4_call_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_decode_swiglu_padding_poison_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_decode_swiglu_row4_call_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_decode_swiglu_vector_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_final_residual_direct_row4_call_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_final_residual_task_count | 448.000 | 448.000 | +0.00% | 448.000 | 448.000 | +0.00% |
| w4u8_gate_up_swiglu_consume_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_gate_up_swiglu_publish_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_input_norm_direct_row4_call_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_input_norm_task_count | 448.000 | 448.000 | +0.00% | 448.000 | 448.000 | +0.00% |
| w4u8_mlp_down_hmx_command_count | 224.000 | 224.000 | +0.00% | 224.000 | 224.000 | +0.00% |
| w4u8_o_batch_count | 448.000 | 112.000 | -75.00% | 448.000 | 112.000 | -75.00% |
| w4u8_o_gate_prefetch_consume_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_o_gate_prefetch_start_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_post_residual_direct_row4_call_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_post_residual_task_count | 448.000 | 448.000 | +0.00% | 448.000 | 448.000 | +0.00% |
| w4u8_qk_padding_poison_pair_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_qkv_ring_batch_count | 896.000 | 168.000 | -81.25% | 896.000 | 168.000 | -81.25% |
| w4u8_qkv_ring_dispatch_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| w4u8_qkv_ring_expand_task_count | 3584.000 | 0.000 | -100.00% | 3584.000 | 0.000 | -100.00% |
| w4u8_qkv_ring_expand_worker_count | 3.000 | 0.000 | -100.00% | 3.000 | 0.000 | -100.00% |
| w4u8_qkv_ring_head_publish_count | 672.000 | 672.000 | +0.00% | 672.000 | 672.000 | +0.00% |
| w4u8_qkv_ring_hmx_dispatch_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| w4u8_qkv_ring_prep_worker_count | 2.000 | 5.000 | +150.00% | 2.000 | 5.000 | +150.00% |
| w4u8_qkv_ring_slot_count | 4.000 | 2.000 | -50.00% | 4.000 | 2.000 | -50.00% |
| weight_ddr_read_bytes | 866032640.000 | 866032640.000 | +0.00% | 866032640.000 | 866032640.000 | +0.00% |
| weight_dma_descriptor_count | 4403.000 | 2275.000 | -48.33% | 4403.000 | 2275.000 | -48.33% |

## decode: physical/resource/task counters (per pass/decode token)

| Counter | Control R1 | Candidate R1 | Change | Control R10 | Candidate R10 | Change |
|---|---|---|---|---|---|---|
| block_invocation_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| boundary_ddr_read_bytes | 4846976.000 | 4846976.000 | +0.00% | 4846976.000 | 4846976.000 | +0.00% |
| boundary_ddr_write_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| boundary_dma_descriptor_count | 226.000 | 226.000 | +0.00% | 226.000 | 226.000 | +0.00% |
| cache_tensor_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| f16_cache_full_prefix_pack_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| f16_cache_native_incremental_append_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| f16_cache_native_prefill_reuse_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| f16_cache_native_prefill_reused_carrier_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| generation_embedding_ddr_read_bytes | 2304.000 | 2304.000 | +0.00% | 2304.000 | 2304.000 | +0.00% |
| generation_lm_head_command_count | 149.000 | 149.000 | +0.00% | 149.000 | 149.000 | +0.00% |
| generation_lm_head_ddr_read_bytes | 156797952.000 | 156797952.000 | +0.00% | 156797952.000 | 156797952.000 | +0.00% |
| generation_lm_head_prefetch_count | 148.000 | 148.000 | +0.00% | 148.000 | 148.000 | +0.00% |
| generation_lm_head_scale_resident_bytes | 1215488.000 | 1215488.000 | +0.00% | 1215488.000 | 1215488.000 | +0.00% |
| hmx_command_count | 1437.000 | 1437.000 | +0.00% | 1437.000 | 1437.000 | +0.00% |
| hmx_fp16_tile_pair_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| hmx_u8s8_tile_pair_count | 1699840.000 | 1699840.000 | +0.00% | 1699840.000 | 1699840.000 | +0.00% |
| intermediate_ddr_read_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| intermediate_ddr_write_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| intermediate_dma_descriptor_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| intermediate_spill_fill_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| repeat_count | 1.000 | 1.000 | +0.00% | 1.000 | 1.000 | +0.00% |
| scan_attention_overlay_capacity_bytes | 2752512.000 | 2752512.000 | +0.00% | 2752512.000 | 2752512.000 | +0.00% |
| scan_attention_overlay_required_bytes | 141994.667 | 141994.667 | +0.00% | 141994.667 | 141994.667 | +0.00% |
| scan_cache_ddr_read_bytes | 8863232.000 | 8863232.000 | +0.00% | 8863232.000 | 8863232.000 | +0.00% |
| scan_cache_ddr_write_bytes | 91392.000 | 91392.000 | +0.00% | 91392.000 | 91392.000 | +0.00% |
| scan_cache_dma_descriptor_count | 1211.000 | 1211.000 | +0.00% | 1211.000 | 1211.000 | +0.00% |
| u8_attention_audit_ddr_write_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_full_prefix_pack_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_cached_head_count | 196.000 | 196.000 | +0.00% | 196.000 | 196.000 | +0.00% |
| u8_cache_k_vtcm_tail_correction_load_bytes | 12152.000 | 12152.000 | +0.00% | 12152.000 | 12152.000 | +0.00% |
| u8_cache_k_vtcm_tail_ddr_write_skip_bytes | 25088.000 | 25088.000 | +0.00% | 25088.000 | 25088.000 | +0.00% |
| u8_cache_k_vtcm_tail_ddr_write_skip_count | 196.000 | 196.000 | +0.00% | 196.000 | 196.000 | +0.00% |
| u8_cache_k_vtcm_tail_fallback_head_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| u8_cache_k_vtcm_tail_hvx_row_update_count | 196.000 | 196.000 | +0.00% | 196.000 | 196.000 | +0.00% |
| u8_cache_k_vtcm_tail_init_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_init_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_k_vtcm_tail_native_load_bytes | 777728.000 | 777728.000 | +0.00% | 777728.000 | 777728.000 | +0.00% |
| u8_cache_k_vtcm_tail_row_update_count | 196.000 | 196.000 | +0.00% | 196.000 | 196.000 | +0.00% |
| u8_cache_k_vtcm_tail_seal_count | 6.125 | 6.125 | +0.00% | 6.125 | 6.125 | +0.00% |
| u8_cache_native_incremental_append_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| u8_cache_native_prefill_build_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_native_prefill_reuse_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_native_prefill_reused_carrier_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_segment_seal_count | 0.875 | 0.875 | +0.00% | 0.875 | 0.875 | +0.00% |
| u8_cache_segment_sealed_bytes | 59136.000 | 59136.000 | +0.00% | 59136.000 | 59136.000 | +0.00% |
| u8_cache_segment_tail_append_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| u8_cache_v_quartet_append_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_quartet_attention_publish_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_quartet_full_tile_rmw_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_quartet_native_load_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_quartet_publish_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_vtcm_tail_init_bytes | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_vtcm_tail_init_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| u8_cache_v_vtcm_tail_native_load_bytes | 487424.000 | 487424.000 | +0.00% | 487424.000 | 487424.000 | +0.00% |
| u8_cache_v_vtcm_tail_publish_count | 56.000 | 56.000 | +0.00% | 56.000 | 56.000 | +0.00% |
| u8_cache_v_vtcm_tail_row_update_count | 224.000 | 224.000 | +0.00% | 224.000 | 224.000 | +0.00% |
| u8_cache_v_vtcm_tail_seal_count | 7.000 | 7.000 | +0.00% | 7.000 | 7.000 | +0.00% |
| vtcm_acquired_bytes | 8388608.000 | 8388608.000 | +0.00% | 8388608.000 | 8388608.000 | +0.00% |
| vtcm_peak_plan_bytes | 8365824.000 | 8365824.000 | +0.00% | 8365824.000 | 8365824.000 | +0.00% |
| vtcm_requested_bytes | 8388608.000 | 8388608.000 | +0.00% | 8388608.000 | 8388608.000 | +0.00% |
| w4u8_av_padding_poison_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_av_requant_call_count | 224.000 | 224.000 | +0.00% | 224.000 | 224.000 | +0.00% |
| w4u8_av_requant_vector_count | 1792.000 | 1792.000 | +0.00% | 1792.000 | 1792.000 | +0.00% |
| w4u8_common_padding_poison_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_decode_direct_n_hmx_command_count | 989.000 | 989.000 | +0.00% | 989.000 | 989.000 | +0.00% |
| w4u8_decode_direct_n_projection_count | 197.000 | 197.000 | +0.00% | 197.000 | 197.000 | +0.00% |
| w4u8_decode_direct_n_weight_ddr_read_bytes | 860225536.000 | 860225536.000 | +0.00% | 860225536.000 | 860225536.000 | +0.00% |
| w4u8_decode_k_pair_row4_call_count | 112.000 | 112.000 | +0.00% | 112.000 | 112.000 | +0.00% |
| w4u8_decode_k_temp_carrier_skipped_count | 224.000 | 224.000 | +0.00% | 224.000 | 224.000 | +0.00% |
| w4u8_decode_q_pair_row4_call_count | 224.000 | 224.000 | +0.00% | 224.000 | 224.000 | +0.00% |
| w4u8_decode_softmax_hvx_tile4_call_count | 224.000 | 224.000 | +0.00% | 224.000 | 224.000 | +0.00% |
| w4u8_decode_swiglu_padding_poison_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_decode_swiglu_row4_call_count | 5376.000 | 5376.000 | +0.00% | 5376.000 | 5376.000 | +0.00% |
| w4u8_decode_swiglu_vector_count | 5376.000 | 5376.000 | +0.00% | 5376.000 | 5376.000 | +0.00% |
| w4u8_final_residual_direct_row4_call_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| w4u8_final_residual_task_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_gate_up_swiglu_consume_count | 168.000 | 168.000 | +0.00% | 168.000 | 168.000 | +0.00% |
| w4u8_gate_up_swiglu_publish_count | 168.000 | 168.000 | +0.00% | 168.000 | 168.000 | +0.00% |
| w4u8_input_norm_direct_row4_call_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| w4u8_input_norm_task_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_mlp_down_hmx_command_count | 224.000 | 224.000 | +0.00% | 224.000 | 224.000 | +0.00% |
| w4u8_o_batch_count | 112.000 | 112.000 | +0.00% | 112.000 | 112.000 | +0.00% |
| w4u8_o_gate_prefetch_consume_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| w4u8_o_gate_prefetch_start_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| w4u8_post_residual_direct_row4_call_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| w4u8_post_residual_task_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_qk_padding_poison_pair_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_qkv_ring_batch_count | 168.000 | 168.000 | +0.00% | 168.000 | 168.000 | +0.00% |
| w4u8_qkv_ring_dispatch_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| w4u8_qkv_ring_expand_task_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_qkv_ring_expand_worker_count | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.0% |
| w4u8_qkv_ring_head_publish_count | 672.000 | 672.000 | +0.00% | 672.000 | 672.000 | +0.00% |
| w4u8_qkv_ring_hmx_dispatch_count | 28.000 | 28.000 | +0.00% | 28.000 | 28.000 | +0.00% |
| w4u8_qkv_ring_prep_worker_count | 5.000 | 5.000 | +0.00% | 5.000 | 5.000 | +0.00% |
| w4u8_qkv_ring_slot_count | 2.000 | 2.000 | +0.00% | 2.000 | 2.000 | +0.00% |
| weight_ddr_read_bytes | 866032640.000 | 866032640.000 | +0.00% | 866032640.000 | 866032640.000 | +0.00% |
| weight_dma_descriptor_count | 1830.000 | 1830.000 | +0.00% | 1830.000 | 1830.000 | +0.00% |

VTCM request/acquisition are prepared-session grants, not per-token allocations. Peak plan is the declared bound, not a measured allocator high-water mark. FastRPC calls/pass: control=1, candidate=1; HMX ownership domains: 1/1; QNN and CPU fallback: none/none (static/runtime contract). Separate weight and cache traffic remain visible; overlapping traffic counters must not be summed without their declared coverage.

## Correctness and evidence limits

| Scope | Field | Control observed values | Candidate observed values | Meaning |
|---|---|---|---|---|
| prefill | output_mismatches | [1] | [1] | raw field; interpretation below |
| prefill | output_max_lsb | [0] | [0] | raw field; interpretation below |
| prefill | output_nonfinite_count | [0] | [0] | raw field; interpretation below |
| prefill | cache_mismatches | [0] | [0] | raw field; interpretation below |
| prefill | cache_prefix_mismatches | [0] | [0] | raw field; interpretation below |
| prefill | cache_structure_mismatches | [0] | [0] | raw field; interpretation below |
| prefill | cache_nonfinite_count | [0] | [0] | raw field; interpretation below |
| prefill | u8_attention_probability_mask_violation_count | [0] | [0] | raw field; interpretation below |
| prefill | u8_attention_fused_k_operand_mismatch_count | [0] | [0] | raw field; interpretation below |
| decode | output_mismatches | [1] | [1] | raw field; interpretation below |
| decode | output_max_lsb | [0] | [0] | raw field; interpretation below |
| decode | output_nonfinite_count | [0] | [0] | raw field; interpretation below |
| decode | cache_mismatches | [0] | [0] | raw field; interpretation below |
| decode | cache_prefix_mismatches | [0] | [0] | raw field; interpretation below |
| decode | cache_structure_mismatches | [0] | [0] | raw field; interpretation below |
| decode | cache_nonfinite_count | [0] | [0] | raw field; interpretation below |
| decode | u8_attention_probability_mask_violation_count | [0] | [0] | raw field; interpretation below |
| decode | u8_attention_fused_k_operand_mismatch_count | [0] | [0] | raw field; interpretation below |

| Check | Result |
|---|---|
| Fresh short gate | 5/5 pairs PASS |
| Formal gate | 10/10 pairs PASS |
| Selected tokens / logit codes / output hashes | All 193 steps in each of 10 paired sessions equal |
| Boundary/hidden/cache audit files | 60 matching tensor files in each cell: 56 full prefill K/V cache buffers plus four last-row hidden files; 17211392 compared bytes; mismatch=0, max byte LSB=0 |
| Full layer hidden outputs | 112/112 nonzero audit FNV hashes match (28 layers × prefill/three decode steps); prefill hashes cover all 64 rows |
| Candidate QKVO/MLP expansion ticks and QKV expansion tasks | 0 in every candidate prefill record |
| Physical gates | Exact 8 MiB; zero timed intermediate DDR/spill; equal HMX tile work; unchanged decode counters |
| Invocation ledger | All 3860 formal records close exactly; unattributed=0 |
| Per-layer ledger/cache lifecycle | All 108080 nested slice_layer records close exactly, zero unattributed/inter-layer hidden DDR; append +64 for prefill and +1 for each decode |
| Independent numerical reference | Inherited valid implementation lineage; EXP-0216 adds byte-exact direct-control boundary audit, not a new standalone full-model reference run |
| W4U8 semantic quality | Not enabled or claimed |

The raw generation_profile output_mismatches field is derived from generation_token_match against the packaged expected sequence; it is not candidate/control tensor mismatch. Its output_hash hashes the returned token, not a captured full hidden tensor. output_max_lsb and several cache comparison fields in this generation profile are initialized defaults, so they alone are not independent zero-LSB/cache-reference evidence. Authoritative EXP-0216 equality is established by paired selected-code/token/hash signatures across 193 steps, the full prefill cache byte comparison, four final-row hidden byte comparisons, and 112 full-layer hidden hashes. Decode VTCM-only mutable cache tails are checked by preserved code/counters and layer lifecycle, not by treating stale DDR mirrors as complete decode cache. No missing diagnostic is fabricated as zero.

## Rotated prefill pairs

| Pair | Control Host us | Candidate Host us | Paired speed | Output/token/code equality |
|---|---|---|---|---|
| 1 | 43433.385 | 39017.500 | +11.318% | PASS |
| 2 | 43506.302 | 39377.604 | +10.485% | PASS |
| 3 | 43763.698 | 39202.448 | +11.635% | PASS |
| 4 | 43729.375 | 39403.802 | +10.978% | PASS |
| 5 | 43755.104 | 39468.437 | +10.861% | PASS |
| 6 | 43673.490 | 39319.947 | +11.072% | PASS |
| 7 | 43823.958 | 39273.021 | +11.588% | PASS |
| 8 | 41449.739 | 37617.344 | +10.188% | PASS |
| 9 | 43753.072 | 39165.416 | +11.714% | PASS |
| 10 | 43811.979 | 39374.375 | +11.270% | PASS |

## Provenance

Formal: `/mnt/d/llm_exp/results/qwen3-block-htp/exp0216/20260905_formal_25b2fdf`; short/audit: `/mnt/d/llm_exp/results/qwen3-block-htp/exp0216/20260905_short_gate_25b2fdf`. Reproduction helper: `/mnt/d/llm_exp/results/qwen3-block-htp/exp0216/complete_profile_report.py`. Original automatic report remains in report.md; this file supplements it with complete repeat-one/repeat-ten tables. The failed f724a22 collection is retained separately and is not used for performance or inferred counters. Artifact paths and SHA256 values are in closure_evidence_sha256.json; immutable source history and selected baselines are preserved.

