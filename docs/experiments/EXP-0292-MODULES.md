# EXP0292 modules

Temporary FP32-residual W16A16; original FP16-residual implementation retained. Identical original weights, FP16 HMX O/Down outputs, KV16, head, OPT4 and pipeline settings; candidate Norm/residual handles only valid rows on decode, while the unchanged control retains its original full-M64 row work. This compares complete implementations, not isolated dtype arithmetic cost. Embedding widens to FP32; O/Down widen/add once in FP32; input/post/final RMSNorm read FP32 and produce FP16. This aligns residual storage/add precision with A8, not the A8 raw projection-output contract. No deliberate slowdown, extra delay, removed control optimization, PPL or quality claim. Original historical formal F16 remains1710.94/27.68; samebinary paired control below is separately measured.
Selected0/14/27 and chain3 M64/M1 pass unchanged wholeblock thresholds. Independent actual-operand residual additions exact; norm outputs <=1 FP16 ULP. Full-model software reference remains failed for BOTH arms; see measured maxima below. Do not call this full-model numerical/quality acceptance. Final norm and independent entire LMhead checks pass all43steps, argmax43/43. FP16 full audit matches2451 original files; FP32 repeated audit matches2451 files. Both KV streams finite, prefix preserved, valid length extends once, padding untouched. Timed paths have full8MiB grant, zero intermediateDDR/spill/output audit, oneRPC/pass, exact own token/logit repeats and complete additive ledgers.
Five short/ten balanced formal rounds, repeat10; repeat1 auxiliary. Complete Host wall includes embedding, all28blocks, finalnorm, head/greedy, FastRPC; excludes coldloading, tokenizer and audit I/O. Audit-enabled times are not speed evidence. HMX/HVX/DMA/worker counters overlap and cannot be added.

All module values are microseconds and percent of complete Host wall. W4A16 and SP2 are non-paired historical formal repeat10 references; EXP0291 faster W4 diagnostic remains archived separately, not substituted for a formal column.
## Prefill M64

| 模块 | W16A16 FP32残差（本次） | W4A16（EXP0289历史） | W4A8-SP2（EXP0288历史） | SP2 相对 W4A16 |
|---|---|---|---|---|
| I/O、metadata | 136.10 (0.36%) | 322.94 (0.85%) | 233.99 (1.10%) | +38.01% |
| Input RMSNorm | 387.81 (1.02%) | 323.03 (0.85%) | 1035.08 (4.88%) | -68.79% |
| QKV＋Q/K Norm-RoPE | 7651.28 (20.21%) | 11478.84 (30.10%) | 6617.08 (31.23%) | +73.47% |
| QK–Softmax–AV | 3998.34 (10.56%) | 3960.49 (10.39%) | 3335.92 (15.74%) | +18.72% |
| O projection | 2993.13 (7.91%) | 2609.92 (6.84%) | 1213.46 (5.73%) | +115.08% |
| Post-attention residual＋RMSNorm | 488.02 (1.29%) | 311.06 (0.82%) | 1025.93 (4.84%) | -69.68% |
| Gate/Up＋SwiGLU | 10079.29 (26.62%) | 9533.26 (25.00%) | 2664.85 (12.58%) | +257.74% |
| Down | 3831.42 (10.12%) | 3588.46 (9.41%) | 1580.89 (7.46%) | +126.99% |
| Final residual | 275.57 (0.73%) | 72.90 (0.19%) | 2.78 (0.01%) | +2524.96% |
| KV carrier conversion | 189.36 (0.50%) | 187.16 (0.49%) | 135.74 (0.64%) | +37.88% |
| KV append DMA | 436.93 (1.15%) | 424.49 (1.11%) | 256.02 (1.21%) | +65.80% |
| Block orchestration | 21.28 (0.06%) | 19.41 (0.05%) | 35.59 (0.17%) | -45.46% |
| Layer bookkeeping | 24.12 (0.06%) | 26.95 (0.07%) | 22.28 (0.11%) | +20.94% |
| Stage-boundary bookkeeping | 7.16 (0.02%) | 8.09 (0.02%) | 20.19 (0.10%) | -59.92% |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | N/A zero denominator |
| Runtime setup/teardown | 92.76 (0.25%) | 107.98 (0.28%) | 114.80 (0.54%) | -5.94% |
| Embedding | 43.69 (0.12%) | 38.35 (0.10%) | 38.83 (0.18%) | -1.24% |
| Final model RMSNorm | 3.14 (0.01%) | 30.82 (0.08%) | 9.64 (0.05%) | +219.55% |
| LM head＋greedy，不含 final norm | 6339.74 (16.75%) | 4335.43 (11.37%) | 1982.58 (9.36%) | +118.68% |
| Host–DSP 边界 | 857.39 (2.26%) | 753.08 (1.97%) | 863.40 (4.07%) | -12.78% |
| 完整 Host wall | 37856.53 (100.00%) | 38132.64 (100.00%) | 21189.05 (100.00%) | +79.96% |

## Decode per step

| 模块 | W16A16 FP32残差（本次） | W4A16（EXP0289历史） | W4A8-SP2（EXP0288历史） | SP2 相对 W4A16 |
|---|---|---|---|---|
| I/O、metadata | 138.36 (0.39%) | 350.91 (0.90%) | 227.23 (2.10%) | +54.43% |
| Input RMSNorm | 75.74 (0.21%) | 318.62 (0.82%) | 186.51 (1.72%) | +70.83% |
| QKV＋Q/K Norm-RoPE | 7611.25 (21.41%) | 11609.55 (29.77%) | 1484.08 (13.71%) | +682.27% |
| QK–Softmax–AV | 2985.64 (8.40%) | 3300.17 (8.46%) | 1924.74 (17.79%) | +71.46% |
| O projection | 2988.30 (8.41%) | 2769.53 (7.10%) | 762.04 (7.04%) | +263.44% |
| Post-attention residual＋RMSNorm | 74.92 (0.21%) | 306.75 (0.79%) | 191.77 (1.77%) | +59.96% |
| Gate/Up＋SwiGLU | 10050.04 (28.28%) | 9670.93 (24.80%) | 1773.61 (16.39%) | +445.27% |
| Down | 3829.81 (10.78%) | 3628.79 (9.31%) | 1094.94 (10.12%) | +231.41% |
| Final residual | 5.91 (0.02%) | 71.52 (0.18%) | 1.92 (0.02%) | +3632.50% |
| KV carrier conversion | 423.44 (1.19%) | 1135.51 (2.91%) | 310.04 (2.86%) | +266.24% |
| KV append DMA | 228.76 (0.64%) | 231.24 (0.59%) | 109.53 (1.01%) | +111.12% |
| Block orchestration | 15.55 (0.04%) | 14.11 (0.04%) | 28.00 (0.26%) | -49.61% |
| Layer bookkeeping | 16.84 (0.05%) | 16.44 (0.04%) | 16.33 (0.15%) | +0.65% |
| Stage-boundary bookkeeping | 1.72 (0.00%) | 1.70 (0.00%) | 1.75 (0.02%) | -2.94% |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | N/A zero denominator |
| Runtime setup/teardown | 49.37 (0.14%) | 60.64 (0.16%) | 71.50 (0.66%) | -15.19% |
| Embedding | 1.61 (0.00%) | 1.49 (0.00%) | 1.40 (0.01%) | +6.59% |
| Final model RMSNorm | 2.66 (0.01%) | 2.32 (0.01%) | 7.72 (0.07%) | -69.97% |
| LM head＋greedy，不含 final norm | 6327.77 (17.80%) | 4807.51 (12.33%) | 1978.20 (18.28%) | +143.02% |
| Host–DSP 边界 | 715.17 (2.01%) | 695.59 (1.78%) | 650.94 (6.01%) | +6.86% |
| 完整 Host wall | 35542.88 (100.00%) | 38993.34 (100.00%) | 10822.27 (100.00%) | +260.31% |

## Paired E2E

| 配置 | Prefill token/s | Decode token/s | 64-token Host ms | 42-step Host ms |
|---|---|---|---|---|
| fp16 | 1709.12 | 27.69 | 37.446 | 1516.971 |
| fp32 | 1690.59 | 28.14 | 37.857 | 1492.801 |
