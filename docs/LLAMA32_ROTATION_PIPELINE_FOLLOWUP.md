# L32-0021 R4 layout / SP2 pipeline follow-up

R3+R4 now passes both single-layer10% speed gates. Continuous3-layer extension fails the independent numerical gate; no full16/frontend/E2E extension. Preserve the optimized opt-in path and all failed/superseded evidence; no default or quality promotion.

Profiled and final single-layer validation source: `5adc46b774a23df9138f1070560fb68a28c165a1`. Three-layer build/reference source: `d77364464d9c73910b3d074b0da6ac4308fbe1ba` (same native implementation, different layer-count build). Evidence: `/mnt/d/llm_exp/results/llama32-htp/l32-0021`.

## Implementation and candidates

1. Two SP2 finish workers consume the previous R4 output while the caller lays out the next input and the sole HMX owner handles the current batch. Alone this yields little benefit because VTCM traffic and longer two-worker finish reduce the overlap gain.
2. Eight-row SP2 lookup batch improves latency hiding. Compiler review subsequently found its variable-index vector array lowered to activation staging on the stack. Those R4/both runs are not physical/performance acceptance evidence, even though explicit DDR counters were zero.
3. Retained candidate: orient the second dense factor as H16 times the group-by-channel intermediate. Since H16 is symmetric, the old transposed-intermediate times H16 result is its transpose; reduction order/scale remain unchanged. New native channel tiles support contiguous vector loads/stores instead of the old scatters/gathers. Reuse both token lanes per load, materialize H16 from immutable native sign tiles, and consume SP2 lookup results directly from phase-dead Down VTCM scratch. No vector array on the stack in the final binary.

The current second GEMM changes physical shape from Mtiles128/Ktiles1/Ntiles1 to1/1/128 per8-token batch. Logical full8192 rotation, HMX tile-pair count, FP16 rounding, SP2 LUT/weights/scales and FP32 residual are unchanged. No butterfly/FWHT. Each finish worker owns2048B inside existing dead Down storage; no allocation growth. Current HMX slot, previous output, next input and scratch have disjoint ownership and completion barriers.

Final disassembly: finish frame88B and layout frame24B contain scalar control/save slots only, no activation vector staging. Superseded finish frame1280B copied1024B of vectors onto stack;0020 control frame48B did not. Evidence in `compiler-physical-review.json` and named disassemblies.20 superseded R4/both CLI invocations are explicitly excluded from physical/performance acceptance; their numeric results remain.

## Correctness and fixed measurement

Final OFF/R3/R4/both, each at layers0/7/15, pass: independent ideal per-row cosine>=.999, rotation component error<=1FP16ULP+minnormal, conditional native-rotation-to-tail exact, and same output hashes as0020. R4 ideal-bitexact CLI exit1 remains failure status; it is not rewritten to success. Formal physical checks enforce8MiB VTCM, no explicit intermediateDDR/spill/W4 expansion, exact cache/structure/replay hashes, and complete additive module accounting. The compiler check supplements these counters.

Formal ten cyclic five-arm cycles: currentOFF/R3/R4/both and sealed0020 both. Each of50 processes executes one warmup plus ten measured M64+decode1 past64 replay pairs.1100 RPCs including100warmups;1000 timed RPCs. Bootstrap paired process means,20000 draws,seed21021.95%upper<=1.10 separately for prefill/decode versus currentOFF. One formal run, no optional stopping. Short run auxiliary only. Old control hashes verified against0020 ledger; no recompilation of that control.

| Arm | Prefill us | vs OFF | 95% ratio CI | Decode us | vs OFF | 95% ratio CI |
|---|---:|---:|---|---:|---:|---|
| off | 1959.78 | +0.00% | 1.00000–1.00000 | 1464.19 | +0.00% | 1.00000–1.00000 |
| r3 | 1899.49 | -3.08% | 0.95057–0.98730 | 1400.94 | -4.32% | 0.92552–0.98827 |
| r4 | 2159.80 | +10.21% | 1.08175–1.12289 | 1507.40 | +2.95% | 1.00120–1.06187 |
| both | 2090.70 | +6.68% | 1.04802–1.08690 | 1497.78 | +2.29% | 0.99073–1.05666 |
| oldboth | 2159.67 | +10.20% | 1.08133–1.12608 | 1469.60 | +0.37% | 0.97848–1.03315 |

Combined vs sealed0020 in the same cycles: prefill latency ratio0.968066 (95%CI0.950082–0.987010),3.19% shorter. Decode ratio1.019180 (CI0.984172–1.061396); no statistically clear improvement or regression. R3 passes both gates. R4-alone prefill+10.21% with upper12.29% fails; that arm stays paused. R3 performance includes its existing selected packing/batch path; it is not a claim that extra arithmetic is intrinsically free.

Prefill Gate/Up+SwiGLU including R4: old613.27us -> new529.86us. Combined R4 diagnostics: prepare6.94us, GEMM submit/wait33.95us, caller layout39.83us, additional exposed finish/dispatch/wait57.38us; worker work sum153.17us overlaps these. Finish counter semantics now attribute only caller time outside layout during overlap; do not compare it to old total finish elapsed as if both were standalone encoder times.

## Complete module table: prefill

Microseconds (share of Host wall). QKV includes R3; Gate/Up+SwiGLU includes R4. Frontend N/A for single-layer scope.

| Module | OFF | R3 | R4 | R3+R4 | sealed0020 R3+R4 |
|---|---:|---:|---:|---:|---:|
| I/O、metadata | 43.41 (2.22%) | 38.10 (2.01%) | 44.48 (2.06%) | 43.28 (2.07%) | 39.36 (1.82%) |
| Input RMSNorm | 76.85 (3.92%) | 76.90 (4.05%) | 76.86 (3.56%) | 76.81 (3.67%) | 76.83 (3.56%) |
| QKV＋RoPE | 237.11 (12.10%) | 180.89 (9.52%) | 236.80 (10.96%) | 180.68 (8.64%) | 181.60 (8.41%) |
| QK–Softmax–AV | 521.08 (26.59%) | 520.69 (27.41%) | 521.68 (24.15%) | 519.57 (24.85%) | 518.85 (24.02%) |
| O projection | 98.42 (5.02%) | 98.65 (5.19%) | 98.17 (4.55%) | 97.84 (4.68%) | 98.11 (4.54%) |
| Post-attention residual＋RMSNorm | 78.76 (4.02%) | 86.77 (4.57%) | 78.86 (3.65%) | 86.74 (4.15%) | 86.88 (4.02%) |
| Gate/Up＋SwiGLU | 323.88 (16.53%) | 324.84 (17.10%) | 529.42 (24.51%) | 529.86 (25.34%) | 613.27 (28.40%) |
| Down | 196.99 (10.05%) | 196.60 (10.35%) | 197.68 (9.15%) | 197.09 (9.43%) | 196.57 (9.10%) |
| Final residual | 0.08 (0.00%) | 0.09 (0.00%) | 0.08 (0.00%) | 0.07 (0.00%) | 0.07 (0.00%) |
| KV carrier conversion | 1.62 (0.08%) | 1.48 (0.08%) | 1.60 (0.07%) | 1.44 (0.07%) | 1.48 (0.07%) |
| KV append DMA | 8.94 (0.46%) | 8.43 (0.44%) | 8.80 (0.41%) | 8.61 (0.41%) | 8.30 (0.38%) |
| Block orchestration | 1.09 (0.06%) | 1.10 (0.06%) | 1.09 (0.05%) | 1.10 (0.05%) | 1.13 (0.05%) |
| Layer bookkeeping | 0.68 (0.03%) | 0.69 (0.04%) | 0.68 (0.03%) | 0.69 (0.03%) | 0.69 (0.03%) |
| Stage-boundary bookkeeping | 0.40 (0.02%) | 0.41 (0.02%) | 0.41 (0.02%) | 0.40 (0.02%) | 0.40 (0.02%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 48.96 (2.50%) | 48.90 (2.57%) | 48.90 (2.26%) | 48.85 (2.34%) | 48.76 (2.26%) |
| Embedding | N/A | N/A | N/A | N/A | N/A |
| Final model RMSNorm | N/A | N/A | N/A | N/A | N/A |
| LM head＋greedy（不含 final norm） | N/A | N/A | N/A | N/A | N/A |
| Host–DSP 边界 | 321.50 (16.41%) | 314.94 (16.58%) | 314.30 (14.55%) | 297.67 (14.24%) | 287.36 (13.31%) |
| 完整 Host wall | 1959.78 (100.00%) | 1899.49 (100.00%) | 2159.80 (100.00%) | 2090.70 (100.00%) | 2159.67 (100.00%) |

## Complete module table: decode

Microseconds (share of Host wall). QKV includes R3; Gate/Up+SwiGLU includes R4. Frontend N/A for single-layer scope.

| Module | OFF | R3 | R4 | R3+R4 | sealed0020 R3+R4 |
|---|---:|---:|---:|---:|---:|
| I/O、metadata | 43.26 (2.95%) | 38.36 (2.74%) | 44.28 (2.94%) | 42.73 (2.85%) | 39.90 (2.72%) |
| Input RMSNorm | 53.04 (3.62%) | 53.01 (3.78%) | 53.08 (3.52%) | 53.04 (3.54%) | 53.03 (3.61%) |
| QKV＋RoPE | 100.78 (6.88%) | 70.06 (5.00%) | 100.71 (6.68%) | 70.33 (4.70%) | 70.33 (4.79%) |
| QK–Softmax–AV | 358.47 (24.48%) | 358.43 (25.59%) | 358.16 (23.76%) | 357.93 (23.90%) | 357.68 (24.34%) |
| O projection | 57.49 (3.93%) | 56.45 (4.03%) | 55.33 (3.67%) | 55.70 (3.72%) | 57.28 (3.90%) |
| Post-attention residual＋RMSNorm | 53.65 (3.66%) | 53.70 (3.83%) | 53.64 (3.56%) | 53.64 (3.58%) | 53.65 (3.65%) |
| Gate/Up＋SwiGLU | 289.65 (19.78%) | 288.80 (20.61%) | 340.43 (22.58%) | 341.65 (22.81%) | 346.07 (23.55%) |
| Down | 158.25 (10.81%) | 157.38 (11.23%) | 157.80 (10.47%) | 159.93 (10.68%) | 158.84 (10.81%) |
| Final residual | 0.07 (0.00%) | 0.08 (0.01%) | 0.07 (0.00%) | 0.07 (0.00%) | 0.07 (0.00%) |
| KV carrier conversion | 3.13 (0.21%) | 3.18 (0.23%) | 3.14 (0.21%) | 3.19 (0.21%) | 3.22 (0.22%) |
| KV append DMA | 6.04 (0.41%) | 6.07 (0.43%) | 6.04 (0.40%) | 6.11 (0.41%) | 5.81 (0.40%) |
| Block orchestration | 1.06 (0.07%) | 1.05 (0.07%) | 1.08 (0.07%) | 1.05 (0.07%) | 1.09 (0.07%) |
| Layer bookkeeping | 0.64 (0.04%) | 0.66 (0.05%) | 0.66 (0.04%) | 0.66 (0.04%) | 0.66 (0.05%) |
| Stage-boundary bookkeeping | 0.41 (0.03%) | 0.41 (0.03%) | 0.41 (0.03%) | 0.40 (0.03%) | 0.40 (0.03%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 48.71 (3.33%) | 48.78 (3.48%) | 48.73 (3.23%) | 48.85 (3.26%) | 48.72 (3.31%) |
| Embedding | N/A | N/A | N/A | N/A | N/A |
| Final model RMSNorm | N/A | N/A | N/A | N/A | N/A |
| LM head＋greedy（不含 final norm） | N/A | N/A | N/A | N/A | N/A |
| Host–DSP 边界 | 289.52 (19.77%) | 264.52 (18.88%) | 283.85 (18.83%) | 302.51 (20.20%) | 272.85 (18.57%) |
| 完整 Host wall | 1464.19 (100.00%) | 1400.94 (100.00%) | 1507.40 (100.00%) | 1497.78 (100.00%) | 1469.60 (100.00%) |

## Consecutive-layer extension and stop

After speed pass, generated missing Down layers1/2 afresh from original Llama BF16, same frozen eight-train-window SP2 calibration and per-channel RTN fold method as0015. Kept sealed layer0 fold. Built independent3-layer dense-factor references before device execution. No recalibration on scored outputs, no reused Qwen/mllm folded weights.

Continuous3-layer OFF remains exactly equal to its independent reference, prefill and decode. Combined R3+R4 completes both hardware calls with3 R3 and3 R4 calls, correct cache structure/prefix, finite output and8MiB/no explicit spill; however minimum row cosine is0.791505 prefill and0.911935 decode (<.999). Cache value mismatches64626/65668 and ideal output mismatches are retained. Thus the numeric gate fails before fullmodel.

Diagnostic: input/RoPE bytes match the separately audited layer0 fixture. Feed only its actual hardware layer0 output into ideal software layers1/2. Relative to the fully ideal chain, prefill minimum cosine drops from0.999987 at layer0 to0.854610 at layer1 and0.803272 at layer2; decode0.999998 ->0.917452 ->0.908457. This reproduces substantial divergence without any cross-layer hardware scheduling. It is evidence of strong amplification of existing rounding by subsequent quantized computation. It is not a full conditional native oracle: prediction vs complete hardware cosine is0.964594 prefill /0.978669 decode, so not every remaining difference is attributed.

Next direction: keep the validated faster pipeline fixed and locate amplification at the next layer input RMSNorm/A8, QKV/attention and FFN boundaries, with actual component-conditioned references. Do not relax the independent gate or label fullmodel validated based on single-layer speed. R4-alone speed can be revisited separately; it does not invalidate the combined speed gate.

111 CLI invocations total:47 exit0,64 exit1 (63 retained single-layer ideal-bitexact failures and one failed3-layer independent gate).1242 completed token-boundary RPCs.20 superseded R4/both runs have the identified stack-staging physical defect. No native crash/build failure. One local diagnostic Python syntax error occurred before computation/output writes and was corrected without a device attempt.

E2E tokens/s: **not measured** in0021; full16/frontend stopped at the3-layer numerical gate. Historical0018 no-rotation FP32/SP2 M64+15 result2069.70prefill /42.51decode tok/s is not a rotated result.
