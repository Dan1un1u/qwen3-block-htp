# L32-0019: Llama dense R3/R4 with FP32 residual and SP2

Implemented opt-in dense R3 H64 and full8192 R4 H16 x H512 on the L32-0018
FP32 O/Down/residual/RMSNorm path. Defaults remain OFF. No butterfly, W4 expansion,
new R1/R2, calibration campaign or quality promotion. Other recipes and frozen
Qwen source unchanged. Rotated Down fixtures are verified L32-0015 fresh-original
BF16 inverse-R4 RTN exports (layers0/7/15), not old Qwen folded weights and not
identical quantized weights to the unrotated GPTQ baseline.

## Result and stop point
All three single layers0/7/15 pass the registered component and ideal cosine
checks in R3, R4 and both modes, M64 prefill plus decode past64. OFF regression
passes exact FP32. R3 final FP32 outputs are exact against independent ideal
reference for these fixtures. R4/full combination are not ideal-reference bitexact:
minimum row cosine across tested rotation cases is 0.9999834747 (>0.999).
All component errors stay within1FP16ULP+minnormal. On the actual HMX rotation
outputs, SP2/integer projections/FP32 tails match exactly; this conditional audit
is not a claim of independent full-chain bitexactness. CLI exit1 for ideal-exact
comparison on R4 is retained. No numerical gate was changed.

Fixed ten cyclic four-arm singlelayer Host-wall cycles completed (40 processes,
80 token boundaries). Each process performs one M64 prefill and one decode;
there are ten independent observations per arm/phase, not a repeat1 decision.
Timing excludes loading and audit dumps; includes per-boundary runtime setup and
FastRPC. Every timed output matches its independently audited functional path's
output hash. This is repeatability evidence in addition to, not a substitute for,
component and conditional arithmetic checks. New process / Host-DSP timing is
noisy; do not claim R3 has a stable10% regression based on its mean.

No arm passes both paired-bootstrap95% upper<=1.10 gates. R4/both prefill fail;
R3 is inconclusive with wide intervals. Per user instruction, stop before
continuous3/full16/frontend/PPL/E2E. No extra optimization iteration or baseline
promotion. This is a single-layer integration checkpoint, not a full-model result.

| Arm | Prefill μs | Change / 95% ratio CI | Decode μs | Change / 95% ratio CI | Both pass |
|---|---:|---|---:|---|---|
| off | 2294.9 | +0.00% [1.0000,1.0000] | 1661.3 | +0.00% [1.0000,1.0000] | True |
| r3 | 2354.3 | +2.59% [0.9197,1.1933] | 1828.2 | +10.05% [0.9351,1.3749] | False |
| r4 | 3100.3 | +35.10% [1.1441,1.5952] | 1676.2 | +0.90% [0.9698,1.0437] | False |
| both | 3336.5 | +45.39% [1.2465,1.6768] | 1706.0 | +2.69% [0.9925,1.0586] | False |

## Physical and implementation evidence
VTCM peak OFF/R3 8,098,272B; R4/both8,229,344B, requested/acquired8,388,608B.
Single HMX owner; FP16 HMX only declared rotations. Native W4 remains packed.
No timed intermediate DDR read/write/spill. Additive ledgers reconcile exactly.
R4 SP2 high plane aliases phase-dead Gate, paying for the larger rotation input
arena while retaining FP32 residuals within budget.

Owned integration repairs retained in history: Q32/KV16 dispatch; three R3 prep
workers; head-ready publication on Llama integer GQA; avoid applying RoPE twice;
aligned R4 gather/scatter windows with group offsets in lanes. R4 crash PC resolved
to qbh_r4_layout_vector scatter, not HMX matmul; badVA0xFF3FF004. Changed FP32
allocation exposed the old region-base assumption. No undocumented hardware
boundary is asserted as a specification.

Diagnostic speed attribution: prefill Gate/Up+SwiGLU includingR4 rises from361.8us
OFF to757.5us R4; Down remains197.5vs197.3us. Thus the measured DSP overhead is
upstream of Down. Host-DSP boundary also increases/noises the wall result; the
35.1% wall increase must not all be assigned to matrix arithmetic. R3 QKV+RoPE
rises260.4→309.3us prefill and101.5→147.3us decode. This port has measurable local
cost; Qwen's near-zero overhead is not established for Llama.

Suggested discussion: first restore/fuse R4 producer scheduling on the FP32
specialized MLP path, then reduce layout/SP2 consumer time; isolate R3 consumer
and native packing. If deeper optimization is authorized, use warmed within-PD
repeated replay as additional fixed protocol to reduce Host boundary uncertainty.
Do not retroactively discard this ten-cycle result or promote a new timing result
without its own preregistered scope. Full model only after both speed gates.

## Complete module tables
Units μs, parentheses share of full Host wall. R3 included in QKV/RoPE;
R4 included in Gate/Up/SwiGLU. Embedding/final model norm/head are N/A because
these are single-layer tests. All zero measured bookkeeping retained.

### prefill

| Module | OFF | R3 | R4 | R3+R4 |
|---|---:|---:|---:|---:|
| I/O、metadata | 45.0 (1.96%) | 38.4 (1.63%) | 41.5 (1.34%) | 38.3 (1.15%) |
| Input RMSNorm | 83.2 (3.63%) | 82.4 (3.50%) | 82.2 (2.65%) | 82.7 (2.48%) |
| QKV＋RoPE | 260.4 (11.35%) | 309.3 (13.14%) | 256.2 (8.26%) | 308.7 (9.25%) |
| QK–Softmax–AV | 530.9 (23.13%) | 525.9 (22.34%) | 531.0 (17.13%) | 526.6 (15.78%) |
| O projection | 104.1 (4.54%) | 104.7 (4.45%) | 102.4 (3.30%) | 104.1 (3.12%) |
| Post-attention residual＋RMSNorm | 79.2 (3.45%) | 87.3 (3.71%) | 79.2 (2.55%) | 87.3 (2.62%) |
| Gate/Up＋SwiGLU | 361.8 (15.77%) | 351.0 (14.91%) | 757.5 (24.43%) | 754.9 (22.63%) |
| Down | 197.5 (8.61%) | 194.9 (8.28%) | 197.3 (6.36%) | 194.2 (5.82%) |
| Final residual | 0.6 (0.02%) | 0.5 (0.02%) | 0.5 (0.02%) | 0.5 (0.01%) |
| KV carrier conversion | 3.0 (0.13%) | 2.3 (0.10%) | 2.7 (0.09%) | 2.3 (0.07%) |
| KV append DMA | 12.1 (0.53%) | 9.9 (0.42%) | 10.3 (0.33%) | 9.0 (0.27%) |
| Block orchestration | 7.7 (0.34%) | 7.4 (0.32%) | 6.6 (0.21%) | 7.2 (0.21%) |
| Layer bookkeeping | 3.4 (0.15%) | 3.3 (0.14%) | 2.9 (0.09%) | 3.1 (0.09%) |
| Stage-boundary bookkeeping | 4.9 (0.21%) | 4.8 (0.20%) | 4.8 (0.16%) | 4.9 (0.15%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 83.2 (3.62%) | 80.5 (3.42%) | 82.0 (2.64%) | 84.4 (2.53%) |
| Embedding | N/A | N/A | N/A | N/A |
| Final model RMSNorm | N/A | N/A | N/A | N/A |
| LM head＋greedy（不含 final norm） | N/A | N/A | N/A | N/A |
| Host–DSP 边界 | 517.9 (22.57%) | 551.6 (23.43%) | 943.3 (30.43%) | 1128.3 (33.82%) |
| 完整 Host wall | 2294.9 (100.00%) | 2354.3 (100.00%) | 3100.3 (100.00%) | 3336.5 (100.00%) |

### decode

| Module | OFF | R3 | R4 | R3+R4 |
|---|---:|---:|---:|---:|
| I/O、metadata | 50.1 (3.02%) | 41.0 (2.24%) | 48.3 (2.88%) | 44.6 (2.61%) |
| Input RMSNorm | 53.0 (3.19%) | 53.0 (2.90%) | 53.0 (3.16%) | 53.0 (3.11%) |
| QKV＋RoPE | 101.5 (6.11%) | 147.3 (8.06%) | 100.8 (6.01%) | 148.4 (8.70%) |
| QK–Softmax–AV | 365.1 (21.98%) | 362.1 (19.81%) | 362.8 (21.64%) | 363.9 (21.33%) |
| O projection | 57.1 (3.44%) | 57.3 (3.14%) | 57.4 (3.42%) | 56.6 (3.32%) |
| Post-attention residual＋RMSNorm | 54.2 (3.26%) | 54.2 (2.96%) | 54.1 (3.23%) | 54.2 (3.18%) |
| Gate/Up＋SwiGLU | 293.8 (17.69%) | 295.9 (16.19%) | 361.8 (21.59%) | 361.8 (21.21%) |
| Down | 159.3 (9.59%) | 159.2 (8.71%) | 158.8 (9.47%) | 157.5 (9.23%) |
| Final residual | 0.1 (0.01%) | 0.1 (0.01%) | 0.2 (0.01%) | 0.2 (0.01%) |
| KV carrier conversion | 3.2 (0.19%) | 3.2 (0.18%) | 3.2 (0.19%) | 3.2 (0.19%) |
| KV append DMA | 6.3 (0.38%) | 6.5 (0.36%) | 6.0 (0.36%) | 6.0 (0.35%) |
| Block orchestration | 1.6 (0.10%) | 1.7 (0.09%) | 1.8 (0.11%) | 1.9 (0.11%) |
| Layer bookkeeping | 0.9 (0.05%) | 1.0 (0.06%) | 1.0 (0.06%) | 1.1 (0.06%) |
| Stage-boundary bookkeeping | 0.4 (0.03%) | 0.5 (0.03%) | 0.4 (0.02%) | 0.4 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 50.2 (3.02%) | 50.0 (2.73%) | 50.3 (3.00%) | 50.5 (2.96%) |
| Embedding | N/A | N/A | N/A | N/A |
| Final model RMSNorm | N/A | N/A | N/A | N/A |
| LM head＋greedy（不含 final norm） | N/A | N/A | N/A | N/A |
| Host–DSP 边界 | 464.3 (27.95%) | 595.2 (32.55%) | 416.4 (24.84%) | 402.7 (23.60%) |
| 完整 Host wall | 1661.3 (100.00%) | 1828.2 (100.00%) | 1676.2 (100.00%) | 1706.0 (100.00%) |

E2E token/s: N/A for rotations; not run. Historical accepted L32-0018 unrotated
FP32 residual+SP2 complete model M64+15:2069.70 prefill tok/s,42.51 decode tok/s,
not a paired comparison to this single-layer experiment.

## Reproduction and provenance
Source profiled8d24121ce0dd776d83de33c619fd671954287d14, native finald1a83002924425b620a51649ee79fd2ea5920b5b.
Tools: llama32_rotated_fp32.py prepare/run; profile_llama32_rotations.py.
Results /mnt/d/llm_exp/results/llama32-htp/l32-0019, formal-layer0-a01.
20 functional/diagnostic CLI attempts:15 complete(30 boundaries),1 argument reject,
2 DSP guard failures,1 owned hung process terminated after verifying PID/package,
1 DSP scatter exception. One complete early R3 run had repeated-RoPE mismatch.
40 formal CLI complete(80 boundaries). Total60CLI;26exit0,32exit1,1exit2,1exit137.
All failed evidence retained. Routine shell search and local Python import errors
before device are not hardware attempts. Nine build logs retained.

Evidence ledger `48e1658f7a9cd80c8e5d66c8d3733ae272c24f92a247946f51f583e9636d0304` (328 files, 10 package manifests verified).
