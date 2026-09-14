# L32-0020 dense R3/R4 pipeline optimization

R3 passes both 10% Host-wall gates. R4 and R3+R4 pass decode, but fail prefill. No full-model extension or default/quality promotion. Two bounded native candidates tested; retain candidate1 and revert the slower candidate2 layout.

Profiled and validated source: `0eb7615c43bb6e36c402f3a5d12d7162c32722dd`. Evidence: `/mnt/d/llm_exp/results/llama32-htp/l32-0020`. Llama W4A8 native packed W4, SP2 mode8, FP32 residual, dense H64 R3 and H16 x H512 R4; fixed L32-0019 weights/scales. No butterfly arithmetic.

## Changes and diagnosis

- R3 materializes H64 by copying four immutable native H128 subtiles instead of constructing 4096 scalar entries. Four-row HVX quantization writes the final A8 tile format directly; scalar tail arithmetic unchanged.
- R4 separates four source gathers and four SP2 lookups to hide dependency latency. Each context uses1024B of phase-dead Down storage; VTCM allocation does not grow.
- Candidate2 register transpose was bit-identical, but auxiliary R4 prefill layout time rose from77.86 to140.36us. Reverted; candidate2 evidence retained.
- Inspection corrects the initial protocol hypothesis: Llama already overlaps SwiGLU/R4 input production with Gate/Up via its SP2 producer. Zero dense_r4_prefill counters describe a different path and do not imply missing overlap.
- One owned unused-parameter build failure was fixed before deployment; all failed build logs remain. No runtime math or numerical gate was relaxed.

## Validation and measurement

OFF/R3/R4/both validated separately at source layers0/7/15, M64+decode1. Components remain within1FP16ULP+minnormal; conditional actual-rotation-to-tail arithmetic exact; independent ideal per-row cosine>=.999. Rotation outputs match the0019 hashes. OFF layers7/15 additionally pass their independent exact FP32 references. R4 ideal-bitexact failures are retained, not converted into pass statuses.

Formal: ten cyclic four-arm cycles (40 processes), each process has one warmup plus ten measured replay sequences (M64 then decode at past64), reset initial cache/input outside timer.880 RPC boundaries including80 warmups;800 timed RPCs. Bootstrap ten paired process means,20000 draws,seed20020; upper95% ratio<=1.10. Original ideal-exact exit1 is retained even in repeated diagnostic mode; physical/RPC/structure failures abort. Prior component/conditional/ideal audits are prerequisites. Short run is auxiliary only.

All formal calls: one HMX owner,8MiB acquired, VTCM peak8,098,272B OFF/R3 and8,229,344B R4/both, no timed intermediate DDR/spill or W4 expansion; exact KV/structure/hash repeatability.64 CLI processes total:30 exit0,34 retained ideal-exact exit1;944 completed token-boundary RPCs. No native crash.

## Formal Host-wall gate

| Arm | Prefill us | vs OFF | 95% ratio CI | Decode us | vs OFF | 95% ratio CI |
|---|---:|---:|---|---:|---:|---|
| off | 1942.87 | +0.00% | 1.00000–1.00000 | 1455.23 | +0.00% | 1.00000–1.00000 |
| r3 | 1868.04 | -3.85% | 0.94113–0.98465 | 1402.32 | -3.64% | 0.93488–0.99523 |
| r4 | 2207.37 | +13.61% | 1.11023–1.16499 | 1499.09 | +3.01% | 1.00044–1.06129 |
| both | 2186.07 | +12.52% | 1.10035–1.14936 | 1524.38 | +4.75% | 1.02152–1.07452 |

R3 speed is a result of the selected native packing/batch path as a whole, not evidence that adding mathematical work is intrinsically faster. R4 packages contain separately folded/requantized Down weights; no quality gain is inferred.0019 used cold processes: its historical35.10%/45.39% prefill overheads cannot be subtracted from this warmed result as a paired optimization gain.

## Complete module table: prefill

Microseconds, parentheses: share of complete Host wall. QKV includes R3; Gate/Up+SwiGLU includes R4. Additive accounting, overlapping subcounters are not added again. Frontend is outside this single-layer experiment.

| Module | OFF | R3 | R4 | R3+R4 |
|---|---:|---:|---:|---:|
| I/O、metadata | 41.15 (2.12%) | 38.29 (2.05%) | 39.33 (1.78%) | 42.95 (1.96%) |
| Input RMSNorm | 76.78 (3.95%) | 76.81 (4.11%) | 76.78 (3.48%) | 76.78 (3.51%) |
| QKV＋RoPE | 237.45 (12.22%) | 180.84 (9.68%) | 236.90 (10.73%) | 180.99 (8.28%) |
| QK–Softmax–AV | 522.66 (26.90%) | 522.12 (27.95%) | 520.91 (23.60%) | 518.60 (23.72%) |
| O projection | 98.70 (5.08%) | 98.05 (5.25%) | 98.92 (4.48%) | 98.29 (4.50%) |
| Post-attention residual＋RMSNorm | 78.80 (4.06%) | 86.77 (4.64%) | 78.75 (3.57%) | 86.73 (3.97%) |
| Gate/Up＋SwiGLU | 326.96 (16.83%) | 325.15 (17.41%) | 606.29 (27.47%) | 602.57 (27.56%) |
| Down | 198.20 (10.20%) | 199.09 (10.66%) | 198.42 (8.99%) | 197.98 (9.06%) |
| Final residual | 0.08 (0.00%) | 0.08 (0.00%) | 0.08 (0.00%) | 0.08 (0.00%) |
| KV carrier conversion | 1.60 (0.08%) | 1.46 (0.08%) | 1.61 (0.07%) | 1.48 (0.07%) |
| KV append DMA | 8.60 (0.44%) | 8.56 (0.46%) | 8.27 (0.37%) | 8.86 (0.41%) |
| Block orchestration | 1.11 (0.06%) | 1.09 (0.06%) | 1.12 (0.05%) | 1.13 (0.05%) |
| Layer bookkeeping | 0.69 (0.04%) | 0.68 (0.04%) | 0.69 (0.03%) | 0.68 (0.03%) |
| Stage-boundary bookkeeping | 0.41 (0.02%) | 0.41 (0.02%) | 0.40 (0.02%) | 0.41 (0.02%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 48.95 (2.52%) | 48.77 (2.61%) | 48.84 (2.21%) | 48.86 (2.24%) |
| Embedding | N/A: outside scope | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| Final model RMSNorm | N/A: outside scope | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| LM head＋greedy（不含 final norm） | N/A: outside scope | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| Host–DSP 边界 | 300.73 (15.48%) | 279.85 (14.98%) | 290.05 (13.14%) | 319.68 (14.62%) |
| 完整 Host wall | 1942.87 (100.00%) | 1868.04 (100.00%) | 2207.37 (100.00%) | 2186.07 (100.00%) |

## Complete module table: decode

Microseconds, parentheses: share of complete Host wall. QKV includes R3; Gate/Up+SwiGLU includes R4. Additive accounting, overlapping subcounters are not added again. Frontend is outside this single-layer experiment.

| Module | OFF | R3 | R4 | R3+R4 |
|---|---:|---:|---:|---:|
| I/O、metadata | 41.20 (2.83%) | 38.39 (2.74%) | 39.21 (2.62%) | 43.08 (2.83%) |
| Input RMSNorm | 53.02 (3.64%) | 53.01 (3.78%) | 53.08 (3.54%) | 53.08 (3.48%) |
| QKV＋RoPE | 101.03 (6.94%) | 70.30 (5.01%) | 101.09 (6.74%) | 70.32 (4.61%) |
| QK–Softmax–AV | 359.27 (24.69%) | 358.80 (25.59%) | 358.86 (23.94%) | 358.63 (23.53%) |
| O projection | 56.43 (3.88%) | 56.55 (4.03%) | 57.37 (3.83%) | 56.15 (3.68%) |
| Post-attention residual＋RMSNorm | 53.66 (3.69%) | 53.67 (3.83%) | 53.64 (3.58%) | 53.67 (3.52%) |
| Gate/Up＋SwiGLU | 291.85 (20.06%) | 289.63 (20.65%) | 347.09 (23.15%) | 347.69 (22.81%) |
| Down | 157.56 (10.83%) | 158.30 (11.29%) | 158.71 (10.59%) | 159.72 (10.48%) |
| Final residual | 0.08 (0.01%) | 0.08 (0.01%) | 0.07 (0.00%) | 0.08 (0.01%) |
| KV carrier conversion | 3.13 (0.22%) | 3.17 (0.23%) | 3.14 (0.21%) | 3.18 (0.21%) |
| KV append DMA | 6.12 (0.42%) | 5.91 (0.42%) | 5.96 (0.40%) | 5.95 (0.39%) |
| Block orchestration | 1.07 (0.07%) | 1.03 (0.07%) | 1.13 (0.08%) | 1.11 (0.07%) |
| Layer bookkeeping | 0.65 (0.04%) | 0.66 (0.05%) | 0.64 (0.04%) | 0.66 (0.04%) |
| Stage-boundary bookkeeping | 0.40 (0.03%) | 0.41 (0.03%) | 0.41 (0.03%) | 0.40 (0.03%) |
| DSP unattributed | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| Runtime setup/teardown | 48.62 (3.34%) | 48.68 (3.47%) | 48.72 (3.25%) | 48.66 (3.19%) |
| Embedding | N/A: outside scope | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| Final model RMSNorm | N/A: outside scope | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| LM head＋greedy（不含 final norm） | N/A: outside scope | N/A: outside scope | N/A: outside scope | N/A: outside scope |
| Host–DSP 边界 | 281.12 (19.32%) | 263.73 (18.81%) | 269.98 (18.01%) | 322.01 (21.12%) |
| 完整 Host wall | 1455.23 (100.00%) | 1402.32 (100.00%) | 1499.09 (100.00%) | 1524.38 (100.00%) |

## Remaining direction

In the combination, formal prefill R4 preparation12.39us, HMX submit/wait33.29us, layout77.29us and SP2 finish94.24us. Layout/finish dominate the dense arithmetic. Matrix time alone cannot explain or remove the overhead. Investigate consuming native stage2 tiles directly into SP2 planes and overlapping layout work with conversion workers; avoid the full register transpose already rejected here. Any next candidate needs a newly registered bounded experiment and fresh fixed comparison, not selective repetition of this failed gate.

Current combined prefill mean would need about49us less to reach the mean10% limit, and additional margin for the confidence-bound gate. No guarantee of that gain.

E2E tokens/s: **not measured in L32-0020**; stopped before continuous3/16/frontend at the prefill speed gate. Historical no-rotation FP32/SP2 L32-0018 full-model M64+15:2069.70 prefill tok/s and42.51 decode tok/s, not a rotated result.
