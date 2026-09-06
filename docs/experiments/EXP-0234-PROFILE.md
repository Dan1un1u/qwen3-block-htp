# EXP0234 complete profiling record: host-only software diagnostic

Source branch codex/exp-0234-w4f16-group128-64k; closure commit c8e57fd0d308df10f0efa82cbbe0f96c49ed2839.
Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0234; artifacts /mnt/d/llm_exp/models/qwen3-block-htp/exp0234. Controls C64 forgrouping andG8 forcalibrationbudget; candidateG64; floatingF reference. Quality execution unit is a packed-weight FP16 software M64+16 teacher-forcing forward. No target DSP runtime for this format exists in this experiment. Paired profiling rounds, repeat-one and repeat-ten: N/A.

|Required section|Repeat1 control/candidate/delta|Repeat10 control/candidate/delta|Reason|
|---|---|---|---|
|Complete Host wall / prefill / continuous decode|N/A|N/A|No grouped device execution|
|DSP invocation / setup / teardown / ledger / unattributed|N/A|N/A|No DSP timestamp recording|
|All additive block stages|N/A|N/A|No grouped runtime|
|Projection DMA / HMX / unpack / waits / lifetime / workers|N/A|N/A|No grouped runtime|
|Attention QK / Softmax / AV / packing / waits / tasks|N/A|N/A|No grouped runtime|
|MLP GateUp / SwiGLU / Down / slots / staging / publication|N/A|N/A|No grouped runtime|
|DDR bytes / DMA descriptors / spill / HMX / VTCM / FastRPC|N/A|N/A|No device use or physical performance claim|
|Device output hashes / mismatch / maximum LSB|N/A|N/A|Software implementation oracles documented in REPORT.md|

Engine/work counters can overlap and would not be additive; none are measured here. Software repeat/causal/CE checks, dense quantizer oracle, inherited tensor identity,196pack roundtrips,112forward checks and finite values pass. Quality scores and confidence intervals are in REPORT.md. N/A never means measured zero.

## Stable historical three-recipe overview

The following M64 table uses microseconds and percent of complete Host wall. It is unchanged EXP230 evidence, **not a G64 profile**. Its W4F16 C64 column is an experimental quality candidate, not a promoted baseline; F16F16/W4U8 are nonpaired EXP218 references. W4bytes differ, so no activation-only attribution. Full repeat1/repeat10 counters and provenance: /mnt/d/llm_exp/results/qwen3-block-htp/exp0230/full_profiling_report.md, SHA256 bf8347a03109d9f816739de6e95d27c322c4740dbfd5f189b6157698c54184c7.

| 模块 | F16A16 历史 EXP0218 | W4A16 C64 EXP0230 | W4A8 历史 EXP0218 | W4A8 相对 W4A16 增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 385.3 (0.61%) | 247.4 (0.63%) | +55.76% |
| Input RMSNorm | 489.7 (0.61%) | 492.0 (0.78%) | 554.0 (1.40%) | -11.19% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11755.2 (18.58%) | 7052.7 (17.82%) | +66.68% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3969.7 (6.27%) | 3214.5 (8.12%) | +23.49% |
| O projection | 5757.7 (7.14%) | 5060.0 (8.00%) | 1256.4 (3.17%) | +302.74% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 473.4 (0.75%) | 654.0 (1.65%) | -27.62% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22435.0 (35.46%) | 14442.7 (36.49%) | +55.34% |
| Down | 13447.9 (16.67%) | 8679.0 (13.72%) | 3428.5 (8.66%) | +153.15% |
| Final residual | 140.1 (0.17%) | 140.2 (0.22%) | 183.8 (0.46%) | -23.73% |
| KV carrier conversion | 174.0 (0.22%) | 172.1 (0.27%) | 203.2 (0.51%) | -15.33% |
| KV append DMA | 343.6 (0.43%) | 343.3 (0.54%) | 463.9 (1.17%) | -25.99% |
| Block orchestration | 16.1 (0.02%) | 19.1 (0.03%) | 34.6 (0.09%) | -44.80% |
| Layer bookkeeping | 23.9 (0.03%) | 24.8 (0.04%) | 23.2 (0.06%) | +7.31% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 9.2 (0.01%) | 22.5 (0.06%) | -59.03% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 94.5 (0.15%) | 105.6 (0.27%) | -10.53% |
| Embedding | 68.1 (0.08%) | 63.6 (0.10%) | 62.4 (0.16%) | +1.79% |
| Final model RMSNorm | 49.7 (0.06%) | 48.0 (0.08%) | 3.7 (0.01%) | +1188.11% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6665.9 (10.54%) | 5284.7 (13.35%) | +26.14% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2440.7 (3.86%) | 2466.6 (6.23%) | -1.05% |
| 完整 Host wall | 80692.2 (100.00%) | 63263.0 (100.00%) | 39575.9 (100.00%) | +59.85% |


## End-to-end speed

G64 prefill/decode: N/A (no grouped device implementation). Historical EXP230 C64:64prompt tokens /63262.995us =1011.649859tok/s;15continuous-decode tokens /1389448.9595us =10.795647tok/s. These are prior per-channel measurements only. No extrapolation from software evaluation time.
