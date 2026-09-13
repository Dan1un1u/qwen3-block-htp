# L32-0010: SP2 Down full-model E2E

Llama 3.2 1B Instruct,16 layers,SM8750/HTP V79. Ten fixed AB/BA pairs,one shared sealed build. M64 prefill plus15 continuous autoregressive decode tokens,capacity80. SP2 mode4 on every Down; per-channel W4/embedding/head/other qparams unchanged.

## prefill

Microseconds; parentheses: percent of complete Host wall. Decode is per generated token.

| Module | Original U8 | SP2 Down |
|---|---:|---:|
| I/O、metadata | 135.7 (0.43%) | 137.8 (0.38%) |
| Input RMSNorm | 1319.7 (4.18%) | 1320.8 (3.69%) |
| QKV＋RoPE | 3845.7 (12.17%) | 3770.2 (10.53%) |
| QK–Softmax–AV | 8101.8 (25.65%) | 8064.0 (22.52%) |
| O projection | 712.4 (2.26%) | 713.6 (1.99%) |
| Post-attention residual＋RMSNorm | 694.5 (2.20%) | 694.2 (1.94%) |
| Gate/Up＋SwiGLU | 7866.5 (24.90%) | 11412.8 (31.88%) |
| Down | 2484.4 (7.86%) | 3567.2 (9.96%) |
| Final residual | 347.1 (1.10%) | 348.8 (0.97%) |
| KV carrier conversion | 25.2 (0.08%) | 25.5 (0.07%) |
| KV append DMA | 104.0 (0.33%) | 104.2 (0.29%) |
| Block orchestration | 21.9 (0.07%) | 21.7 (0.06%) |
| Layer bookkeeping | 12.9 (0.04%) | 12.9 (0.04%) |
| Stage-boundary bookkeeping | 7.1 (0.02%) | 7.2 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 79.8 (0.25%) | 80.3 (0.22%) |
| Embedding | 43.0 (0.14%) | 42.7 (0.12%) |
| Final model RMSNorm | 3.9 (0.01%) | 4.0 (0.01%) |
| LM head＋greedy（不含 final norm） | 3263.8 (10.33%) | 3229.7 (9.02%) |
| Host–DSP 边界 | 2519.6 (7.98%) | 2246.3 (6.27%) |
| 完整 Host wall | 31588.8 (100.00%) | 35803.9 (100.00%) |

## decode

Microseconds; parentheses: percent of complete Host wall. Decode is per generated token.

| Module | Original U8 | SP2 Down |
|---|---:|---:|
| I/O、metadata | 114.1 (0.53%) | 114.3 (0.52%) |
| Input RMSNorm | 84.9 (0.39%) | 84.9 (0.39%) |
| QKV＋RoPE | 1643.3 (7.60%) | 1633.0 (7.49%) |
| QK–Softmax–AV | 5929.0 (27.43%) | 5937.6 (27.24%) |
| O projection | 709.5 (3.28%) | 709.3 (3.25%) |
| Post-attention residual＋RMSNorm | 169.0 (0.78%) | 168.9 (0.77%) |
| Gate/Up＋SwiGLU | 4864.8 (22.51%) | 4877.8 (22.38%) |
| Down | 2472.0 (11.44%) | 2563.8 (11.76%) |
| Final residual | 81.3 (0.38%) | 80.8 (0.37%) |
| KV carrier conversion | 48.4 (0.22%) | 48.3 (0.22%) |
| KV append DMA | 84.8 (0.39%) | 85.0 (0.39%) |
| Block orchestration | 16.7 (0.08%) | 16.7 (0.08%) |
| Layer bookkeeping | 9.3 (0.04%) | 9.2 (0.04%) |
| Stage-boundary bookkeeping | 1.2 (0.01%) | 1.3 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 51.3 (0.24%) | 51.2 (0.23%) |
| Embedding | 1.4 (0.01%) | 1.0 (0.00%) |
| Final model RMSNorm | 2.7 (0.01%) | 2.8 (0.01%) |
| LM head＋greedy（不含 final norm） | 3263.2 (15.10%) | 3226.8 (14.81%) |
| Host–DSP 边界 | 2065.3 (9.56%) | 2181.9 (10.01%) |
| 完整 Host wall | 21612.1 (100.00%) | 21794.5 (100.00%) |

## E2E

| Mode | U8 token/s | SP2 token/s | Latency increase | Throughput decrease | Latency ratio 95% CI | 10% gate |
|---|---:|---:|---:|---:|---|---|
| prefill | 2026.03 | 1787.51 | +13.34% | +11.77% | [1.11694, 1.14594] | FAIL |
| decode | 46.27 | 45.88 | +0.84% | +0.84% | [0.99939, 1.01867] | PASS |

All320 timed full-model boundaries reconcile exactly. Full Host wall includes embedding,16 transformer layers,final norm,LM head,greedy and FastRPC; excludes tokenizer,weight loading and session preparation. One prompt/context shape only; does not establish long-context scaling or PPL/text recovery. Baseline and SP2 use their own independent integer greedy goldens; all16 IDs and selected U8 logit codes agree. Continuous3-layer output and KV exact. SP2 text remains repetitive; no quality promotion.

## Interpretation and evidence

Full-model prefill fails the retained10% gate: +13.3437% latency,95% CI +11.6943% to +14.5942%. Decode passes:+0.8440%,CI -0.0613% to +1.8670%. Thus small single-layer overhead does not establish small full-model prefill overhead. Main additive prefill increase: Gate/Up+SwiGLU +3546.3us and Down +1082.8us; complete Host +4215.1us. These module timings localize the cost but do not alone establish a DMA/HVX stall mechanism. No further optimization or unchanged formal rerun in this experiment.

Correctness: consecutive3-layer M64+1 outputs/KV exact; both16-step generation arms match their independent integer token/code goldens. Twenty formal processes /320 full-model token boundaries all exact and all ledgers additive. Total23 device processes /354 boundaries including functional checks. SP2 all16 partial-dot/Q31 bounds pass.8MiB acquired,max8212960B VTCM;no intermediate tensorDDR/spill or W4 expansion. Native source unchanged from L32-0009; this experiment adds package/oracle/profiling integration only. Weights,tokenizer,head and non-Down qparams unchanged. Original U8 baseline remains default. PPL not run; SP2 text is repetitive and no quality acceptance is implied.

Tested source05383b783dfba81c2985931b8e12782d249226c9. Three/16-layer binaries,seals,commands,oracle,rawstdout/results and packages are archived under L32-0010. One initial local log-redirection failure occurred before build/device; directory creation repaired it. Rebuilding the3-layer configuration for binary archival matched every tested hash exactly;16-layer final resealed before deployment. No failed native runs or numerical retries.

Evidence: `/mnt/d/llm_exp/results/llama32-htp/l32-0010`. Full-model timing is warm token-to-token runtime, not cold-start or external tokenizer latency.
