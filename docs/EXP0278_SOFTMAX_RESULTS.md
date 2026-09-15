# EXP-0278：现有 log2 / FP32 HVX softmax 全模型消融

冻结 W4/SP2、FP32 残差、所有非 softmax 权重与尺度，无旋转。LOG2 保留 Qwen wide NR64，在 raw QK 的宽差值上离散化指数；FP 从 raw HMX QK 的原尺度计算连续 max/exp/sum/normalize，最终统一输出 U8 概率，保留整数 AV。这是 softmax 路径对照，不是全浮点 attention / FlashAttention。

两臂均采用正常向量实现及冻结流水。每条 greedy / fixed 轨迹各 5 short + 10 formal，交替配对、repeat10 主计时、repeat1 辅助；fixed 保持输入 token，KV 值由各臂自己的计算产生。全模型包括 embedding、所有层、final norm、LM head、greedy 与 FastRPC；排除 tokenizer、冷加载及 ADB。
配对 bootstrap 20000，seed278；9600 个计时 profile 全部保留且通过物理审计。

## greedy · prefill

| 模块 | LOG2 μs（Host占比） | FP32 HVX μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 251.9 (0.73%) | 250.4 (0.71%) |
| Input RMSNorm | 2019.8 (5.84%) | 2088.0 (5.95%) |
| QKV＋Q/K Norm-RoPE | 7046.0 (20.38%) | 7053.7 (20.09%) |
| QK–Softmax–AV | 3492.1 (10.10%) | 3855.2 (10.98%) |
| O projection | 2084.0 (6.03%) | 2081.0 (5.93%) |
| Post-attention residual＋RMSNorm | 2115.6 (6.12%) | 2129.4 (6.06%) |
| Gate/Up＋SwiGLU | 7035.6 (20.35%) | 7048.9 (20.07%) |
| Down | 3776.6 (10.92%) | 3794.6 (10.81%) |
| Final residual | 2.8 (0.01%) | 2.8 (0.01%) |
| KV carrier conversion | 137.1 (0.40%) | 136.6 (0.39%) |
| KV append DMA | 294.1 (0.85%) | 294.2 (0.84%) |
| Block orchestration | 35.3 (0.10%) | 35.8 (0.10%) |
| Layer bookkeeping | 27.6 (0.08%) | 27.4 (0.08%) |
| Stage-boundary bookkeeping | 21.2 (0.06%) | 21.7 (0.06%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 115.4 (0.33%) | 118.8 (0.34%) |
| Embedding | 55.4 (0.16%) | 55.4 (0.16%) |
| Final model RMSNorm | 13.5 (0.04%) | 13.6 (0.04%) |
| LM head＋greedy，不含 final norm | 5189.4 (15.01%) | 5193.2 (14.79%) |
| Host–DSP 边界 | 865.8 (2.50%) | 913.8 (2.60%) |
| 完整 Host wall | 34579.1 (100.00%) | 35114.5 (100.00%) |

FP/LOG2 Host wall = 1.0154843，95%CI [1.0134265, 1.0176821]。

## greedy · decode

| 模块 | LOG2 μs（Host占比） | FP32 HVX μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 248.7 (1.17%) | 249.0 (1.18%) |
| Input RMSNorm | 369.0 (1.74%) | 365.4 (1.73%) |
| QKV＋Q/K Norm-RoPE | 2500.4 (11.79%) | 2499.9 (11.81%) |
| QK–Softmax–AV | 1928.9 (9.09%) | 1881.6 (8.89%) |
| O projection | 1346.1 (6.35%) | 1344.2 (6.35%) |
| Post-attention residual＋RMSNorm | 369.2 (1.74%) | 371.7 (1.76%) |
| Gate/Up＋SwiGLU | 6406.2 (30.20%) | 6407.1 (30.26%) |
| Down | 3491.6 (16.46%) | 3494.0 (16.50%) |
| Final residual | 1.9 (0.01%) | 1.9 (0.01%) |
| KV carrier conversion | 291.9 (1.38%) | 291.9 (1.38%) |
| KV append DMA | 122.1 (0.58%) | 122.0 (0.58%) |
| Block orchestration | 27.9 (0.13%) | 27.9 (0.13%) |
| Layer bookkeeping | 16.6 (0.08%) | 16.6 (0.08%) |
| Stage-boundary bookkeeping | 1.7 (0.01%) | 1.7 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 73.7 (0.35%) | 73.7 (0.35%) |
| Embedding | 1.8 (0.01%) | 1.9 (0.01%) |
| Final model RMSNorm | 15.4 (0.07%) | 15.5 (0.07%) |
| LM head＋greedy，不含 final norm | 3227.4 (15.21%) | 3227.9 (15.25%) |
| Host–DSP 边界 | 773.6 (3.65%) | 777.1 (3.67%) |
| 完整 Host wall | 21214.1 (100.00%) | 21171.2 (100.00%) |

FP/LOG2 Host wall = 0.9979797，95%CI [0.9965539, 0.9992703]。

## fixed · prefill

| 模块 | LOG2 μs（Host占比） | FP32 HVX μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 252.4 (0.73%) | 252.9 (0.72%) |
| Input RMSNorm | 2019.7 (5.82%) | 2088.1 (5.93%) |
| QKV＋Q/K Norm-RoPE | 7052.6 (20.34%) | 7051.1 (20.04%) |
| QK–Softmax–AV | 3493.1 (10.07%) | 3857.2 (10.96%) |
| O projection | 2084.7 (6.01%) | 2086.9 (5.93%) |
| Post-attention residual＋RMSNorm | 2115.7 (6.10%) | 2129.8 (6.05%) |
| Gate/Up＋SwiGLU | 7054.0 (20.34%) | 7060.7 (20.06%) |
| Down | 3778.3 (10.90%) | 3796.2 (10.79%) |
| Final residual | 2.9 (0.01%) | 2.9 (0.01%) |
| KV carrier conversion | 136.5 (0.39%) | 135.7 (0.39%) |
| KV append DMA | 295.1 (0.85%) | 294.9 (0.84%) |
| Block orchestration | 35.5 (0.10%) | 35.5 (0.10%) |
| Layer bookkeeping | 27.7 (0.08%) | 27.7 (0.08%) |
| Stage-boundary bookkeeping | 21.2 (0.06%) | 21.3 (0.06%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 116.6 (0.34%) | 116.6 (0.33%) |
| Embedding | 55.5 (0.16%) | 55.4 (0.16%) |
| Final model RMSNorm | 13.5 (0.04%) | 13.6 (0.04%) |
| LM head＋greedy，不含 final norm | 5197.8 (14.99%) | 5201.9 (14.78%) |
| Host–DSP 边界 | 922.5 (2.66%) | 960.9 (2.73%) |
| 完整 Host wall | 34675.6 (100.00%) | 35189.2 (100.00%) |

FP/LOG2 Host wall = 1.0148099，95%CI [1.0117180, 1.0177549]。

## fixed · decode

| 模块 | LOG2 μs（Host占比） | FP32 HVX μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 249.8 (1.17%) | 249.9 (1.18%) |
| Input RMSNorm | 369.1 (1.73%) | 368.1 (1.73%) |
| QKV＋Q/K Norm-RoPE | 2506.9 (11.78%) | 2507.3 (11.81%) |
| QK–Softmax–AV | 1931.2 (9.08%) | 1882.5 (8.87%) |
| O projection | 1347.6 (6.33%) | 1349.3 (6.36%) |
| Post-attention residual＋RMSNorm | 369.3 (1.74%) | 370.2 (1.74%) |
| Gate/Up＋SwiGLU | 6426.0 (30.21%) | 6434.4 (30.31%) |
| Down | 3499.4 (16.45%) | 3500.8 (16.49%) |
| Final residual | 1.9 (0.01%) | 1.9 (0.01%) |
| KV carrier conversion | 292.0 (1.37%) | 292.0 (1.38%) |
| KV append DMA | 122.4 (0.58%) | 122.3 (0.58%) |
| Block orchestration | 27.8 (0.13%) | 27.9 (0.13%) |
| Layer bookkeeping | 16.6 (0.08%) | 16.6 (0.08%) |
| Stage-boundary bookkeeping | 1.7 (0.01%) | 1.7 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 73.7 (0.35%) | 73.7 (0.35%) |
| Embedding | 1.8 (0.01%) | 1.8 (0.01%) |
| Final model RMSNorm | 15.3 (0.07%) | 15.9 (0.07%) |
| LM head＋greedy，不含 final norm | 3234.2 (15.20%) | 3233.6 (15.23%) |
| Host–DSP 边界 | 786.4 (3.70%) | 780.5 (3.68%) |
| 完整 Host wall | 21272.9 (100.00%) | 21230.2 (100.00%) |

FP/LOG2 Host wall = 0.9979927，95%CI [0.9957243, 1.0004015]。

## 实际端到端吞吐

| 轨迹 | 配置 | Prefill token/s | Decode token/s |
|---|---|---:|---:|
| greedy | LOG2 | 1850.8303 | 47.1385 |
| greedy | FP | 1822.6086 | 47.2339 |
| fixed | LOG2 | 1845.6776 | 47.0081 |
| fixed | FP | 1818.7423 | 47.1026 |

M64+15，28 层。fixed 行为受控输入重放，greedy 为实际自由生成路径，两者不混合。
Qwen selected layer14 and chain3 independent exact; full28 physical/front-end finalNorm and full-vocabulary head oracle; no full28 CPU transformer equivalence claim
相同 HMX 工作量与权重 DDR 字节，无中间 tensor DDR/spill，8 MiB VTCM。FP32 exp 与归约使用供应商 QHL HVX；概率码行和实现见下方 Implementation；组件输出门槛独立记录。未做模型质量/PPL 验收，不晋升基线。A8 额外支持 shape 仍待执行。

Implementation: FP vector probability and vector rowmass (primary)
