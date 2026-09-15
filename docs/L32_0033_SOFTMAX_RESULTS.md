# L32-0033：现有 log2 / FP32 HVX softmax 全模型消融

冻结 W4/SP2、FP32 残差、所有非 softmax 权重与尺度，无旋转。LOG2 保留各层原有除法及 score 重量化；FP 从 raw HMX QK 的原尺度计算连续 max/exp/sum/normalize，最终统一输出 U8 概率，保留整数 AV。这是 softmax 路径对照，不是全浮点 attention / FlashAttention。

两臂均采用正常向量实现及冻结流水。每条 greedy / fixed 轨迹各 5 short + 10 formal，交替配对、repeat10 主计时、repeat1 辅助；fixed 保持输入 token，KV 值由各臂自己的计算产生。全模型包括 embedding、所有层、final norm、LM head、greedy 与 FastRPC；排除 tokenizer、冷加载及 ADB。
配对 bootstrap 20000，seed30033；9600 个计时 profile 全部保留且通过物理审计。

## greedy · prefill

| 模块 | LOG2 μs（Host占比） | FP32 HVX μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 130.0 (0.44%) | 129.1 (0.46%) |
| Input RMSNorm | 1268.7 (4.29%) | 1290.3 (4.60%) |
| QKV＋RoPE | 3865.4 (13.08%) | 3871.2 (13.81%) |
| QK–Softmax–AV | 8390.3 (28.39%) | 6873.8 (24.53%) |
| O projection | 1580.9 (5.35%) | 1582.0 (5.65%) |
| Post-attention residual＋RMSNorm | 1276.1 (4.32%) | 1274.3 (4.55%) |
| Gate/Up＋SwiGLU | 5485.2 (18.56%) | 5477.1 (19.54%) |
| Down | 3131.1 (10.59%) | 3127.8 (11.16%) |
| Final residual | 1.6 (0.01%) | 1.7 (0.01%) |
| KV carrier conversion | 27.6 (0.09%) | 27.7 (0.10%) |
| KV append DMA | 107.9 (0.37%) | 107.9 (0.39%) |
| Block orchestration | 22.9 (0.08%) | 22.7 (0.08%) |
| Layer bookkeeping | 14.3 (0.05%) | 13.9 (0.05%) |
| Stage-boundary bookkeeping | 6.8 (0.02%) | 6.7 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 90.0 (0.30%) | 87.8 (0.31%) |
| Embedding | 55.7 (0.19%) | 55.1 (0.20%) |
| Final model RMSNorm | 60.3 (0.20%) | 55.4 (0.20%) |
| LM head＋greedy（不含 final norm） | 3207.2 (10.85%) | 3201.6 (11.42%) |
| Host–DSP 边界 | 831.0 (2.81%) | 817.2 (2.92%) |
| 完整 Host wall | 29553.0 (100.00%) | 28023.4 (100.00%) |

FP/LOG2 Host wall = 0.9482431，95%CI [0.9462983, 0.9501916]。

## greedy · decode

| 模块 | LOG2 μs（Host占比） | FP32 HVX μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 124.8 (0.56%) | 124.5 (0.57%) |
| Input RMSNorm | 871.0 (3.93%) | 871.9 (3.98%) |
| QKV＋RoPE | 1667.1 (7.53%) | 1669.2 (7.62%) |
| QK–Softmax–AV | 5996.4 (27.07%) | 5760.3 (26.30%) |
| O projection | 883.6 (3.99%) | 882.9 (4.03%) |
| Post-attention residual＋RMSNorm | 874.5 (3.95%) | 874.8 (3.99%) |
| Gate/Up＋SwiGLU | 4865.7 (21.97%) | 4874.5 (22.25%) |
| Down | 2610.4 (11.79%) | 2615.8 (11.94%) |
| Final residual | 1.0 (0.00%) | 1.0 (0.00%) |
| KV carrier conversion | 48.4 (0.22%) | 48.5 (0.22%) |
| KV append DMA | 91.2 (0.41%) | 91.2 (0.42%) |
| Block orchestration | 16.3 (0.07%) | 16.2 (0.07%) |
| Layer bookkeeping | 9.3 (0.04%) | 9.3 (0.04%) |
| Stage-boundary bookkeeping | 1.2 (0.01%) | 1.2 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 50.9 (0.23%) | 50.9 (0.23%) |
| Embedding | 1.5 (0.01%) | 1.6 (0.01%) |
| Final model RMSNorm | 55.3 (0.25%) | 55.8 (0.25%) |
| LM head＋greedy（不含 final norm） | 3198.6 (14.44%) | 3195.6 (14.59%) |
| Host–DSP 边界 | 781.9 (3.53%) | 760.7 (3.47%) |
| 完整 Host wall | 22149.1 (100.00%) | 21906.0 (100.00%) |

FP/LOG2 Host wall = 0.9890238，95%CI [0.9874232, 0.9909410]。

## fixed · prefill

| 模块 | LOG2 μs（Host占比） | FP32 HVX μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 128.8 (0.44%) | 128.4 (0.46%) |
| Input RMSNorm | 1268.1 (4.29%) | 1290.5 (4.60%) |
| QKV＋RoPE | 3867.9 (13.09%) | 3873.0 (13.80%) |
| QK–Softmax–AV | 8388.1 (28.39%) | 6863.6 (24.46%) |
| O projection | 1580.8 (5.35%) | 1580.7 (5.63%) |
| Post-attention residual＋RMSNorm | 1276.2 (4.32%) | 1274.4 (4.54%) |
| Gate/Up＋SwiGLU | 5499.7 (18.61%) | 5493.6 (19.58%) |
| Down | 3127.9 (10.59%) | 3126.7 (11.14%) |
| Final residual | 1.7 (0.01%) | 1.6 (0.01%) |
| KV carrier conversion | 27.5 (0.09%) | 27.9 (0.10%) |
| KV append DMA | 107.8 (0.36%) | 107.7 (0.38%) |
| Block orchestration | 22.9 (0.08%) | 23.4 (0.08%) |
| Layer bookkeeping | 14.1 (0.05%) | 14.2 (0.05%) |
| Stage-boundary bookkeeping | 6.8 (0.02%) | 7.0 (0.03%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 89.6 (0.30%) | 90.7 (0.32%) |
| Embedding | 55.2 (0.19%) | 55.0 (0.20%) |
| Final model RMSNorm | 60.3 (0.20%) | 55.4 (0.20%) |
| LM head＋greedy（不含 final norm） | 3215.9 (10.88%) | 3206.1 (11.43%) |
| Host–DSP 边界 | 805.4 (2.73%) | 837.4 (2.98%) |
| 完整 Host wall | 29544.7 (100.00%) | 28057.1 (100.00%) |

FP/LOG2 Host wall = 0.9496487，95%CI [0.9479124, 0.9516427]。

## fixed · decode

| 模块 | LOG2 μs（Host占比） | FP32 HVX μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 124.0 (0.56%) | 124.1 (0.57%) |
| Input RMSNorm | 870.9 (3.93%) | 871.0 (3.98%) |
| QKV＋RoPE | 1669.8 (7.53%) | 1667.8 (7.61%) |
| QK–Softmax–AV | 5992.8 (27.04%) | 5752.6 (26.26%) |
| O projection | 886.4 (4.00%) | 880.0 (4.02%) |
| Post-attention residual＋RMSNorm | 874.5 (3.95%) | 878.5 (4.01%) |
| Gate/Up＋SwiGLU | 4879.0 (22.01%) | 4883.3 (22.29%) |
| Down | 2616.3 (11.81%) | 2616.1 (11.94%) |
| Final residual | 1.0 (0.00%) | 1.0 (0.00%) |
| KV carrier conversion | 48.4 (0.22%) | 48.5 (0.22%) |
| KV append DMA | 90.9 (0.41%) | 90.7 (0.41%) |
| Block orchestration | 16.3 (0.07%) | 16.3 (0.07%) |
| Layer bookkeeping | 9.2 (0.04%) | 9.3 (0.04%) |
| Stage-boundary bookkeeping | 1.2 (0.01%) | 1.2 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 51.0 (0.23%) | 50.9 (0.23%) |
| Embedding | 1.5 (0.01%) | 1.5 (0.01%) |
| Final model RMSNorm | 55.3 (0.25%) | 55.9 (0.26%) |
| LM head＋greedy（不含 final norm） | 3208.0 (14.48%) | 3207.8 (14.64%) |
| Host–DSP 边界 | 765.7 (3.46%) | 751.9 (3.43%) |
| 完整 Host wall | 22162.4 (100.00%) | 21908.3 (100.00%) |

FP/LOG2 Host wall = 0.9885331，95%CI [0.9871226, 0.9899538]。

## 实际端到端吞吐

| 轨迹 | 配置 | Prefill token/s | Decode token/s |
|---|---|---:|---:|
| greedy | LOG2 | 2165.6036 | 45.1485 |
| greedy | FP | 2283.8063 | 45.6496 |
| fixed | LOG2 | 2166.2095 | 45.1214 |
| fixed | FP | 2281.0641 | 45.6448 |

M64+15，16 层。fixed 行为受控输入重放，greedy 为实际自由生成路径，两者不混合。
Llama independent selected layer7, chain3, chain16 two-step exact; full16 all-layer greedy/fixed CPU teacher selected token/logit codes exact
相同 HMX 工作量与权重 DDR 字节，无中间 tensor DDR/spill，8 MiB VTCM。FP32 向量 exp/归约/概率码行和使用供应商 QHL HVX；组件输出门槛独立记录。未做模型质量/PPL 验收，不晋升基线。A8 额外支持 shape 仍待执行。
