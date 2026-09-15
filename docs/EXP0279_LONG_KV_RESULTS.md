# EXP-0279：更长 KV 的完整模型流水消融

原生 W4/SP2 + FP32 残差，无旋转，保留原 log2 softmax；all-on 对比仅关闭 Gate/Up 提前交错（FFN mask2）。计算核、权重、尺度、HMX 工作量、正常核内双缓冲不变。M64+33，cache128，KV 实际到97。

5 short + 10 formal，交替配对、repeat10 主计时、repeat1 辅助；真实 greedy 全模执行，两臂数值合同相同。完整 Host wall 包含 embedding、所有层、final norm、LM head、greedy、FastRPC；排除 tokenizer、冷加载、ADB。
10200 个计时 profile 全部通过物理检查；配对 bootstrap20000 seed279。

## prefill

| 模块 | all-on μs（Host占比） | FFN 关闭 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 241.4 (0.70%) | 239.4 (0.65%) |
| Input RMSNorm | 2020.8 (5.84%) | 2020.1 (5.45%) |
| QKV＋Q/K Norm-RoPE | 7048.6 (20.38%) | 7051.8 (19.03%) |
| QK–Softmax–AV | 3489.5 (10.09%) | 3490.4 (9.42%) |
| O projection | 2083.0 (6.02%) | 2086.6 (5.63%) |
| Post-attention residual＋RMSNorm | 2114.6 (6.11%) | 2114.2 (5.70%) |
| Gate/Up＋SwiGLU | 7119.7 (20.59%) | 9543.9 (25.75%) |
| Down | 3813.8 (11.03%) | 3825.1 (10.32%) |
| Final residual | 3.0 (0.01%) | 3.1 (0.01%) |
| KV carrier conversion | 135.2 (0.39%) | 136.2 (0.37%) |
| KV append DMA | 293.6 (0.85%) | 291.3 (0.79%) |
| Block orchestration | 35.7 (0.10%) | 35.9 (0.10%) |
| Layer bookkeeping | 28.7 (0.08%) | 28.1 (0.08%) |
| Stage-boundary bookkeeping | 21.3 (0.06%) | 21.3 (0.06%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 118.8 (0.34%) | 119.3 (0.32%) |
| Embedding | 55.8 (0.16%) | 55.5 (0.15%) |
| Final model RMSNorm | 13.5 (0.04%) | 13.4 (0.04%) |
| LM head＋greedy，不含 final norm | 5211.0 (15.07%) | 5205.7 (14.04%) |
| Host–DSP 边界 | 732.8 (2.12%) | 783.3 (2.11%) |
| 完整 Host wall | 34580.9 (100.00%) | 37064.5 (100.00%) |

## decode

| 模块 | all-on μs（Host占比） | FFN 关闭 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 236.1 (1.11%) | 235.8 (1.08%) |
| Input RMSNorm | 366.2 (1.73%) | 365.7 (1.67%) |
| QKV＋Q/K Norm-RoPE | 2510.6 (11.83%) | 2517.7 (11.53%) |
| QK–Softmax–AV | 1951.4 (9.20%) | 1961.8 (8.98%) |
| O projection | 1347.2 (6.35%) | 1353.3 (6.20%) |
| Post-attention residual＋RMSNorm | 369.6 (1.74%) | 369.7 (1.69%) |
| Gate/Up＋SwiGLU | 6463.8 (30.47%) | 7004.6 (32.07%) |
| Down | 3520.2 (16.59%) | 3541.7 (16.22%) |
| Final residual | 1.9 (0.01%) | 1.9 (0.01%) |
| KV carrier conversion | 314.2 (1.48%) | 314.2 (1.44%) |
| KV append DMA | 132.3 (0.62%) | 133.0 (0.61%) |
| Block orchestration | 27.8 (0.13%) | 27.7 (0.13%) |
| Layer bookkeeping | 16.6 (0.08%) | 16.3 (0.07%) |
| Stage-boundary bookkeeping | 1.7 (0.01%) | 1.7 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 73.5 (0.35%) | 73.5 (0.34%) |
| Embedding | 1.7 (0.01%) | 1.6 (0.01%) |
| Final model RMSNorm | 15.3 (0.07%) | 14.7 (0.07%) |
| LM head＋greedy，不含 final norm | 3252.0 (15.33%) | 3259.7 (14.92%) |
| Host–DSP 边界 | 613.4 (2.89%) | 646.7 (2.96%) |
| 完整 Host wall | 21215.2 (100.00%) | 21841.2 (100.00%) |

## 配对效应

| 范围 | FFN关闭 / all-on 墙钟 | 95%CI |
|---|---:|---|
| prefill | 1.0718219 | [1.0699142, 1.0743945] |
| decode | 1.0295060 | [1.0272191, 1.0322866] |
| decode_KV65_96 | 1.0295348 | [1.0271863, 1.0323482] |
| decode_KV97 | 1.0285813 | [1.0256988, 1.0315150] |

KV97 分段每条轨迹仅末尾一个 token，仅作边界诊断；不能把不同 token 的差异全部归因于长度。跨实验 M64+15 与本实验绝对速度不作配对加速分母。
same frozen LOG2 DSP arithmetic; own selected/chain3 gates inherited from EXP0278; full28 paired outputs/hidden bytes exact and all34 actual finalNorm/full-vocabulary head audited; no full28 CPU transformer equivalence claim
无模型质量验收，不晋升基线。

## 端到端 token/s

| 配置 | Prefill | Decode |
|---|---:|---:|
| ALL | 1850.7348 | 47.1360 |
| FFN | 1726.7186 | 45.7851 |
