# L32-0034：更长 KV 的完整模型流水消融

原生 W4/SP2 + FP32 残差，无旋转，保留原 log2 softmax；all-on 对比仅关闭 Gate/Up 提前交错（FFN mask2）。计算核、权重、尺度、HMX 工作量、正常核内双缓冲不变。M64+33，cache128，KV 实际到97。

5 short + 10 formal，交替配对、repeat10 主计时、repeat1 辅助；真实 greedy 全模执行，两臂数值合同相同。完整 Host wall 包含 embedding、所有层、final norm、LM head、greedy、FastRPC；排除 tokenizer、冷加载、ADB。
10200 个计时 profile 全部通过物理检查；配对 bootstrap20000 seed30034。

## prefill

| 模块 | all-on μs（Host占比） | FFN 关闭 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 120.5 (0.41%) | 119.6 (0.38%) |
| Input RMSNorm | 1268.5 (4.31%) | 1267.4 (4.01%) |
| QKV＋RoPE | 3868.5 (13.13%) | 3865.2 (12.21%) |
| QK–Softmax–AV | 8397.1 (28.51%) | 8383.9 (26.49%) |
| O projection | 1580.6 (5.37%) | 1579.4 (4.99%) |
| Post-attention residual＋RMSNorm | 1276.5 (4.33%) | 1275.7 (4.03%) |
| Gate/Up＋SwiGLU | 5516.3 (18.73%) | 7729.9 (24.43%) |
| Down | 3128.9 (10.62%) | 3127.3 (9.88%) |
| Final residual | 1.4 (0.00%) | 1.4 (0.00%) |
| KV carrier conversion | 27.6 (0.09%) | 27.4 (0.09%) |
| KV append DMA | 115.6 (0.39%) | 114.8 (0.36%) |
| Block orchestration | 22.8 (0.08%) | 22.9 (0.07%) |
| Layer bookkeeping | 14.4 (0.05%) | 14.1 (0.04%) |
| Stage-boundary bookkeeping | 6.5 (0.02%) | 6.3 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 90.5 (0.31%) | 90.3 (0.29%) |
| Embedding | 56.5 (0.19%) | 55.5 (0.18%) |
| Final model RMSNorm | 60.2 (0.20%) | 60.2 (0.19%) |
| LM head＋greedy（不含 final norm） | 3216.5 (10.92%) | 3209.7 (10.14%) |
| Host–DSP 边界 | 683.1 (2.32%) | 692.9 (2.19%) |
| 完整 Host wall | 29451.9 (100.00%) | 31643.8 (100.00%) |

## decode

| 模块 | all-on μs（Host占比） | FFN 关闭 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 116.9 (0.52%) | 116.4 (0.51%) |
| Input RMSNorm | 871.0 (3.90%) | 870.7 (3.82%) |
| QKV＋RoPE | 1669.5 (7.48%) | 1668.3 (7.31%) |
| QK–Softmax–AV | 6359.9 (28.48%) | 6358.0 (27.87%) |
| O projection | 880.6 (3.94%) | 886.7 (3.89%) |
| Post-attention residual＋RMSNorm | 875.0 (3.92%) | 875.0 (3.84%) |
| Gate/Up＋SwiGLU | 4900.6 (21.95%) | 5376.2 (23.57%) |
| Down | 2633.7 (11.80%) | 2639.9 (11.57%) |
| Final residual | 1.0 (0.00%) | 1.0 (0.00%) |
| KV carrier conversion | 48.4 (0.22%) | 48.5 (0.21%) |
| KV append DMA | 90.8 (0.41%) | 90.3 (0.40%) |
| Block orchestration | 16.2 (0.07%) | 16.2 (0.07%) |
| Layer bookkeeping | 9.3 (0.04%) | 9.3 (0.04%) |
| Stage-boundary bookkeeping | 1.2 (0.01%) | 1.2 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 50.8 (0.23%) | 50.9 (0.22%) |
| Embedding | 1.8 (0.01%) | 1.8 (0.01%) |
| Final model RMSNorm | 55.7 (0.25%) | 55.7 (0.24%) |
| LM head＋greedy（不含 final norm） | 3209.2 (14.37%) | 3200.8 (14.03%) |
| Host–DSP 边界 | 535.5 (2.40%) | 544.0 (2.38%) |
| 完整 Host wall | 22327.4 (100.00%) | 22810.8 (100.00%) |

## 配对效应

| 范围 | FFN关闭 / all-on 墙钟 | 95%CI |
|---|---:|---|
| prefill | 1.0744214 | [1.0723032, 1.0764817] |
| decode | 1.0216507 | [1.0204680, 1.0228252] |
| decode_KV65_96 | 1.0217601 | [1.0205503, 1.0229683] |
| decode_KV97 | 1.0182668 | [1.0164580, 1.0200280] |

KV97 分段每条轨迹仅末尾一个 token，仅作边界诊断；不能把不同 token 的差异全部归因于长度。跨实验 M64+15 与本实验绝对速度不作配对加速分母。
independent full16 CPU greedy selected token/logit code exact, first16 matches old frozen model
无模型质量验收，不晋升基线。

## 端到端 token/s

| 配置 | Prefill | Decode |
|---|---:|---:|
| ALL | 2173.0344 | 44.7880 |
| FFN | 2022.5160 | 43.8389 |
