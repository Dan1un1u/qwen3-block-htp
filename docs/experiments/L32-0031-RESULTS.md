# L32-0031：SP2 decode 共装与两独立流

固定原生W4/SP2mode8、FP32残差1、无旋转，权重码/尺度、LUT和所有非目标优化不变。SPLIT由SwiGLU直接输出独立high平面，Down对同一已搬入权重执行两次矩阵累积；ALL用空闲物理行共装一次矩阵流。Prefill合同不变。

全模5short+10formal，交替顺序，repeat10主计时，repeat1辅助。95%CI按十个配对轮次bootstrap20000seed30031，保留全部样本。

## 完整16层 prefill 模块表

| 模块 | 共装 μs（Host占比） | 两独立流 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 130.2 (0.44%) | 130.7 (0.44%) |
| Input RMSNorm | 1267.8 (4.29%) | 1268.0 (4.28%) |
| QKV＋RoPE | 3865.2 (13.07%) | 3863.9 (13.05%) |
| QK–Softmax–AV | 8410.1 (28.44%) | 8416.1 (28.43%) |
| O projection | 1580.1 (5.34%) | 1579.0 (5.33%) |
| Post-attention residual＋RMSNorm | 1276.3 (4.32%) | 1276.3 (4.31%) |
| Gate/Up＋SwiGLU | 5486.7 (18.56%) | 5480.5 (18.52%) |
| Down | 3126.5 (10.57%) | 3132.5 (10.58%) |
| Final residual | 2.0 (0.01%) | 1.9 (0.01%) |
| KV carrier conversion | 26.1 (0.09%) | 26.1 (0.09%) |
| KV append DMA | 108.3 (0.37%) | 108.2 (0.37%) |
| Block orchestration | 22.5 (0.08%) | 22.5 (0.08%) |
| Layer bookkeeping | 13.5 (0.05%) | 13.6 (0.05%) |
| Stage-boundary bookkeeping | 6.7 (0.02%) | 6.8 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 89.9 (0.30%) | 90.5 (0.31%) |
| Embedding | 57.7 (0.19%) | 58.3 (0.20%) |
| Final model RMSNorm | 60.8 (0.21%) | 60.7 (0.20%) |
| LM head＋greedy（不含 final norm） | 3192.0 (10.80%) | 3192.6 (10.79%) |
| Host–DSP 边界 | 844.1 (2.85%) | 869.8 (2.94%) |
| 完整 Host wall | 29566.4 (100.00%) | 29597.9 (100.00%) |

两流/共装 Host wall：1.001063，95%CI [0.998009, 1.003802]。

## 完整16层 decode 模块表

| 模块 | 共装 μs（Host占比） | 两独立流 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 125.2 (0.57%) | 126.0 (0.56%) |
| Input RMSNorm | 871.2 (3.94%) | 871.3 (3.89%) |
| QKV＋RoPE | 1669.0 (7.55%) | 1669.2 (7.46%) |
| QK–Softmax–AV | 5976.7 (27.02%) | 5978.9 (26.72%) |
| O projection | 885.1 (4.00%) | 883.8 (3.95%) |
| Post-attention residual＋RMSNorm | 874.5 (3.95%) | 874.5 (3.91%) |
| Gate/Up＋SwiGLU | 4865.7 (22.00%) | 4866.7 (21.75%) |
| Down | 2608.1 (11.79%) | 2875.1 (12.85%) |
| Final residual | 1.1 (0.01%) | 1.1 (0.00%) |
| KV carrier conversion | 48.4 (0.22%) | 48.4 (0.22%) |
| KV append DMA | 92.0 (0.42%) | 91.9 (0.41%) |
| Block orchestration | 16.0 (0.07%) | 16.1 (0.07%) |
| Layer bookkeeping | 9.3 (0.04%) | 9.3 (0.04%) |
| Stage-boundary bookkeeping | 1.2 (0.01%) | 1.2 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 51.2 (0.23%) | 51.3 (0.23%) |
| Embedding | 1.5 (0.01%) | 1.5 (0.01%) |
| Final model RMSNorm | 55.4 (0.25%) | 55.4 (0.25%) |
| LM head＋greedy（不含 final norm） | 3183.1 (14.39%) | 3186.3 (14.24%) |
| Host–DSP 边界 | 781.1 (3.53%) | 765.6 (3.42%) |
| 完整 Host wall | 22115.9 (100.00%) | 22373.7 (100.00%) |

两流/共装 Host wall：1.011657，95%CI [1.007580, 1.016246]。

## 完整E2E

| 配置 | Prefill token/s | Decode token/s |
|---|---:|---:|
| ALL | 2164.618 | 45.216 |
| SPLIT | 2162.319 | 44.695 |

M64+15、cache80、原始提示词。包含embedding/16层/finalnorm/head/greedy/FastRPC，排除tokenizer/冷加载/ADB。HMXcommand计数是worker提交次数；真正额外工作看tilepair计数，decode每层增加16384，16层增加262144。权重搬运不增加。结果只解释固定SP2合同的执行方式，不是SP2相对普通A8的成本（A5另测），也不代表模型质量。

## 结论边界

两流使decode Down局部从2608.07增至2875.15μs（+10.24%），完整decode墙钟只增加1.17%（95%CI +0.76%～+1.62%），prefill无稳定差异。这支持闲置物理行共装降低计算成本，但不是完整decode低开销的唯一或主要原因；两个独立流依然复用已驻留权重，没有重搬W4。共装2164.618/45.216tps，两流2162.319/44.695tps。本轮不评估SP2相对普通A8或模型质量。
