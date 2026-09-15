# EXP0276：SP2 decode 共装与两独立流

固定原生W4/SP2mode8、FP32残差2、无旋转，权重码/尺度、LUT和所有非目标优化不变。SPLIT由SwiGLU直接输出独立high平面，Down对同一已搬入权重执行两次矩阵累积；ALL用空闲物理行共装一次矩阵流。Prefill合同不变。

全模5short+10formal，交替顺序，repeat10主计时，repeat1辅助。95%CI按十个配对轮次bootstrap20000seed276，保留全部样本。

## 完整28层 prefill 模块表

| 模块 | 共装 μs（Host占比） | 两独立流 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 259.4 (0.75%) | 257.9 (0.74%) |
| Input RMSNorm | 2020.4 (5.83%) | 2020.4 (5.82%) |
| QKV＋Q/K Norm-RoPE | 7051.7 (20.34%) | 7058.9 (20.34%) |
| QK–Softmax–AV | 3493.8 (10.08%) | 3494.2 (10.07%) |
| O projection | 2090.3 (6.03%) | 2093.0 (6.03%) |
| Post-attention residual＋RMSNorm | 2117.1 (6.11%) | 2116.3 (6.10%) |
| Gate/Up＋SwiGLU | 7045.5 (20.32%) | 7045.4 (20.30%) |
| Down | 3788.4 (10.93%) | 3789.3 (10.92%) |
| Final residual | 2.6 (0.01%) | 2.7 (0.01%) |
| KV carrier conversion | 136.5 (0.39%) | 136.1 (0.39%) |
| KV append DMA | 295.8 (0.85%) | 295.2 (0.85%) |
| Block orchestration | 35.3 (0.10%) | 35.4 (0.10%) |
| Layer bookkeeping | 25.0 (0.07%) | 25.0 (0.07%) |
| Stage-boundary bookkeeping | 20.8 (0.06%) | 20.8 (0.06%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 116.3 (0.34%) | 116.7 (0.34%) |
| Embedding | 56.5 (0.16%) | 56.6 (0.16%) |
| Final model RMSNorm | 13.7 (0.04%) | 13.7 (0.04%) |
| LM head＋greedy，不含 final norm | 5199.0 (15.00%) | 5204.0 (15.00%) |
| Host–DSP 边界 | 901.0 (2.60%) | 917.8 (2.65%) |
| 完整 Host wall | 34669.1 (100.00%) | 34699.3 (100.00%) |

两流/共装 Host wall：1.000872，95%CI [0.999180, 1.002771]。

## 完整28层 decode 模块表

| 模块 | 共装 μs（Host占比） | 两独立流 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 255.3 (1.20%) | 255.8 (1.19%) |
| Input RMSNorm | 369.2 (1.74%) | 369.3 (1.72%) |
| QKV＋Q/K Norm-RoPE | 2505.1 (11.79%) | 2508.9 (11.70%) |
| QK–Softmax–AV | 1921.0 (9.04%) | 1924.7 (8.98%) |
| O projection | 1345.1 (6.33%) | 1346.7 (6.28%) |
| Post-attention residual＋RMSNorm | 369.1 (1.74%) | 369.2 (1.72%) |
| Gate/Up＋SwiGLU | 6417.1 (30.21%) | 6425.1 (29.97%) |
| Down | 3495.3 (16.45%) | 3664.6 (17.09%) |
| Final residual | 1.7 (0.01%) | 1.7 (0.01%) |
| KV carrier conversion | 291.1 (1.37%) | 291.2 (1.36%) |
| KV append DMA | 125.4 (0.59%) | 125.8 (0.59%) |
| Block orchestration | 27.9 (0.13%) | 28.0 (0.13%) |
| Layer bookkeeping | 16.4 (0.08%) | 16.4 (0.08%) |
| Stage-boundary bookkeeping | 1.8 (0.01%) | 1.8 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 73.7 (0.35%) | 73.7 (0.34%) |
| Embedding | 1.8 (0.01%) | 1.9 (0.01%) |
| Final model RMSNorm | 14.8 (0.07%) | 14.9 (0.07%) |
| LM head＋greedy，不含 final norm | 3224.9 (15.18%) | 3226.5 (15.05%) |
| Host–DSP 边界 | 785.2 (3.70%) | 793.3 (3.70%) |
| 完整 Host wall | 21241.8 (100.00%) | 21439.5 (100.00%) |

两流/共装 Host wall：1.009305，95%CI [1.007675, 1.010686]。

## 完整E2E

| 配置 | Prefill token/s | Decode token/s |
|---|---:|---:|
| ALL | 1846.025 | 47.077 |
| SPLIT | 1844.417 | 46.643 |

M64+15、cache128、冻结EOS前缀。包含embedding/28层/finalnorm/head/greedy/FastRPC，排除tokenizer/冷加载/ADB。HMXcommand计数是worker提交次数；真正额外工作看tilepair计数，decode每层增加12288，28层增加344064。权重搬运不增加。结果只解释固定SP2合同的执行方式，不是SP2相对普通A8的成本（A5另测），也不代表模型质量。

## 结论边界

两独立流使decode Down局部从3495.25增至3664.63μs（+4.85%），完整decode墙钟增加0.93%（95%CI +0.77%～+1.07%）；prefill无稳定差异。共装有效，但在固定FP32残差和已复用权重的良好两流实现下，其完整模型贡献约1%，不能写成decode低开销的唯一或主要原因。算法额外矩阵pass数量与墙钟不成比例，须结合权重搬运复用和非Down成本解释。
