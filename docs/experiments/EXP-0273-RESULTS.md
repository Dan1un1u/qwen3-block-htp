# EXP0273 A0：Qwen 基线复现

固定 EXP0272 无旋转 SP2mode8 + FP32 residual mode2，原生 src/include 未变。1 次 repeat1 辅助、5 次 repeat10 复现；不是十轮配对消融或新优化验收。已验证旧证据922文件、复用权重本地和设备 hash，新构建独立封存。原独立单层/chain3证明在原生树相同的前提下显式继承；本轮全部16步 token/logit 与旧封存逐项一致。

6 CLI / 816 profiles；8MiB VTCM，峰值8365824B，中间 tensor DDR/spill=0。未运行整数残差、PPL、softmax 或其他对比臂。Llama 与后续消融仍待执行。

| 模块 | Prefill μs（Host占比） | Decode μs/token（Host占比） |
|---|---:|---:|
| I/O、metadata | 255.2 (0.74%) | 251.5 (1.19%) |
| Input RMSNorm | 2020.1 (5.84%) | 368.5 (1.74%) |
| QKV＋Q/K Norm-RoPE | 7025.7 (20.30%) | 2491.9 (11.78%) |
| QK–Softmax–AV | 3483.4 (10.07%) | 1924.4 (9.10%) |
| O projection | 2093.0 (6.05%) | 1340.4 (6.34%) |
| Post-attention residual＋RMSNorm | 2115.5 (6.11%) | 369.5 (1.75%) |
| Gate/Up＋SwiGLU | 6977.1 (20.16%) | 6358.6 (30.06%) |
| Down | 3764.9 (10.88%) | 3469.2 (16.40%) |
| Final residual | 2.8 (0.01%) | 2.0 (0.01%) |
| KV carrier conversion | 134.8 (0.39%) | 291.0 (1.38%) |
| KV append DMA | 295.7 (0.85%) | 126.8 (0.60%) |
| Block orchestration | 35.5 (0.10%) | 28.6 (0.14%) |
| Layer bookkeeping | 27.3 (0.08%) | 16.3 (0.08%) |
| Stage-boundary bookkeeping | 20.6 (0.06%) | 1.7 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 115.8 (0.33%) | 73.5 (0.35%) |
| Embedding | 61.8 (0.18%) | 2.0 (0.01%) |
| Final model RMSNorm | 15.1 (0.04%) | 14.8 (0.07%) |
| LM head＋greedy，不含 final norm | 5194.5 (15.01%) | 3193.1 (15.10%) |
| Host–DSP 边界 | 962.2 (2.78%) | 827.4 (3.91%) |
| 完整 Host wall | 34601.0 (100.00%) | 21151.2 (100.00%) |

E2E：prefill 1849.6557 token/s；decode 47.2787 token/s。

范围：M64+15，28层，cache128，冻结EOS前缀，warm embedding/transformer/finalnorm/head/greedy/FastRPC；不含外部tokenizer和冷加载。与历史速度只能说明复现接近，不能按非配对差值声称优化收益。
