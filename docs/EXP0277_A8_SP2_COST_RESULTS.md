# EXP-0277：固定 FP32 残差的普通 A8 / SP2 公平成本

冻结原生 W4 权重/尺度与所有非目标边界，无旋转。仅恢复普通 A8 middle 参数与 SwiGLU LUT；两臂同等使用流水化向量 gather、Gate/Up 提前发布、Down 原始累加器与 FP32 epilogue。无重新校准或权重量化。

每种轨迹独立完成 5 short + 10 formal 配对轮次，交替顺序，repeat10 主计时；repeat1 辅助。固定 token 重放 mode3 保留相同 LM head/greedy 工作、不计算额外 NLL；各臂 KV 数值随合同变化，固定的是 token 输入序列。实际 greedy 另报，不混合两个估计。
配对 bootstrap 20000，seed277。全部 9600 个计时 profile 通过物理核查，保留全部轮次。

## greedy · prefill 全模型模块表

| 模块 | 普通 A8 μs（Host占比） | SP2 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 259.5 (0.76%) | 256.4 (0.74%) |
| Input RMSNorm | 2037.3 (5.94%) | 2018.4 (5.83%) |
| QKV＋Q/K Norm-RoPE | 7028.3 (20.49%) | 7032.2 (20.33%) |
| QK–Softmax–AV | 3493.8 (10.19%) | 3492.4 (10.10%) |
| O projection | 2089.2 (6.09%) | 2085.0 (6.03%) |
| Post-attention residual＋RMSNorm | 2073.5 (6.05%) | 2117.4 (6.12%) |
| Gate/Up＋SwiGLU | 7016.8 (20.46%) | 7045.6 (20.37%) |
| Down | 3518.1 (10.26%) | 3785.5 (10.94%) |
| Final residual | 2.2 (0.01%) | 2.2 (0.01%) |
| KV carrier conversion | 137.5 (0.40%) | 135.6 (0.39%) |
| KV append DMA | 296.6 (0.86%) | 296.0 (0.86%) |
| Block orchestration | 35.5 (0.10%) | 35.2 (0.10%) |
| Layer bookkeeping | 25.6 (0.07%) | 25.6 (0.07%) |
| Stage-boundary bookkeeping | 21.4 (0.06%) | 21.1 (0.06%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 116.1 (0.34%) | 116.6 (0.34%) |
| Embedding | 56.0 (0.16%) | 56.3 (0.16%) |
| Final model RMSNorm | 18.5 (0.05%) | 13.7 (0.04%) |
| LM head＋greedy，不含 final norm | 5153.0 (15.02%) | 5154.7 (14.90%) |
| Host–DSP 边界 | 920.4 (2.68%) | 905.0 (2.62%) |
| 完整 Host wall | 34299.2 (100.00%) | 34594.9 (100.00%) |

SP2/A8 Host wall = 1.0086212，95%CI [1.0057344, 1.0116684]。

## greedy · decode 全模型模块表

| 模块 | 普通 A8 μs（Host占比） | SP2 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 255.8 (1.21%) | 255.1 (1.20%) |
| Input RMSNorm | 366.6 (1.74%) | 369.2 (1.74%) |
| QKV＋Q/K Norm-RoPE | 2495.5 (11.81%) | 2502.4 (11.79%) |
| QK–Softmax–AV | 1916.5 (9.07%) | 1918.0 (9.04%) |
| O projection | 1341.2 (6.35%) | 1345.2 (6.34%) |
| Post-attention residual＋RMSNorm | 368.2 (1.74%) | 369.1 (1.74%) |
| Gate/Up＋SwiGLU | 6385.7 (30.23%) | 6405.2 (30.19%) |
| Down | 3437.9 (16.27%) | 3490.3 (16.45%) |
| Final residual | 1.8 (0.01%) | 1.8 (0.01%) |
| KV carrier conversion | 290.8 (1.38%) | 290.7 (1.37%) |
| KV append DMA | 126.0 (0.60%) | 126.0 (0.59%) |
| Block orchestration | 28.1 (0.13%) | 28.0 (0.13%) |
| Layer bookkeeping | 16.5 (0.08%) | 16.5 (0.08%) |
| Stage-boundary bookkeeping | 1.9 (0.01%) | 1.9 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 73.6 (0.35%) | 73.6 (0.35%) |
| Embedding | 2.0 (0.01%) | 1.8 (0.01%) |
| Final model RMSNorm | 14.3 (0.07%) | 14.8 (0.07%) |
| LM head＋greedy，不含 final norm | 3210.0 (15.20%) | 3217.6 (15.16%) |
| Host–DSP 边界 | 792.2 (3.75%) | 791.8 (3.73%) |
| 完整 Host wall | 21124.6 (100.00%) | 21219.1 (100.00%) |

SP2/A8 Host wall = 1.0044715，95%CI [1.0016027, 1.0080510]。

## fixed · prefill 全模型模块表

| 模块 | 普通 A8 μs（Host占比） | SP2 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 256.3 (0.75%) | 257.3 (0.74%) |
| Input RMSNorm | 2037.7 (5.93%) | 2019.8 (5.83%) |
| QKV＋Q/K Norm-RoPE | 7030.9 (20.47%) | 7029.0 (20.28%) |
| QK–Softmax–AV | 3489.7 (10.16%) | 3497.2 (10.09%) |
| O projection | 2085.0 (6.07%) | 2089.8 (6.03%) |
| Post-attention residual＋RMSNorm | 2074.6 (6.04%) | 2118.6 (6.11%) |
| Gate/Up＋SwiGLU | 7047.6 (20.52%) | 7052.1 (20.35%) |
| Down | 3527.5 (10.27%) | 3789.3 (10.93%) |
| Final residual | 2.3 (0.01%) | 2.3 (0.01%) |
| KV carrier conversion | 133.4 (0.39%) | 136.0 (0.39%) |
| KV append DMA | 295.5 (0.86%) | 295.8 (0.85%) |
| Block orchestration | 35.8 (0.10%) | 35.3 (0.10%) |
| Layer bookkeeping | 24.9 (0.07%) | 25.1 (0.07%) |
| Stage-boundary bookkeeping | 21.4 (0.06%) | 21.1 (0.06%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 117.3 (0.34%) | 116.2 (0.34%) |
| Embedding | 56.2 (0.16%) | 55.7 (0.16%) |
| Final model RMSNorm | 18.5 (0.05%) | 13.7 (0.04%) |
| LM head＋greedy，不含 final norm | 5146.1 (14.98%) | 5156.2 (14.88%) |
| Host–DSP 边界 | 950.7 (2.77%) | 948.5 (2.74%) |
| 完整 Host wall | 34351.7 (100.00%) | 34659.1 (100.00%) |

SP2/A8 Host wall = 1.0089482，95%CI [1.0065433, 1.0113939]。

## fixed · decode 全模型模块表

| 模块 | 普通 A8 μs（Host占比） | SP2 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 255.3 (1.20%) | 256.1 (1.20%) |
| Input RMSNorm | 365.9 (1.73%) | 369.2 (1.74%) |
| QKV＋Q/K Norm-RoPE | 2504.9 (11.81%) | 2510.6 (11.80%) |
| QK–Softmax–AV | 1920.1 (9.05%) | 1920.0 (9.03%) |
| O projection | 1345.7 (6.35%) | 1348.6 (6.34%) |
| Post-attention residual＋RMSNorm | 370.1 (1.74%) | 369.2 (1.74%) |
| Gate/Up＋SwiGLU | 6415.6 (30.25%) | 6427.9 (30.22%) |
| Down | 3452.6 (16.28%) | 3502.2 (16.46%) |
| Final residual | 1.8 (0.01%) | 1.8 (0.01%) |
| KV carrier conversion | 290.8 (1.37%) | 290.6 (1.37%) |
| KV append DMA | 125.1 (0.59%) | 125.6 (0.59%) |
| Block orchestration | 28.1 (0.13%) | 27.9 (0.13%) |
| Layer bookkeeping | 16.5 (0.08%) | 16.5 (0.08%) |
| Stage-boundary bookkeeping | 1.9 (0.01%) | 1.9 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 73.6 (0.35%) | 73.6 (0.35%) |
| Embedding | 1.8 (0.01%) | 1.8 (0.01%) |
| Final model RMSNorm | 15.0 (0.07%) | 14.9 (0.07%) |
| LM head＋greedy，不含 final norm | 3222.2 (15.19%) | 3225.5 (15.16%) |
| Host–DSP 边界 | 801.6 (3.78%) | 788.3 (3.71%) |
| 完整 Host wall | 21208.5 (100.00%) | 21272.6 (100.00%) |

SP2/A8 Host wall = 1.0030221，95%CI [1.0014471, 1.0047096]。

## 端到端吞吐

| 轨迹 | 配置 | Prefill token/s | Decode token/s |
|---|---|---:|---:|
| greedy | A8 | 1865.9314 | 47.3382 |
| greedy | SP2 | 1849.9823 | 47.1274 |
| fixed | A8 | 1863.0823 | 47.1510 |
| fixed | SP2 | 1846.5589 | 47.0089 |

M64+15、28层，包含 embedding、全部 transformer、final norm、LM head、greedy、FastRPC；排除外部 tokenizer、冷加载、ADB。fixed 行是受控 token 重放吞吐，不是自由生成文本速度。
Prefill SP2 额外 344064 个 HMX tile pair；decode 通过物理行共装，tile pair 与普通 A8 相同。两阶段权重 DDR 字节和 worker dispatch 次数相同。不据此推断 HMX 内部如何处理 W4。
Qwen selected layer14 and chain3 independent exact; full28 physical/front-end finalNorm and full-vocabulary head oracle; no full28 CPU transformer equivalence claim
未做 PPL/模型质量验收，不晋升基线。A9 FP 向量 softmax 与 A8 额外 shape 仍待执行。
