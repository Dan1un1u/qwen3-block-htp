# L32-0032：固定 FP32 残差的普通 A8 / SP2 公平成本

冻结原生 W4 权重/尺度与所有非目标边界，无旋转。仅恢复普通 A8 middle 参数与 SwiGLU LUT；两臂同等使用流水化向量 gather、Gate/Up 提前发布、Down 原始累加器与 FP32 epilogue。无重新校准或权重量化。

每种轨迹独立完成 5 short + 10 formal 配对轮次，交替顺序，repeat10 主计时；repeat1 辅助。固定 token 重放 mode3 保留相同 LM head/greedy 工作、不计算额外 NLL；各臂 KV 数值随合同变化，固定的是 token 输入序列。实际 greedy 另报，不混合两个估计。
配对 bootstrap 20000，seed30032。全部 9600 个计时 profile 通过物理核查，保留全部轮次。

## greedy · prefill 全模型模块表

| 模块 | 普通 A8 μs（Host占比） | SP2 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 131.5 (0.45%) | 131.2 (0.44%) |
| Input RMSNorm | 1282.5 (4.39%) | 1267.9 (4.28%) |
| QKV＋RoPE | 3868.2 (13.24%) | 3863.3 (13.03%) |
| QK–Softmax–AV | 8493.4 (29.07%) | 8493.4 (28.64%) |
| O projection | 1582.8 (5.42%) | 1583.8 (5.34%) |
| Post-attention residual＋RMSNorm | 1308.0 (4.48%) | 1277.0 (4.31%) |
| Gate/Up＋SwiGLU | 5485.3 (18.77%) | 5484.6 (18.50%) |
| Down | 2638.4 (9.03%) | 3133.3 (10.57%) |
| Final residual | 1.7 (0.01%) | 1.7 (0.01%) |
| KV carrier conversion | 27.7 (0.09%) | 27.3 (0.09%) |
| KV append DMA | 108.2 (0.37%) | 108.4 (0.37%) |
| Block orchestration | 23.1 (0.08%) | 22.9 (0.08%) |
| Layer bookkeeping | 13.4 (0.05%) | 13.3 (0.05%) |
| Stage-boundary bookkeeping | 7.1 (0.02%) | 6.9 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 91.3 (0.31%) | 90.2 (0.30%) |
| Embedding | 57.9 (0.20%) | 58.3 (0.20%) |
| Final model RMSNorm | 56.0 (0.19%) | 60.6 (0.20%) |
| LM head＋greedy（不含 final norm） | 3168.6 (10.84%) | 3185.1 (10.74%) |
| Host–DSP 边界 | 875.7 (3.00%) | 842.8 (2.84%) |
| 完整 Host wall | 29220.9 (100.00%) | 29652.1 (100.00%) |

SP2/A8 Host wall = 1.0147549，95%CI [1.0121397, 1.0172584]。

## greedy · decode 全模型模块表

| 模块 | 普通 A8 μs（Host占比） | SP2 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 126.7 (0.57%) | 126.7 (0.57%) |
| Input RMSNorm | 869.8 (3.93%) | 871.3 (3.94%) |
| QKV＋RoPE | 1669.3 (7.54%) | 1667.9 (7.54%) |
| QK–Softmax–AV | 5993.6 (27.09%) | 5997.1 (27.12%) |
| O projection | 886.7 (4.01%) | 885.1 (4.00%) |
| Post-attention residual＋RMSNorm | 877.0 (3.96%) | 874.9 (3.96%) |
| Gate/Up＋SwiGLU | 4873.0 (22.02%) | 4862.2 (21.99%) |
| Down | 2609.8 (11.79%) | 2606.1 (11.79%) |
| Final residual | 1.0 (0.00%) | 1.0 (0.00%) |
| KV carrier conversion | 48.2 (0.22%) | 48.2 (0.22%) |
| KV append DMA | 92.0 (0.42%) | 91.8 (0.42%) |
| Block orchestration | 15.8 (0.07%) | 15.9 (0.07%) |
| Layer bookkeeping | 9.4 (0.04%) | 9.4 (0.04%) |
| Stage-boundary bookkeeping | 1.2 (0.01%) | 1.2 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 51.2 (0.23%) | 51.3 (0.23%) |
| Embedding | 1.7 (0.01%) | 1.5 (0.01%) |
| Final model RMSNorm | 56.1 (0.25%) | 55.4 (0.25%) |
| LM head＋greedy（不含 final norm） | 3161.7 (14.29%) | 3177.5 (14.37%) |
| Host–DSP 边界 | 784.1 (3.54%) | 767.3 (3.47%) |
| 完整 Host wall | 22128.5 (100.00%) | 22111.7 (100.00%) |

SP2/A8 Host wall = 0.9992388，95%CI [0.9976420, 1.0009743]。

## fixed · prefill 全模型模块表

| 模块 | 普通 A8 μs（Host占比） | SP2 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 128.3 (0.44%) | 128.6 (0.43%) |
| Input RMSNorm | 1282.2 (4.39%) | 1268.2 (4.29%) |
| QKV＋RoPE | 3874.3 (13.28%) | 3869.5 (13.08%) |
| QK–Softmax–AV | 8502.3 (29.13%) | 8500.8 (28.75%) |
| O projection | 1586.1 (5.44%) | 1586.5 (5.36%) |
| Post-attention residual＋RMSNorm | 1307.8 (4.48%) | 1276.7 (4.32%) |
| Gate/Up＋SwiGLU | 5509.1 (18.88%) | 5501.0 (18.60%) |
| Down | 2656.1 (9.10%) | 3133.9 (10.60%) |
| Final residual | 1.7 (0.01%) | 1.7 (0.01%) |
| KV carrier conversion | 27.6 (0.09%) | 27.4 (0.09%) |
| KV append DMA | 108.0 (0.37%) | 108.3 (0.37%) |
| Block orchestration | 23.1 (0.08%) | 22.8 (0.08%) |
| Layer bookkeeping | 13.5 (0.05%) | 13.5 (0.05%) |
| Stage-boundary bookkeeping | 6.8 (0.02%) | 6.9 (0.02%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 91.7 (0.31%) | 90.8 (0.31%) |
| Embedding | 57.8 (0.20%) | 57.9 (0.20%) |
| Final model RMSNorm | 55.9 (0.19%) | 60.6 (0.20%) |
| LM head＋greedy（不含 final norm） | 3182.6 (10.91%) | 3204.9 (10.84%) |
| Host–DSP 边界 | 767.5 (2.63%) | 712.9 (2.41%) |
| 完整 Host wall | 29182.7 (100.00%) | 29572.7 (100.00%) |

SP2/A8 Host wall = 1.0133644，95%CI [1.0101526, 1.0163887]。

## fixed · decode 全模型模块表

| 模块 | 普通 A8 μs（Host占比） | SP2 μs（Host占比） |
|---|---:|---:|
| I/O、metadata | 123.4 (0.56%) | 123.9 (0.56%) |
| Input RMSNorm | 870.5 (3.95%) | 871.2 (3.95%) |
| QKV＋RoPE | 1670.7 (7.58%) | 1671.6 (7.58%) |
| QK–Softmax–AV | 5998.6 (27.20%) | 5998.9 (27.20%) |
| O projection | 886.7 (4.02%) | 889.1 (4.03%) |
| Post-attention residual＋RMSNorm | 874.9 (3.97%) | 874.9 (3.97%) |
| Gate/Up＋SwiGLU | 4882.0 (22.14%) | 4881.6 (22.13%) |
| Down | 2618.3 (11.87%) | 2617.2 (11.87%) |
| Final residual | 1.0 (0.00%) | 1.0 (0.00%) |
| KV carrier conversion | 48.2 (0.22%) | 48.2 (0.22%) |
| KV append DMA | 91.8 (0.42%) | 91.8 (0.42%) |
| Block orchestration | 15.8 (0.07%) | 15.8 (0.07%) |
| Layer bookkeeping | 9.4 (0.04%) | 9.4 (0.04%) |
| Stage-boundary bookkeeping | 1.2 (0.01%) | 1.2 (0.01%) |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) |
| Runtime setup/teardown | 51.1 (0.23%) | 51.2 (0.23%) |
| Embedding | 1.5 (0.01%) | 1.5 (0.01%) |
| Final model RMSNorm | 55.8 (0.25%) | 55.4 (0.25%) |
| LM head＋greedy（不含 final norm） | 3175.0 (14.40%) | 3170.9 (14.38%) |
| Host–DSP 边界 | 674.1 (3.06%) | 683.0 (3.10%) |
| 完整 Host wall | 22050.1 (100.00%) | 22057.8 (100.00%) |

SP2/A8 Host wall = 1.0003501，95%CI [0.9977827, 1.0029492]。

## 端到端吞吐

| 轨迹 | 配置 | Prefill token/s | Decode token/s |
|---|---|---:|---:|
| greedy | A8 | 2190.2103 | 45.1906 |
| greedy | SP2 | 2158.3639 | 45.2250 |
| fixed | A8 | 2193.0798 | 45.3513 |
| fixed | SP2 | 2164.1572 | 45.3354 |

M64+15、16层，包含 embedding、全部 transformer、final norm、LM head、greedy、FastRPC；排除外部 tokenizer、冷加载、ADB。fixed 行是受控 token 重放吞吐，不是自由生成文本速度。
Prefill SP2 额外 262144 个 HMX tile pair；decode 通过物理行共装，tile pair 与普通 A8 相同。两阶段权重 DDR 字节和 worker dispatch 次数相同。不据此推断 HMX 内部如何处理 W4。
Llama independent selected layer7, chain3, chain16 two-step exact; full16 all-layer greedy/fixed CPU teacher selected token/logit codes exact
未做 PPL/模型质量验收，不晋升基线。A9 FP 向量 softmax 与 A8 额外 shape 仍待执行。
