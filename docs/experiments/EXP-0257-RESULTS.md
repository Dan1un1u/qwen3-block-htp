# EXP-0257：完整模型真机文本完成，正式速度验证中断

完整28层真机文本已经执行；候选组件/切片/独立head检查与五轮短测通过，但十轮正式profiling在第七轮因原始A8对照不确定输出停止。修复后的A8文本仍不可用：六条固定样本0/6，同提示词冻结W4A16软件对照6/6（按内容判定，忽略格式差异）。这不是新的PPL验收，不晋升基线。

C64 per-output-channel W4 / original OFF A8 / offline fixed EOS W4A16-to-U8 prefix / wide NR64 / no R3

本次补齐此前硬件移植遗漏的固定EOS前缀：离线用原始C64 W4A16计算一次K/V，存成56KiB U8种子，设备预填充覆盖row0。无在线FP16运行、权重更新、分组量化或R3。初次无种子的坏输出保留为移植未完成的诊断，不混入正式候选。

独立证据：单层20文件精确复现、三层连续执行和缓存切换、种子单列快补与完整K重打包精确一致、实际QK/概率/AV对独立整数参考逐字节一致。64步完整模型NR64标量/HVX token完全一致；三步独立LMhead算术/greedy对照通过。这证明已检查范围的实现一致性，不证明全模型每个张量等价于teacher或软件浮点模型。

5轮短测完成；正式测试完成6轮，第7轮repeat10原始对照的第8条序列在step10/11/14生成token不一致后停止。全部已采集6016个完整模型RPC和168448个逐层计时账本均闭合，但token确定性gate失败，8MiB VTCM，零计时中间DDR/spill，每次完整模型一个RPC。repeat10是复用已加载权重后重建10个独立会话；每会话均为M64 prefill加15步真实token反馈decode。

速度主对照C0：原始整数attention、无种子；B：宽差值SOLE加正确种子；A：宽差值NR64加正确种子。A/C0计入全部修复开销，A/B用于归一化归因。下表仅为失败发生前6个完整轮次的诊断统计，不能替代预定10轮正式验收；失败原始文件未丢弃。五轮短测速度门槛通过，正式速度结论不予通过。

| 比较与阶段 | Host wall变化 | 配对95%区间 |
|---|---|---|
| c0_r1_prefill_ns | +0.528% | [+0.249%, +1.311%] |
| c0_r1_decode_ns | +0.016% | [-1.061%, +1.810%] |
| c0_r10_prefill_ns | +0.753% | [+0.513%, +1.227%] |
| c0_r10_decode_ns | -0.309% | [-0.631%, -0.153%] |
| c1_r1_prefill_ns | -0.486% | [-0.785%, -0.333%] |
| c1_r1_decode_ns | -0.386% | [-1.233%, +1.365%] |
| c1_r10_prefill_ns | -0.202% | [-0.635%, +0.322%] |
| c1_r10_decode_ns | -0.069% | [-0.521%, +0.167%] |

M64模块单位μs，括号占各自Host wall。F16A16 EXP0218和Selected W4A16 EXP0166为非配对历史参照；当前A8取失败前6个完整轮次repeat10中位数，仅作已测速度展示。各行分别取中位数，因此舍入前也不必恰好相加；原始调用账本均逐条闭合。

| 模块 | F16A16 EXP0218 | W4A16 EXP0166 | W4A8 EXP0257 | A8相对W4A16增速 |
|---|---|---|---|---|
| I/O、metadata | 99.1 (0.12%) | 366.8 (0.58%) | 227.3 (0.61%) | +61.36% |
| Input RMSNorm | 489.7 (0.61%) | 493.9 (0.78%) | 546.9 (1.47%) | -9.69% |
| QKV＋Q/K Norm-RoPE | 11456.4 (14.20%) | 11772.6 (18.69%) | 7041.6 (18.89%) | +67.18% |
| QK–Softmax–AV | 3983.1 (4.94%) | 3974.7 (6.31%) | 3477.4 (9.33%) | +14.30% |
| O projection | 5757.7 (7.14%) | 5044.9 (8.01%) | 1245.4 (3.34%) | +305.10% |
| Post-attention residual＋RMSNorm | 473.3 (0.59%) | 471.8 (0.75%) | 657.7 (1.76%) | -28.26% |
| Gate/Up＋SwiGLU | 29617.4 (36.70%) | 22459.5 (35.66%) | 13933.3 (37.38%) | +61.19% |
| Down | 13447.9 (16.67%) | 8516.2 (13.52%) | 3361.8 (9.02%) | +153.32% |
| Final residual | 140.1 (0.17%) | 140.3 (0.22%) | 184.3 (0.49%) | -23.90% |
| KV carrier conversion | 174.0 (0.22%) | 148.0 (0.24%) | 135.1 (0.36%) | +9.58% |
| KV append DMA | 343.6 (0.43%) | 338.9 (0.54%) | 284.5 (0.76%) | +19.12% |
| Block orchestration | 16.1 (0.02%) | 16.8 (0.03%) | 36.5 (0.10%) | -54.02% |
| Layer bookkeeping | 23.9 (0.03%) | 22.1 (0.04%) | 26.2 (0.07%) | -15.60% |
| Stage-boundary bookkeeping | 8.3 (0.01%) | 4.4 (0.01%) | 20.6 (0.06%) | -78.79% |
| DSP unattributed | 0.0 (0.00%) | 0.0 (0.00%) | 0.0 (0.00%) | N/A |
| Runtime setup/teardown | 82.6 (0.10%) | 99.9 (0.16%) | 117.8 (0.32%) | -15.21% |
| Embedding | 68.1 (0.08%) | 77.5 (0.12%) | 39.4 (0.11%) | +96.59% |
| Final model RMSNorm | 49.7 (0.06%) | 48.3 (0.08%) | 3.0 (0.01%) | +1518.13% |
| LM head＋greedy，不含 final norm | 11993.8 (14.86%) | 6702.0 (10.64%) | 5258.7 (14.11%) | +27.45% |
| Host–DSP 边界 | 2374.1 (2.94%) | 2288.5 (3.63%) | 675.8 (1.81%) | +238.63% |
| 完整 Host wall | 80692.2 (100.00%) | 62974.7 (100.00%) | 37272.7 (100.00%) | +68.96% |

## 文本检查

六题包含中英文首都、12+9及植物光合作用。为固定总M64，提示词含重复的简短回答指令；提示词在输出前冻结，同样提示词W4A16对照可用。不给只有格式差异的内容扣分：W4A16英文解释只有一句也按内容可用。A8问题为语义缺失/循环。原始64token固定预算用于缓存测试；文本仅取首次EOS之前，EOS之后不能作为正确答案。

### en_fact

Please give a clear and direct answer to the following question. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Now What is the capital of France? Answer briefly.

A8：
```text
Okay, now I need to to do this. Now, what is the capital of France? Let me try to think..

Okay, now, I need to do this. Now,, what is the capital of France? Now, I need to give a clear and direct answer.

Okay, now, what,
```
判定：Repeated deliberation/question, no Paris answer

W4A16软件对照：
```text
The capital of France is Paris.<|im_end|>
```

### en_math

Please give a clear and direct answer to the following question. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Now What is 12 plus 9? Answer with the number.

A8：
```text
以此为则，以此为则。请看此，请看此。请看此，请看此。请看此，请看此。请看此，请看此。请看此，请看此。请看此，请看此。请看此，请看
```
判定：Repetitive unrelated Chinese, no21

W4A16软件对照：
```text
12 + 9 = 21.<|im_end|>
```

### zh_fact

Please give a clear and direct answer to the following question. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. 中国的首都是哪里？请简短回答。

A8：
```text
中国首都是哪里？


上述问题的解答如下：

中国首都是哪里？？

中国首都是哪里？？

中国首都是哪里？？

中国首都是哪里？？

中国首都是哪里？？

中国首都是哪里？？

中国首都是哪里？？

中国首都是哪里？？

中国首都是哪里
```
判定：Question echo and repetition, no Beijing answer

W4A16软件对照：
```text
北京。<|im_end|>
```

### zh_math

Please give a clear and direct answer to the following question. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Now, 12加9等于多少？请只回答数字。

A8：
```text
Okay, let's see. I need to find out if a number is a multiple of 3, or if it's a multiple of 3, or if it is a multiple of 3, or if it is divisible by 3, or if it is divisible by 3, or if it is divisible
```
判定：Off-topic divisibility discussion, no21

W4A16软件对照：
```text
12 + 9 = 21.<|im_end|>
```

### en_prose

Please give a clear and direct answer to the following question. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Now Explain in two simple sentences why plants need sunlight.

A8：
```text
**

**1.** **1.** **2**. **2.** **1.** **2.** **2.** **2.** **2.** **2.** **2.** **2.** **2.** **2.** **2.** **2.
```
判定：Repeated numbering/formatting, no explanation

W4A16软件对照：
```text
Plants need sunlight to perform photosynthesis, which is the process of converting light into energy.<|im_end|>
```

### zh_prose

Please give a clear and direct answer to the following question. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. Please be concise. 请用两句话解释植物为什么需要阳光。

A8：
```text
请用两句话解释植物为什么需要阳光。

```
判定：Question echo before first EOS; later continuation excluded

W4A16软件对照：
```text
植物需要阳光进行光合作用，以制造养分。阳光是植物进行光合作用的能量来源。<|im_end|>
```

## 解释边界与后续讨论

此结果已是完整模型设备文本验证，不再是单block速度外推。正式测试先因原始A8速度对照的偶发不确定输出中断；短测和已完成候选运行一致不代表该共用运行时隐患已排除。须定位并修复后重新做完整正式测试。当前不能仅凭组件通过，将全模退化完全归于模型量化本身。设备head使用U8 logits(scale0.5/zero128)，此前软件PPL保留浮点logits；还存在完整设备HMX与软件仿真边界差异。下一步先定位原始速度对照的瞬态数值不一致，再做同token teacher-forcing设备/软件逐层对齐和输出head消融，区分transformer累积误差、归一化/转换及logit网格影响；本轮不擅自改变算法。无新设备PPL。

## 实测E2E速度

| arm/repeat | prefill64 tok/s | decode15 tok/s | 16输出/模型总Host tok/s | 16输出/完整generation loop tok/s |
|---|---|---|---|---|
| w0_r1 | 1654.049 | 47.359 | 45.032 | 44.108 |
| w1_r1 | 1638.779 | 47.419 | 45.000 | 44.088 |
| w4_r1 | 1646.663 | 47.356 | 44.981 | 44.108 |
| w0_r10 | 1730.102 | 50.094 | 47.578 | 45.965 |
| w1_r10 | 1716.453 | 50.215 | 47.619 | 45.965 |
| w4_r10 | 1717.077 | 50.320 | 47.717 | 45.895 |

prefill分子64包含1个固定EOS种子位置；首个输出已在prefill产生，decode分子15。模型Host包含embedding/28层/finalnorm/head/greedy及Host-DSP边界；generation loop另含设备Host循环和profiling序列化。两者均为实际整模型计时，无逐层外推。上述热态速度不含模型加载、ADB、WSL tokenizer/detokenizer；这些不能混合平台后伪称设备推理耗时。cold startup和独立WSL前后处理实测值保存在summary.json。
