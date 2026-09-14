# 从数值表示到硬件流水：移动 NPU 低比特 LLM Runtime 的论文叙事（V2）

日期：2026-09-14。整理任务：L32-0027。依据用户提供的旧讨论与封存实验更新；本次没有新增硬件测量。

## 1. 论文现在应回答什么问题

**如何让面向量化的数值表示，以硬件原生格式及时进入矩阵引擎，并把增加的转换和修正工作尽量移出完整推理的关键路径？**

这比“优化 HVX、HMX、DMA 并行”更完整，也能够统一我们已经做过的普通 W4A8、SP2、在线旋转与高精度残差工作。

建议核心论点：

> 移动 NPU 上低比特推理的执行成本，取决于数值表示如何映射到原生矩阵指令、生产者如何直接形成消费者需要的物理格式，以及有限片上存储中的数据何时就绪。联合设计这三个方面，能够在保持 packed W4 权重路径的同时，以较低的实测增量成本支持非均匀激活表示和更丰富的精度边界。

与旧稿相比，SP2 应从扩展案例提升为核心技术贡献；Qwen 的普通 U8 回移实验应成为关键归因证据；FP32 残差和 R3/R4 则展示执行机制的扩展能力。论文的实证重点是系统性能与映射机制。现有结果尚不足以主张完整 W4A8 方案达到可用生成质量、软件 PPL 已在硬件复现，或同质量优于现有系统。

用户已接受原快速实现作为当前论文工作版本，暂停继续软硬数值对齐。原实现的 full16 数值差异仍如实保留，不因此改变历史 gate 或模型质量结论。该选择不要求先解决舍入问题才能继续写作。

## 2. 建议收敛成三项贡献

### 贡献一：面向消费者的物理张量接口与有限 VTCM 流水

逻辑 shape/dtype 不足以决定高效执行。实现还必须共同约定：量化编码及尺度、原生 tile 排列、有效行、就绪发布粒度、缓冲区生命周期及复用条件。我们把这些约定贯穿相邻算子，组织一个有序 HMX 消费者与 DMA/HVX 生产者协同工作。

用户提出的两点可以正式写为：

- **以消费者就绪时间为目标的异构调度。** Gate/Up 按消费顺序交错，让 HVX 在一组 Up tile 完成后即可生产 SwiGLU；HMX 计算下一块时，HVX 完成上一块 raw accumulator 的输出处理。优化目标是完整 Host wall，而非单个线程等待计数。
- **跨算子的原生物理格式生产与消费。** Norm、SwiGLU、旋转和 KV 边界尽量直接生产下一消费者可读取的 tile，减少通用中间表示及重复重排。这里包括地址、布局、有效区域和发布粒度，不能只写成“内存地址对齐”。

VTCM 生命周期是调度的一部分：缓冲区只有在最后一个消费者完成后才可复用；ready 表示数据可消费，done 表示最终写入完成且可覆盖。当前是针对目标结构设计的 runtime 与内核实现，尚非自动推导这些计划的通用编译器。

### 贡献二：非均匀 SP2 激活对原生 W4 矩阵路径的精确映射

SP2 码本索引为 8 bit，但它所代表的重构值并不是普通均匀 INT8。对当前码本，提取公共尺度后，重构值可用有符号 16 位整数 v 表示：

\[
v=l+256h,\quad l\in[0,255],\quad h\in[-128,127],\quad u_h=h+128.
\]

对每个输出通道 n：

\[
\sum_k v_k w_{kn}=\sum_k l_k w_{kn}
+256\sum_k u_{h,k}w_{kn}-32768\sum_k w_{kn}.
\]

因此可以用两个 U8 操作数分量复用原生 packed signed W4 乘法，加上预计算的权重列和补偿，精确形成指定 SP2 重构值的整数点积。无需将 W4 展开成 S8，也无需引入沿 K 维分组的权重尺度。

“精确”限定在所选量化码及已证明的整数范围内，不意味着量化前浮点模型无误差。两个分量来自 16 位整数的 radix-256 分解，不是“两个幂项天然对应两次矩阵乘”。更宽的码本不能直接沿用两分量结论；A8 码本索引、16 位重构值及两张临时 U8 操作数应分开标注。

工程上，SwiGLU 的 LUT 直接生产 low/high 原生 tile；Down 保持 packed W4 的 DMA 流，在整数合并后进行输出重定标。当前实现按权重正负和证明原始点积落在 signed24 范围，利用 retain-store 精确读出并在 HVX 中重构，不能笼统描述为一个未经验证的“原生 INT32 输出指令”。

**decode 还有一个特别有力的观察：额外逻辑分量可以填入矩阵指令原本闲置的物理行。** high 分量放入第 4–7 行，复用同一 HMX 流；prefill 的有效行已经占满，则需要两个计算流。因此 decode 的增量不仅来自工作重叠，还来自避免增加矩阵流和重复权重搬运。这是阶段特定的映射，不能外推到任意 batch。

这项贡献是“表示到现有乘法器的映射 + 原生生产端融合 + 阶段特定的执行计划”。SP2/APoT 码本思想以及进制分解本身不是我们的发明。

### 贡献三：跨模型的成本归因与精度边界扩展

在 Qwen3-1.7B 与 Llama-3.2-1B-Instruct 上验证上述机制，分别覆盖不同 head 维度、GQA 比例和层数。实验区分共享流水收益与新表示的真实增量成本，并探索原生 W4×A8 点积与 FP32 残差、在线 dense R3/R4 的组合。

这项贡献的重点是可验证的执行行为和成本边界，而非提出一个已完成质量验收的新量化算法。两个模型说明机制可迁移，但不足以证明适用于所有 Transformer、上下文长度或移动 NPU。

## 3. 新实验给出的关键结论

### 3.1 新增算术不一定等比例增加墙钟，但必须给普通基线同等优化

Qwen EXP-0267 中 SP2 相对旧 U8 的 prefill 墙钟下降 15.59%。如果只展示这一结果，很容易误写为 SP2 本身比均匀量化更便宜。

EXP-0268 把相同生产与调度改进移植回 U8：所有 U8 模式保持原有量化、单次 Down、HMX 工作量、指令数和权重搬运字节数；优化 U8 的完整 prefill 墙钟下降 **18.27%**。与这个更公平的 U8 对照相比，SP2 的真实增量是 prefill **2.71%**、decode **0.82%**。

因此最有力的论点是：**我们缩短了旧流水的等待关键路径，并把 SP2 新增工作的大部分成本覆盖在执行过程中；额外计算仍然存在。**

[Qwen EXP-0268 记录](/home/daniuniu/work/qwen3-block-htp-project-memory/docs/experiments/EXP-0268.md)

### 3.2 Prefill 与 decode 应使用不同的表示映射和调度

Llama L32-0012 中，仅重排 gather 的 mode6 未获得稳定 E2E 收益。提前交错 Gate/Up 的 mode7 相比 mode6 使完整 prefill 延迟下降 **5.03%**；加入 Down HMX/HVX 两槽重叠的 mode8 再下降 **2.07%**。最终 SP2 比 U8 的完整 prefill 只慢 **1.89%**，decode 的延迟差 **−0.02%**，其置信区间跨零。

这支持以完整依赖关系来优化，不能将收益归因于未测量的 gather bank conflict。decode 的原生行打包与 prefill 的提前发布/输出处理重叠，应分别画清楚。

[Llama L32-0012 实现与消融](/home/daniuniu/work/llama32-htp/docs/LLAMA32_SP2_PIPELINE_DIRECTIONS.md)

### 3.3 点积输入精度与残差存储精度可以分开选择

原生 W4×A8 线性乘法并不强迫 O/Down 出口、残差和 RMSNorm 也使用 A8。L32-0018 保留原生乘法，读取宽整数结果后在 FP32 中缩放、相加和计算 Norm，在 **Norm 之后** 重新量化为固定 A8，交给后续线性层。

无旋转、SP2 的全模配对结果为 prefill 延迟 **−3.23%**、decode **+8.13%**。它证明特定高精度边界可以以有限总体成本实现；prefill 变快包含不同数值轨迹/布局下其他模块的观测变化，不能写成 FP32 算术天然快于整数，也不能据此声称模型质量已经恢复。

[Llama L32-0018 报告](/home/daniuniu/work/llama32-htp-project-memory/docs/LLAMA32_FP32_PIPELINE_FOLLOWUP.md)

### 3.4 旋转的代价取决于执行边界，当前快速实现的质量结论仍有限

Llama 已把 Qwen dense R3/R4 的格式与流水经验迁移到 FP32 残差/SP2 路径。R4 采用分解后的 dense H512/H16 矩阵乘，不使用蝶形；不能描述成一次完整 8192×8192 稠密矩阵乘。

L32-0022 的单层配对结果为 R3+R4 prefill 延迟 **+6.21%**、decode **−0.93%**；后者 CI 跨零，应表述为未观察到稳定额外延迟。该组合尚无包含 embedding/head 的完整 Llama E2E 测量。

在线旋转准备、格式收尾与消费融合的重要性可以讨论，但没有同硬件、同误差合同的 dense-versus-butterfly 配对结果，不能声称 dense 普遍胜过蝶形。

[Llama L32-0022 报告](/home/daniuniu/work/llama32-htp/docs/LLAMA32_ROTATION_NUMERICAL_REPAIR.md)

### 3.5 量化的硬件成本由尺度进入归约的位置决定

旧稿中的 LPBQ32 负面实验仍有价值：沿 K 分组的尺度需要在归约内部参与运算，不能像 per-output-channel 单尺度一样只在最终结果上处理。已有展开与多次 packed W4 方案显示严重开销，说明直接套用量化格式会破坏既有执行计划。

应将它作为特定映射的反例，而不是“分组量化必然慢”的普遍定律。SP2 提供正面对照：其有限码本重构可以分解为少量原生整数分量，从而保持权重格式与主要搬运路径。

## 4. 可直接进入论文的实测表

### 4.1 用于因果讨论的同轮配对结果

目标平台为 SM8750 / HTP V79。近期实验实际申请 8 MiB VTCM；例如 Qwen EXP-0268 峰值 8,365,824 B、Llama SP2 L32-0012 峰值 8,229,344 B。对应计时路径无普通中间激活 DDR 读写或张量 spill，诊断导出排除在计时外；这不表示模型权重不从 DDR 搬运，也不表示 Host/FastRPC 开销为零。

百分比均为完整 Host 延迟变化，负数表示减少；括号为配对 bootstrap 95% CI。近期性能判断使用固定十轮与既定上界 10% 标准，repeat1 仅作辅助。不同实验行不能连乘成为一个未经测量的组合结果。

| 实验与对照 | 范围 | Prefill 延迟变化 | Decode 延迟变化 |
|---|---|---:|---:|
| Qwen EXP-0268：优化 U8 / 原 U8 | 28 层，M64+15 | −18.273% [−18.525, −18.042] | −0.005% [−0.280, +0.241] |
| Qwen EXP-0268：SP2 / 优化 U8 | 同轮 28 层，M64+15 | +2.710% [+2.330, +3.072] | +0.819% [+0.632, +1.009] |
| Llama L32-0012：SP2 mode8 / U8 | 16 层，M64+15 | +1.885% [+0.191, +3.445] | −0.023% [−0.430, +0.385] |
| Llama L32-0018：FP32 残差 SP2 / 原 SP2，无旋转 | 16 层，M64+15 | −3.231% [−4.609, −2.276] | +8.128% [+7.183, +9.276] |
| Llama L32-0022：FP32 残差 SP2 上 R3+R4 / OFF | **单层 0**，M64+decode1 | +6.207% [+3.891, +8.768] | −0.930% [−3.888, +2.164] |

最后一行 OFF/R3R4 的原始 Host 时间分别为 prefill 1945.139/2065.881 μs，decode 1464.737/1451.117 μs。这是已知 full16 参考不对齐的快速旋转实现的局部性能证据。

SP2 的全模正确性证据独立于后来的旋转问题：L32-0012 保持冻结的 SP2 整数合同，单层、连续层和全模生成码通过核对；EXP-0268 的各 U8 变体与旧 U8 一致，SP2 与封存 SP2 一致。它们不是跨 recipe 相同输出或相同 PPL 的证据。

### 4.2 完整 runtime 吞吐快照

单位 token/s。warm runtime 包含 embedding、全部 transformer 层、final RMSNorm、LM head、greedy 与 FastRPC；排除外部 tokenizer、冷加载、ADB 和会话准备。Prefill 用 64/该次完整 prefill 时间，decode 用后续 decode token 数/对应时间；不是含冷启动的用户文本到文本吞吐，也不是“只统计 GEMM”。

| 模型 | 配置 | 实验证据 / 形状 | Prefill | Decode |
|---|---|---|---:|---:|
| Qwen3-1.7B | W16A16 | EXP-0218，M64+15，历史 | 793.14 | 9.12 |
| Qwen3-1.7B | W4A16 OPT2 | EXP-0260，M64+15，历史 | 1040.24 | 15.30 |
| Qwen3-1.7B | W4A8 U8 优化，无旋转 | EXP-0268，M64+15 | **2085.26** | **48.57** |
| Qwen3-1.7B | W4A8 SP2，无旋转 | EXP-0268，同轮 M64+15 | **2030.24** | **48.17** |
| Qwen3-1.7B | W4A8 R3，非 SP2 | EXP-0259，M64+15，历史 | 1703.36 | 47.87 |
| Qwen3-1.7B | W4A8 R3+R4，非 SP2 | EXP-0265，M64+15，历史 | 1581.07 | 46.06 |
| Llama-3.2-1B-Instruct | W16A16 | 本表未纳入已核实的对应正式速度记录 | N/A | N/A |
| Llama-3.2-1B-Instruct | W4A16 OPT2 | L32-0006，M64+7 | 1261.16 | 23.60 |
| Llama-3.2-1B-Instruct | W4A8 U8，无旋转 | L32-0012，M64+15 | 2042.63 | 46.08 |
| Llama-3.2-1B-Instruct | W4A8 SP2，无旋转 | L32-0012，同轮 M64+15 | **2004.83** | **46.09** |
| Llama-3.2-1B-Instruct | W4A8 SP2 + FP32 残差，无旋转 | L32-0018，M64+15 | **2069.70** | **42.51** |
| Llama-3.2-1B-Instruct | 上述路径 + 快速 R3+R4 | 当前论文工作实现；未做完整 frontend E2E | **N/A** | **N/A** |

这是不同阶段的证据快照，不是全部同轮、同质量、同 workload 的排名表。尤其不能把 EXP-0268 U8 与历史 R3/R4 的差值称为旋转的净开销。Qwen W16A16 的历史十轮测量协议也不同，保留其原始协议，不按当前 repeat10 重命名。

L32-0018 的同轮无旋转 SP2 对照为 2002.83/45.97 token/s；FP32 成本只用这一对照计算。完整模块账本继续保留在各实验 PROFILE 中，主文可展示配对变化与吞吐，附录提供完整模块表。

## 5. 当前数值问题如何影响论文表述

现有证据更符合“局部舍入扰动穿过离散阈值，随后沿层与 KV 轨迹放大”，不能简单概括为普通浮点误差线性累加。

L32-0023 的一个已定位例子中，R4 输出差 2.384e−7，跨过 SP2 阈值后重构码由 10 变为 12。后续量化与反馈放大差异：全 16 层独立参考比较的最小余弦为 prefill **0.5051**、decode **0.7883**，未通过 0.999 门槛；用实际前层输入逐层核对，局部最小余弦仍至少 **0.999999971**。局部通过不能覆盖全模失败。

精确 R4 重算的诊断对照恢复了该冻结 fixture 的 full16 一致性，支持上述误差传播解释，但该实现成本极高。它既不是部署方案，也不是精确旋转必然昂贵的理论下界；不应把它当低效基线来夸大论文加速倍数。

建议论文明确区分四种证据：

| 证据层面 | 当前可说什么 |
|---|---|
| 表示映射的正确性 | 固定 SP2 码本的整数分解、累加读出、重定标与冻结 oracle 有相应精确验证 |
| 执行与性能 | 已有 VTCM/所有权/中间 DDR 审计，以及指定 workload 的配对硬件延迟 |
| 快速旋转全模数值一致性 | 已知失败，用户接受作为研究实现；未改写为通过 |
| 模型质量 | W4A8/SP2/快速旋转未完成可用质量验收；不存在可报告的当前组合硬件 PPL |

旧的 Llama L32-0007 独立轻量设备评测中，匹配 BF16/W16A16/W4A16 PPL 为 26.6980/26.6917/31.0391（2048 targets）；这是旧配方的独立结果。它既不是完整 WikiText-2 validation，也不能与另一软件配置的 13.6347/17.6424 横向比较。本文不新跑 PPL，也不把软件质量与不同硬件合同的速度拼成一个点。

可直接采用的限制表述：

> The fast rotation path exhibits local rounding differences that can be amplified by discrete quantization boundaries across layers. We therefore report its measured execution cost separately from end-to-end numerical agreement and model-quality claims. The exact integer SP2 mapping and its matched performance studies are evaluated under their own frozen arithmetic contracts.

[全模误差及归因](/home/daniuniu/work/llama32-htp/docs/LLAMA32_ROTATION_FULLMODEL_VALIDATION.md)

## 6. 与相关工作的关系及创新边界

“并行计算和搬运”不是足够独立的创新点。FlashAttention-3 已通过异步机制重叠数据移动、矩阵计算与 softmax；我们应将其作为相关设计背景。[FlashAttention-3](https://arxiv.org/abs/2407.08608)

MARLIN 已强调低比特权重重排、反量化与流水，不能把低比特布局优化本身称为首创。我们的比较重点应是移动 HTP 的原生 W4×U8 操作数、HVX/HMX/DMA 分工、有限 VTCM，以及跨算子原生生产消费与 SP2 映射。[MARLIN](https://github.com/IST-DASLab/marlin)

APoT 已研究多个 2 的幂之和构成的非均匀量化。我们的 SP2 工作贡献在于将指定码本降低为既有 W4 乘法器能够执行的合同，并在 prefill/decode 上给出不同映射与实测成本，而不是发明非均匀幂码本。[APoT](https://arxiv.org/abs/1909.13144)

llama.cpp 已有 Snapdragon/Hexagon 后端，不能称“首个移动 NPU LLM runtime”。代码中继承或借鉴的 htp-ops-lib/FastRPC 基础设施应按实际来源保留归属；现有相关工作检查不足以支持任何“首个”宣称。[llama.cpp Snapdragon backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)

建议使用“we design / implement / demonstrate”，把贡献落到具体机制和对照证据上。相对外部系统的 SOTA、能效领先或同质量加速，目前均没有本轮可引用的匹配证据。

## 7. 可直接改写进论文的叙事

### 标题建议

**From Quantized Representations to Native Execution: Layout and Pipeline Co-Design for LLMs on Mobile NPUs**

中文：**从量化表示到原生执行：移动 NPU 大模型的布局与流水协同设计**。

### 中文引言草稿

低比特矩阵指令为移动端大模型推理提供了较高的潜在计算吞吐，但这一优势并不自动转化为完整推理的低延迟。矩阵引擎消费的是具有固定物理格式的操作数，而量化、归一化、门控、旋转与缓存管理通常沿逻辑算子边界产生数据。两者之间的格式转换、数据就绪延迟和有限片上存储中的生命周期冲突，会使矩阵计算之外的工作进入关键路径。

我们研究如何围绕硬件消费者重组这些边界。在一个由 DMA、HVX 向量单元和 HMX 矩阵引擎组成的移动 NPU runtime 中，我们联合设计原生操作数生产、tile 级发布与消费，以及缓冲区的复用和输出处理流水。进一步，我们将 SP2 非均匀激活重构为可由原生 W4×U8 乘法器处理的整数分量，在保持 packed W4 权重的同时，融合 SwiGLU 生产与 Down 投影，并分别利用 decode 的空闲物理行和 prefill 的异构重叠降低额外成本。

在 Qwen3-1.7B 与 Llama-3.2-1B-Instruct 的完整 warm runtime 上，SP2 相对各自对照的 prefill 额外延迟分别为 2.71% 和 1.89%，decode 分别为 0.82% 和统计上未见稳定变化。共享优化回移到普通 U8 的对照进一步表明，主要收益来自就绪顺序与物理接口的调整。我们还展示了 FP32 残差边界的完整模型执行成本，以及快速在线旋转的单层成本，并明确区分这些性能证据与尚未完成的全模数值及模型质量验收。

### English abstract draft

Low-bit matrix instructions do not automatically deliver low end-to-end latency for LLM inference on mobile NPUs. Operand preparation, physical layout conversion, and buffer lifetimes can delay the matrix consumer and dominate the critical path. We present a runtime design that jointly organizes native operand production, tile-level readiness, and DMA/vector/matrix execution within bounded on-chip memory. We further map a finite SP2 activation codebook to integer components consumable by native packed-W4 matrix operations, fusing their production with SwiGLU and using phase-specific schedules for prefill and decode. On Qwen3-1.7B and Llama-3.2-1B-Instruct, matched full-runtime measurements show prefill overheads of 2.71% and 1.89%, respectively, with decode overhead of 0.82% on Qwen and no statistically resolved change on Llama. A scheduling backport to the uniform-activation baseline separates shared pipeline improvements from the incremental SP2 cost. Additional experiments characterize FP32 residual boundaries and fast dense online rotations. We report execution efficiency separately from unresolved full-model numerical agreement and quantized model quality.

## 8. 按当前时间约束组织主文与图表

主文建议顺序：问题与硬件约束 → 物理接口及调度 → SP2 原生映射与双阶段计划 → 两模型完整 runtime 与公平归因 → FP32/旋转扩展 → 数值限制及相关工作。

优先使用已有证据制作四张图，不需要为此继续软硬对齐：

1. **表示到执行的数据流图。** 区分 SP2 索引、low/high tile、原生 W4 点积、整数合并和输出边界；旁边标注 FP32 残差的 Norm 后 A8 边界。
2. **Prefill/decode 两种执行计划。** 展示 Gate/Up 交错、HVX 生产、HMX 消费及 raw-output 两槽生命周期；示意时序明确标注为示意，不能用模块计数伪造实测硬件 timeline。
3. **Qwen 五臂公平归因图。** 旧 U8 → 多 HVX → 按 Up 就绪消费 → Gate/Up 提前交错，以及 SP2 对优化 U8 的增量。这是主文最重要的因果证据。
4. **两模型成本与扩展表。** SP2 的全模配对 CI 为主体；FP32 全模、旋转单层分面展示，避免暗示存在尚未测量的完整组合速度。

旧稿提出的完整 2×2 因子消融、长上下文、能耗及外部系统对照仍可增强论文，但目前没有就标为缺失，不能把已有顺序消融重命名为完整因子实验。若只允许一项后续补测，当前组合的完整 Llama frontend/E2E 是直接填补表格的候选；本次不启动，也不作为交付稿件的前置条件。

## 9. 证据入口与冻结范围

- 当前论文工作源码父提交：`cb8be962f3f31a1b89c8f8cd2bebb9cceef049a7`，`codex/llama32-no-rotation`。本稿只增加文档，不改变任何 native 数学、流水或默认开关。
- [Qwen 历史 recipe/速度清单](/home/daniuniu/work/llama32-htp/baselines/qwen3-frozen/manifest.json)
- [Qwen SP2 移植 EXP-0267](/home/daniuniu/work/qwen3-block-htp-project-memory/docs/experiments/EXP-0267.md)
- [Qwen 公平 U8 回移 EXP-0268](/home/daniuniu/work/qwen3-block-htp-project-memory/docs/experiments/EXP-0268.md)
- [Llama SP2 原生融合 L32-0009](/home/daniuniu/work/llama32-htp/docs/LLAMA32_SP2_DOWN_FUSED.md)
- [Llama SP2 完整优化 L32-0012](/home/daniuniu/work/llama32-htp/docs/LLAMA32_SP2_PIPELINE_DIRECTIONS.md)
- [Llama FP32 流水 L32-0018](/home/daniuniu/work/llama32-htp-project-memory/docs/LLAMA32_FP32_PIPELINE_FOLLOWUP.md)
- [Llama 旋转局部修复与速度 L32-0022](/home/daniuniu/work/llama32-htp/docs/LLAMA32_ROTATION_NUMERICAL_REPAIR.md)
- [Llama 全模独立数值失败 L32-0023](/home/daniuniu/work/llama32-htp/docs/LLAMA32_ROTATION_FULLMODEL_VALIDATION.md)
- [Llama 原快速路径恢复 L32-0026](/home/daniuniu/work/llama32-htp/docs/LLAMA32_H512_FAST_ROUNDING.md)

所有速度仍绑定原实验源码、二进制、输入、模型和计时协议。本次 paper 工作选择不等于重新封装或晋升旧实验默认配置，不引入新的硬件结果。证据文件索引及内容 SHA256 见同目录 `PAPER_STORY_RUNTIME_V2_EVIDENCE.json`。
