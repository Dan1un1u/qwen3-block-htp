# 无旋转 SP2 / FP32 残差论文消融计划

日期：2026-09-15。状态：用户批准执行（2026-09-15 修订）；各阶段分别注册并经 preflight 后执行。
用户已验收 EXP-0272，并批准消融表，但取消 W4/S8 展开对照和整数残差对照，新增 log2 softmax / FP 向量 softmax 消融。

## 1. 冻结对象及验收

| 模型 | 论文工作基线 | Prefill token/s | Decode token/s | 测量范围 |
|---|---|---:|---:|---|
| Qwen3，28 层，hidden2048 / FFN6144 | EXP-0272，SP2mode8 + FP32 residual mode2，无旋转 | 1849.4159 | 47.0701 | M64+15，cache128，冻结 EOS 前缀 |
| Llama-3.2-1B-Instruct，16 层，hidden2048 / FFN8192 | L32-0018，SP2mode8 + FP32 residual mode1，无旋转 | 2069.7026 | 42.5101 | M64+15，cache80 |

两行各自绑定原始实测，不能作同 workload 的跨模型速度排名。完整 warm runtime 包含 embedding、全部层、final norm、LM head、greedy、FastRPC；排除外部 tokenizer、冷加载和会话准备。两个 W4A8 配置均未完成模型质量验收。

Qwen 测试源码 5f859339a62461b3c6fb7dc99c9dc243c1385c45，闭合源码 149b7f72eb9853874c44025153d32d22229b61d6，分支 codex/exp-0272-fp32-fullmodel-norm。
Llama 测试源码 e3d065a515221aa1e97c63fa4d58d23e434c45df，当前分支 codex/llama32-no-rotation，HEAD 0ae07a92e4d99fd3fd23c89adc3ce4a2b44f7b4f。
L32-0028 已审计后续旋转/撤回修改没有遗漏的已验证 OFF 优化；Qwen EXP0269–0272 新方法尚未在 Llama 上验证，不能把 Llama 历史速度标到移植后的新构建。

Qwen 对同轮整数残差控制的完整 Host wall 比率：prefill 1.08996576，95%CI [1.08725796,1.09257814]；decode 1.01352815，[1.01015659,1.01626911]，满足正式 10% gate。相对同轮旧 FP32，prefill 降低 0.7297%，decode 无稳定变化。不得把跨次测量差异全部归因于新核。
历史 EXP0268 无 FP32 残差配置保留为参照，当前论文 FP32 工作基线更新为 EXP0272。本次接受不改写封存 SUMMARY/REPORT 中当时尚未晋升的记录，也不改二进制默认参数。
旧基线摘要中 Qwen3-0.6B 标签与源码的 hidden2048/FFN6144 及旧论文 Qwen3-1.7B 标签不一致；本计划使用已验证结构与实验 ID 标识。论文定稿前应由原始 checkpoint config 与 shard provenance 统一名称，不修改旧权重或证据 hash。

## 2. 论文主线

核心问题：原生低比特矩阵指令的优势，如何穿过非线性、量化、格式转换及残差边界，成为完整模型的低延迟？

建议论点：联合设计数值表示到原生乘法器的映射、消费者所需的物理张量接口，以及有限片上存储中的数据就绪顺序，能够保留 packed W4 主路径，并把新增 SP2 与高精度残差工作的较大部分移出关键路径。

三项方法贡献：
1. 指定 SP2 码本的精确整数映射：v=l+256h，high 采用 h+128 的 U8 输入，列和补偿 -32768 sum(w)。两个整数分量复用原生 W4×U8；码本、进制分解不是本项目发明，贡献在其完整硬件实现与阶段特定执行计划。
2. 跨算子原生物理接口：SwiGLU 直接生产 low/high HMX tile，明确有效行、发布粒度与缓冲生命周期。原生格式对齐包含布局与就绪协议，远多于地址对齐。
3. 以消费者就绪为目标的 HVX/HMX/DMA 调度：Gate/Up 提前交错，raw accumulator 两槽输出处理，decode 利用闲置物理行共装 low/high。Prefill 已用满物理行，依靠重叠覆盖额外计算；两阶段不能用同一个解释。

FP32 残差作为上述机制的精度边界扩展：原生 W4×A8 点积与残差存储 dtype 可以分开选择，Norm 后再 A8。现阶段只证明实现与成本，不宣称恢复 PPL。R3/R4 不进入本轮主线或实验矩阵。

## 3. 计划实验清单

所有序号为本方案编号，不是已注册的 Experiment ID。主消融固定 SP2+FP32 数学合同，从当前源码显式关闭机制；不直接切换旧版本充当单因素对照。

| 编号 / 优先级 | 要检验的机制 | 对照设计 | 主要证据与范围 |
|---|---|---|---|
| A0 / 必做 | 基线可复现且没有暗中换路径 | 两个冻结基线；新实验构建的 all-on 与冻结输出核对，复测 warm E2E | 模型/源码/二进制/参数 hash、输出与物理审计、完整模块表、token/s |
| A1 / 核心 | 原生格式和异构流水是否分别有效、是否互相增强 | 2×2：直接原生 tile / 紧凑重构值后消费端 pack × 算子级发布 / 跨算子分块发布；四格保持相同 SP2 计算与各核内部合理流水 | 两模型 FFN 组件与全模型。直接效应、交互效应、物化字节、关键路径；其他路径和 FP32 Norm 固定 |
| A2 / 核心 | Prefill 的收益究竟来自哪段重叠 | 在 all-on 上分别关闭 Gate/Up 提前交错、O/Down raw epilogue 重叠、权重 DMA 预取；每次只关一项 | 两模型单层定位；三项中有代表性的全模型结果。固定 HMX 工作量/权重搬运字节，采集 ready/wait 与时间线 |
| A3 / 核心 | Decode 的 SP2 低开销是否主要来自闲置物理行 | 当前 low/high 共装一条矩阵流，对比相同 W4/SP2 的两条独立矩阵流；保留相同 epilogue 与预计算列和 | 先 Down 组件，再两模型完整 decode。HMX tile/命令数、权重搬运/复用、Host wall；不能只给 kernel 时间 |
| A5 / 公平成本 | 普通 A8 与 SP2 成本 | 固定 FP32 残差，优化普通 A8 与 SP2 两臂，共享调度优化同等提供 | 两模型全模配对速度；旧整数残差上的配对结果仅作历史证据，非当前合同测量 |
| A6 / 可叠加性 | 为何单层与全模退化不同 | 复用 A1/A2 的关键开关，测 1 层、连续 3 层和完整 28/16 层；另记录完整 frontend 固定成本 | 同层输入 replay 的局部比较 + 各自真实 full E2E；逐层增量、Norm/FFN/head/Host 边界。禁止单层外推全模 tps |
| A7 / 补充 | 当前 FP32 优化具体做对了什么 | Qwen ordered reduce8 与双流交错 reduce16；Llama 保留当前 FP32 核对照，有适用性证据后才注册移植 | Qwen EXP0272 已有同轮完整证据，直接引用；不将尚未测试的移植称为 Llama latest |
| A8 / 附录 | 收益适用的 shape 边界 | 主实验完成后，选择已支持的额外 prefill 长度/更长 KV，并以同一 all-on/control 配对 | 至少一个较小/较大有效 M 或较长上下文；先检查原生行填充、8MiB、正确性 fixture，避免为了扫参数重写 runtime |

A1 的主对照不关闭 GEMM 内部 DMA/HMX 重叠、HVX 向量化或已有正常双缓冲。调度因子界定为算子完成发布与跨算子分块就绪发布，而非设备全串行。格式因子候选为每元素一份 uint16(v+32768) 的紧凑重构值接口与直接 low/high HMX-native 接口；两者对应相同冻结 SP2 码值。紧凑接口由 HVX 生产，消费侧用向量拆分/pack，禁止先写 native 再刻意转回通用格式制造无效工作。A2 独立检验局部依赖/预取优化，单独关闭机制只作归因，不把最慢诊断臂当作竞争性总加速分母。

如果 A1 关闭格式融合需要超出片上存储，应先复用已死亡的缓冲区或限于 FFN 组件，不为了凑齐全模型四格违反物理合同。完整 2×2 未完成前不能把已有顺序消融重命名为因子实验。

A4 已取消，不运行 W4/S8 对照。

A5 改变数值合同，不能要求跨格输出相同；每格对自身参考验证。固定权重 codes/scales、输入和非目标边界，明确记账必要的 qparams/LUT 差异。增加固定 token replay 的 kernel 归因，以免自由生成后的 token/KV 差异被当成单个改动的计算收益；保留真实 greedy E2E 作为产品执行成本。

## 4. 测量协议草案

- 先按未来实验 ID 完成 bootstrap/preflight，检查冻结源码、模型、开关及数值/物理门槛；此次已批准执行，先由 EXP0273 登记 Qwen 基线复现与后续数值/计时接口准备；后续阶段在该批准范围内另行冻结协议，无需重复请求许可。
- 主测量沿用各模型 M64+15 和原 cache/prefix，不将两个不同模型的绝对速度相除声称机制加速。
- 固定十轮配对/多臂平衡顺序，repeat10 为主，repeat1 辅助。所有成功/失败样本与退出原因保留；不能测到显著就停止。
- 机制消融是主动关闭优化，用户批准其完成测量不受“慢于 10% 就停止”的候选筛选规则约束；这在实验注册时明确为诊断对照，不能事后给失败优化放宽 gate。数值与物理约束保持。未来候选晋升仍按正式 full-model 10% gate。
- 主结果完整 Host wall、模块 μs/占比、prefill/decode token/s、配对 95%CI。A1 预先定义 2×2 延迟交互项及配对置信区间，不用顺序瀑布图替代交互估计。
- 归因证据为 HMX tile/命令、实际 DMA 字节、HVX 转换工作、ready/done 等待及 VTCM 峰值。没有可靠计数器时不虚构利用率、bank conflict 或 pipeline stall。
- 时间线使用单独诊断 run，开关 profiler 导致的额外开销不进入正式速度；同一时钟和真实事件区间，不把重叠模块计数相加伪造时序。
- 等价消融必须维持原输出、物理 KV、溢出界与 FP32 舍入合同；更改求和顺序获取加速不属于此批消融。各边界 oracle 与全模型冻结输出证据区分登记。

## 5. 实施顺序与论文图表

建议最小主文包：A0 → A1 + A2 + A3 → A5；在这些测量中覆盖 A6 的关键 1/3/full 点。A4 取消；A7 只引用已完成的同 FP32 合同证据，A8 最后做。新增 A9 softmax 对照为主文实验。不要先拉开所有参数的笛卡尔积。

论文图：
1. SP2 表示→low/high 原生 tile→packed W4→FP32 残差→Norm后A8的数据流与生命周期。
2. 两模型 A1 四格因子图和真实 prefill 三引擎时间线，显示独立及交互收益。
3. Decode 共装/两流对照图，附 HMX 工作及权重字节，而非仅吞吐。
4. A5 完整模型成本图与 A6 层数扩展图，附原始模块总表。

已有反例可进附录：Llama 单独 gather 重排未见稳定 E2E 改善；Qwen C2 合并写回虽精确却更慢，已撤回。这支持按真实消费者路径和调度结果选择优化，不能宣称减少指令/存储必然变快。

## 6. 证据入口

- Qwen memory docs/experiments/EXP-0272-RESULTS.md 及 EXP-0272-PROFILE.md。
- Qwen /mnt/d/llm_exp/results/qwen3-block-htp/exp0272/EVIDENCE_SHA256.json，SHA256 7fa56656cbc385a2733146df45d1c5b399a302a0a16898e9fc135ae496016d61。
- Llama /mnt/d/llm_exp/results/llama32-htp/l32-0018/evidence-ledger-a01.json，SHA256 5572319f6bb94f623089199ef729a13c3c5d20b82134be14e75028ac707bd734。
- /home/daniuniu/work/llama32-htp-project-memory/docs/LLAMA32_FP32_PIPELINE_FOLLOWUP.md。
- /home/daniuniu/work/llama32-htp/docs/LLAMA32_SP2_PIPELINE_DIRECTIONS.md。
- /home/daniuniu/work/llama32-htp/docs/PAPER_STORY_RUNTIME_V2.md 为历史叙事，本计划以新无旋转基线替换其主线；不重写旧稿或其证据 hash。

## 2026-09-15 用户修订与 A9 softmax 协议

- A4 原生 W4 / 在线 W4→S8 对照取消；不可推断 HMX 内部物理实现。
- A5 取消残差维度，全部主文消融固定 FP32 残差，普通 A8/SP2 两臂。用户认为高精度残差才是合理部署合同。历史整数残差结果保留原记录，不再安排其新测量。
- 其他消融及诊断慢臂不受候选10%停止规则约束的测量方案已批准，数值/物理合同不变。
- 新增 A9：现有 low-bit log2 softmax 对比标准 FP32 HVX 向量 softmax。使用相同实际 HMX QK score 输入、其固定尺度、因果 mask、KV 长度和 native probability 输出布局。FP 对照对未进行 log2 整数 exponent rounding/cap 的 score 差值计算 max/exp/sum/normalize；最终统一量化至现有 probability U8 接口，后续整数 AV 不变。此实验不是完整浮点 attention，也不声称 FlashAttention 性能。
- FP 实现必须为合理向量 exp/归约实现，不以逐元素 scalar expf 充当最终速度基线；标量/Float64 仅用于独立参考。两臂数学近似不同，分别验证自身参考，不要求两臂输出相同。
- 单独记录 FP 概率相对 Float64 的误差、row sum/mask、U8 输出及 AV 误差。目标 FP softmax 概率 max_abs<=2e-6、row-sum error<=2e-6，所有值有限，masked 精确零；若逼近达不到则修复 FP 实现，不放宽门槛。输出量化临界值单独记录/验证。
- 独立组件固定输入测 softmax 核心、边界转换与 native pack；完整集成报告 attention 和全模型 Host wall/tps。固定 token replay 用于路径归因，实际 greedy E2E 单独报告。
- 第一阶段执行 A0；A1/A2/A3/A5/A6/A8/A9 是已批准后续任务，不能以 A0 完成宣称整批实验完成。

## A0 Qwen 执行记录

EXP0273 已完成 Qwen 基线复现：1849.6557/47.2787 token/s，输出与封存一致，物理审计通过。无新对比结果；Llama A0 和其余已批准消融待执行，后续无需重复申请批准。完整记录 docs/experiments/EXP-0273-RESULTS.md。

## 2026-09-15 基线公平性澄清与 SwiGLU 接口核查

用户质疑不能专门负优化慢基线。主对照定义收紧为“优化过的模块化算子实现”，不得全局串行化、逐元素标量化、刻意劣化 DMA 粒度/线程数、添加多余同步/DDR往返或 native→通用→native 无效 roundtrip。两边固定数学合同、FP32残差、资源上限、同等合法调优预算，正式测试前锁定各自配置。消费侧转换允许与 DMA 等独立任务合理重叠，不人为禁用。

四格：紧凑重构值+算子级发布，native输出+算子级发布，紧凑重构值+分块发布，native输出+分块发布。前两格的每个算子内依然使用优化后的向量化和搬运/矩阵流水。若四格不能在同一合法工作范围内构造自然可用接口，不为统计表强造劣质实现；报告两组局部消融及实际缺失的交互证据，不能称完整2x2。历史 L32-0012 m5/6/7/8 和 Qwen EXP0268 可证明自然演进来源，但其旧整数残差计时不直接充当新 FP32 消融结果。逐项关机制是消融解释，不是“比优秀系统快多少”的竞争性 baseline。

只读核查 Qwen dd1e3a56a5db61877cb160fb60718c4b00c33e98 与 Llama0ae07a92e4d99fd3fd23c89adc3ce4a2b44f7b4f：src/dsp/mlp_u8.c 文件完全相同。qbh_mlp_gate_up_sp2_lut_pipelined_hvx（line183）从已量化 Gate/Up 码查 LUT，表项为 v+32768，packe/packo 直接写 low l / high h+128（line198-200）。prefill 调用处按 HMX tile 地址输出（Qwen block_imp.c4555）；decode high=middle+128（9928），按有效物理行共装两分量。Down 原生指令直接消费 low/high；prefill 两流，decode 共装一流。Qwen llama_fp32_residual.inc14-28 完成 low+256high-32768sum(w)、缩放和 FP32残差累加；Down 输出不是SP2。

这已实现冻结输入码/查表合同下的 SwiGLU+SP2+消费格式生产融合，不是硬件原生SiLU/SP2指令，也不代表零转换指令或零开销。两张U8操作数合计2bytes/元素，不得写成1byte的SP2索引被HMX直接消费。该接口已经是两模型当前方法的一部分，无需再实现一次。此轮仅澄清文档，无新源码/设备测量。

## Full-model clarification and execution
User confirms B00/B10/B01/B11 are complete-model configurations, not an FFN-only experiment. EXP0274 owns the first Qwen factorial; its enumerated interface and schedule sites apply throughout all28layers, frontend/head remain included. Per-module counters only explain complete Host effects. Preserve intrinsic optimized vector/GEMM/DMA kernels in modular baseline.
