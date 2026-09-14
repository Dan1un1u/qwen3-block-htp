# L32-0018：FP32 Norm 出口与 O 流水优化，两阶段速度门槛通过

源码 `e3d065a515221aa1e97c63fa4d58d23e434c45df`，分支 codex/llama32-no-rotation。用户授权继续速度优化，
保留 L32-0012 SP2 权重/尺度，沿用 L32-0016 FP32 残差合同，无旋转或 PPL/校准。

## 实现与归因

Norm 原来每32个A8输出经向量栈写入、标量搬运，并立即做向量归约/标量提取
来判断除法舍入临界值。现在直接用带掩码的 HVX store 写32字节 native 段，
每行汇总临界标记，复用已完成转置的 VTCM tile 保存2048个字节标记。
只有该行存在临界值才同步到标量，依照原先非融合FP32乘法、除法和roundf修复对应位置。
因此减少HVX与标量之间的同步和往返；FP32逐通道累加顺序不变，没有树形归约、FMA或门槛放宽。
SDK模拟器分段诊断显示输出量化阶段为主要开销；模拟周期只用于定位，不作为真机速度。

O prefill 复用 Down 的两槽 raw accumulator流水：HMX原生packedW4单次乘法，
三次signed24 retain-store后发布；HVX并行完成zero compensation、FP32缩放与残差加法。
消费者最终写入/barrier后确认槽可重用，每条HMX命令在其bias DMA槽复用前排空。
Down继续原SP2两pass及decode合并矩阵方案，O decode沿用原单行实现。

## 算术、物理及失败记录

第0/7/15层M64+decode、连续3层、全16层FP32输出与封存oracle逐元素完全一致。
完整生成token ID和logit码一致；正式20次CLI的320个token边界全部精确且物理检查通过。
本轮29次硬件CLI，28成功、1次O调度状态检查失败；成功共364个token边界。
另有1次ADB部署连接失败，尚未启动硬件CLI，重试通过。
O首候选错误地把已join的attention线程池历史dispatch count视为忙锁；
改为逐worker检查idle job，未放宽并发安全检查。全部失败记录/二进制均保留。
VTCM峰值FP32为8098272B，control为8229344B；8MiB限制保持，
中间DDR读写/spill=0，packedW4无展开、无FP16 HMX回退、无R3/R4。

## 固定十轮全模型结果

同一构建原0012-SP2作control，M64+15连续decode、容量80，AB/BA十轮；
bootstrap20000次，seed17017。完整Host墙钟包含embedding、16层、finalnorm、head、
greedy及FastRPC，排除加载和外部tokenizer。repeat1仅辅助。

| 阶段 | 0012-SP2 token/s | FP32优化后 token/s | Host耗时变化 | 配对ratio95%CI | 10%gate |
|---|---:|---:|---:|---|---|
| Prefill |2002.83|2069.70|−3.23%|[0.953911,0.977237]|通过|
| Decode |45.97|42.51|+8.13%|[1.071833,1.092762]|通过|

L32-0017的1639.74/42.26token/s是非配对历史参照。该次prefill两处Norm合计10.221ms，
本轮2.542ms；O由1.923ms降至1.579ms。Decode两处Norm由1.958ms降至1.746ms。
本轮相对同次control仍有O+0.859ms、post-Norm+0.581ms、attention+0.313ms，
但Gate/Up＋SwiGLU观测少2.452ms（0017已经存在），故净prefill略快。
后者属于不同数值轨迹/布局下的总体观测，不能说FP32算术本身更快，也不能归因于本轮新核。

## 结论与下一步

L32-0012速度参照上的无旋转FP32残差/Norm方案已通过双阶段10%性能门槛。
这不是量化精度或可用文本验收；既有固定尺度文本仍不可用，本轮不跑PPL。
不自动晋升默认baseline，也不恢复旋转/校准任务。
若继续速度研究，decode约1.49ms额外Norm仍是主要余量，prefill主要剩O的raw-store出口；
目前无需为达到既定10%目标继续改动数学合同。可先固定此次速度结论，再讨论下一阶段融合。

模块总表 `/mnt/d/llm_exp/results/llama32-htp/l32-0018/PROFILE.md`，原始固定十轮 `formal-a01/`，摘要 `summary.json`。
证据ledger `/mnt/d/llm_exp/results/llama32-htp/l32-0018/evidence-ledger-a01.json`，181个结果文件及7个复用模型manifest，
SHA256 `5572319f6bb94f623089199ef729a13c3c5d20b82134be14e75028ac707bd734`。无活动硬件任务。Qwen与Llama rotation分支冻结。
