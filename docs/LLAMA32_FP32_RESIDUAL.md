# L32-0016：0012-SP2 的 FP32 残差精度原型

第一阶段实现与评测已完成；算术合同通过，模型质量仍不可用。
源码分支 codex/llama32-no-rotation，实测提交 `e3cf74916aebbc9254615d68b1c4c135b43fee03`。
L32-0012 速度基线及其它 recipe 保留原结论。本轮没有正式 profiling，
没有性能验收或新基线晋升。L32-0015 旋转继续 deferred。

## 已实现的数据流

原始 FP16 embedding → FP32 hidden → FP32 RMSNorm → 原冻结 A8
→ 原 W4 QKV / RoPE / 整数 attention / KV8 → 原生 W4 O 的完整整数累加结果
→ FP32 缩放及残差相加 → FP32 RMSNorm → 原冻结 A8 → 原 W4 Gate/Up
→ 原 SwiGLU/SP2 LUT（241-level radix256）→ 原生 W4 Down 的完整整数结果
→ FP32 缩放及残差相加 → FP32 跨层 hidden。最后 FP32 final norm → 原冻结
A8 → 原 W4 LM head → 原 U8 logits / greedy / NLL。

运行开关 `QBH_LLAMA_FP32_RESIDUAL=1`，配合 `QBH_LLAMA_SP2=8`、direct_n
mask63 及 native MLP/QKV/O 边界；Llama ABI130。R1/R2/R3/R4 均关闭。
权重、SP2码本/alpha、A8 scales 全部固定，无重新量化、平滑或重新校准。
高精度输出从 HMX retain 累加器读取，绝非先裁剪 U8 再转 float。
Host 对所有可能 U8 输入证明 signed24 部分累加界、SP2 合成 S32 界；
没有 S8 权重展开或浮点 GEMM 回退。O/Down 的 FP32 add 融合在 projection
出口，归入对应 projection ledger，残差阶段不重复相加。

VTCM residual 从128KiB扩为512KiB；无旋转 native U8 activation 保留量从
1MiB缩为512KiB。实测峰值 **8,098,272 B / 8,388,608 B**，中间 DDR/spill=0。
首版采用易审计的 scalar FP32 Norm/epilogue，暂不使用原 Down U8 异步
消费 epilogue；速度优化不是本轮验收项。Native W4 乘法及 SP2 producer 保留。

## 真机正确性

- 单层0/7/15：各 M64 prefill + decode1，FP32输出逐值一致、KV精确。
- 连续3层、完整16层：M64+decode1，FP32输出逐值一致、KV精确。
- 显式舍入修复后 full16 回归再次精确通过（chain16-fp32-a02）。
- 原 FP16 embedding 到 head 的完整16-token生成：token ID和U8 logit码均精确。
- 同构建原0012-SP2对照也通过完整生成算术检查。
- 两组完整生成均16层/step，无 FP16 HMX tile、权重展开、中间DDR/spill，
  ledger_unattributed_ticks=0，R3/R4调用=0。

## 保留的失败及修复

layer0-fp32-a01：FP32输出已精确，但测试包漏写decode新增KV参考，CLI退出1。
新建a02包补全后通过；原结果保留。一次chain16包尚未写出manifest时的
提前启动在部署前退出，未产生硬件调用，随后等待封装完成后正常执行。

frontend-fp32-a01：实际执行16token完成，参考code从step0、token从step5
开始分歧。审计证明 finalnorm 在实际hidden上精确，KV从layer13开始分歧。
反汇编确认新 scalar Norm 的平方和被合并成 FMA，与参考的分步FP32舍入
不同；固定 `#pragma clang fp contract(off)`，并统一 reciprocal/multiply顺序。
此修复用于算术合同对齐，不宣称 FMA 数学精度更差。修复后16层所有prefill
KV、final hidden、head code以及完整16token序列全部精确。旧失败包/二进制/
结果未删除。诊断模式的额外hidden/Norm输出单独登记，不作为速度结果。

## 同集轻量 PPL

同一构建、同一冻结 EN/ZH Wikipedia heldout：128篇，M64上下文，每篇16个
teacher-forced目标，共2048token（EN/ZH各1024）。不是完整WikiText-2 PPL。
数据/teacher来自既有 llama32-wiki-quant-v1，保持文档及32gram训练/评测隔离。
两组评测均2048个唯一sample/step、全有限、硬件/缓存检查通过。

| 配置 | 总PPL | EN | ZH |
|---|---:|---:|---:|
| 原始 BF16 teacher |26.6980|17.7874|40.0723|
| 0012-SP2 原U8残差 |130712.2867|129575.7949|131858.7466|
| SP2 + 本轮FP32链路 |1766.7649|645.9578|4832.2942|

总PPL降低98.6484%，但仍为BF16的66.18倍。新方案的固定prompt输出仍含重复
`user.  -  - ...`，不能认为文本可用。工具的`pass:true`表示A8无质量门槛时
执行/算术/评测完整通过，绝非质量验收通过。

收益同时包含embedding恢复、O/Down出口、残差、Norm精度变化，不能全部
归因于残差存储单项。冻结旧尺度是第一阶段控制变量；本轮未校准/选择权重。
参考范围审计中layer7旧残差U8步长约3，O/Down幅度约1，说明量化分辨率会
吞掉小更新；这不是裁剪溢出或完整因果消融的结论。

## 后续边界

FP32精度原型已完成。继续修精度时，应先在训练校准集按新FP32数据流检查/
重校准A8边界，分别定位Norm/QKV/GateUp/SwiGLU、attention/KV和head。
保留本轮固定尺度对照；FP16残差存储、R3/R4融合及性能优化为独立后续工作，
不可把它们与本轮控制混为同一改善。当前没有证据支持直接接受A8模型质量。

## 可复核证据

结果根目录 `/mnt/d/llm_exp/results/llama32-htp/l32-0016`：
`summary.json`、`block-verification-a01.json`、`chain16-fp32-a02/result.json`、
`frontend-fp32-a02/result.json`、`frontend-base-device-a01/result.json`、
`frontend-physical-a01.json`、`remote-verification-a01.json`。
归档构建 `build-e3cf749-16/`。权重包 `/mnt/d/llm_exp/models/llama32-htp/l32-0016/`。
Evidence ledger SHA256 `69b591d67d1ce3f45a0e2cc5e467f7b2d90ccddff854a581d79637769ec77de8`（192个结果/二进制/参考文件及模型manifest索引）。
所有发布到设备的最终两组package文件、三个二进制和heldout文件远端hash已核对。

复用工具：`tools/llama32_fp32_residual.py`（prepare/chain/frontend/basefront/run）、
`tools/run_llama32_frontend.py --fp32-residual`。新 run 必须重新登记实验并preflight；
不可重写这些封存 attempt/model 目录。
