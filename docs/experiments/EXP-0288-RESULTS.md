# EXP-0288：Qwen3-0.6B 原始权重迁移与流水优化

结论：完成 no-rotation W4A8-SP2、FP32 残差链路及独立全模数值验证。保留 OPT1，停止本轮有界优化。模型质量/PPL 未验收，不自动替换已有1.7B或Llama基线。

## 正式端到端速度

SM8750 / HTP V79；完整 embedding、28层、final RMSNorm、LM head、greedy、FastRPC。M64+42、cache128、batch1、固定与真实greedy一致的轨迹；5轮short、10轮正式ABBA，每次repeat10；repeat1仅辅助。排除冷加载、外部分词、ADB和日志拉取。

| 配置 | Prefill token/s | Decode token/s |
|---|---:|---:|
| 初始正确移植 CONTROL | 2754.1715 | 92.4637 |
| OPT1：原生W4 prefill head | 3020.4273 | 92.4021 |

Prefill：候选/原版墙钟配对比 0.91184908，95%CI [0.90996052, 0.91365183]；吞吐变化 +9.6673%。10% gate 通过。
Decode：候选/原版墙钟配对比 1.00066666，95%CI [0.99877056, 1.00265854]；吞吐变化 -0.0666%。10% gate 通过。

## 长 decode 补充

M64+63、cache128，单次repeat10；独立参考覆盖全部64次生成。此项不是十轮正式配对统计。

| 配置 | Prefill token/s | Decode token/s |
|---|---:|---:|
| CONTROL | 2744.8741 | 91.5916 |
| OPT1 | 3021.5299 | 92.1258 |

## 迁移与优化

原始文件来自 Qwen 官方 ModelScope Qwen/Qwen3-0.6B，逐文件 revision、SHA256、大小固定于 D:/llm_exp/models/Qwen3-0.6B-origin/DOWNLOAD_PROVENANCE.json。全部11文件校验通过。从原始权重重新做现有C64增强GPTQ/per-output-channel W4与静态A8校准，SP2 mode8；未复用1.7B量化权重或尺度。冻结校准token语义与新tokenizer一致。

模型 residual hidden=1024，但Q/attention width=2048；FFN=3072，28层，16Q/8KV heads，head_dim128，vocab151936，embedding tied。拆分hidden与attention维度、修正VTCM规划和投影元数据，保留Q/K Norm及标准Qwen RoPE。0.6B编译身份ABI135，显式限制到本轮recipe。

以EXP0284保留的快FP32路径为父版本，避免EXP0285已拒绝的FP16/dispatch退化。仅一个性能候选：将Llama已验证的LM head原生W4、双缓冲及有效行路径用于0.6B prefill最终token。代表性配对记录中head命令594→149，展开ticks归零，weight DDR读取仍79006720B，VTCM峰值仍5744384B。完整互斥账本见MODULES.md，worker重叠计数不可直接求和。

核查确认：原生QKV/O/Down、SP2 decode两行共装、Gate/Up-SwiGLU依赖流、持久原生KV、decode有效行4、FP32向量残差路径均保留。没有发现其它可直接移植的明确1B遗漏；更深层调参不纳入本轮。1.7B兼容编译通过，恢复0.6B后二进制哈希与测量封存完全相同；未重新声称1.7B硬件性能。

## 数值与物理证据

首层/第14层/末层及连续3层prefill+decode独立输出通过。两完整版本均通过：64个生成步骤的完整28层递推最后token FP32 hidden逐值精确、final norm逐码精确、全词表参考选出的token/code精确、28层prefill原生KV逐字节精确。另核对完整prefill最终64x1024 FP32矩阵哈希。实际硬件日志的成功状态不单独作为独立正确性证据。

所有正式步骤：单次全栈RPC、单HMX所有者、8MiB申请/获得，计划峰值5744384B，中间张量DDR读写/spill=0，逐层与整体互斥账本闭合。零spill指显式中间张量合同，不声称消除编译器局部stack访存。

保留的失败与修复：初版KV包缺原生分段文件，停在host校验；长评测目标数DSP仍限定16，扩展为有界长度；全模首次独立分歧为layer13（从0计）K的一个码。旧CPU参考FP32 sqrt后再FP32除法发生两次舍入，而QHL直接rsqrt；修正为高精度求倒数平方根后一次FP32舍入，完整64步重新独立生成并与未改数学的硬件精确匹配。未注入真机中间值、未调整门槛；旧包/日志全部保留。参考修正不构成新的模型质量改善。

## 复现与保留

原始权重：D:/llm_exp/models/Qwen3-0.6B-origin。
最终部署包：D:/llm_exp/models/qwen3-block-htp/exp0288/frontend64-a03。
量化/校准：同实验models根目录，manifest.json/calibration.json；具体源码与输入SHA见日志。
CONTROL measured source: 85b9142987b8b37951e21a7a65d36aeb22a01f96。
OPT1 measured source: 85f7e2d9742d9c33523e3e0c6125402f75543156。
源码分支：codex/exp-0288-qwen3-06b-sp2。
构建：QBH_QWEN_MODEL_SIZE=0.6B bash scripts/build_qwen3_sp2.sh 28；硬件运行须先登记新实验并成功preflight。
SUMMARY.json统计、MODULES.md模块账本、implementation-audit.json优化核查、reference-correction-provenance.json参考修复、两套runtime seal及所有失败/正式日志均纳入EVIDENCE_SHA256.json。
