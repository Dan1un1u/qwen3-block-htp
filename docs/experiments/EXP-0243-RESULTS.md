# EXP-0243：固定前缀隔离有效，但 W4A8 仍未通过精度验收

在冻结 C64 per-output-channel W4 权重、原始 gamma、embedding 和 LM head 的条件下完成。未进行旋转、训练、分组、混合精度部署或设备测试。开发集在五种前缀 × 三种静态 U8 策略中选择 1 个 EOS token（151645）+ MSE clipping；其他 recipe 未改变，未提升任何 Selected baseline。

最终结果使用提前冻结并独立审计的 512 篇新文档、8,192 个评分 token，中英 × 新闻/百科四组等权。每篇正文保持 64 context + 16 targets，前缀不计分、不替换正文。实际执行 prefill64 后 15 次 teacher-forced 追加，前缀与正文 KV 真正以 uint8 存储，读取时用于 FP16 软件 attention。

| 配置 | 独立复核 PPL |
|---|---:|
| F16A16，无前缀 | 25.4912 |
| W4A16 C64，无前缀 | 27.3164 |
| F16A16，EOS 前缀 | 27.5411 |
| W4A16 C64，EOS 前缀 | 29.1041 |
| W4A8，无前缀 min-max | 151,935.9505 |
| W4A8，无前缀 p99.99 | 34,193.3632 |
| W4A8，EOS + MSE | 39.2779 |

同 EOS 前缀下，A8 相对 C64 A16 的 PPL 增加 **34.96%**，配对文档 bootstrap 95% CI **30.78%–39.46%**；相对 F16 增加 **42.62%**，95% CI **37.38%–48.24%**。所有语言/领域组也未满足 10% 门槛。总体 5% / 分组 10% 的精度验收失败。

EOS 改变了条件分布：它使 F16 的 PPL 增加 **8.04%**，使 C64 A16 增加 **6.54%**。相对无前缀 F16 的总差距是 **54.08%**；这与同前缀下的量化损失分开报告，不能只展示从崩坏值到 39.28 的改善。

## 首位置机制与剩余问题

F16 和 C64 的 16/16 个诊断窗口都在第三层（L02）出现超过 1 万的首 token SwiGLU 激活。实际完整聊天模板也触发；保留 KV 连续解码没有再次触发同等级尖峰，清空 KV 从后半段重新开始又触发。因此它不是 W4 才创造的现象。这支持首位置/历史状态相关机制，但不单凭这些观测声称已证明 attention sink 的完整内部因果链。

EOS 的在线正文 L02 SwiGLU 最大绝对值从 16,688 降至 21.578，残差 min-max 步长从 54.548 降至 0.1574，16,384 个校准 token 中该层 >1000 的数量降为零。普通句号/高频词前缀仍会遇到重复 token 导致的残留巨值，不能认为任意一个前缀都等效。

剩余异常值分布在后续层与大量正文位置。EOS 下 L27 SwiGLU 最大绝对值仍为 2,158，元素绝对值中位数约 0.152（完整直方图估计），1,892/16,384 个 token 超过 1000；L27 residual 最大绝对值 4,628。统计长尾提示值得优先检查的边界，尚未通过消融证明这些边界分别贡献多少 PPL 损失。

![EOS 后各层的 SwiGLU 长尾](D:/llm_exp/results/qwen3-block-htp/exp0243/figures/remaining_layer_tails.png)

## 实现与验证边界

32 组配置均通过首 4 篇哨兵的重复、未来 token 因果对照和独立 CE 检查；全量逐 token NLL/PPL 汇总另做独立复算。前缀来自固定 token 本身，没有正文/未来信息。15 份前缀 U8 文件序列化往返精确；1-token EOS KV 原始载荷为 57,344 bytes（56 KiB），不代表完整设备成本。

标量 U8 oracle 精确；FP32 attention cache 代数 oracle 最大误差 2.384e-7，小于预先声明的 1e-5 容差。额外的真实模型交叉检查使用标准浮点 DynamicCache 存储同样 QDQ 后的数值，仅作为参考，在相同逐 token 调用形状下，logits 和全部 28 层解码后的 K/V 与真实 U8 cache 完全一致。没有部署 FP16 prefix KV。

批量与逐 token 的 FP16 路径并非逐位相同：EOS A16 四文档检查最大 logit/NLL 差 0.08594/0.03746；EOS MSE A8 为 12.1953/4.03073。相同调用形状的缓存交叉检查精确，结合 FP32 代数检查，支持量化阈值放大有限精度差异的解释；不能据此宣称软件 QDQ 等价于 DSP/HMX。最终 PPL 始终使用真实逐 token KV 路径，不把开发集批量得分当作部署验证。

辅助交叉检查第一次与主评测同时加载时显存不足；失败日志保留，待主评测结束后使用完全相同代码串行重跑通过。没有更换任何评分、权重、参数或门槛。

## 建议下一步

先在已隔离首 token 的基线上做按边界/层段的 A8 误差消融，重点检查后段 SwiGLU、Down、残差与 head input。A16 恢复只作软件因果对照，不作为混合精度部署。由贡献大小决定优先局部 R4/通道缩放还是 residual R1；单独 R4 的等价变换不改变原坐标 Down 输出，不能默认解决 residual 量化。另需评估已有提示词头作为前缀的适用性，避免额外 EOS 的条件分布代价；重新选择或调整均需新开发/最终数据协议，不回调本轮最终集。

本轮在有效证据、精度失败状态结束；下一方法和设备阶段未启动。原生 HMX W4、per-channel 权重和 >10% 单层延迟停止门槛保持。新 E2E prefill/decode token/s 均为 N/A；完整 N/A profiling 与历史参照见 FULL_PROFILE.md。

# Detailed numerical appendix

Selected on development only: `eos` [151645], `mse`. Frozen per-output-channel C64 W4 weights; static U8 online boundaries and actual U8 prefix/body KV storage. Software FP16 arithmetic is not DSP/HMX bit equivalence.

## Fresh final PPL

| Configuration | PPL | en wiki | en news | zh wiki | zh news |
|---|---:|---:|---:|---:|---:|
| C64 empty a16 | 27.3164 | 17.4892 | 26.8563 | 27.3335 | 43.3690 |
| C64 empty minmax | 151935.9505 | 151935.9505 | 151935.9505 | 151935.9505 | 151935.9505 |
| C64 empty percentile | 34193.3632 | 29127.0283 | 58940.1806 | 23268.6287 | 34220.6968 |
| C64 eos a16 | 29.1041 | 19.6462 | 29.3593 | 28.0468 | 44.3513 |
| C64 eos mse | 39.2779 | 25.5018 | 42.4476 | 38.9285 | 56.4808 |
| F empty a16 | 25.4912 | 16.5331 | 25.4611 | 24.3074 | 41.2658 |
| F eos a16 | 27.5411 | 18.6664 | 28.5094 | 25.3799 | 42.5977 |

## Paired attribution

| Comparison | PPL ratio | 95% paired document CI |
|---|---:|---|
| selected_vs_same_prefix_C64 | 1.349566 | [1.3078429538905885, 1.3945561200728385] |
| selected_vs_same_prefix_F16 | 1.426155 | [1.3737647313287151, 1.4824214182667432] |
| selected_vs_empty_F16 | 1.540840 | [1.4842917922440726, 1.6023659517909483] |
| C64_prefix_conditioning | 1.065445 | [1.0553534964843976, 1.0761752157336153] |
| F16_prefix_conditioning | 1.080416 | [1.06928777435973, 1.0923162009244956] |
| empty_C64_vs_F16 | 1.071599 | [1.0496809483643887, 1.0945662163593302] |

## Development selection (exposed, not final)

| Prefix | F16 | C64 A16 | A8 minmax | A8 MSE | A8 p99.99 |
|---|---:|---:|---:|---:|---:|
| empty | 25.3375 | 27.1502 | 151935.9505 | 151491.1139 | 29112.0065 |
| dot | 25.0425 | 27.0144 | 151935.9505 | 277197483.8522 | 40.4517 |
| eos | 27.4387 | 28.8486 | 80.0454 | 40.0994 | 45.8118 |
| mode_first | 24.8677 | 26.9284 | 151935.9505 | 18752092.7948 | 42.4419 |
| chat_header | 28.7185 | 30.9010 | 82.7568 | 42.7327 | 52.6984 |

## First-position diagnostics

| Model/case | First >1000 /16 | Later >1000 count | First median | Later maximum |
|---|---:|---:|---:|---:|
| F/bare128 | 16 | 0 | 15160.0000 | 19.9688 |
| F/bare64 | 16 | 0 | 15160.0000 | 19.0469 |
| F/cached64_then64steps | 16 | 0 | 15160.0000 | 19.9844 |
| F/chat | 16 | 0 | 17632.0000 | 19.7812 |
| F/replace_first | 16 | 1 | 14568.0000 | 14568.0000 |
| F/restart_second64 | 16 | 0 | 15232.0000 | 19.6250 |
| F/shift1_64 | 16 | 0 | 15432.0000 | 18.7031 |
| C64/bare128 | 16 | 0 | 14944.0000 | 19.8750 |
| C64/bare64 | 16 | 0 | 14944.0000 | 18.1719 |
| C64/cached64_then64steps | 16 | 0 | 14944.0000 | 19.8594 |
| C64/chat | 16 | 0 | 17376.0000 | 19.8438 |
| C64/replace_first | 16 | 1 | 14800.0000 | 14800.0000 |
| C64/restart_second64 | 16 | 0 | 15072.0000 | 19.9219 |
| C64/shift1_64 | 16 | 0 | 15372.0000 | 17.7969 |

## Evidence and limits

All final rows score the same 64 context + 16 target body tokens, using prefill64 and 15 teacher-forced cache append/read steps. Prefix is unscored and never replaces body context. All models use matching prefix controls. Fresh data were frozen before inference and audited for document/text/32-token and Wikitext-title overlap. Development alone chose the policy. Paired bootstrap: 10000, seed243, document-stratified by four cells.

Repeated logits/NLL and causal future-token controls are exact; independent cross entropy error <5e-6. F16/C64 model weights remain unchanged. FP32 cached/uncached attention oracle tolerance1e-5; measured error recorded in checks. Bulk/sequential FP16 and full-prefix/seeded-prefix differences are reported in cache checks, not hidden by a false byte-exact claim. Prefix U8 NPZ bytes roundtrip exactly; runtime stores prefix and body in uint8.

PPL acceptance remains overall <=5% and every language/domain cell <=10% versus the paired floating reference; matching the imperfect W4A16 alone is insufficient. Short questions not used. No device throughput is inferred from software evaluator duration. New prefill/decode E2E token/s: N/A. Device feasibility, if justified, requires a separate experiment and the existing >10% single-layer latency stop rule.

Figures: figures/positions.png, figures/prefix_distributions.png, figures/final_ppl.png. Full numerical results: summary.json and scores/.
