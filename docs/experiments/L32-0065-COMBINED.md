# 整数非线性与AV尺度融合消融

L32-0065与EXP-0307，2026-09-22。两模型均完成五轮短测、十轮正式配对测量，每轮repeat10。固定M64 prefill＋42个decode、相同token轨迹、原生per-channel W4、均匀INT16 Down、FP32残差/Norm、无旋转。保留生产流水与向量实现。E表及其它历史工作表均未修改。

| 模型 | 配置 | Prefill token/s | Decode token/s |
|---|---|---:|---:|
| Llama3.2-1B | F：浮点非线性及必要QDQ | 1553.23 | 46.00 |
| Llama3.2-1B | I：整数非线性，保留AV独立RQ | 2318.92 | 46.14 |
| Llama3.2-1B | M：整数非线性，AV倍率融合进O | 2315.35 | 46.33 |
| Qwen3-1.7B | F：浮点非线性及必要QDQ | 1222.48 | 46.82 |
| Qwen3-1.7B | I：整数非线性，保留AV独立RQ | 1837.27 | 47.21 |
| Qwen3-1.7B | M：不满足原饱和语义，未纳入测速 | — | — |

## 可支持的结论

- Llama3.2-1B：整数非线性及其边界处理使prefill墙钟下降33.02%（吞吐提高49.30%），decode墙钟下降0.31%。这是Softmax/SwiGLU实现及必要转换的组合效应，不是单独QDQ的成本。
- Qwen3-1.7B：整数非线性及其边界处理使prefill墙钟下降33.46%（吞吐提高50.29%），decode墙钟下降0.83%。这是Softmax/SwiGLU实现及必要转换的组合效应，不是单独QDQ的成本。
- 公平AV对照补齐有效行处理后，Llama AV融合的decode墙钟收益为0.408%，配对倍率95%CI为[0.993435, 0.998477]。Prefill CI跨过1，不能确认收益。历史约4%的decode结论不能继续作为当前公平对照的收益。
- Qwen直接后移AV倍率会绕过原U8饱和：候选轨迹中1310/6078464个值超出原范围，首层相同输入已出现6个。截断与O矩阵乘不交换，故候选在数值资格检查阶段被否决。未通过自己的参考或改变原语义的版本不得作为有效加速结果。
- 矩阵命令数、tile工作量、权重搬运、DMA描述符、VTCM峰值及发布次数保持一致；原始profile逐项核验。性能差异不能归因于减少工作量或给浮点组加入低效标量循环。
- FP32残差、Norm等仍保留高精度。不得把这些实验证据表述为整个block无浮点、没有任何RQ，或量化精度/PPL已验收。

## 已有证据复用范围

已完成的布局与流水联合基线、Gate/Up交错、raw epilogue重叠、decode lookahead与双分量共装消融保留原测量身份。它们验证共享执行载体和流水，不因SP2码表换成均匀INT16而重复。SP2/INT16近似同速不等于把SP2原始速度改写成INT16实测。历史M64+15/M64+33与本轮M64+42是独立实验，百分比不相加，绝对速度不混用分母。

- 历史流水报告：/home/daniuniu/work/qwen3-block-htp/docs/paper_no_rotation_ablations/REPORT.md
- 独立INT16/SP2成本对照：D:/llm_exp/results/paper-int16-down-20260916/REPORT.md

## 数值与归档

Llama三组全16层各688个边界哈希，Qwen两组全28层各1204个边界哈希，与各自独立参考完全一致。组件共核验2883584个Gate/Up码对。所有正式计时均通过完整Host/DSP ledger、8MiB VTCM和零中间张量DDR/溢出检查。F/I并非相同非线性近似；Llama I/M保留浮点scale重结合的舍入差异说明，未声称二者逐位等价。

Llama模型包及设备文件在计时前后均复核。Qwen复用已验证909个历史权重/配置产物，按既有退休记录排除32个旧chain数组，重新计算参考；control-v2仅扩展expected-token元数据。所有失败尝试保留，未放宽数值门槛、未筛除正式样本。

- [Llama完整报告](../llama32-htp/l32-0065/REPORT.md)及[分模块表](../llama32-htp/l32-0065/MODULES.md)
- [Qwen完整报告](../qwen3-block-htp/exp0307/REPORT.md)及[分模块表](../qwen3-block-htp/exp0307/MODULES.md)

本轮无自动基线晋升；Qwen无条件AV融合结论为不适用。已有流水消融无需补测。本轮有效对照已完成，失败的融合条件也已有明确反例。
