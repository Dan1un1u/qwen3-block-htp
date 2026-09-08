Source branch codex/exp-0242-static-a8-localization, final source 394253ea340e1bdd1ddca1923dc0e9cb0bd9c13b. Numerical evaluation/traces source 37e6689983abfa7668abaf0be5175a4f9f1538e8; calibration source f0239e60bb72832c5ba3e3c9443646843bef07eb. PC057; completed, evidence valid, software quality gate fail, adoption pending. No device run or promotion. Evidence /mnt/d/llm_exp/results/qwen3-block-htp/exp0242, SHA256 ledger 5c6e1f92088a9e7ff49aa27cf1ab6a59f13ace3b46ef34f997474ec022d92977.

# EXP-0242 results

## 主要结论与解释

EXP-0242 已完成固定 C64 per-output-channel W4 的 A8 软件误差定位、静态 clipping 与可视化。478 个激活边界；校准 128 篇 / 16,384 tokens，开发 128 篇 / 2,048 targets，固定历史复核面板 512 篇 / 8,192 targets。三者文档及连续 32-token 无交集。本轮最终面板未选择阈值，但它是历史已暴露面板，不宣称全新未见测试。

最终 PPL：F16 24.774821；C64 W4A16 26.296039；同一 C64 的全部边界 minmax A8 151935.950477；开发集选出的 99.99% 分位裁剪 A8 32468.413936。最佳静态裁剪仍为 C64 的 1234.73 倍，分层配对文档 bootstrap 95% 区间 1030.78–1475.38 倍。静态 clipping 不足以恢复可用精度；没有接受或部署新 recipe。

最明确的早期异常位于 L02（从零计数，即第 3 层）。128 个平衡中英文 Wiki/news 校准窗口中，SwiGLU 绝对值超过 1000 的 token 全部是窗口第 0 个位置（128/128），其余位置为零个。最大绝对值 16688，p99.9 仅约 1.146；通道 1821 占总平方能量 99.992%。这是窗口起始位置的观察，尚未用 F16 激活对照、固定前缀或其它因果干预确认机制，不能直接推广为所有部署输入的固定现象。

L02 的巨大值经 Down 进入残差。残差 minmax 步长 54.548，而 |x| 中位数仅 0.281；按完整校准直方图 bin 边界保守计数，至少 99.989% 的元素落在零码区间。峰值保存得好并不等于普通信号被保存：该层残差相对 RMSE 可仅约 5%，PPL 却已严重损坏。全张量能量指标在此被少量峰值主导。

开发集隔离消融：仅 residual A8 的 PPL 151935.95；仅 SwiGLU A8 58.59，经 MSE clipping 为 36.63；仅 attention probability/context A8 27.18（参照 27.15）。全部边界量化时，仅保留残差为 A16 的诊断对照 PPL 160.43，说明残差是主因之一，仍有其它 A8 损失；这只是归因实验，没有提出或部署混合精度 recipe。前 1/2/4 层累计 minmax 的 PPL 为 31.39/37.96/25303832.48，证实早层已发生灾难性损失。后续层把结果压到接近均匀分布不代表恢复。

只对 O/Gate/Up/Down 输出做 MSE clipping，开发 PPL 从 38.83 恶化到 723.61；同时在独立的 16 篇开发张量轨迹上，聚合 NMSE 从 0.02250 降到 0.00726。局部张量重建目标和最终语言建模质量确实存在不一致，不能只看 MSE 挑方案。

全部权重摘要保持一致；F16/C64 开发集逐 token NLL、top1 完全复现 EXP0230。33 份评测全部通过 logits/NLL 重复、因果掩码独立性和独立 CE 检查。两次推理前/校准首张量处的工具实现错误均留档，并按 PC037 修复；没有改动阈值搜索范围或降低数值门槛。

## 图表

[首 token 与量化步长机制](D:/llm_exp/results/qwen3-block-htp/exp0242/figures/early_outlier_mechanism.png) · [完整分布图](D:/llm_exp/results/qwen3-block-htp/exp0242/figures/activation_distributions.png) · [token/通道热图](D:/llm_exp/results/qwen3-block-htp/exp0242/figures/activation_heatmaps.png) · [消融与误差传播](D:/llm_exp/results/qwen3-block-htp/exp0242/figures/a8_error_localization.png) · [最终 PPL](D:/llm_exp/results/qwen3-block-htp/exp0242/figures/primary_ppl.png)。每幅另有 PDF。分布图按校准中每类最重尾部选例，并非声称所有层都一样；热图是一条固定英文 Wiki 窗口，全部统计使用平衡中英文数据。

本轮是 FP16 eager 浮点 GEMM 上注入静态 U8 QDQ 的软件诊断，并非 DSP/HMX 输出转换、非线性 LUT 或 KV append 的逐位模拟。没有新的硬件 profiling，E2E token/s 为 N/A。下一步需讨论已定位的起始位置异常与残差量化边界；本轮未启动前缀、旋转、SmoothQuant、LET 或分组方案。


static affine U8 QDQ, fixed C64 W4, FP16 software, no DSP measurements

128 calibration documents / 16384 tokens; 128 development documents / 2048 targets; exposed independent primary 512 documents / 8192 targets

## Primary PPL

| Variant | Overall | English Wiki | Chinese Wiki | English news | Chinese news | vs F16 | vs C64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 24.774821 | 17.201958 | 26.697411 | 22.827868 | 35.935995 | +0.00% | -5.78% |
| C64 | 26.296039 | 18.854208 | 28.443771 | 23.914707 | 37.282139 | +6.14% | +0.00% |
| all_minmax | 151935.950477 | 151935.950477 | 151935.950477 | 151935.950477 | 151935.950477 | +613167.61% | +577690.25% |
| all_percentile | 32468.413936 | 33286.851498 | 15414.138630 | 54311.352390 | 39880.601521 | +130954.08% | +123372.64% |

Selected complete-scope policy: percentile. No weight changes or device deployment.

## Development attribution

| Family | Minmax PPL | MSE PPL |
|---|---:|---:|
| norm_input | 29.824091 | 28.054869 |
| qkv_output | 27.468787 | 27.563175 |
| projection_output | 38.825296 | 723.608504 |
| swiglu | 58.594440 | 36.629062 |
| qk_kv | 32.816565 | 33.006118 |
| attention | 27.177458 | 27.060170 |
| residual | 151935.950477 | 151893.326124 |
| head_input | 27.871304 | 27.565753 |

## Files

Calibration thresholds/full moments: calibration.json; full histograms/channel/token traces: activation_arrays.npz; independent tensor errors: local_tensor_error.json; residual propagation: residual_drift.json; per-document scores: scores/; paired 95% intervals: summary.json.

Histogram MSE is an approximate local reconstruction criterion, not a guarantee of lower model NLL. Raw tensor error is independently measured on development. Quantized softmax probabilities retain zero and are not renormalized. QKV V and KV carrier share a boundary; combined paths quantize it once.

## Scope and limits

This diagnostic injects static U8 QDQ into eager FP16 execution of fixed C64 effective weights. It is not bit-exact HMX conversion, DSP nonlinear LUTs or KV append validation, and cannot certify final W4A8 device quality. The old EXP0218 W4U8 weights differ from C64; this is not a matched reconstruction of that historical collapse. All activation sites remain enabled in final A8 candidates. Short answers were not used for selection.

No device execution: complete Host wall, module profiling, physical VTCM/DMA counters and E2E token/s are N/A. Existing >10% single-layer latency stop rule remains for future device work.
