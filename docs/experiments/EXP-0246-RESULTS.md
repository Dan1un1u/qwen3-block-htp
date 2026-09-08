# EXP0246 dense online R3 results

Fixed C64 per-channel W4, EOS151645 and all non-Q/K MSE A8 parameters. Dense X@S128 then normalization before FP16 handoff; no butterfly or learned matrix. Q and K share the transform after RoPE, with prefix keys transformed once before U8 storage.

| Variant | PPL | vs A8 | vs C64 | vs F16 |
|---|---:|---:|---:|---:|
| F | 26.8074 | -28.52% | -3.93% | +0.00% |
| C64 | 27.9047 | -25.60% | +0.00% | +4.09% |
| A8 | 37.5051 | +0.00% | +34.40% | +39.91% |
| recal_L0 | 37.5051 | +0.00% | +34.40% | +39.91% |
| recal_ALL | 37.7765 | +0.72% | +35.38% | +40.92% |
| R3_L0 | 33.2076 | -11.46% | +19.00% | +23.88% |
| R3_ALL | 32.3915 | -13.63% | +16.08% | +20.83% |

| Scope | R3 / matched recal PPL (95% CI) | Recal / original A8 PPL (95% CI) | Device progression eligible |
|---|---:|---:|---|
| L0 | 0.8854 [0.8452, 0.9264] | 1.0000 [1.0000, 1.0000] | True |
| ALL | 0.8575 [0.8206, 0.8969] | 1.0072 [0.9755, 1.0404] | True |

| Variant | en wiki | zh wiki | en news | zh news |
|---|---:|---:|---:|---:|
| F | 23.8702 | 18.7940 | 27.4262 | 41.9734 |
| C64 | 25.4032 | 20.4877 | 27.7724 | 41.9481 |
| A8 | 31.6635 | 28.9115 | 38.6394 | 55.9373 |
| recal_L0 | 31.6635 | 28.9115 | 38.6394 | 55.9373 |
| recal_ALL | 31.7391 | 28.7598 | 39.3775 | 56.6575 |
| R3_L0 | 28.2165 | 25.8625 | 33.1923 | 50.2044 |
| R3_ALL | 28.0776 | 24.8991 | 31.0949 | 50.6393 |

Calibration:128 frozen documents x128 tokens, batch4 prefill64+64 sequential steps; same unquantized C64 trajectories, record raw and rotated Q/K at each call shape. Only Q/K recalibrated; K range includes prefix extrema. Development32 exposed documents,9 fixed variants; final256 new documents,7 fixed variants. No tuning or selection on final data.

Independent matrix parity/FP64 orthogonality/dense arithmetic and paired-attention checks passed; profiler observes aten::mm. Parent F/C64/A8 per-token development controls exact. A16 R3 controls satisfy abs(delta meanNLL)<=0.005. Repeated/causal/CE and independent float-QDQ cache checks, prefix-once/count/dtype/length assertions, immutable whole weights and remaining parameters passed.

Confidence intervals are nominal paired document bootstrap10000 seed246, stratified four cells. Existing5%/10% quality gates unchanged. Device progression requires R3 upper ratio CI below1 against both original A8 and matched recalibration; eligibility is not deployment quality acceptance or promotion.

No HMX implementation, hardware timings, or device accuracy claimed for this software phase. E2E token/s: N/A.

![Final PPL](/mnt/d/llm_exp/results/qwen3-block-htp/exp0246/figures/final_ppl.png)

![Key distributions](/mnt/d/llm_exp/results/qwen3-block-htp/exp0246/figures/key_distributions.png)

# 结果解读

本轮证据支持在线 R3 改善静态 A8 精度。固定 per-channel W4 权重，使用朴素稠密矩阵乘 QH、KH；没有蝶形或 FWHT，也没有重新量化权重。

新最终集256篇、4096个目标token：F16 26.8074，W4A16 C64 27.9047，原始A8 37.5051，全层重校准 37.7765，仅L0 R3 33.2076，全层R3 32.3915。仅L0重校准与原始A8逐token完全相同。

全层 R3 相对原始 A8 的 PPL 降低13.63%，配对比值95%区间[0.8262,0.9042]；相对匹配重校准降低14.25%，区间[0.8206,0.8969]。仅L0的独立旋转收益也成立，比值区间[0.8452,0.9264]。四个语种/领域格子的全层R3相对两种对照均改善，名义逐项95%区间均低于1。这些区间不是多重比较校正后的同时保证。

按额外 A8 NLL 衡量，全层R3消除了约49.57%的退化；仍比同批W4A16高16.08%、比F16高20.83%。这说明方向有效，但尚未通过现有5%/10%质量验收。全层与L0之间的点估计差距不能自动解释为全层显著优越或完整部署收益。

机制证据：L0 K 校准体样本峰值从394.25降至49.15625，量化步长从2.723529降至0.358333。典型分量的幅度会上升，因为能量被分散；这不表示能量消失。quantization_mse_summary.json保留全56个Q/K位置的对数量化直方图中点MSE估计，明确不作为精确逐元素MSE。前缀K极值参与范围保护。

数学上(H正交)保持未量化QK点积；实际FP16舍入不严格逐位等价。独立F64正交/稠密算术、A16整模控制、实际U8缓存与同形状浮点QDQ缓存、前缀只旋转一次、重复/因果/CE及权重不可变检查全部通过。最终数据与校准/开发/既往评测做文档、文本、32-token重叠排除。

本阶段只证明软件质量收益。没有HTP时间或E2E token/s；硬件FP16舍入、数据布局和流水开销需要下一独立设备实验验证。F16A16和W4A16 recipe冻结，未提升任何baseline。
