# EXP0244 fixed-prefix A8 layer/boundary attribution

Frozen EOS=151645, C64 per-channel W4 and EXP0243 MSE A8 parameters. All PPL rows use the same prefix and fresh final body tokens. Restorations are software diagnostics; weights never change.

| Variant | PPL | vs C64 | vs F16 | Excess A8 NLL removed |
|---|---:|---:|---:|---:|
| F | 27.7853 | -4.13% | +0.00% | 112.4% |
| C64 | 28.9810 | +0.00% | +4.30% | 100.0% |
| A8 | 40.7428 | +40.58% | +46.63% | 0.0% |
| global_swiglu | 38.1622 | +31.68% | +37.35% | 19.2% |
| global_down | 40.9175 | +41.19% | +47.26% | -1.3% |
| global_residual | 36.8760 | +27.24% | +32.72% | 29.3% |
| union | 34.4711 | +18.94% | +24.06% | 49.1% |
| down_L09 | 40.5400 | +39.88% | +45.90% | 1.5% |
| layer_L00 | 34.4329 | +18.81% | +23.92% | 49.4% |
| global_kv_and_qrope | 32.8458 | +13.34% | +18.21% | 63.3% |

Ratios are paired, same-prefix references. The fraction is a conditional intervention result, not additive attribution or deployable mixed precision. Full 95% intervals and four cells are in summary.json.

Fresh final: 256 documents, 4096 scored targets, prefill64 then15 cached steps. Development128 documents; sequential rerank32 documents. Final list frozen before final scoring: {'local': 'down_L09', 'layer': 'layer_L00', 'global_group': 'global_kv_and_qrope'}.

## Proxy correction

Historical EXP0242/243 residual_mid pre-hook affected the norm input only. EXP0244 quantizes once before the midpoint fanout, feeding norm and skip the same tensor. The archived EXP0243 development panel is reproduced exactly using the legacy mode; historical evidence is unchanged.

| Development execution | C64 A16 PPL | Legacy A8 PPL | Shared-midpoint A8 PPL |
|---|---:|---:|---:|
| development bulk | 26.4844 | 35.9261 | 37.0563 |
| rerank sequential | 30.3588 | 41.8446 | 38.6764 |

Bulk and sequential use different FP16 GEMM shapes; bulk is screening only and sequential rerank determines the selected masks. The rerank subset is smaller than bulk and final; their PPL values are not a matched comparison.

Union benefit minus sum of individual benefits = +0.006286 NLL; paired95%CI [-0.06413877487066201, 0.07563138242694548]. This documents interactions, not an additive partition.

Checks: disabled/all-restored exact; shared-fanout sentinel; legacy128-document exact reproduction; independent same-shape U8 versus float-QDQ cache; per-layer K/V dtype/length/appends; repeated and causal sequential logits; independent CE; immutable prefix, fixed parameters and whole-state digest; document/text/32-token exclusions and reconstruction audit.

Existing overall5% / per-cell10% versus F16 thresholds unchanged. No hardware run, speed estimate, weight artifact, rotation or baseline promotion. E2E token/s: N/A.

![Layer/boundary screening](/mnt/d/llm_exp/results/qwen3-block-htp/exp0244/figures/layer_boundary_heatmap.png)

![Final PPL and paired effects](/mnt/d/llm_exp/results/qwen3-block-htp/exp0244/figures/final_ppl_effects.png)

# EXP-0244 结果与下一步讨论

**结论：优先细查早层 Q/K/V 激活量化，暂不直接启动后段局部 R4 或全局 residual R1。** 残差与 SwiGLU 都有损失贡献，但当前最强证据指向早层及 Q-RoPE/K-cache/V 组合。此组合尚未拆开，不能把全部收益归给 K-cache，更不能声称已经证明 R3 有效。

独立最终集256篇文档、4096个目标token；四类中英文新闻/百科均衡；固定EOS前缀、C64 per-channel W4及EXP0243静态MSE A8参数。均按prefill64＋15次真实缓存续写评分，前缀不计入损失。以下恢复仅为软件归因，不是混合精度部署方案。

| 配置 | PPL | 消除的额外A8 NLL（95% CI） |
|---|---:|---:|
| F16A16 对照 | 27.7853 | — |
| C64 W4A16 对照 | 28.9810 | — |
| 完整 A8（共享残差修正后） | 40.7428 | 0.0% [0.0%, 0.0%] |
| 全部层 SwiGLU 恢复 A16 | 38.1622 | 19.2% [8.2%, 29.2%] |
| 全部层 Down 输出恢复 A16 | 40.9175 | -1.3% [-12.3%, 8.6%] |
| 全部层残差 mid＋out 恢复 A16 | 36.8760 | 29.3% [19.8%, 38.4%] |
| 全部层 SwiGLU＋Down＋残差恢复 A16 | 34.4711 | 49.1% [40.9%, 56.7%] |
| 仅第9层 Down 输出恢复 A16 | 40.5400 | 1.5% [-7.2%, 9.4%] |
| 仅第0层全部激活边界恢复 A16 | 34.4329 | 49.4% [39.5%, 58.6%] |
| 全部层 Q-RoPE＋K-cache＋V 恢复 A16 | 32.8458 | 63.3% [54.7%, 71.4%] |

完整A8相对同前缀C64的PPL增加40.58%，相对同前缀F16增加46.63%。原5%总体/10%单格验收门槛不变，诊断性恢复不构成可部署方案验收。

第0层整体恢复的PPL与跨全模型SwiGLU/Down/残差联合恢复接近。开发集28层热图中，最后一层三个重点边界的单独恢复均无改善；这只是筛查观察，未对最后一层追加最终集测试。不能仅凭后段长尾最大就认定那里主导任务损失。

全局Down输出与第9层Down输出的改善证据较弱，需看其区间是否跨零。残差恢复比SwiGLU恢复更有收益；联合恢复在独立最终集更好，与较小开发子集不同。
联合NLL收益减去三个单项收益之和为+0.006286，95%CI为[-0.06413877487066201, 0.07563138242694548]。这些都是固定其他量化边界条件下的恢复效果，不能视作可相加的因果份额，也不是旋转可达到的理论上界。

KV/Q-RoPE组相对联合恢复的PPL比值为0.9528，名义配对95%CI为[0.9163625325875245, 0.9890773912544425]；此为已预声明最终候选之间的探索性比较，没有根据最终集更换候选。

## 接入与数值边界

旧EXP0242/243的中间残差QDQ仅接到RMSNorm输入，旁路保留未量化值。本轮先修正为一次QDQ后同时供给norm与skip，再做消融；旧代理仍完整复现历史128篇开发集，PPL40.09935697084471。历史证据未改写。新旧代理的差异随数据与计算路径变化，不能使用固定修正系数套回历史结果。

通过关闭量化/全恢复等价、共享旁路哨兵、实际U8缓存对独立同形状浮点QDQ缓存、重复与未来token因果性、独立CE与math.fsum汇总、全模型权重摘要不变及文档/文本/32-token排除检查。

相同32篇开发文档，C64的bulk/逐token PPL分别为30.366071/30.358781，而A8为41.103654/38.676441。FP16 GEMM形状的微小变化会被后续QDQ放大，因此bulk仅用于筛查，选择按逐token复核，最终也采用同一路径。该软件归因仍不是DSP整数累加、非线性或吞吐验证。

冻结校准统计中，第0层K-cache步长2.723529、绝对值中位数0.842321、每token最大值中位数288.5，提示持续的通道幅度不均衡；这是支持进一步定位的线索，不是K单独主导的因果证明。

## 建议的顺序（尚未启动新实验）

1. 保留C64权重与EOS，将第0层及早层的Q-RoPE、K-cache、V投影/缓存分别消融，并加Q/K联合对照，区分单边界与配对相互作用。V输出与缓存保持同一逻辑边界。

2. 若Q/K被确认主导，再优先评估Q/K配对的在线变换。SpinQuant的R3方法在Q/K上应用同一归一化Hadamard；在未量化精确算术下，(QH)(KH)^T=QK^T。这类方法可作为保持当前W4权重不变的候选，但本轮没有测试它；官方动态token/head量化代码也不能直接替换本项目的静态U8语义。[官方实现](https://github.com/facebookresearch/SpinQuant/blob/main/train_utils/apply_r3_r4.py)，[论文](https://arxiv.org/html/2405.16406v4)。

3. residual R1作为后续候选，局部R4优先级更低。若后续旋转需要改权重，必须从原始浮点权重重新折叠、再按本项目per-channel方法量化，独立检查F16等价、W4A16代价和A8收益；不复用旧LPBQ/旋转折叠权重。

原生HMX W4约束保留；本轮没有设备执行，没有新权重或基线提升。未来硬件评测仍保留任一单层prefill/decode Host wall稳定回退超过10%即暂停的门槛。E2E token/s：N/A。

![逐层逐边界热图](/mnt/d/llm_exp/results/qwen3-block-htp/exp0244/figures/layer_boundary_heatmap.png)

![独立最终PPL与配对差异](/mnt/d/llm_exp/results/qwen3-block-htp/exp0244/figures/final_ppl_effects.png)
