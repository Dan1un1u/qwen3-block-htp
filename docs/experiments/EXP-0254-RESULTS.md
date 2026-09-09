# EXP0254 结论与路线

按顺序完成联合 attention 恢复及条件后续研究。两套互不重叠的128文档/2048目标token面板均在推理前冻结；第一套决定路线，第二套独立确认。所有实验固定原始 per-channel C64 W4、R3、EOS前缀和EXP0253浮点 attention 内部计算，只恢复指定激活边界为FP16。这里的基准B不是设备NR64模型；本轮不代表硬件PPL或可部署混合精度方案。

第一阶段J/C64=1.170251，配对95%CI[1.115126,1.231598]，按预声明规则进入残差/SwiGLU分支。J/B=1.018960，CI[0.972865,1.067624]，不支持把联合恢复的点估计变差解释为稳定退化。

独立确认：F16=27.704303，C64=29.257888，B=34.120655，attention联合恢复J=32.625730，残差R=31.250432，SwiGLU S=32.946848，RS=30.570799，JR=31.094948，JS=32.587528，JRS=30.601240。

- 残差恢复R/B=0.915880，95%CI[0.875928,0.957095]，改善8.41%，本轮证据最明确的单项优先方向。S/B=0.965598，CI[0.920750,1.011868]，尚未确认单独SwiGLU恢复的稳定改善。
- RS/B=0.895962，CI[0.854671,0.939206]，改善10.40%。RS点估计最好，但追加S相对R的次级探索性比较RS/R=0.978252，CI[0.941745,1.015659]，不能声称已证明额外收益。
- JRS与RS接近。次级探索性JRS/RS=1.000996，CI[0.975864,1.026789]，未显示继续扩大attention恢复的额外收益。
- 预声明的全部联合交互项区间均跨0，尚无显著超加性收益证据；也不能以此认定不存在相互作用。所有区间为文档分层配对bootstrap的名义点区间，未做多重比较校正；次级比较不用于候选选择或验收。
- 所有诊断恢复仍未通过原模型门槛。RS/F16=1.103468，CI[1.054795,1.153401]，仍退化10.35%；RS/C64=1.044874，仍有约4.49%的额外PPL。C64/F16点估计本身为1.056077，略超总体5%门槛，不能把全部差距归给A8。C64是匹配权重对照，不是数学下界。

建议优先把残差误差定位转化为可部署A8修复：限定 residual_mid/out 两个边界，先评估实际量化执行轨迹下的静态范围/零点校准，并对残差共享分支及后续归一化的输出误差做验证。保持W4权重、量化粒度和原生HMX读W4语义，SwiGLU作为预声明的次要对照；暂不投入更复杂的联合旋转或FlashAttention整体重写。具体参数候选、校准/开发/确认角色须在新实验开始前登记冻结。候选需要同时复核浮点attention参考和实际整数attention代理，不能把本轮条件收益直接套到设备。

本轮FP16恢复不是A8算法实现，也不是硬件集成候选。若将来真正A8候选通过软件质量与原有数值门槛，再进行单层5short/10formal测量，保持稳定>10% Host-wall回退即暂停的规则。现有R3设备整层数值门槛问题仍单独待解；其它recipes/权重/设备二进制冻结。

24份评分完成独立NLL/PPL/交互项重算；F/C64/B精确复现旧开发集；全部掩码通过独立混合缓存oracle、边界哨兵、完整调用次数、重复/因果性/CE、权重及前缀不变检查。证据有效不等于A8部署验收通过。E2E token/s：N/A，本轮无硬件执行。

# EXP-0254 ordered joint restoration

Software diagnostics with fixed per-channel W4, dense R3, EOS prefix and floating attention core. Restorations are not deployed A8 candidates. Both panels frozen before inference and independently held out from all old data.

Prespecified route: **downstream**, triage J/C64=1.170251; routing only, not model acceptance.

## Triage

128documents,2048targets; equal-cell NLL.

| Configuration | PPL | PPL/F16 | PPL/C64 | Model point gate |
|---|---:|---:|---:|---|
| F16 | 21.342110 | 1.000000 | 0.947165 | reference |
| W4A16 control | 22.532613 | 1.055782 | 1.000000 | False |
| R3 floating core, A8 boundaries | 25.878158 | 1.212540 | 1.148476 | False |
| Joint attention restored | 26.368808 | 1.235530 | 1.170251 | False |

| Paired intervention | PPL ratio (95% CI) |
|---|---:|
| J_vs_B | 1.018960 [0.972865, 1.067624] |

| Configuration | en_wiki | zh_wiki | en_news | zh_news |
|---|---:|---:|---:|---:|
| F | 15.251474 | 29.319572 | 21.657981 | 21.422098 |
| C64 | 15.473500 | 32.891879 | 21.375327 | 23.694970 |
| B | 17.655086 | 37.335275 | 25.672651 | 26.501687 |
| J | 17.208409 | 40.939406 | 24.890739 | 27.570329 |
## Independent confirmation

128documents,2048targets; equal-cell NLL.

| Configuration | PPL | PPL/F16 | PPL/C64 | Model point gate |
|---|---:|---:|---:|---|
| F16 | 27.704303 | 1.000000 | 0.946900 | reference |
| W4A16 control | 29.257888 | 1.056077 | 1.000000 | False |
| R3 floating core, A8 boundaries | 34.120655 | 1.231601 | 1.166204 | False |
| Joint attention restored | 32.625730 | 1.177641 | 1.115109 | False |
| Residual restored | 31.250432 | 1.127999 | 1.068103 | False |
| SwiGLU restored | 32.946848 | 1.189232 | 1.126084 | False |
| Residual + SwiGLU | 30.570799 | 1.103468 | 1.044874 | False |
| Joint attention + residual | 31.094948 | 1.122387 | 1.062789 | False |
| Joint attention + SwiGLU | 32.587528 | 1.176262 | 1.113803 | False |
| Joint attention + residual + SwiGLU | 30.601240 | 1.104566 | 1.045914 | False |

| Paired intervention | PPL ratio (95% CI) |
|---|---:|
| J_vs_B | 0.956187 [0.914863, 0.999087] |
| R_vs_B | 0.915880 [0.875928, 0.957095] |
| S_vs_B | 0.965598 [0.920750, 1.011868] |
| RS_vs_B | 0.895962 [0.854671, 0.939206] |
| JR_vs_B | 0.911323 [0.874493, 0.948518] |
| JS_vs_B | 0.955067 [0.911865, 0.998068] |
| JRS_vs_B | 0.896854 [0.857836, 0.936661] |
| JR_vs_J | 0.953081 [0.917262, 0.990448] |
| JS_vs_J | 0.998829 [0.953722, 1.046549] |
| JRS_vs_J | 0.937948 [0.901598, 0.977264] |

| Interaction: positive = extra joint NLL benefit | Estimate (95% CI) |
|---|---:|
| J_with_R | -0.039814 [-0.096943, +0.018918] |
| J_with_S | -0.033836 [-0.094821, +0.027752] |
| R_with_S | -0.013019 [-0.070235, +0.045823] |
| R_with_S_given_J | +0.014833 [-0.037958, +0.069664] |
| total_joint | -0.058816 [-0.140841, +0.023013] |

| Configuration | en_wiki | zh_wiki | en_news | zh_news |
|---|---:|---:|---:|---:|
| F | 17.366683 | 38.276392 | 22.306343 | 39.729459 |
| C64 | 18.759038 | 40.137646 | 24.044439 | 40.475739 |
| B | 21.075206 | 49.700197 | 28.894773 | 44.783724 |
| J | 20.559050 | 44.469465 | 27.021537 | 45.863368 |
| R | 19.762818 | 44.433198 | 25.159803 | 43.167852 |
| S | 20.312669 | 47.534693 | 27.245032 | 44.790970 |
| RS | 19.480752 | 43.639957 | 24.500110 | 41.934271 |
| JR | 19.817720 | 44.787073 | 24.946298 | 42.222784 |
| JS | 21.094685 | 46.942993 | 26.498422 | 42.977569 |
| JRS | 19.309151 | 42.757925 | 24.082841 | 44.103048 |

Original model gates: overall5percent and eachcell10percent vs matched F16; pointwise paired confidence reported separately in summary.json. C64 is a matched weight control, not a mathematical floor. Conditional restoration effects and interactions are not additive causal fractions or bounds on deployed quantization.

Exact EXP0253 development reproduction for F/C64/B, independent mixed-cache oracle for every evaluated mask, restored-versus-QDQ hook sentinels, full invocation counts, repeat/causality/CE, immutable weights/prefix and independently reconstructed disjoint data are recorded under checks/ and scores/.

E2E tokens/s: N/A; no device run, new weights or native FlashAttention implementation.
