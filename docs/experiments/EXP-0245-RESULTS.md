# EXP0245 early Q/K/V attribution

Fixed EOS151645, C64 per-channel W4, EXP0243 MSE A8 parameters and EXP0244 shared midpoint fanout. Q is post-RoPE; K is post-RoPE before cache; V projection output/cache are tied. Restored cache includes prefix and body.

| Variant | PPL | vs C64 | vs F16 | Excess A8 NLL removed (95% CI) |
|---|---:|---:|---:|---:|
| F | 25.5761 | -3.35% | +0.00% | N/A |
| C64 | 26.4619 | +0.00% | +3.46% | N/A |
| A8 | 36.9908 | +39.79% | +44.63% | 0.0% [0.0, 0.0] |
| L0_Q | 36.7727 | +38.97% | +43.78% | 1.8% [-11.3, 13.4] |
| L0_K | 31.8492 | +20.36% | +24.53% | 44.7% [34.2, 54.2] |
| L0_V | 37.3674 | +41.21% | +46.10% | -3.0% [-16.9, 8.9] |
| L0_QK | 31.4355 | +18.80% | +22.91% | 48.6% [37.9, 57.8] |
| L0_QKV | 32.5263 | +22.92% | +27.17% | 38.4% [26.9, 49.0] |
| ALL_Q | 36.5204 | +38.01% | +42.79% | 3.8% [-8.1, 14.7] |
| ALL_K | 29.9174 | +13.06% | +16.97% | 63.4% [54.2, 72.3] |
| ALL_V | 37.1577 | +40.42% | +45.28% | -1.3% [-13.5, 9.2] |
| ALL_QK | 30.6862 | +15.96% | +19.98% | 55.8% [46.8, 63.9] |
| ALL_QKV | 29.9503 | +13.18% | +17.10% | 63.0% [54.5, 71.1] |

## Four-cell PPL

| Variant | English wiki | Chinese wiki | English news | Chinese news |
|---|---:|---:|---:|---:|
| F | 26.4266 | 21.8672 | 20.4968 | 36.1259 |
| C64 | 26.4594 | 22.3731 | 21.2264 | 39.0211 |
| A8 | 36.6149 | 31.7716 | 30.0326 | 53.5902 |
| L0_Q | 35.4465 | 31.7147 | 29.7523 | 54.6700 |
| L0_K | 30.4441 | 27.0477 | 25.2228 | 49.5413 |
| L0_V | 36.3401 | 32.7136 | 30.5019 | 53.7685 |
| L0_QK | 31.0118 | 26.5942 | 25.0503 | 47.2666 |
| L0_QKV | 32.2498 | 27.0751 | 27.0300 | 47.4236 |
| ALL_Q | 33.8865 | 31.7622 | 30.4214 | 54.3284 |
| ALL_K | 28.8277 | 25.2351 | 24.1747 | 45.5531 |
| ALL_V | 35.8031 | 31.4300 | 30.6923 | 55.1952 |
| ALL_QK | 30.3278 | 25.2977 | 24.9329 | 46.3532 |
| ALL_QKV | 29.2524 | 24.0986 | 24.4170 | 46.7475 |

## Predeclared interactions

| Scope | QK benefit minus Q and K benefits (95% CI) | Added V benefit over QK (95% CI) |
|---|---:|---:|
| L0 | +0.007163 [-0.047716, +0.062363] | -0.034111 [-0.071742, +0.004687] |
| ALL | -0.038172 [-0.091881, +0.014653] | +0.024276 [-0.006120, +0.054503] |

## Layer0 versus all layers

| Boundary | PPL(layer0 restore) / PPL(all restore), 95% CI |
|---|---:|
| Q | 1.0069 [0.9707, 1.0445] |
| K | 1.0646 [1.0301, 1.0998] |
| V | 1.0056 [0.9679, 1.0463] |
| QK | 1.0244 [0.9922, 1.0581] |
| QKV | 1.0860 [1.0473, 1.1260] |

Exploratory conditional Q addition to K-only restoration (fixed final arms; not a new preregistered hypothesis or a selection rule):

- L0: PPL(QK)/PPL(K) = 0.9870, nominal95% paired CI [0.9534, 1.0223].
- ALL: PPL(QK)/PPL(K) = 1.0257, nominal95% paired CI [0.9917, 1.0599].

Final13 configurations and development22 were frozen before inference; no final-set selection. Development is exposed historical32 documents; final256 documents/4096 targets is new, with document/text/32-token exclusions and independent reconstruction. Identical batch4 sequential prefill64+15 path throughout.

Five historical controls reproduced exactly per token. Independent same-shape float-QDQ cache oracles for Q/K/V/QK/QKV masks, cache dtype/length/appends, repeated/causal logits, independent CE and math.fsum scoring, immutable prefix and full weight digests all checked.

Intervals: paired stratified document bootstrap10000, seed245; nominal pointwise95%, no multiplicity adjustment. Conditional restoration effects are not additive causal shares, deployed mixed precision, or bounds on rotation gains. Prefix/body contributions remain combined.

Original5% overall/10% cell thresholds remain descriptive; diagnostic gate does not accept a deployable A8 model. No hardware, new weights, rotations, calibration or baseline promotion. E2E token/s: N/A.

![Development scope and boundary](/mnt/d/llm_exp/results/qwen3-block-htp/exp0245/figures/scope_boundary_heatmap.png)

![Final PPL and paired effects](/mnt/d/llm_exp/results/qwen3-block-htp/exp0245/figures/final_ppl_effects.png)

# EXP-0245 结果与下一步讨论

**K-cache 是本轮最明确的优先修复边界。** 全部候选均为恢复A16的软件归因；没有实施旋转、训练、权重重新量化、设备部署或基线提升。

独立最终集256篇文档、4096个评分token，四类中英文新闻/百科均衡；同一EOS前缀、固定C64 per-channel W4、EXP0243静态MSE U8参数。所有配置均为batch4、prefill64＋15次真实缓存续写。前缀不参与PPL评分。开发22项、最终13项都在推理前冻结，未依据最终结果更换候选。

| 条件恢复配置 | PPL | 消除的额外A8 NLL（95% CI） |
|---|---:|---:|
| F16A16 对照 | 25.5761 | — |
| 固定 C64 W4A16 对照 | 26.4619 | — |
| 完整 A8 软件代理 | 36.9908 | 0.0% [0.0%, 0.0%] |
| 第0层 Q-RoPE恢复A16 | 36.7727 | 1.8% [-11.3%, 13.4%] |
| 第0层 K-cache恢复A16 | 31.8492 | 44.7% [34.2%, 54.2%] |
| 第0层 V输出/缓存恢复A16 | 37.3674 | -3.0% [-16.9%, 8.9%] |
| 第0层 Q/K联合恢复A16 | 31.4355 | 48.6% [37.9%, 57.8%] |
| 第0层 Q/K/V联合恢复A16 | 32.5263 | 38.4% [26.9%, 49.0%] |
| 全部28层 Q-RoPE恢复A16 | 36.5204 | 3.8% [-8.1%, 14.7%] |
| 全部28层 K-cache恢复A16 | 29.9174 | 63.4% [54.2%, 72.3%] |
| 全部28层 V输出/缓存恢复A16 | 37.1577 | -1.3% [-13.5%, 9.2%] |
| 全部28层 Q/K联合恢复A16 | 30.6862 | 55.8% [46.8%, 63.9%] |
| 全部28层 Q/K/V联合恢复A16 | 29.9503 | 63.0% [54.5%, 71.1%] |

完整A8相对同前缀C64的PPL增加39.79%；全部K恢复后仍比C64高13.06%。所有诊断恢复中PPL最低的是全部28层 K-cache恢复A16（29.9174），这不构成可部署A8方案的验收。

## 第0层与全层

第0层K恢复PPL为31.8492，全层K恢复为29.9174。二者PPL比值为1.0646，名义配对95%CI为[1.030059648921002, 1.0997889188678132]。这比单看开发集分段热图更适合判断全层干预是否有额外价值。

开发集只有32篇历史文档、每格8篇，明确作为已暴露数据使用。中段Q/K联合在该小面板变差、后段单项较弱，均是筛查现象；不能据此认定这些层没有误差，或把分段收益相加预测全层结果。

## Q/K相互作用与V

| 范围 | QK收益减去Q、K单项收益之和（NLL，95% CI） | 在QK之上再恢复V的收益（NLL，95% CI） |
|---|---:|---:|
| L0 | +0.007163 [-0.047716, +0.062363] | -0.034111 [-0.071742, +0.004687] |
| ALL | -0.038172 [-0.091881, +0.014653] | +0.024276 [-0.006120, +0.054503] |

额外检查固定最终候选中QK相对K的配对差异；以下为探索性比较，未新增候选或重新选择：

- L0：PPL(QK)/PPL(K)=0.9870，95%CI [0.9534, 1.0223]；区间跨1，尚不能确认额外恢复Q有稳定收益。
- ALL：PPL(QK)/PPL(K)=1.0257，95%CI [0.9917, 1.0599]；区间跨1，尚不能确认额外恢复Q有稳定收益。

所有区间均为四格分层、文档级配对bootstrap10000次（seed245）的名义点区间，未做多重比较校正。恢复收益是固定其余量化边界条件下的结果，不是可相加的因果份额，也不是旋转可达到的上界。V恢复若变差，不能解释为V没有误差或A8比A16本质更准确。

## 建议下一步：有界验证配对Q/K在线Hadamard

建议先测试针对K-cache量化损失的R3候选，把第0层和全层作为两个明确范围。保留当前W4权重和其余A8边界，暂不回到权重折叠、学习全局R1或后段R4。当前结论定位了边界，尚未证明R3能够改善PPL。

Q不是主要单边界误差来源，也不妨碍把配对Q/K变换作为K修复候选：旋转K时同步旋转Q，是为了保持注意力点积关系，并非要把Q恢复到A16。实际A8中两边的舍入/裁剪都会改变，仍须用PPL验证净收益。

方法上，在RoPE之后、Q/K量化之前，对每个头的Q和K施加相同的归一化Hadamard H；未量化精确算术中，(QH)(KH)^T=QK^T。当前模型head_dim=128、16个Q头、8个KV头；变换应与GQA配对一致，并一致覆盖前缀与正文缓存。这一步可保持现有线性W4权重不变，区别于此前需要折叠并重新量化权重的R1/R2实验。[SpinQuant官方实现](https://github.com/facebookresearch/SpinQuant/blob/main/train_utils/apply_r3_r4.py)，[论文](https://arxiv.org/html/2405.16406v4)。

量化语义仍按本项目静态U8；不照搬官方动态token/head K量化。旋转后的Q/K必须用独立校准数据重新求参数，并保留未旋转、同流程重校准对照，以区分旋转收益与重新校准收益。其它边界参数与C64权重保持冻结；开发与最终数据仍分开。先检查未量化配对变换的数值等价、前缀/缓存一致性，再看全A8的独立PPL。

当前K恢复同时作用于前缀与正文，尚未把两者贡献拆开。R3改变的是二者的表示，不能只旋转正文而保留旧基底的前缀K缓存。即便软件PPL改善，HMX整数路径、缓存布局和在线Hadamard开销仍需单独验证；继续保留单层prefill或decode完整Host wall稳定回退超过10%即暂停、先讨论的门槛。

## 证据与边界

五个历史对照逐token完全复现；Q/K/V/QK/QKV的U8缓存与独立同形状浮点QDQ缓存完全一致；每种配置均有重复/因果性/独立CE检查。35份评分独立math.fsum重汇总，四次模型加载前后完整权重摘要不变，各权重recipe跨阶段前缀摘要相同。最终集完成历史文档/文本/32-token排除与重建审计。

原总体5%/单格10%的精度门槛未变；本轮local gate通过仅表示诊断实现及证据有效，不是部署精度验收。没有新增硬件计时，E2E token/s：N/A。详细四格PPL、配对区间、缓存及哈希证据见REPORT.md、summary.json和checks/。

![开发集边界与层范围](/mnt/d/llm_exp/results/qwen3-block-htp/exp0245/figures/scope_boundary_heatmap.png)

![最终PPL与配对效果](/mnt/d/llm_exp/results/qwen3-block-htp/exp0245/figures/final_ppl_effects.png)
