"""Consolidate sealed paired evidence; never pool experiments or rewrite raw data."""
import json,hashlib,yaml
from pathlib import Path
BASE=Path('/mnt/d/llm_exp/results');R=BASE/'paper-no-rotation-ablation-20260915';R.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def win(p):return 'D:/llm_exp/results/'+str(p.relative_to(BASE))
Q=BASE/'qwen3-block-htp';L=BASE/'llama32-htp'
paths={'Qwen':{'A1':Q/'exp0274/consumer-row1-a02','A2/A6':Q/'exp0275','A3':Q/'exp0276','A5':Q/'exp0277','A9':Q/'exp0278/vsum','A8':Q/'exp0279'},'Llama':{'A1':L/'l32-0029','A2/A6':L/'l32-0030','A3':L/'l32-0031','A5':L/'l32-0032','A9':L/'l32-0033','A8':L/'l32-0034'}}
expected={'exp0274':'ce6873fee5fb0c66d0fdb63043341b7353aaf664875580b6b8fa00e25978f5d6','exp0275':'75b66ce1ef9f3e3c7491b63b22106199aaa51d0ac81a2d62a366f9e201336d3b','exp0276':'07fa8eb7877a063519f0b450fcd087e33506ae5931591644cae68bcc93077814','exp0277':'160ef619a2017a11913246d7459ca031c14f3fa00f7a9dedf9da5fb11e6a8eef','exp0278':'dc52fb7ad240babd04f54d0b06363e148ff0e41365213b6738410a03ee9f13c2','l32-0029':'bfef7a0973d7d41eb00061eba218e2a94b75c083bced81be1e29fe755e34c0ee','l32-0030':'7d6bc80d63ecdfa9d1c7e3040483a30dc68cf1dd0a14264d7aee862f1bee6ed2','l32-0031':'660a885cbb7ea74b3533a40d9ad9c6a5dda71ed5724399b7c4e79c1a23696474','l32-0032':'a11ee12fd162617993b247e4b60074a815493205050517fa0643d8af0343f557','l32-0033':'c647b984f5cc875a366f6829589c66a6b5428961b867ff1159cc4f9813866904','l32-0034':'6bbd9ddfa2d750b9f55877fb2277bdb0ef90c3a2e0c5fa1acb9b1aaab1711da6'}
# The newly finished EXP0279 ledger is pinned in authority before consolidation.
qm=yaml.safe_load(Path('/home/daniuniu/work/qwen3-block-htp-project-memory/PROJECT_STATUS.yaml').read_text());expected['exp0279']=qm['current_scope']['latest_paper_long_kv_result']['evidence_ledger_sha256']
data={};bindings=[]
for model,phases in paths.items():
 data[model]={}
 for phase,p in phases.items():
  root=p.parent if p.name in ['consumer-row1-a02','vsum'] else p;ledger=root/'EVIDENCE_SHA256.json';assert sha(ledger)==expected[root.name];manifest=read(ledger)
  for name in ['SUMMARY.json','REPORT.md']:
   q=p/name;k=str(q.relative_to(root));assert sha(q)==manifest['files'][k]['sha256']
  z=read(p/'SUMMARY.json');data[model][phase]=z;bindings.append(dict(model=model,phase=phase,root=str(root),primary=str(p),ledger_sha256=sha(ledger),summary_sha256=sha(p/'SUMMARY.json'),report_sha256=sha(p/'REPORT.md')))
def effect(model,phase,mode,arm=None):
 z=data[model][phase]
 if phase=='A1':return z['effects'][mode][arm or 'B11_vs_B00_latency']
 if phase in ['A2/A6','A3']:return z['scopes']['28' if model=='Qwen' else '16']['effects'][mode][arm or 'SPLIT']
 if phase in ['A5','A9']:
  e=z['trajectories']['fixed']['effects'][mode];return dict(ratio=e['sp2_over_a8_wall' if phase=='A5' else 'fp_over_log2_wall'],ci95=e['ci95'])
 e=z['effects'][mode];return dict(ratio=e['ffn_disabled_over_all_wall'],ci95=e['ci95'])
def fmt(e):return f"{100*(e['ratio']-1):+.2f}% [{100*(e['ci95'][0]-1):+.2f}, {100*(e['ci95'][1]-1):+.2f}]"
lines=['# 无旋转 SP2 / FP32 残差：完整硬件消融结果','','已完成用户批准的两模型主消融和附录形状验证。固定高精度残差；未进行 W4→S8 展开对照或整数残差对照。每阶段的有效构建、独立数值范围、失败记录和完整原始计时均封存。新 softmax FP 路径是对照，并未自动晋升为论文基线。','','## 建议的论文主张','','**原生低比特矩阵乘法的系统收益，取决于表示如何落到物理操作数、生产者何时发布有效数据，以及消费者如何在有限片上空间内排程。** 我们在冻结 W4、SP2 和 FP32 残差的条件下，以正常向量化、正常核内双缓冲的模块化实现为对照，验证完整模型的延迟变化。','','直接原生格式本身不是通用加速器：流水是更稳定的收益来源，格式收益及其与流水的交互依赖阶段和模型。SP2 的额外整数工作没有导致成比例的全模成本，但 decode 共装只解释约 1% 的全模收益，不能把全部低开销归功于单一技巧。低比特 softmax 也不具有普遍优势，必须比较实际优化程度和形状。','','## 全模消融矩阵','','数字为“表中分子/分母”的 Host wall 变化及配对95%CI，单位百分比；正值表示分子更慢。每行是独立冻结实验，不能相加成瀑布总加速，也不能跨行相除作配对结论。A5/A9 使用固定 token 输入，实际 greedy 另列在详细报告。','','| 机制及比值方向 | Qwen prefill | Qwen decode | Llama prefill | Llama decode |','|---|---:|---:|---:|---:|']
rows=[('A1：完整 B11 / 优化模块化 B00','A1',None),('A2：关闭 Gate/Up 交错 / all-on','A2/A6','FFN'),('A2：关闭 raw epilogue 重叠 / all-on','A2/A6','EPILOGUE'),('A2：关闭 decode 权重 lookahead / all-on','A2/A6','LOOKAHEAD'),('A3：SP2 两独立流 / 物理行共装','A3',None),('A5：SP2 / 普通 A8，固定 FP32 残差','A5',None),('A9：FP32 HVX softmax / 保留 log2','A9',None),('A8：长 KV 关闭 Gate/Up 交错 / all-on','A8',None)]
for label,phase,arm in rows:lines.append('| '+label+' | '+' | '.join(fmt(effect(model,phase,mode,arm)) for model in ['Qwen','Llama'] for mode in ['prefill','decode'])+' |')
lines += ['','A1–A9 主形状为 M64+15，Qwen cache128、Llama cache80。A8 为 M64+33/cache128，两模型均实际到 KV97；其末尾单 token 只作 tile 边界诊断。A2 的 prefill LOOKAHEAD 和 decode EPILOGUE 对应当前未启用的站点；不把无效开关当作“该机制无用”的证据。Llama QKV ring mask 同样是 no-op，不列入有效机制主表。','','## 公平基线与格式×流水交互','','B00：紧凑重构值接口＋算子级完成发布；B10：直接原生 tile＋算子级发布；B01：紧凑接口＋分块就绪流水；B11：原生 tile＋分块流水。四格都是完整模型配置。B00 保留正常 HVX、GEMM 内部 DMA/HMX 双缓冲、相同资源上限；没有全局串行化、无效 tensor roundtrip、刻意降低 DMA 粒度或处理无效 decode 行。Qwen 最初不公平的 decode row64 消费打包样本已全部保留并标记无效；主结果只取 consumer-row1-a02 完整重测。','','| 条件格式效应 / 交互比值 | Qwen prefill | Qwen decode | Llama prefill | Llama decode |','|---|---:|---:|---:|---:|']
for label,key in [('native / 紧凑：算子级调度','format_given_conventional'),('native / 紧凑：流水调度','format_given_pipeline'),('延迟交互项','interaction')]:lines.append('| '+label+' | '+' | '.join(fmt(effect(m,'A1',mode,key)) for m in ['Qwen','Llama'] for mode in ['prefill','decode'])+' |')
lines += ['','Qwen prefill 存在支持性的交互，Llama prefill 交互CI跨零效应。因此不能写成“两模型均证明显著协同”。Decode 中 native 格式的条件效应略差，也是有效结果。A6 已验证1层、3层与完整模型；局部百分比没有被外推为全模吞吐。','','## 可明确归属的方法贡献','','1. **指定 SP2 码本的原生 W4 整数执行映射。** 对公共尺度下可重构为 signed16 的值分解 v=l+256h，high以h+128输入，输出补偿−32768·sum(w)。复用两个 U8 操作数和 packed W4 路径。我们不声称发明 SP2 或进制分解，也不推断 HMX 内部是否展开 W4。','2. **SwiGLU→SP2→消费者原生 tile 的生产接口。** 从冻结 Gate/Up 输入码查表，直接写 low/high，明确有效行、发布粒度和缓冲生命周期；没有独立中间 SP2 索引再展开阶段。两操作数合计2 bytes/元素，不能宣称 HMX 原生读取1 byte SP2 索引，也不能宣称零转换指令。','3. **依照阶段安排 HVX/HMX/DMA 工作。** Gate/Up 交错、raw accumulator 双槽 epilogue、权重 lookahead 和 decode 闲置物理行共装，各有独立完整模型消融。Prefill 与 decode 的关键机制不同；它们共同控制临界依赖与真实墙钟。','4. **将乘法输入精度与残差存储合同解耦。** 原生 W4×A8 点积仍可进入 FP32 epilogue、残差与 Norm 后 A8，保持8MiB且不产生中间 tensor DDR spill。这一批固定 FP32 残差，证明实现与成本，不将旧A8残差当精度合理性对照。','','## 表示成本与 softmax 图','','![完整模型配对成本](paired_representation_costs.png)','','每个面板使用自己的同轮对照。Qwen A9 主对照为向量概率码行和；早期带标量统计的 FP 原型完整样本单独封存，不进入主分母。Llama log2 保留各层原除法配置（selected7 EXACT），Qwen 使用宽差值 NR64，不能当作同一个内核跨模型泛化。','','FP 概率相对 Float64 的最大误差：Qwen 1.096e-7，Llama 2.194e-7；门槛2e-6，mask零、有限性与自己的 U8 half-up 均通过。同一实际选中层 QKV 的 AV 最大码差分别为 Qwen6/3、Llama7/6（prefill/decode）。这些是两个 softmax 近似之间的差异，不是设备实现错误，不代表 PPL 改善。','','## 已完成范围与证据入口','','| 项目 | Qwen | Llama |','|---|---|---|']
for phase in ['A1','A2/A6','A3','A5','A9','A8']:lines.append('| '+phase+' | '+' | '.join(f'[{data[m][phase]["experiment"]} 完整模块表]({win(paths[m][phase]/"REPORT.md")})' for m in ['Qwen','Llama'])+' |')
lines += ['','A0 基线复现由 Qwen EXP0273 及两模型 A1 全格数值/历史输出核对覆盖；A7 引用 EXP0272 同轮既有证据。A4 与残差维度按用户要求取消。A8/A9 已完成，不再作为待办。所有失败尝试保留，包括不公平旧控制、trace缺失、组件原型、构建与参考准备修复。','','完整 HVX/HMX/DMA 时间线因 FARF 输出截断而不可用，不能给出 overlap/utilization 百分比或虚构 bank conflict、硬件 stall。计数器与配对开关能支持条件性能归因，但不能替代缺失的时间线。','','数值范围：Llama 包含独立全16层参考与全生成轨迹 token/logit 码；Qwen 包含独立选中层/连续3层、全28层物理与确定性检查、每步实际 final norm/full-vocabulary head。Qwen 未宣称全28层独立CPU transformer等价。两模型 W4A8/SP2 均未在本批做PPL或通用文本质量验收，不能写成“高精度无损量化”。','','结构标识：Qwen3 实际28层、hidden2048/FFN6144，Llama3.2-1B-Instruct实际16层、hidden2048/FFN8192。原始config匹配这些维度；不沿用旧文档里冲突的Qwen0.6B标签。不同模型/上下文的绝对速度不作机制加速的分母。','','## 完整端到端 token/s','','包含 embedding、全部层、final norm、LM head、greedy、FastRPC；排除外部 tokenizer、冷加载和ADB。下面列实际 greedy；所有全模块μs/Host占比和固定输入吞吐见上方各自报告。','','| 模型 | 实验/配置 | 形状 | Prefill token/s | Decode token/s |','|---|---|---|---:|---:|']
for model in ['Qwen','Llama']:
 for phase in ['A5','A9']:
  z=data[model][phase]
  for a,v in z['trajectories']['greedy']['arms'].items():lines.append(f'| {model} | {z["experiment"]} {a} | M64+15 | {v["prefill"]["e2e_tps"]:.2f} | {v["decode"]["e2e_tps"]:.2f} |')
 z=data[model]['A8']
 for a,v in z['arms'].items():lines.append(f'| {model} | {z["experiment"]} {a} | M64+33 | {v["prefill"]["e2e_tps"]:.2f} | {v["decode"]["e2e_tps"]:.2f} |')
lines += ['', '[全部实验完整模块表合订本](MODULE_TABLES.md)']
with (R/'MODULE_TABLES.md').open('x') as f:
 f.write('# Complete module tables, original paired scopes retained\n\n')
 for model in ['Qwen','Llama']:
  for phase,p in paths[model].items():f.write('\n\n---\n\n# '+model+' '+phase+'\n\n'+(p/'REPORT.md').read_text())
with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
with (R/'EVIDENCE_INDEX.json').open('x') as f:json.dump(dict(campaign_complete=True,stage_bindings=bindings,canceled=['W4-to-S8 comparison','integer-residual comparison'],unavailable=['complete FARF timeline/utilization'],quality_claim=False,baseline_promoted=False),f,indent=2)
print('CONSOLIDATED',R,flush=True)
