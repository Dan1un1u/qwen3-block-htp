"""Reconstruct every A5 fixed/greedy round; never pool their trajectories."""
import sys,json,statistics,hashlib
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/llama32-htp');sys.path.insert(0,str(S/'tools'))
from run_llama32_frontend import records as parse_records
from report_llama32_pipeline_profile import MODULES as GROUPS
def records(p):return parse_records(Path(p).read_text())
def normalized(vals):
 x=dict(vals[0]);x['generation_lm_head_ticks']-=x['generation_final_norm_ticks'];return x
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0032')
ID='L32-0032';LAYERS=16;SEED=30032;DELTA=262144
STDOUT='stdout.txt'
ARMS=['A8','SP2'];out=dict(experiment=ID,trajectories={},quality_claim=False)
rng=np.random.default_rng(SEED);idx=rng.integers(0,10,(20000,10));count=0;seals=set();measured=set();physical=[]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def exclusive(p,z):
 with p.open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False);f.write('\n')
for f in ['single_gate.json','slice_gate.json','full_gate.json']:assert read(R/f)['pass_all']
for trajectory in ['greedy','fixed']:
 for phase,n in [('short',5),('formal',10)]:
  assert read(R/f'{trajectory}_{phase}.json')['pass_all']
  for c in range(n):
   paired=[]
   for a in ARMS:
    d=R/f'{trajectory}-{phase}/{c:02d}-{a}';cfg=read(d/'protocol.json');seal=cfg['runtime']['seal'];assert seal['paper_trace'] is False;measured.add(seal['source_head'])
    for name,h in seal['files'].items():
     p=Path(cfg['runtime']['archive'])/Path(name).name
     if str(p) not in seals:assert sha(p)==h;seals.add(str(p))
    ps=[v for v in records(d/STDOUT) if v.get('record')=='generation_profile'];assert len(ps)==160
    assert read(d/'validated.json')['pass_all']
    for v in ps:
     assert v['paper_format_disable']==v['paper_pipeline_disable']==0
     assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
     assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode'])
    paired.append(ps);count+=len(ps)
   for x,y in zip(*paired):
    assert x['mode']==y['mode']
    for k in ['weight_ddr_read_bytes','w4u8_decode_direct_n_weight_ddr_read_bytes','hmx_command_count','hmx_fp16_tile_pair_count']:assert x[k]==y[k],k
    delta=y['hmx_u8s8_tile_pair_count']-x['hmx_u8s8_tile_pair_count'];assert delta==(DELTA if x['mode']=='prefill' else 0),(x['mode'],delta)
    if phase=='formal' and c==0:physical.append(dict(trajectory=trajectory,mode=x['mode'],a8_pairs=x['hmx_u8s8_tile_pair_count'],sp2_pairs=y['hmx_u8s8_tile_pair_count'],weight_bytes=x['weight_ddr_read_bytes']))
 z=read(R/f'{trajectory}_formal.json');assert len(z['runs'])==20;arms={}
 for a in ARMS:
  ps=[]
  for c in range(10):ps.extend(v for v in records(R/f'{trajectory}-formal/{c:02d}-{a}'/STDOUT) if v.get('record')=='generation_profile')
  assert len(ps)==1600;arms[a]={}
  for mode in ['prefill','decode']:
   vals=[v for v in ps if v['mode']==mode];ns=[normalized([v]) for v in vals];host=statistics.mean(v['host_wall_ns']/1000 for v in vals)
   rows=[dict(module=n,us=statistics.mean(sum(v[k] for k in keys)/19.2 for v in ns)) for n,keys in GROUPS]
   rows += [dict(module='Host–DSP 边界',us=host-sum(v['us'] for v in rows)),dict(module='完整 Host wall',us=host)]
   for v in rows:v['host_share_pct']=100*v['us']/host
   assert all(v['us']>=0 for v in rows)
   arms[a][mode]=dict(host_us=host,modules=rows,e2e_tps=(64 if mode=='prefill' else 1)*1e6/host)
 effects={}
 for mode in ['prefill','decode']:
  t={a:np.array([next(v[mode+'_ns'] for v in z['runs'] if v['arm']==a and v['cycle']==c) for c in range(10)]) for a in ARMS}
  effects[mode]=dict(sp2_over_a8_wall=float(t['SP2'].mean()/t['A8'].mean()),ci95=np.quantile(t['SP2'][idx].mean(1)/t['A8'][idx].mean(1),[.025,.975]).tolist())
 out['trajectories'][trajectory]=dict(arms=arms,effects=effects)
assert len(measured)==1
out.update(measured_source=next(iter(measured)),timed_profiles=count,paired_profiles=count//2,physical_gate=True,numerical_gate_scope='Llama independent selected layer7, chain3, chain16 two-step exact; full16 all-layer greedy/fixed CPU teacher selected token/logit codes exact',precision='FP32 residual1; frozen nativeW4; middle ordinaryA8 versus SP2; no rotation',weight_bytes_identical=True,worker_dispatches_identical=True,prefill_extra_pairs=DELTA,decode_extra_pairs=0,first_formal_steps=physical)
exclusive(R/'SUMMARY.json',out)
exclusive(R/'FINAL_AUDIT.json',{k:v for k,v in out.items() if k!='trajectories'})
lines=[f'# {ID}：固定 FP32 残差的普通 A8 / SP2 公平成本','','冻结原生 W4 权重/尺度与所有非目标边界，无旋转。仅恢复普通 A8 middle 参数与 SwiGLU LUT；两臂同等使用流水化向量 gather、Gate/Up 提前发布、Down 原始累加器与 FP32 epilogue。无重新校准或权重量化。','','每种轨迹独立完成 5 short + 10 formal 配对轮次，交替顺序，repeat10 主计时；repeat1 辅助。固定 token 重放 mode3 保留相同 LM head/greedy 工作、不计算额外 NLL；各臂 KV 数值随合同变化，固定的是 token 输入序列。实际 greedy 另报，不混合两个估计。',f'配对 bootstrap 20000，seed{SEED}。全部 {count} 个计时 profile 通过物理核查，保留全部轮次。']
for trajectory,v in out['trajectories'].items():
 for mode in ['prefill','decode']:
  lines += ['',f'## {trajectory} · {mode} 全模型模块表','','| 模块 | 普通 A8 μs（Host占比） | SP2 μs（Host占比） |','|---|---:|---:|']
  for x,y in zip(v['arms']['A8'][mode]['modules'],v['arms']['SP2'][mode]['modules']):lines.append(f"| {x['module']} | {x['us']:.1f} ({x['host_share_pct']:.2f}%) | {y['us']:.1f} ({y['host_share_pct']:.2f}%) |")
  e=v['effects'][mode];lines += ['',f"SP2/A8 Host wall = {e['sp2_over_a8_wall']:.7f}，95%CI [{e['ci95'][0]:.7f}, {e['ci95'][1]:.7f}]。"]
lines += ['','## 端到端吞吐','','| 轨迹 | 配置 | Prefill token/s | Decode token/s |','|---|---|---:|---:|']
for trajectory,v in out['trajectories'].items():
 for a in ARMS:
  u=v['arms'][a];lines.append(f"| {trajectory} | {a} | {u['prefill']['e2e_tps']:.4f} | {u['decode']['e2e_tps']:.4f} |")
lines += ['',f'M64+15、{LAYERS}层，包含 embedding、全部 transformer、final norm、LM head、greedy、FastRPC；排除外部 tokenizer、冷加载、ADB。fixed 行是受控 token 重放吞吐，不是自由生成文本速度。',f'Prefill SP2 额外 {DELTA} 个 HMX tile pair；decode 通过物理行共装，tile pair 与普通 A8 相同。两阶段权重 DDR 字节和 worker dispatch 次数相同。不据此推断 HMX 内部如何处理 W4。',out['numerical_gate_scope'],'未做 PPL/模型质量验收，不晋升基线。A9 FP 向量 softmax 与 A8 额外 shape 仍待执行。']
with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({t:v['effects'] for t,v in out['trajectories'].items()},indent=2))
