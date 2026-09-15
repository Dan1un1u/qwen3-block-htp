"""Reconstruct every A5 fixed/greedy round; never pool their trajectories."""
import sys,json,statistics,hashlib
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/qwen3-block-htp');sys.path.insert(0,str(S/'scripts'))
from device_exp0278_vsum import records
from summarize_exp0217 import normalized
from report_exp0273 import GROUPS
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0278/vsum')
ID='EXP-0278';LAYERS=28;SEED=278;DELTA=0
STDOUT='stdout.jsonl'
ARMS=['LOG2','FP'];out=dict(experiment=ID,trajectories={},quality_claim=False)
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
    for k in ['weight_ddr_read_bytes','w4u8_decode_direct_n_weight_ddr_read_bytes','hmx_command_count','hmx_fp16_tile_pair_count']+[k for k in x if 'dispatch' in k and 'ticks' not in k]:assert x[k]==y[k],k
    delta=y['hmx_u8s8_tile_pair_count']-x['hmx_u8s8_tile_pair_count'];assert delta==(DELTA if x['mode']=='prefill' else 0),(x['mode'],delta)
    if phase=='formal' and c==0:physical.append(dict(trajectory=trajectory,mode=x['mode'],log2_pairs=x['hmx_u8s8_tile_pair_count'],fp_pairs=y['hmx_u8s8_tile_pair_count'],weight_bytes=x['weight_ddr_read_bytes']))
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
  effects[mode]=dict(fp_over_log2_wall=float(t['FP'].mean()/t['LOG2'].mean()),ci95=np.quantile(t['FP'][idx].mean(1)/t['LOG2'][idx].mean(1),[.025,.975]).tolist())
 out['trajectories'][trajectory]=dict(arms=arms,effects=effects)
assert len(measured)==1
out.update(measured_source=next(iter(measured)),timed_profiles=count,paired_profiles=count//2,physical_gate=True,numerical_gate_scope='Qwen selected layer14 and chain3 independent exact; full28 physical/front-end finalNorm and full-vocabulary head oracle; no full28 CPU transformer equivalence claim',precision='FP32 residual2; frozen nativeW4/SP2; retained log2 versus FP32 vector softmax; no rotation',weight_bytes_identical=True,worker_dispatches_identical=True,prefill_extra_pairs=DELTA,decode_extra_pairs=0,first_formal_steps=physical)
out['implementation']='FP vector probability and vector rowmass (primary)' if R.name=='vsum' else 'FP vector probability with scalar rowmass telemetry (prototype; supplementary only)'
exclusive(R/'SUMMARY.json',out)
exclusive(R/'FINAL_AUDIT.json',{k:v for k,v in out.items() if k!='trajectories'})
lines=[f'# {ID}：现有 log2 / FP32 HVX softmax 全模型消融','','冻结 W4/SP2、FP32 残差、所有非 softmax 权重与尺度，无旋转。LOG2 保留 Qwen wide NR64，在 raw QK 的宽差值上离散化指数；FP 从 raw HMX QK 的原尺度计算连续 max/exp/sum/normalize，最终统一输出 U8 概率，保留整数 AV。这是 softmax 路径对照，不是全浮点 attention / FlashAttention。','','两臂均采用正常向量实现及冻结流水。每条 greedy / fixed 轨迹各 5 short + 10 formal，交替配对、repeat10 主计时、repeat1 辅助；fixed 保持输入 token，KV 值由各臂自己的计算产生。全模型包括 embedding、所有层、final norm、LM head、greedy 与 FastRPC；排除 tokenizer、冷加载及 ADB。',f'配对 bootstrap 20000，seed{SEED}；{count} 个计时 profile 全部保留且通过物理审计。']
for trajectory,v in out['trajectories'].items():
 for mode in ['prefill','decode']:
  lines += ['',f'## {trajectory} · {mode}','','| 模块 | LOG2 μs（Host占比） | FP32 HVX μs（Host占比） |','|---|---:|---:|']
  for x,y in zip(v['arms']['LOG2'][mode]['modules'],v['arms']['FP'][mode]['modules']):lines.append(f"| {x['module']} | {x['us']:.1f} ({x['host_share_pct']:.2f}%) | {y['us']:.1f} ({y['host_share_pct']:.2f}%) |")
  e=v['effects'][mode];lines += ['',f"FP/LOG2 Host wall = {e['fp_over_log2_wall']:.7f}，95%CI [{e['ci95'][0]:.7f}, {e['ci95'][1]:.7f}]。"]
lines += ['','## 实际端到端吞吐','','| 轨迹 | 配置 | Prefill token/s | Decode token/s |','|---|---|---:|---:|']
for trajectory,v in out['trajectories'].items():
 for a in ARMS:
  u=v['arms'][a];lines.append(f"| {trajectory} | {a} | {u['prefill']['e2e_tps']:.4f} | {u['decode']['e2e_tps']:.4f} |")
lines += ['',f'M64+15，{LAYERS} 层。fixed 行为受控输入重放，greedy 为实际自由生成路径，两者不混合。',out['numerical_gate_scope'],'相同 HMX 工作量与权重 DDR 字节，无中间 tensor DDR/spill，8 MiB VTCM。FP32 exp 与归约使用供应商 QHL HVX；概率码行和实现见下方 Implementation；组件输出门槛独立记录。未做模型质量/PPL 验收，不晋升基线。A8 额外支持 shape 仍待执行。']
lines += ['', 'Implementation: '+out['implementation']]
with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({t:v['effects'] for t,v in out['trajectories'].items()},indent=2))
