import sys,json,statistics
from pathlib import Path
import numpy as np
model,root,exp=sys.argv[1:4];R=Path(root);S=Path('/home/daniuniu/work')/model
sys.path.insert(0,str(S/('scripts' if model.startswith('qwen') else 'tools')))
if model.startswith('qwen'):
 from report_exp0273 import GROUPS
 from summarize_exp0217 import normalized
 suffix='stdout.jsonl'
else:
 from report_llama32_pipeline_profile import MODULES as GROUPS
 suffix='stdout.txt'
 def normalized(v):
  x=dict(v[0]);x['generation_lm_head_ticks']-=x['generation_final_norm_ticks'];return x
def read(p):return json.loads(p.read_text())
def records(p):
 for l in p.read_text().splitlines():
  try:v=json.loads(l)
  except ValueError:continue
  if isinstance(v,dict) and v.get('record')=='generation_profile':yield v
z=read(R/'fixed_formal.json');assert z['pass_all'];ARMS=list(dict.fromkeys(x['arm'] for x in z['runs']));assert len(z['runs'])==10*len(ARMS)
idx=np.random.default_rng(28035).integers(0,10,(20000,10));out=dict(experiment=exp,shape='M64+15',rotation=False,sp2=8,FP32_residual=True,quality_claim=False,baseline_promoted=False,arms={},effects={})
for a in ARMS:
 ps=[v for c in range(10) for v in records(R/f'fixed-formal/{c:02d}-{a}'/suffix)];assert len(ps)==1600;out['arms'][a]={}
 for mode in ['prefill','decode']:
  vs=[v for v in ps if v['mode']==mode];ns=[normalized([v]) for v in vs];h=statistics.mean(v['host_wall_ns']/1000 for v in vs)
  rows=[dict(module=n,us=statistics.mean(sum(v[k] for k in ks)/19.2 for v in ns)) for n,ks in GROUPS]
  rows += [dict(module='Host-DSP boundary',us=h-sum(v['us'] for v in rows)),dict(module='Complete Host wall',us=h)]
  for row in rows:row['host_share_pct']=100*row['us']/h
  assert abs(sum(row['us'] for row in rows[:-1])-h)<1e-5
  counters={k:statistics.mean(v[k] for v in vs) for k in vs[0] if isinstance(vs[0][k],(int,float)) and any(t in k for t in ['wait_ticks','worker_ticks','overlap','command_count','tile_pair_count','weight_ddr_read_bytes'])}
  out['arms'][a][mode]=dict(host_us=h,tps=(64 if mode=='prefill' else 1)*1e6/h,modules=rows,counters=counters)
for mode in ['prefill','decode']:
 t={a:np.array([next(x[mode+'_ns'] for x in z['runs'] if x['arm']==a and x['cycle']==c) for c in range(10)]) for a in ARMS};out['effects'][mode]={}
 for a,b in [('FAST','LOG2'),('FAST','FP'),('FP','LOG2')]:
  if a not in t or b not in t:continue
  ratio=t[a].mean()/t[b].mean();ci=np.quantile(t[a][idx].mean(1)/t[b][idx].mean(1),[.025,.975]).tolist();out['effects'][mode][a+'_over_'+b]=dict(ratio=float(ratio),ci95=ci)
# Include repeat1 as auxiliary, never a selection gate.
out['repeat1_auxiliary']={a:read(R/f'fixed-aux-{a}-r1/validated.json') for a in ARMS}
lines=[f'# {exp} exact log2 softmax optimization','', 'No rotation; per-output-channel native W4, SP2mode8, FP32 residual, latest native-row1 decode. LOG2 retains original clipped-score exact log2; FAST preserves its full integer probabilities; FP uses vendor HVX exp and its frozen own reference, same U8 output/AV. All weights/qparams fixed. Fixed-token primary trajectory,5short10formal cyclic paired rounds, repeat10; repeat1 auxiliary. Greedy and fixed full16 independent teacher gates both pass; selectedlayer7/chain3/chain16 pass; complete8MiB/nointermediateDDR/no spill/exact ledger. No PPL, quality or baseline promotion.','', 'The original C expression uses uint64, but compiler already lowers its16 divisions to unsigned32 helpers. The first candidate reduces16 divisions to one; do not claim a64-to32 instruction-width optimization. Component diagnostics enable numerical telemetry whereas formal full-model runs disable it, so component time is not directly extrapolated.','']
for mode in ['prefill','decode']:
 lines += ['## '+mode,'','| Module | '+' | '.join(ARMS)+' |','|---|'+'---:|'*len(ARMS)]
 for i,row in enumerate(out['arms'][ARMS[0]][mode]['modules']):
  lines.append('| '+row['module']+' | '+' | '.join(f"{out['arms'][a][mode]['modules'][i]['us']:.2f} ({out['arms'][a][mode]['modules'][i]['host_share_pct']:.2f}%)" for a in ARMS)+' |')
 lines += ['','| Paired Host wall ratio | Estimate | 95% CI |','|---|---:|---:|']
 for n,v in out['effects'][mode].items():lines.append(f"| {n} | {v['ratio']:.6f} | {v['ci95']} |")
lines += ['','## Warm complete-model E2E','','| Configuration | Prefill token/s | Decode token/s |','|---|---:|---:|']
for a in ARMS:lines.append(f"| {a} | {out['arms'][a]['prefill']['tps']:.3f} | {out['arms'][a]['decode']['tps']:.3f} |")
lines += ['','Includes embedding, complete stack, finalNorm, full-vocabulary head, greedy and FastRPC. Excludes tokenizer, cold model loading and ADB. Prefill is64/Hostwall; decode is15/summed continuous decode Hostwall. Overlapping work counters are not additive wall contributions. Complete FARF timeline remains unavailable.']
out['component']={}
lines += ['','## Matched4-head component, audit telemetry enabled','','| Shape | LOG2 us | FAST us | FP us |','|---|---:|---:|---:|']
for case in ['m64-p0-random','m1-p64-random']:
 vals=read(R/f'component-{case}-formal.json')['runs'];means={a:statistics.mean(v['ticks']/v['repeat']/19.2 for v in vals if v['arm']==a) for a in ['LOG2','FAST','FP']};out['component'][case]=means;lines.append('| '+case+' | '+' | '.join(f'{means[a]:.3f}' for a in ['LOG2','FAST','FP'])+' |')
out['source_head']=read(R/'runtime-l16.json')['seal']['source_head']
with (R/'SUMMARY.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(tps={a:{m:out['arms'][a][m]['tps'] for m in ['prefill','decode']} for a in ARMS},effects=out['effects']),indent=2))
