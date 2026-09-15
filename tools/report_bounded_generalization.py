"""Complete-module summaries for the pre-registered bounded cases."""
import sys,json,statistics
from pathlib import Path
import numpy as np
model,root=sys.argv[1:];R=Path(root);S=Path('/home/daniuniu/work')/model;sys.path.insert(0,str(S/('scripts' if model.startswith('qwen') else 'tools')))
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
 for line in p.read_text().splitlines():
  try:v=json.loads(line)
  except ValueError:continue
  if isinstance(v,dict) and v.get('record')=='generation_profile':yield v
out=dict(model=model,quality_claim=False,baseline_promoted=False,cases={},measurement='warm fullmodel includes embedding,alllayers,finalnorm,head,greedy,FastRPC; excludes tokenizer,coldload,ADB; cache128 maxKV127')
idx=np.random.default_rng(28137).integers(0,10,(20000,10));lines=['# Bounded latest runtime generalization','',out['measurement'], '', 'Two fixed real-text prompts truncated to64 tokenizer tokens; this is numerical/performance generalization, not instruction-following or PPL. Latest and original execute identical quantized arithmetic. Repeat5,5short10formal cyclic paired runs. No factorial rerun.','']
for case,steps in [('A16',16),('A64',64),('B64',64)]:
 z=read(R/f'{case}-formal.json');assert z['pass_all'];out['cases'][case]={};allps={}
 for arm in ['LATEST','ORIGINAL']:
  ps=[v for c in range(10) for v in records(R/f'{case}/formal/{c:02d}-{arm}'/suffix)];assert len(ps)==10*5*steps;allps[arm]=ps;out['cases'][case][arm]={}
  for mode in ['prefill','decode']:
   vs=[v for v in ps if v['mode']==mode];ns=[normalized([v]) for v in vs];wall=statistics.mean(v['host_wall_ns']/1000 for v in vs);rows=[dict(module=n,us=statistics.mean(sum(v[k] for k in keys)/19.2 for v in ns)) for n,keys in GROUPS];rows += [dict(module='Host-DSP boundary',us=wall-sum(v['us'] for v in rows)),dict(module='Complete Host wall',us=wall)]
   for row in rows:row['host_share_pct']=row['us']/wall*100
   out['cases'][case][arm][mode]=dict(host_us=wall,tps=(64 if mode=='prefill' else 1)*1e6/wall,modules=rows)
 effects={}
 for mode in ['prefill','decode']:
  ts={a:np.array([next(v[mode+'_ns'] for v in z['runs'] if v['cycle']==c and v['arm']==a) for c in range(10)]) for a in ['LATEST','ORIGINAL']};a,b=ts['LATEST'],ts['ORIGINAL'];effects[mode]=dict(latest_over_original_wall=float(a.mean()/b.mean()),ci95=np.quantile(a[idx].mean(1)/b[idx].mean(1),[.025,.975]).tolist())
 out['cases'][case]['effects']=effects
 for mode in ['prefill','decode']:
  lines += ['## '+case+' '+mode,'','| Module | Original μs (Host share) | Latest μs (Host share) |','|---|---:|---:|']
  for i,row in enumerate(out['cases'][case]['LATEST'][mode]['modules']):lines.append('| '+row['module']+' | '+' | '.join(f"{out['cases'][case][a][mode]['modules'][i]['us']:.2f} ({out['cases'][case][a][mode]['modules'][i]['host_share_pct']:.2f}%)" for a in ['ORIGINAL','LATEST'])+' |')
 bins={}
 for low,high in [(65,96),(97,127)]:
  vals={a:[v for v in ps if v['mode']=='decode' and low<=v['valid_length']<=high] for a,ps in allps.items()}
  if vals['LATEST']:bins[f'KV{low}-{high}']={a:dict(host_us=statistics.mean(v['host_wall_ns']/1000 for v in vs),tps=1e9/statistics.mean(v['host_wall_ns'] for v in vs)) for a,vs in vals.items()}
 out['cases'][case]['decode_bins']=bins
lines += ['','## Complete-model E2E','','| Case | Original prefill tps | Latest prefill tps | Original decode tps | Latest decode tps |','|---|---:|---:|---:|---:|']
for c,z in out['cases'].items():lines.append('| '+c+' | '+' | '.join(f"{z[a][m]['tps']:.2f}" for m in ['prefill','decode'] for a in ['ORIGINAL','LATEST'])+' |')
for f,z in [('SUMMARY.json',out)]:
 with (R/f).open('x') as h:json.dump(z,h,ensure_ascii=False,indent=2);h.write('\n')
with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({c:dict(tps={a:{m:z[a][m]['tps'] for m in ['prefill','decode']} for a in ['ORIGINAL','LATEST']},effects=z['effects']) for c,z in out['cases'].items()},indent=2))
