#!/usr/bin/env python3
"""Summarize exclusive L32-0063 phase fractions without adding worker timers."""
import csv,json,sys
from pathlib import Path
import numpy as np
from report_llama32_pipeline_profile import MODULES
from profile_fp_islands import R,save

def main():
 result={};rows=[];rng=np.random.default_rng(63);idx=rng.integers(0,10,(10000,10))
 for arm,prefix in [('pipelined',''),('operator_boundaries','isolated-phases-')]:
  rounds=[json.loads((R/f'{prefix}formal-{i:02d}/validated.json').read_text())['profiles'] for i in range(10)]
  out={}
  for step,mode in enumerate(['prefill','decode_one_step']):
   batches=[v[step::2] for v in rounds];profiles=[p for v in batches for p in v]
   wall=np.array([np.mean([p['host_wall_ns']/1000 for p in v]) for v in batches]);h=float(wall.mean())
   modules={}
   for name,keys in MODULES:
    vals=np.array([np.mean([sum(p[k] for k in keys)-(p['generation_final_norm_ticks'] if 'LM head' in name else 0) for p in v])/19.2 for v in batches])
    modules[name]=float(vals.mean())
   modules['Host-DSP boundary']=h-sum(modules.values());assert modules['Host-DSP boundary']>=0
   assert abs(sum(modules.values())-h)<1e-7
   stats=dict(host_wall_us=h,ci95_us=np.percentile(wall[idx].mean(1),[2.5,97.5]).tolist(),tps=(64 if step==0 else 1)*1e6/h,modules_us=modules,modules_percent={k:100*v/h for k,v in modules.items()},peak_vtcm=max(p['vtcm_peak_plan_bytes'] for p in profiles))
   if arm=='operator_boundaries' and step==0:
    measures={'FP Softmax + QDQ':'attention_softmax_ticks','FP SwiGLU + QDQ':'activation_ticks'}
    nonlinear={}
    for name,key in measures.items():
     vals=np.array([np.mean([p[key] for p in v])/19.2 for v in batches]);pct=100*vals.mean()/h
     ci=np.percentile(100*vals[idx].mean(1)/wall[idx].mean(1),[2.5,97.5]).tolist()
     nonlinear[name]=dict(us=float(vals.mean()),percent=float(pct),ci95_percent=ci)
    stats['fp_nonlinear_including_qdq']=nonlinear
    stats['combined_percent']=sum(x['percent'] for x in nonlinear.values())
    sw=nonlinear['FP SwiGLU + QDQ']['us'];so=nonlinear['FP Softmax + QDQ']['us']
    plot=dict(modules);plot['Gate/Up matrix pipeline']=plot.pop('Gate/Up＋SwiGLU')-sw
    plot['QK/AV and attention preparation']=plot.pop('QK–Softmax–AV')-so
    plot['FP SwiGLU + QDQ']=sw;plot['FP Softmax + QDQ']=so
    assert all(x>=0 for x in plot.values()) and abs(sum(plot.values())-h)<1e-7
    stats['exclusive_plot_us']=plot
    for name,value in plot.items():rows.append(dict(model='Llama3.2-1B',prompt=64,schedule=arm,component=name,us=value,percent=100*value/h))
   # These are work sums only. They MUST NOT be added to exclusive wall.
   stats['overlapping_worker_work_us']={k:float(np.mean([p[k] for p in profiles])/19.2) for k in ['u8_attention_softmax_ticks','u8_attention_qk_hmx_ticks','u8_attention_av_hmx_ticks']}
   out[mode]=stats
  result[arm]=out
 save(R/'summary.json',result)
 with (R/'figure-b-llama1b.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 text=['# L32-0063 full additive modules','', 'Microseconds; percentages use complete Host wall. Prefill M64, one subsequent decode for correctness/diagnostics. Softmax worker work is overlapping, excluded from this ledger.','']
 text+=['| Module | Pipelined prefill | Operator-boundary prefill |','|---|---:|---:|']
 for name in result['pipelined']['prefill']['modules_us']:
  cells=[]
  for arm in result:
   v=result[arm]['prefill'];cells.append(f"{v['modules_us'][name]:.2f} ({v['modules_percent'][name]:.2f}%)")
  text.append('| '+name+' | '+' | '.join(cells)+' |')
 text+=['','Both columns include embedding, all 16 blocks, final norm, LM head, greedy and FastRPC. They are diagnostic schedules, not an optimized-vs-baseline speedup experiment.','']
 (R/'MODULES.md').write_text('\n'.join(text))
 print(json.dumps({a:{m:{k:v for k,v in s.items() if k not in ['modules_us','modules_percent','exclusive_plot_us','overlapping_worker_work_us']} for m,s in data.items()} for a,data in result.items()},indent=2))
if __name__=='__main__':main()
