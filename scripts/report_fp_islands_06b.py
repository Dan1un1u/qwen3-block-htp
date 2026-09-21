#!/usr/bin/env python3
"""EXP0306 exclusive phase ledger; no summation of overlapping worker times."""
import json,csv,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S/'scripts'))
from measure_exp0218 import LEDGER,OVERVIEW
from summarize_exp0217 import normalized
from profile_fp_islands_06b import R,save
rounds=[json.loads((R/f'formal-{i:02d}/validated.json').read_text())['profiles'] for i in range(10)]
rng=np.random.default_rng(305);idx=rng.integers(0,10,(10000,10));result={};csvrows=[]
for step,mode in enumerate(['prefill','decode_one_step']):
 batches=[v[step::2] for v in rounds];profiles=[p for v in batches for p in v]
 wall=np.array([np.mean([p['host_wall_ns']/1000 for p in v]) for v in batches]);h=float(wall.mean())
 modules={name:float(np.mean([normalized(v)[key]/19.2 for v in batches])) for name,key in LEDGER}
 modules['Host-DSP boundary']=h-sum(modules.values());assert modules['Host-DSP boundary']>=0
 assert abs(sum(modules.values())-h)<1e-7
 out=dict(host_wall_us=h,ci95_us=np.percentile(wall[idx].mean(1),[2.5,97.5]).tolist(),tps=(64 if step==0 else 1)*1e6/h,modules_us=modules,modules_percent={k:100*v/h for k,v in modules.items()},peak_vtcm=max(p['vtcm_peak_plan_bytes'] for p in profiles))
 if step==0:
  nonlinear={}
  for name,key in [('FP Softmax + QDQ','attention_softmax_ticks'),('FP SwiGLU + QDQ','activation_ticks')]:
   vals=np.array([np.mean([p[key] for p in v])/19.2 for v in batches]);nonlinear[name]=dict(us=float(vals.mean()),percent=float(100*vals.mean()/h),ci95_percent=np.percentile(100*vals[idx].mean(1)/wall[idx].mean(1),[2.5,97.5]).tolist())
  out['fp_nonlinear_including_qdq']=nonlinear;out['combined_percent']=sum(x['percent'] for x in nonlinear.values())
  plot=dict(modules);so=nonlinear['FP Softmax + QDQ']['us'];sw=nonlinear['FP SwiGLU + QDQ']['us']
  plot['QK/AV and attention preparation']=plot.pop('QK-Softmax-AV')-so
  assert abs(plot.pop('SwiGLU')-sw)<1e-7
  plot['FP SwiGLU + QDQ']=sw;plot['FP Softmax + QDQ']=so
  assert all(x>=0 for x in plot.values()) and abs(sum(plot.values())-h)<1e-7
  out['exclusive_plot_us']=plot
  for name,v in plot.items():csvrows.append(dict(model='Qwen3-0.6B',prompt=64,schedule='operator_boundaries',component=name,us=v,percent=100*v/h))
 out['overlapping_worker_work_us']={k:float(np.mean([p[k] for p in profiles])/19.2) for k in ['u8_attention_softmax_ticks','u8_attention_qk_hmx_ticks','u8_attention_av_hmx_ticks']}
 result[mode]=out
save(R/'summary.json',result)
with (R/'figure-b-qwen06b.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
text=['# EXP-0306 full additive modules','','Microseconds; percentages use complete Host wall. Prefill M64, one subsequent decode for correctness and diagnostics. Four vector contexts; explicit QK/Softmax/AV phase boundaries and GateUp/SwiGLU boundary. Worker work sums are excluded from the additive ledger.','','| Module | Prefill | One-step decode |','|---|---:|---:|']
for name in result['prefill']['modules_us']:
 cells=[f"{result[m]['modules_us'][name]:.2f} ({result[m]['modules_percent'][name]:.2f}%)" for m in result]
 text.append('| '+name+' | '+' | '.join(cells)+' |')
text+=['','E2E: '+', '.join(f"{m}: {s['tps']:.2f} token/s" for m,s in result.items()),'','Includes embedding, all 28 blocks, final norm, head, greedy and FastRPC. Diagnostic phase costs are not critical-path shares of a production-overlapped schedule. No model-quality or baseline promotion claim.']
(R/'MODULES.md').write_text('\n'.join(text)+'\n')
print(json.dumps({m:{k:v for k,v in s.items() if k not in ['modules_us','modules_percent','exclusive_plot_us','overlapping_worker_work_us']} for m,s in result.items()},indent=2))
