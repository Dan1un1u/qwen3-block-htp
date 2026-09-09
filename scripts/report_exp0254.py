#!/usr/bin/env python3
"""Paired inference and prespecified factorial contrasts, independent panels."""
import json,math
from pathlib import Path
import numpy as np
import data_exp0254 as data
import attention_exp0254 as a
R=data.RESULT
read=a.read
write=a.write
CELLS=data.CELLS

def analyze(phase,names):
 samples=[x for x in read('dataset.json')['samples'] if x['split']==phase]
 scores={n:read(f'scores/{phase}_{n}.json') for n in names}
 arrays={n:np.array([[np.mean(x['nll']) for x in d['samples'] if x['cell']==c] for c in CELLS]) for n,d in scores.items()}
 for n,d in scores.items():
  assert [(x['id'],x['cell']) for x in d['samples']]==[(x['id'],x['cell']) for x in samples]
  assert arrays[n].shape==(4,32) and abs(arrays[n].mean()-d['mean_nll'])<1e-12
 rng=np.random.default_rng(254);ix=rng.integers(0,32,(10000,4,32))
 def contrast(weights):
  x=sum(w*arrays[n] for n,w in weights.items());boot=np.take_along_axis(np.broadcast_to(x,(10000,4,32)),ix,axis=2).mean(2)
  return dict(weights=weights,mean_nll=float(x.mean()),ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),cells={c:dict(mean_nll=float(x[j].mean()),ci95=np.quantile(boot[:,j],[.025,.975]).tolist()) for j,c in enumerate(CELLS)})
 def pair(n,ref):
  d=contrast({n:1,ref:-1});d.update(candidate=n,reference=ref,ppl_ratio=math.exp(d['mean_nll']),ratio_ci95=np.exp(d['ci95']).tolist())
  for v in d['cells'].values():v.update(ppl_ratio=math.exp(v['mean_nll']),ratio_ci95=np.exp(v['ci95']).tolist())
  return d
 vsf={n:pair(n,'F') for n in names if n!='F'};vsc={n:pair(n,'C64') for n in names if n not in ['F','C64']}
 pairs={n+'_vs_B':pair(n,'B') for n in names if n not in ['F','C64','B']}
 interactions={}
 if phase=='confirmation':
  if read('route.json')['branch']=='downstream':
   for n in ['JR','JS','JRS']:pairs[n+'_vs_J']=pair(n,'J')
   interactions=dict(J_with_R=contrast({'J':1,'R':1,'JR':-1,'B':-1}),J_with_S=contrast({'J':1,'S':1,'JS':-1,'B':-1}),R_with_S=contrast({'R':1,'S':1,'RS':-1,'B':-1}),R_with_S_given_J=contrast({'JR':1,'JS':1,'JRS':-1,'J':-1}),total_joint=contrast({'J':1,'R':1,'S':1,'JRS':-1,'B':-2}))
  else:
   pairs['QK_vs_K']=pair('QK','K');interactions['joint_over_sum_QK_V_C']=contrast({'QK':1,'V':1,'C':1,'J':-1,'B':-2})
 quality={n:dict(point_pass=d['ppl_ratio']<=1.05 and all(x['ppl_ratio']<=1.1 for x in d['cells'].values()),confident_pass=d['ratio_ci95'][1]<=1.05 and all(x['ratio_ci95'][1]<=1.1 for x in d['cells'].values())) for n,d in vsf.items()}
 return dict(names=names,ppl={n:d['ppl'] for n,d in scores.items()},cell_ppl={n:{c:math.exp(v) for c,v in d['cell_nll'].items()} for n,d in scores.items()},vs_F=vsf,vs_C64=vsc,pairs=pairs,interactions_positive_extra_joint_benefit=interactions,quality_gate=quality,documents=128,targets=2048)

def main():
 a.preflight();a.frozen();assert read('confirmation_complete.json')['pass_all']
 route=read('route.json');triage=analyze('triage',a.BASE_NAMES);confirmation=analyze('confirmation',a.selected())
 source=subprocess_head();heads=sorted({read(str(p.relative_to(R)))['source_head'] for p in (R/'scores').glob('*.json')})
 summary=dict(experiment='EXP-0254',route=route,triage=triage,confirmation=confirmation,inference_source_heads=heads,source_head=source,bootstrap=dict(seed=254,repetitions=10000,unit='document',strata=CELLS),numerical_gate='pass',device_runs=0,device_ppl=None,e2e_tokens_per_second=None,baseline_promoted=False)
 write('summary.json',summary)
 labels={'F':'F16','C64':'W4A16 control','B':'R3 floating core, A8 boundaries','J':'Joint attention restored','K':'K projection/cache restored','QK':'Q/K projection/cache restored','V':'V output/cache restored','C':'Attention context restored','R':'Residual restored','S':'SwiGLU restored','RS':'Residual + SwiGLU','JR':'Joint attention + residual','JS':'Joint attention + SwiGLU','JRS':'Joint attention + residual + SwiGLU'}
 lines=['# EXP-0254 ordered joint restoration','','Software diagnostics with fixed per-channel W4, dense R3, EOS prefix and floating attention core. Restorations are not deployed A8 candidates. Both panels frozen before inference and independently held out from all old data.','',f"Prespecified route: **{route['branch']}**, triage J/C64={route['J_over_C64']:.6f}; routing only, not model acceptance.",'']
 for phase,d in [('Triage',triage),('Independent confirmation',confirmation)]:
  lines += ['## '+phase,'','128documents,2048targets; equal-cell NLL.','', '| Configuration | PPL | PPL/F16 | PPL/C64 | Model point gate |','|---|---:|---:|---:|---|']
  for n in d['names']:
   lines.append(f"| {labels[n]} | {d['ppl'][n]:.6f} | {d['ppl'][n]/d['ppl']['F']:.6f} | {d['ppl'][n]/d['ppl']['C64']:.6f} | {d['quality_gate'].get(n,{}).get('point_pass','reference')} |")
  lines += ['','| Paired intervention | PPL ratio (95% CI) |','|---|---:|']
  for n,x in d['pairs'].items():lines.append(f"| {n} | {x['ppl_ratio']:.6f} [{x['ratio_ci95'][0]:.6f}, {x['ratio_ci95'][1]:.6f}] |")
  if d['interactions_positive_extra_joint_benefit']:
   lines += ['','| Interaction: positive = extra joint NLL benefit | Estimate (95% CI) |','|---|---:|']
   for n,x in d['interactions_positive_extra_joint_benefit'].items():lines.append(f"| {n} | {x['mean_nll']:+.6f} [{x['ci95'][0]:+.6f}, {x['ci95'][1]:+.6f}] |")
  lines += ['','| Configuration | en_wiki | zh_wiki | en_news | zh_news |','|---|---:|---:|---:|---:|']
  for n in d['names']:lines.append('| '+n+' | '+' | '.join(f"{d['cell_ppl'][n][c]:.6f}" for c in CELLS)+' |')
 lines += ['','Original model gates: overall5percent and eachcell10percent vs matched F16; pointwise paired confidence reported separately in summary.json. C64 is a matched weight control, not a mathematical floor. Conditional restoration effects and interactions are not additive causal fractions or bounds on deployed quantization.','', 'Exact EXP0253 development reproduction for F/C64/B, independent mixed-cache oracle for every evaluated mask, restored-versus-QDQ hook sentinels, full invocation counts, repeat/causality/CE, immutable weights/prefix and independently reconstructed disjoint data are recorded under checks/ and scores/.','', 'E2E tokens/s: N/A; no device run, new weights or native FlashAttention implementation.']
 with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print(json.dumps(dict(route=route['branch'],triage=triage['ppl'],confirmation=confirmation['ppl']),indent=2),flush=True)
def subprocess_head():
 import subprocess
 return subprocess.check_output(['git','rev-parse','HEAD'],cwd=data.SOURCE,text=True).strip()
if __name__=='__main__':main()
