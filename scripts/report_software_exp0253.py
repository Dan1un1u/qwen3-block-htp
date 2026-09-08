#!/usr/bin/env python3
import json, math, hashlib, subprocess
from pathlib import Path
import numpy as np
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0253')
CELLS=['en_wiki','zh_wiki','en_news','zh_news']
MODES=['legacy','wide_nr64','float_core']
NAMES=['F','C64']+[x+'_'+m for x in ['OFF','R3'] for m in MODES]
def read(n):return json.loads((R/n).read_text())
def write(n,x):
 with (R/n).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ds=read('dataset.json');scores={n:read('scores/final_'+n+'.json') for n in NAMES}
 assert read('development_complete.json')['pass_all'] and read('checks/numerical.json')['pass_all']
 arrays={n:np.array([[np.mean(r['nll']) for r in d['samples'] if r['cell']==c] for c in CELLS]) for n,d in scores.items()}
 for n,d in scores.items():
  assert [(r['id'],r['cell']) for r in d['samples']]==[(r['id'],r['cell']) for r in ds['samples']]
  assert arrays[n].shape==(4,32) and abs(arrays[n].mean()-d['mean_nll'])<1e-12
  assert d['freeze_sha256']==sha(R/'freeze.json')
 rng=np.random.default_rng(253);ix=rng.integers(0,32,(10000,4,32))
 def comparison(a,b):
  delta=arrays[a]-arrays[b];boot=np.take_along_axis(np.broadcast_to(delta,(10000,4,32)),ix,axis=2).mean(2)
  ci=np.quantile(boot.mean(1),[.025,.975]);mean=float(delta.mean())
  return dict(candidate=a,reference=b,delta_nll=mean,delta_nll_ci95=ci.tolist(),ppl_ratio=math.exp(mean),ratio_ci95=np.exp(ci).tolist(),
   cells={c:dict(delta_nll=float(delta[j].mean()),ppl_ratio=float(np.exp(delta[j].mean())),ratio_ci95=np.exp(np.quantile(boot[:,j],[.025,.975])).tolist()) for j,c in enumerate(CELLS)})
 pairs=[('legacy','wide_nr64'),('legacy','float_core'),('wide_nr64','float_core')]
 ladder={x:{a+'->'+b:comparison(x+'_'+b,x+'_'+a) for a,b in pairs} for x in ['OFF','R3']}
 rotation={m:comparison('R3_'+m,'OFF_'+m) for m in MODES}
 baseline={n:comparison(n,'F') for n in NAMES if n!='F'}
 quality={n:dict(point_pass=d['ppl_ratio']<=1.05 and all(v['ppl_ratio']<=1.10 for v in d['cells'].values()),
  confident_pass=d['ratio_ci95'][1]<=1.05 and all(v['ratio_ci95'][1]<=1.10 for v in d['cells'].values())) for n,d in baseline.items()}
 ins=read('inputs.json');scales={n:[ins['parameters'][n][f'L{i:02d}.attention_prob']['mse']['scale']*255 for i in range(28)] for n in ['A8','R3_ALL']}
 summary=dict(experiment='EXP-0253',scope='conditional full-model software PPL comparing standard FP32 attention core with retained U8 cache and output boundaries; not DSP PPL or a native FlashAttention kernel',
  final_ppl={n:d['ppl'] for n,d in scores.items()},cell_ppl={n:{c:math.exp(v) for c,v in d['cell_nll'].items()} for n,d in scores.items()},
  ladder=ladder,rotation_plus_frozen_QK_recalibration=rotation,vs_F=baseline,quality_gate=quality,probability_scale_times255=scales,
  samples=128,targets=2048,bootstrap=dict(repetitions=10000,seed=253,unit='document',strata=CELLS),
  vs_C64={n:comparison(n,'C64') for n in NAMES if n not in ['F','C64']},
  numerical_gate='pass',device_runs=0,short_rounds=0,formal_rounds=0,device_ppl=None,e2e_tokens_per_second=None,baseline_promoted=False,
  inference_source_heads=sorted(set(d['source_head'] for d in scores.values())),freeze_sha256=sha(R/'freeze.json'))
 write('software_summary.json',summary)
 lines=['# EXP-0253 floating attention diagnostic','',summary['scope'],'',f"F16 PPL {scores['F']['ppl']:.6f}; C64 W4A16 PPL {scores['C64']['ppl']:.6f}.",'',
 '| Attention path | OFF PPL | R3 PPL | R3 / OFF (95% CI) |','|---|---:|---:|---:|']
 for m in MODES:
  x=rotation[m];ci=x['ratio_ci95'];lines.append(f"| {m} | {scores['OFF_'+m]['ppl']:.6f} | {scores['R3_'+m]['ppl']:.6f} | {x['ppl_ratio']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |")
 lines+=['','## Conditional increments (positive means worse)','', '| Added stage | OFF delta NLL (95% CI) | R3 delta NLL (95% CI) |','|---|---:|---:|']
 for m in ladder['OFF']:
  parts=[]
  for x in ['OFF','R3']:
   d=ladder[x][m];ci=d['delta_nll_ci95'];parts.append(f"{d['delta_nll']:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}]")
  lines.append('| '+m+' | '+' | '.join(parts)+' |')
 lines+=['','## Cells','', '| Configuration | en_wiki | zh_wiki | en_news | zh_news |','|---|---:|---:|---:|---:|']
 for n in NAMES:lines.append('| '+n+' | '+' | '.join(f"{summary['cell_ppl'][n][c]:.6f}" for c in CELLS)+' |')
 lines+=['','Eight prespecified arms. All six historical development controls reproduce exactly (EXP0249 legacy; EXP0252 F/C64/NR64). Fresh128documents/2048targets; frozen weights, parameters, prefix, independent data audit. Numerical, repeat, causal, CE, immutable weight/prefix checks. No final-data selection. C64 denotes the fixed per-channel W4A16 weight control, not group64. The floating core decodes identical U8 cache semantics and removes only attention-internal quantization; output FP16 cast and context QDQ remain.','', 'No hardware execution in this experiment. E2E tokens/s: N/A (not measured).' ]
 lines+=['','## Unchanged quality gates versus F16','', '| Arm | PPL / F16 (95% CI) | Point pass | Confidence pass |','|---|---:|---|---|']
 for n in NAMES[1:]:
  d=baseline[n];ci=d['ratio_ci95'];g=quality[n]
  lines.append(f"| {n} | {d['ppl_ratio']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] | {g['point_pass']} | {g['confident_pass']} |")
 lines+=['','Float-core QK uses FP32 TwoSum residual reduction before common maximum subtraction to satisfy the unchanged Float64 numerical reference gate; see ARITHMETIC_REPAIR.md. Its arithmetic is a standard attention diagnostic reference, not the measured precision or speed of a native FlashAttention implementation. Retained U8 input codes and one output context QDQ are checked on every invocation.']
 (R/'SOFTWARE_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(summary['final_ppl'],indent=2),flush=True)
if __name__=='__main__':main()
