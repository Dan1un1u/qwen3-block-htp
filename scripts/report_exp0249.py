#!/usr/bin/env python3
import json, math, hashlib, subprocess
from pathlib import Path
import numpy as np
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0249')
CELLS=['en_wiki','zh_wiki','en_news','zh_news']
MODES=['legacy','carrier','score','exponent','sole']
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
  assert arrays[n].shape==(4,64) and abs(arrays[n].mean()-d['mean_nll'])<1e-12
  assert d['freeze_sha256']==sha(R/'freeze.json')
 rng=np.random.default_rng(249);ix=rng.integers(0,64,(10000,4,64))
 def comparison(a,b):
  delta=arrays[a]-arrays[b];boot=np.take_along_axis(np.broadcast_to(delta,(10000,4,64)),ix,axis=2).mean(2)
  ci=np.quantile(boot.mean(1),[.025,.975]);mean=float(delta.mean())
  return dict(candidate=a,reference=b,delta_nll=mean,delta_nll_ci95=ci.tolist(),ppl_ratio=math.exp(mean),ratio_ci95=np.exp(ci).tolist(),
   cells={c:dict(delta_nll=float(delta[j].mean()),ppl_ratio=float(np.exp(delta[j].mean())),ratio_ci95=np.exp(np.quantile(boot[:,j],[.025,.975])).tolist()) for j,c in enumerate(CELLS)})
 ladder={x:{b:comparison(x+'_'+b,x+'_'+a) for a,b in zip(MODES,MODES[1:])} for x in ['OFF','R3']}
 rotation={m:comparison('R3_'+m,'OFF_'+m) for m in MODES}
 baseline={n:comparison(n,'F') for n in NAMES if n!='F'}
 quality={n:dict(point_pass=d['ppl_ratio']<=1.05 and all(v['ppl_ratio']<=1.10 for v in d['cells'].values()),
  confident_pass=d['ratio_ci95'][1]<=1.05 and all(v['ratio_ci95'][1]<=1.10 for v in d['cells'].values())) for n,d in baseline.items()}
 ins=read('inputs.json');scales={n:[ins['parameters'][n][f'L{i:02d}.attention_prob']['mse']['scale']*255 for i in range(28)] for n in ['A8','R3_ALL']}
 summary=dict(experiment='EXP-0249',scope='conditional full-model software PPL with independently validated integer attention, not DSP PPL',
  final_ppl={n:d['ppl'] for n,d in scores.items()},cell_ppl={n:{c:math.exp(v) for c,v in d['cell_nll'].items()} for n,d in scores.items()},
  ladder=ladder,rotation_plus_frozen_QK_recalibration=rotation,vs_F=baseline,quality_gate=quality,probability_scale_times255=scales,
  samples=256,targets=4096,bootstrap=dict(repetitions=10000,seed=249,unit='document',strata=CELLS),
  numerical_gate='pass',device_runs=0,short_rounds=0,formal_rounds=0,device_ppl=None,e2e_tokens_per_second=None,baseline_promoted=False,
  inference_source_heads=sorted(set(d['source_head'] for d in scores.values())),freeze_sha256=sha(R/'freeze.json'))
 write('summary.json',summary)
 lines=['# EXP-0249 integer attention attribution','',summary['scope'],'',f"F16 PPL {scores['F']['ppl']:.6f}; C64 W4A16 PPL {scores['C64']['ppl']:.6f}.",'',
 '| Attention path | OFF PPL | R3 PPL | R3 / OFF (95% CI) |','|---|---:|---:|---:|']
 for m in MODES:
  x=rotation[m];ci=x['ratio_ci95'];lines.append(f"| {m} | {scores['OFF_'+m]['ppl']:.6f} | {scores['R3_'+m]['ppl']:.6f} | {x['ppl_ratio']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |")
 lines+=['','## Conditional increments (positive means worse)','', '| Added stage | OFF delta NLL (95% CI) | R3 delta NLL (95% CI) |','|---|---:|---:|']
 for m in MODES[1:]:
  parts=[]
  for x in ['OFF','R3']:
   d=ladder[x][m];ci=d['delta_nll_ci95'];parts.append(f"{d['delta_nll']:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}]")
  lines.append('| '+m+' | '+' | '.join(parts)+' |')
 lines+=['','## Cells','', '| Configuration | en_wiki | zh_wiki | en_news | zh_news |','|---|---:|---:|---:|---:|']
 for n in NAMES:lines.append('| '+n+' | '+' | '.join(f"{summary['cell_ppl'][n][c]:.6f}" for c in CELLS)+' |')
 lines+=['','## Evidence and limits','',
 'All four legacy development controls reproduce EXP0246 per-token NLL and top1 exactly. Independent int64/scalar and GPU reference agree, including actual EXP0042/EXP0248 prefill QK/probability/AV captures, random causal prefill/decode, all 56 layer configurations and 384 SOLE/exact division boundary checks.',
 'Fresh 256 documents, 64 per language/domain cell, 4096 targets. Documents, text hashes and 32-token windows excluded against historical/calibration data. Frozen configurations and independent source reconstruction precede scoring. Repeat, causal, CE, immutable weights and prefix checks pass for every mode.',
 'Carrier includes signed K clipping, V recentering, integer AV conversion, FP32-exact integer dot products and the existing probability normalization/scale discrepancy. Later score/exponent/SOLE increments change only the named factor relative to the preceding path. They are conditional effects, not independent additive component errors.',
 'R3 vs OFF includes previously frozen Q/K recalibration. Linear, norm and dense R3 operations retain the software arithmetic and EOS uses offline raw-prefix KV. Thus full-device accuracy and speed remain unmeasured. No final-based tuning, no new quantization range, no promotion.',
 'The failed initial parameter-freeze assertion is retained; it incorrectly assumed every layer probability scale was 1/255. Twenty-six of 28 frozen layers have slightly clipped probability ranges. Original dataset protocol and amended configuration protocol remain separately hashed.',
 '', 'E2E token/s: N/A (no device/full-model timing).']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n')
 profile='# EXP-0249 profiling scope\n\nSoftware arithmetic attribution and PPL only. New device runs:0; short rounds:0; formal profiling rounds:0. All DSP module timings, Host wall, physical counters and E2E token/s are N/A. GPU scoring elapsed seconds are experiment runtime, not inference throughput. Prior recipe timing references are unchanged and not rerun. No performance extrapolation or baseline promotion.\n'
 (R/'FULL_PROFILE.md').write_text(profile)
 print(json.dumps(summary['final_ppl'],indent=2),flush=True)
if __name__=='__main__':main()
