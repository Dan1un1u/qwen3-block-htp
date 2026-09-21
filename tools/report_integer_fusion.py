#!/usr/bin/env python3
"""Immutable paired-ablation tables from retained raw profiles, no resampling selection."""
import json,sys,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
from report_llama32_pipeline_profile import MODULES

def read(p):return json.loads(p.read_text())
def write(p,text):
 assert not p.exists(),p;p.write_text(text)
def report(r,model,modes,layers):
 summary=read(r/'formal-summary.json');tables={};counter={};totalprofiles=0;peak=0
 for phase in ['prefill','decode']:
  tables[phase]={};counter[phase]={}
  for mode in modes:
   samples=[]
   for i in range(10):
    raw=read(r/f'formal-{i:02d}-{mode}/records.json');pp=[p for p in raw if p.get('record')=='generation_profile'];assert len(pp)==430
    totalprofiles+=len(pp) if phase=='prefill' else 0
    for p in pp:
     assert sum(sum(p[k] for k in f) for _,f in MODULES)-p['generation_final_norm_ticks']==p['invocation_ticks']
     peak=max(peak,p['vtcm_peak_plan_bytes'])
    samples += pp[::43] if phase=='prefill' else [p for j,p in enumerate(pp) if j%43]
   host=np.mean([p['host_wall_ns']/1000 for p in samples]);rows=[]
   for name,fields in MODULES:
    us=float(np.mean([sum(p[k] for k in fields)-(p['generation_final_norm_ticks'] if fields==['generation_lm_head_ticks'] else 0) for p in samples])/19.2)
    rows.append(dict(module=name,us=us,host_percent=100*us/host))
   boundary=host-sum(x['us'] for x in rows);assert boundary>=0
   rows += [dict(module='Host–DSP boundary',us=boundary,host_percent=100*boundary/host),dict(module='Full Host wall',us=float(host),host_percent=100)]
   tables[phase][mode]=rows
   counter[phase][mode]={k:float(np.mean([p[k] for p in samples])/19.2) for k in ['u8_attention_av_requant_ticks','u8_attention_softmax_ticks','w4u8_gate_up_swiglu_worker_ticks','o_projection_ticks']}
 doc=['# '+model+' matched integer and scale-fusion ablations','',
 'M64 prefill +42 fixed-input decode. Native per-channel W4, uniform INT16 Down, no rotation, FP32 residual and RMSNorm, unchanged KV and high precision embedding. Same production layout, scheduling flags, workers and token trace within the model. Host wall includes embedding, all layers, final norm, LM head, greedy and FastRPC; excludes loading, tokenizer and ADB. No model-quality/PPL claim or baseline promotion.','',
 'F: vector floating Softmax/SwiGLU with fused necessary QDQ. I: existing integer nonlinear producers and standalone AV conversion. M: AV common multiplier and zero correction folded into O, where numerically eligible. F/I attributes the combined nonlinear implementation and associated boundary handling, not QDQ alone; I/M attributes only AV-to-O. Other fusion/layout evidence remains historical and is not relabeled as measured INT16.','',
 'Five short rounds and ten formal rounds, repeat10 each, warmup before measurement. Formal rounds alternate or rotate complete arms. All samples retained;95% intervals bootstrap the ten paired round means (20,000 draws). No independence assumption across the ten repeats inside a round. Decode latency below is the complete42-token phase.','',
 '| Configuration | Prefill ms | Prefill token/s | Decode42 ms | Decode token/s |','|---|---:|---:|---:|---:|']
 for m in modes:
  a,b=summary['prefill'][m],summary['decode'][m];doc.append(f"| {m} | {a['wall_ms']:.6f} | {a['tps']:.2f} | {b['wall_ms']:.6f} | {b['tps']:.2f} |")
 doc += ['','| Comparison | Phase | Wall reduction | Optimized/control wall ratio,95%CI |','|---|---|---:|---|']
 for phase in ['prefill','decode']:
  for pair in ['F->I','I->M','F->M']:
   if pair not in summary[phase]:continue
   d=summary[phase][pair];lo,hi=d['ci95'];doc.append(f"| {pair} | {phase} | {d['wall_reduction_pct']:.4f}% | {d['wall_ratio']:.6f} [{lo:.6f}, {hi:.6f}] |")
 doc += ['',f'All {totalprofiles:,} formal invocations pass the hardware status,8MiB/no intermediate DDR/no spill and exact additive ledger checks. Peak VTCM {peak:,}bytes. Single-layer,three-layer and full-model audits use independent own-contract references. Each full arm checks {layers*43:,} layer outputs plus43 head token/code pairs. The FP component exhaustively checks all {layers*65536:,} Gate/Up input code pairs. Own-contract correctness does not assert F/I or I/M bit equality.','']
 if 'M' in modes:
  doc += ['The generic Llama scan fallback formerly honored live-row AV conversion only under the3B compile guard. This experiment uses the common audited4-row HVX granule for1B too and separates AV conversion timing from matrix timing. Padding-poison tests pass. The fair control removes most of the previously reported AV-fold speedup: prefill has no confirmed benefit; decode improves only about0.4%. Do not reuse the older~4% decode claim as this matched result.','',
  'Llama I/M reuse independently computed L32-0062 references under verified unchanged payloads. On the checked trajectories the fold bypasses no AV saturation. Floating scale reassociation can still change rounding and downstream quantized decisions; bitwise I/M equivalence and PPL acceptability are not claimed.','']
 else:
  rej=read(r/'M-rejection.json');doc += [f"Qwen unconditional AV folding is rejected before timing: {rej['candidate_trajectory_saturated']:,}/{rej['candidate_trajectory_elements']:,} candidate AV values exceed the original U8 output range, including6 in the first layer on identical inputs. Saturation cannot commute through O. Candidate package/reference are preserved as negative evidence, not a valid speedup arm. No tolerance or saturation rule was relaxed.",'',
  'Qwen starts from the verified909 recovered EXP0284 payloads. The32 retired old chain arrays are explicitly excluded according to the prior recovery record. New own references are recomputed. Only the expected-token capacity metadata was extended from16 to64 in control-v2; the16 historical fixed inputs are continued by repeating the final frozen token. Arithmetic payloads are unchanged.','']
 doc += ['## Scope of performance evidence','',
 'These are end-to-end comparisons with the same production overlap. Component counters below are worker or substage observations; they must not be stacked as additional Host-wall percentages. In particular, eliminated AV conversion is not guaranteed to reduce the full critical path by its entire duration.','',
 '| Phase | Arm | AV RQ µs | Softmax worker µs | SwiGLU worker µs | O stage µs |','|---|---|---:|---:|---:|---:|']
 for phase in counter:
  for m,c in counter[phase].items():doc.append('| '+phase+' | '+m+' | '+' | '.join(f'{v:.3f}' for v in c.values())+' |')
 doc += ['','Per-module additive tables: MODULES.md and modules.json. Raw profiles, commands, build seals, independent references and failed attempts remain beside this report.','']
 write(r/'REPORT.md','\n'.join(doc));write(r/'modules.json',json.dumps(tables,ensure_ascii=False,indent=2)+'\n');write(r/'component-counter-summary.json',json.dumps(counter,indent=2)+'\n')
 lines=['# Additive module latency','', 'Units µs. Parentheses show share of complete Host wall. Prefill is per64-token invocation; decode is per single-token invocation, averaged over all42 positions and100 formal trajectories.','']
 for phase in tables:
  lines+=['## '+phase,'','| Module | '+' | '.join(modes)+' |','|---|'+'---:|'*len(modes)]
  for j,row in enumerate(tables[phase][modes[0]]):
   lines.append('| '+row['module']+' | '+' | '.join(f"{tables[phase][m][j]['us']:.3f} ({tables[phase][m][j]['host_percent']:.3f}%)" for m in modes)+' |')
  lines += ['', 'E2E token/s: '+', '.join(f"{m}={summary[phase][m]['tps']:.2f}" for m in modes), '']
 write(r/'MODULES.md','\n'.join(lines))
 with (r/'chart-data.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['model','configuration','prefill_ms','prefill_tps','decode42_ms','decode_tps'])
  for m in modes:w.writerow([model,m,summary['prefill'][m]['wall_ms'],summary['prefill'][m]['tps'],summary['decode'][m]['wall_ms'],summary['decode'][m]['tps']])
 print('REPORT',model,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='llama':report(Path('/mnt/d/llm_exp/results/llama32-htp/l32-0065'),'Llama3.2-1B',['F','I','M'],16)
 else:report(Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0307'),'Qwen3-1.7B',['F','I'],28)
