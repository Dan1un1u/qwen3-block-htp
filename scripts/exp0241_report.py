#!/usr/bin/env python3
"""EXP0241 complete three-way report from retained, immutable device rounds."""
import json, math, statistics, subprocess, hashlib, shutil
from pathlib import Path
import numpy as np
import exp0241_device as d
from exp0240_report import FIELDS, MODULES, means, coherent_center
R=d.RESULT
CELLS=['control','lpbq32','direct']
def records(tag):
 return [json.loads(l) for l in (R/tag/'stdout.jsonl').read_text().splitlines() if '\"record\":\"exp0240_profile\"' in l]
def main():
 audit=json.loads((R/'projection_audit_01_integer_audit.json').read_text());assert audit['pass']
 provenance=json.loads((R/'final_formal_provenance.json').read_text())
 z={'experiment':'EXP-0241','formal_source':provenance['source_head'],'reporting_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.SOURCE,text=True).strip(),'threshold_percent':10,'full_model_executed':False,'quality_evaluated':False,'baseline_promoted':False,'rounds':{'short':5,'formal':10},'performance':{},'modules':{},'physical':{},'correctness':{'checks':len(audit['projections']),'elements':sum(v['elements'] for v in audit['projections']),'mismatches':sum(v['mismatches'] for v in audit['projections'])}}
 rng=np.random.default_rng(241);allrows=[];allruns={}
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   runs={}
   for cell in CELLS:
    runs[cell]=[]
    for i in range(10):
     rs=records(f'final_formal_{i:02d}_r{rep}_{cell}');assert len(rs)==rep*9
     for x in rs:
      assert sum(x[k] for k in FIELDS)==x['ledger_named_ticks']==x['invocation_ticks']
      assert x['ledger_unattributed_ticks']==0 and x['vtcm_acquired_bytes']==8388608
      assert x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==x['intermediate_spill_fill_count']==0
      assert x['block_invocation_count']==1
     subset=[x for x in rs if (x['replay_step']==0)==(mode=='prefill')];allrows.extend(subset);runs[cell].append(means(subset))
   key=f'repeat{rep}_{mode}';allruns[key]=runs
   values={c:np.array([x['host_wall_ns']/1000 for x in v]) for c,v in runs.items()}
   z['performance'][key]={'median_us':{c:float(np.median(v)) for c,v in values.items()},'comparisons':{}}
   for candidate,control in [('direct','control'),('direct','lpbq32'),('lpbq32','control')]:
    ratios=values[candidate]/values[control];ci=np.quantile(np.median(ratios[rng.integers(0,10,(50000,10))],axis=1),[.025,.975])
    z['performance'][key]['comparisons'][candidate+'_vs_'+control]={'paired_ratio':float(np.median(ratios)),'regression_percent':float((np.median(ratios)-1)*100),'ratio_ci95':ci.tolist()}
   z['modules'][key]={}
   for cell,rr in runs.items():
    center,indices=coherent_center(rr);host=center['host_wall_ns']/1000
    modules={name:sum(center[f] for f in fields)/19.2 for name,fields in MODULES}
    modules['Host-DSP boundary']=host-center['invocation_ticks']/19.2
    assert math.isclose(sum(modules.values()),host,abs_tol=1e-6)
    modules['Complete Host wall']=host
    z['modules'][key][cell]={'round_indices':indices,'values_us':modules}
 for cell in CELLS:
  rows=[x for i in range(10) for x in records(f'final_formal_{i:02d}_r10_{cell}')]
  z['physical'][cell]={k:sorted({x[k] for x in rows}) for k in ['vtcm_requested_bytes','vtcm_acquired_bytes','vtcm_peak_plan_bytes','weight_ddr_read_bytes','hmx_u8s8_tile_pair_count','intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','block_invocation_count','ledger_unattributed_ticks']}
 stop=any(z['performance'][f'repeat10_{m}']['comparisons']['direct_vs_control']['ratio_ci95'][0]>1.1 for m in ['prefill','decode'])
 within=all(z['performance'][f'repeat10_{m}']['comparisons']['direct_vs_control']['ratio_ci95'][1]<=1.1 for m in ['prefill','decode'])
 z['decision']='stop_before_full_model_discuss' if stop else 'eligible_for_separate_full_model_continuation' if within else 'uncertain_requires_additional_pairs'
 (R/'closure.json').write_text(json.dumps(z,indent=2,ensure_ascii=False)+'\n')
 text=['# EXP-0241: exact LPBQ32 direct-W4 experiment','',f"Decision: {z['decision']}. No baseline promotion. Full model not started.",'','Quantizer and integer semantics are unchanged from EXP0240. For m in1..16 and signed q in[-8,7], q*m=sum_b(2**b)*(q if bit_b(m) else0). Five packed-W4 masks are produced inside timed DSP VTCM. Each mask is consumed2**b times with weight.n into a single accumulator, then one unchanged bias/scale/U8 conversion. This implementation issues31 HMX passes including zero-mask work; no S8 weight buffer is materialized. It is a bounded exact algorithm, not a lower bound on optimal LPBQ execution.','', 'DDR weights retain the original compressed size plus multiplier metadata. Host only reorders nibbles to direct-n lane order; group products/masks are never pre-expanded in DDR. All seven projections use mode3. Historical expansion counter names measure packed-W4 masking for mode3, not S8 expansion. HMX tile-pair counters include every repeated pass; attention U8xS8 is unchanged. QKV batches4; O/Gate/Up batches4; Down batches2. Mask buffers fit existing1MiB slots (QKV4=655360 bytes; Down2=983040 bytes); zero intermediate DDR/spill,8MiB VTCM and oneRPC/step remain.','', 'Scope: real layer14 M64 prefill followed by eight teacher-input M1 steps with self-computed persistent KV. Repeat10 means ten complete replays within one loaded Prepared Runtime, state/cache reset before each replay; no discarded warmup or outliers. Five short and ten formal rounds, three-way rotated/reversed order. Latencies are medians of ten round means; ratios are medians of paired round ratios. Bootstrap50000 seed241. No token boundary or extrapolated throughput.','',f"Correctness: {z['correctness']}. Independent S8 reconstruction, int64 cross-check and SDK native final conversion; complete layer/KV against EXP0240 and unit-multiplier against per-channel verified before collection. Original failed diagnostics, if any, are retained separately.",'','| Scope | Per-channel us | LPBQ S8 us | LPBQ direct-W4 us | Direct vs per-channel | Ratio95% CI |','|---|---:|---:|---:|---:|---|']
 for key,v in z['performance'].items():
  a=v['median_us'];q=v['comparisons']['direct_vs_control'];text.append(f"|{key}|{a['control']:.4f}|{a['lpbq32']:.4f}|{a['direct']:.4f}|{q['regression_percent']:+.3f}%|{q['ratio_ci95']}|")
 for key,table in z['modules'].items():
  text+=['',f'## Complete additive module table: {key}','','Units us; parentheses percent of complete Host wall. Same median-Host-ranked two rounds for every field within each cell.','', '| Module | Per-channel | LPBQ S8 | LPBQ direct-W4 |','|---|---:|---:|---:|']
  for name in table['control']['values_us']:
   vals=[]
   for c in CELLS:
    mm=table[c]['values_us'];v=mm[name];vals.append(f"{v:.4f} ({100*v/mm['Complete Host wall']:.3f}%)")
   text.append('|'+name+'|'+'|'.join(vals)+'|')
 text+=['','## Frozen three-recipe scope overview','','| Recipe | Measurement in this experiment |','|---|---|','|F16F16|N/A: frozen, no equivalent single-layer measurement|','|W4F16|N/A: frozen, no equivalent single-layer measurement|','|W4U8|The three complete paired single-layer tables above|','','Historical full-model evidence is not mixed into this single-layer comparison.','','## Physical gates','','```json',json.dumps(z['physical'],indent=2),'```','','## Exact binary and command provenance','','```json',json.dumps(provenance,indent=2),'```']
 for key,runs in allruns.items():
  text+=['',f'## All numeric counters: {key}','','Overlapping work counters are diagnostic and cannot be summed into the additive ledger. Values below independently median each counter across round means.','','| Counter | Per-channel | LPBQ S8 | LPBQ direct-W4 |','|---|---:|---:|---:|']
  for field in sorted(runs['control'][0]):
   text.append('|'+field+'|'+'|'.join(f"{statistics.median(x[field] for x in runs[c]):.6f}" for c in CELLS)+'|')
 text+=['','E2E token/s: N/A. No full model executed. PPL/quality: N/A, speed-only experiment.']
 (R/'full_profiling_report.md').write_text('\n'.join(text)+'\n')
 print(json.dumps({'performance':z['performance'],'decision':z['decision']},indent=2),flush=True)
if __name__=='__main__':main()
