#!/usr/bin/env python3
from pathlib import Path
import sys,json,statistics,subprocess,hashlib
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from exp0240_report import FIELDS,MODULES,means,coherent_center
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0248')
S=Path('/home/daniuniu/work/qwen3-block-htp')
def records(tag,kind='exp0240_profile'):
 return [json.loads(l) for l in (R/tag/'stdout.jsonl').read_text().splitlines() if l.startswith('{') and json.loads(l).get('record')==kind]
def main():
 numerical=json.loads((R/'numerical_audit.json').read_text());assert numerical['pass_all']
 assert json.loads((R/'collection_pass.json').read_text())['pass_all']
 allruns={};allrecords=[];perf={};modules={};rng=np.random.default_rng(248)
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   runs={c:[] for c in ['control','refined']}
   for c in runs:
    for i in range(10):
     rs=records(f'formal_{i:02d}_r{rep}_{c}')
     for x in rs:
      assert sum(x[k] for k in FIELDS)==x['invocation_ticks']==x['ledger_named_ticks']
      assert x['ledger_unattributed_ticks']==0
     rows=[x for x in rs if (x['replay_step']==0)==(mode=='prefill')];runs[c].append(means(rows));allrecords.extend(rows)
   a=np.array([x['host_wall_ns']/1000 for x in runs['control']]);b=np.array([x['host_wall_ns']/1000 for x in runs['refined']]);ratios=b/a
   sampled=np.median(ratios[rng.integers(0,10,(10000,10))],axis=1);ci=np.quantile(sampled,[.025,.975])
   key=f'repeat{rep}_{mode}';perf[key]=dict(control_us=float(np.median(a)),refined_us=float(np.median(b)),paired_regression_percent=float((np.median(ratios)-1)*100),paired_ratio_ci95=ci.tolist(),stable_over10percent=bool(ci[0]>1.1),round_control_us=a.tolist(),round_refined_us=b.tolist());allruns[key]=runs
   if rep==10:
    modules[mode]={}
    for c in runs:
     center,indices=coherent_center(runs[c]);host=center['host_wall_ns']/1000
     values={name:sum(center[k] for k in fields)/19.2 for name,fields in MODULES}
     values['Host–DSP boundary']=host-center['invocation_ticks']/19.2;values['Complete Host wall']=host
     assert abs(sum(v for k,v in values.items() if k!='Complete Host wall')-host)<1e-7
     modules[mode][c]=dict(round_indices=indices,host_us=host,modules_us=values)
 assert all(perf[f'repeat10_{m}']['stable_over10percent'] for m in ['prefill','decode'])
 details={}
 for mode in ['prefill','decode']:
  rows=[]
  for i in range(10):rows.extend(x for x in records(f'formal_{i:02d}_r10_refined','dense_r3') if (x['step']==0)==(mode=='prefill'))
  details[mode]={k:statistics.mean(x[k] for x in rows) for k in ['rows','hmx_calls','refined_values','prepare_ticks','matmul_ticks','finish_ticks']}
 summary=dict(experiment='EXP-0248',source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),runtime_source_head=json.loads((R/'formal_provenance.json').read_text())['source_head'],execution_state='completed',evidence_validity='valid',local_gate='fail',adoption_status='rejected',numerical_gate='pass',physical_gate='pass',speed_gate='fail',short_rounds=5,formal_rounds=10,timed_layer_RPCs=2970,audit_runs=6,audit_layer_RPCs=54,performance=perf,module_centers=modules,dense_r3=details,baseline_promoted=False,device_PPL=None,E2E_tokens_per_second=None,reason='Refinement removes whole-layer replay mismatch but stable slowdown exceeds10percent in both scopes; full model not started')
 summary['physical']={k:sorted({x[k] for x in allrecords}) for k in ['vtcm_requested_bytes','vtcm_acquired_bytes','vtcm_peak_plan_bytes','intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','block_invocation_count','w4u8_decode_direct_n_projection_count','ledger_unattributed_ticks']}
 (R/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 text=['# EXP0248 complete profiling comparison','','Scope: one real C64 layer0, EOS+71 tokens; M64 then eight teacher-input M1 steps. Control original A8; candidate dense HMX R3 with guarded HVX direct-dot refinement. F16/W4A16 frozen. No full-model token boundary.','Runtime source '+summary['runtime_source_head']+'. Reporting source '+summary['source_head']+'.','Five short and ten alternating paired formal rounds, each repeat1 and repeat10. No discarded warmups/outliers. Primary Host wall is median of ten paired-round means; decode averages eight positions. Paired median ratios,10000 bootstrap resamples seed248. Additive tables use identical two median-Host-ranked rounds for every field in a cell. Engine counters can overlap and are separate.','All2970 timed RPCs match frozen audited per-step hashes; unused legacy output/cache zeros are NOT golden checks. One RPC/step,8MiB requested/granted,peak6682752bytes,zero intermediate DDR/spill,seven native W4 projections. Audit exports are disabled for all timed runs. R3 dense matrix remains explicit; no butterfly/FWHT or scalar oracle fallback.','','|Scope|Control us|Refined us|Paired time regression|Ratio95% CI|','|---|---:|---:|---:|---|']
 for k,v in perf.items():text.append(f"|{k}|{v['control_us']:.3f}|{v['refined_us']:.3f}|{v['paired_regression_percent']:+.2f}%|{v['paired_ratio_ci95']}|")
 user=[]
 for mode in ['prefill','decode']:
  centers=modules[mode];tab=[f'## {mode}: repeat10 additive modules','','Unit us; parentheses share complete Host wall. Frozen recipes N/A; same-scope historical measurements are unavailable.','','|Module|F16A16|W4A16|W4A8 control|W4A8 refined R3|','|---|---|---|---:|---:|']
  for name in centers['control']['modules_us']:
   vals=[]
   for c in ['control','refined']:
    z=centers[c];v=z['modules_us'][name];vals.append(f"{v:.1f} ({100*v/z['host_us']:.2f}%)")
   tab.append('|'+name+'|N/A|N/A|'+'|'.join(vals)+'|')
  for name in ['Embedding','Final model RMSNorm','LM head + greedy']:tab.append('|'+name+'|N/A|N/A|N/A outside layer|N/A outside layer|')
  text.extend(['']+tab);user.extend(['']+tab)
 text.extend(['','## Complete counter diagnostics','','Time counters below use19.2ticks/us. Counts/bytes retain native units. Independent per-field medians need not sum. Legacy unused golden fields are excluded.'])
 for key,runs in allruns.items():
  text.extend(['','### '+key,'','|Field|Control|Refined|Change|','|---|---:|---:|---:|'])
  for k in runs['control'][0]:
   if k in ['experiment','replay_step','first_position','valid_length','repeat_count'] or k.startswith(('output_','cache_','fp16_')):continue
   a=statistics.median(x[k] for x in runs['control']);b=statistics.median(x[k] for x in runs['refined']);delta=f'{100*(b/a-1):+.2f}%' if a else 'N/A zero denominator'
   text.append(f'|{k}|{a:.4f}|{b:.4f}|{delta}|')
 text.extend(['','## R3 diagnostics','','These counters overlap QKV and must not be added to the ledger.',json.dumps(details,indent=2),'','E2E prefill/decode token/s: N/A. Device PPL: N/A. >10percent speed gate stopped full-model work.'])
 (R/'FULL_PROFILE.md').write_text('\n'.join(text)+'\n');(R/'USER_PROFILE.md').write_text('\n'.join(user)+'\nE2E token/s: N/A (single layer only).\n')
 print(json.dumps(perf,indent=2));print('DENSE',details)
if __name__=='__main__':main()
