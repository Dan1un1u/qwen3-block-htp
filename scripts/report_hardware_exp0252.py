#!/usr/bin/env python3
from pathlib import Path
import sys,json,statistics,subprocess,hashlib
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from exp0240_report import FIELDS,MODULES,means,coherent_center
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0252')
S=Path('/home/daniuniu/work/qwen3-block-htp')
def records(tag,kind='exp0240_profile'):
 return [json.loads(l) for l in (R/tag/'stdout.jsonl').read_text().splitlines() if l.startswith('{') and json.loads(l).get('record')==kind]
def main():
 numerical=json.loads((R/'numerical_audit.json').read_text());assert numerical['pass_all']
 assert json.loads((R/'collection_pass.json').read_text())['pass_all']
 allruns={};allrecords=[];perf={};modules={};rng=np.random.default_rng(252)
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   runs={c:[] for c in ['control','nr64']}
   for c in runs:
    for i in range(10):
     rs=records(f'formal_{i:02d}_r{rep}_{c}')
     for x in rs:
      assert sum(x[k] for k in FIELDS)==x['invocation_ticks']==x['ledger_named_ticks']
      assert x['ledger_unattributed_ticks']==0
     rows=[x for x in rs if (x['replay_step']==0)==(mode=='prefill')];runs[c].append(means(rows));allrecords.extend(rows)
   a=np.array([x['host_wall_ns']/1000 for x in runs['control']]);b=np.array([x['host_wall_ns']/1000 for x in runs['nr64']]);ratios=b/a
   sampled=np.median(ratios[rng.integers(0,10,(10000,10))],axis=1);ci=np.quantile(sampled,[.025,.975])
   key=f'repeat{rep}_{mode}';perf[key]=dict(control_us=float(np.median(a)),nr64_us=float(np.median(b)),paired_regression_percent=float((np.median(ratios)-1)*100),paired_ratio_ci95=ci.tolist(),stable_over10percent=bool(ci[0]>1.1),round_control_us=a.tolist(),round_nr64_us=b.tolist());allruns[key]=runs
   if rep==10:
    modules[mode]={}
    for c in runs:
     center,indices=coherent_center(runs[c]);host=center['host_wall_ns']/1000
     values={name:sum(center[k] for k in fields)/19.2 for name,fields in MODULES}
     values['Host–DSP boundary']=host-center['invocation_ticks']/19.2;values['Complete Host wall']=host
     assert abs(sum(v for k,v in values.items() if k!='Complete Host wall')-host)<1e-7
     modules[mode][c]=dict(round_indices=indices,host_us=host,modules_us=values)
 speed_pass=all(perf[f'repeat10_{m}']['paired_ratio_ci95'][1]<1.1 for m in ['prefill','decode'])
 speed_fail=any(perf[f'repeat10_{m}']['stable_over10percent'] for m in ['prefill','decode'])
 details={'r3_nr64_eligible':numerical['eligible']['r3_nr64'],'r3_nr64_whole_gate':numerical['r3_amplification']}
 summary=dict(experiment='EXP-0252',source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),runtime_source_head=json.loads((R/'formal_provenance.json').read_text())['source_head'],execution_state='completed',evidence_validity='valid',local_gate='pass' if speed_pass else 'fail',adoption_status='not_applicable',numerical_gate='pass',physical_gate='pass',speed_gate='pass' if speed_pass else ('fail' if speed_fail else 'uncertain'),short_rounds=5,formal_rounds=10,timed_layer_RPCs=2970,audit_runs=13,audit_layer_RPCs=117,performance=perf,module_centers=modules,dense_r3=details,baseline_promoted=False,device_PPL=None,E2E_tokens_per_second=None,reason='Incremental no-R3 NR64 reciprocal normalization measured. Plain dense HMX R3 combination remains numerically ineligible. R3 numerical gate remains failed; conditional integration has not started.')
 summary['physical']={k:sorted({x[k] for x in allrecords}) for k in ['vtcm_requested_bytes','vtcm_acquired_bytes','vtcm_peak_plan_bytes','intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','block_invocation_count','w4u8_decode_direct_n_projection_count','ledger_unattributed_ticks']}
 (R/'hardware_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 text=['# EXP0252 complete profiling comparison','','Scope: one real C64 layer0, EOS+71 tokens; M64 then eight teacher-input M1 steps. Control wide-score SOLE; candidate same no-R3 package with NR64 reciprocal normalization; plain dense HMX R3 plus NR64 failed the independent whole-layer gate and is not formally timed. F16/W4A16 frozen. No full-model token boundary.','Runtime source '+summary['runtime_source_head']+'. Reporting source '+summary['source_head']+'.','Five short and ten alternating paired formal rounds, each repeat1 and repeat10. No discarded warmups/outliers. Primary Host wall is median of ten paired-round means; decode averages eight positions. Paired median ratios,10000 bootstrap resamples seed252. Additive tables use identical two median-Host-ranked rounds for every field in a cell. Engine counters can overlap and are separate.','All2970 timed RPCs match frozen audited per-step hashes; unused legacy output/cache zeros are NOT golden checks. One RPC/step,8MiB requested/granted,peak6682752bytes,zero intermediate DDR/spill,seven native W4 projections. Audit exports are disabled for all timed runs. Both timed arms have R3 disabled. No butterfly/FWHT or scalar oracle fallback.','','|Scope|Control us|Wide us|Paired time regression|Ratio95% CI|','|---|---:|---:|---:|---|']
 for k,v in perf.items():text.append(f"|{k}|{v['control_us']:.3f}|{v['nr64_us']:.3f}|{v['paired_regression_percent']:+.2f}%|{v['paired_ratio_ci95']}|")
 user=[]
 for mode in ['prefill','decode']:
  centers=modules[mode];tab=[f'## {mode}: repeat10 additive modules','','Unit us; parentheses share complete Host wall. Frozen recipes N/A; same-scope historical measurements are unavailable.','','|Module|F16A16|W4A16|W4A8 control|W4A8 NR64|W4A16 / W4A8 - 1|','|---|---|---|---:|---:|---|']
  for name in centers['control']['modules_us']:
   vals=[]
   for c in ['control','nr64']:
    z=centers[c];v=z['modules_us'][name];vals.append(f"{v:.1f} ({100*v/z['host_us']:.2f}%)")
   tab.append('|'+name+'|N/A|N/A|'+'|'.join(vals)+'|N/A no matched W4A16|' )
  for name in ['Embedding','Final model RMSNorm','LM head + greedy']:tab.append('|'+name+'|N/A|N/A|N/A outside layer|N/A outside layer|N/A|')
  text.extend(['']+tab);user.extend(['']+tab)
 text.extend(['','Identity: source branch codex/exp-0252-normalization-r3; evidence '+str(R)+'; frozen artifacts /mnt/d/llm_exp/models/qwen3-block-htp/exp0247/{control,r3}. Project Variant W4U8/native per-channel weight.n. Full correctness, recovery and scope limits in REPORT.md; exact binary/package identity in ARTIFACT_PROVENANCE.json.'])
 text.extend(['','## Complete counter diagnostics','','Time counters below use19.2ticks/us. Counts/bytes retain native units. Independent per-field medians need not sum. Legacy unused golden fields are excluded.'])
 for key,runs in allruns.items():
  text.extend(['','### '+key,'','|Field|Control|Wide|Change|','|---|---:|---:|---:|'])
  for k in runs['control'][0]:
   if k in ['experiment','replay_step','first_position','valid_length','repeat_count'] or k.startswith(('output_','cache_','fp16_')):continue
   a=statistics.median(x[k] for x in runs['control']);b=statistics.median(x[k] for x in runs['nr64']);delta=f'{100*(b/a-1):+.2f}%' if a else 'N/A zero denominator'
   text.append(f'|{k}|{a:.4f}|{b:.4f}|{delta}|')
 text.extend(['','## R3 diagnostics','','Untimed plain dense HMX R3 numerical audit; not valid formal timing evidence.',json.dumps(details,indent=2),'','E2E prefill/decode token/s: N/A. Device PPL: N/A. Full-model work has not started; R3 numerical gate remains failed. The score repair does not establish model quality acceptance.'])
 (R/'HARDWARE_PROFILE.md').write_text('\n'.join(text)+'\n');(R/'USER_PROFILE.md').write_text('\n'.join(user)+'\nE2E token/s: N/A (single layer only).\n')
 print(json.dumps(perf,indent=2));print('DENSE',details)
if __name__=='__main__':main()
