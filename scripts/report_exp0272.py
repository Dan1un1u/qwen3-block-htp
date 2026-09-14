"""Reconstruct full-model gate and complete module/counter evidence."""
from common_exp0272 import *
from device_exp0272 import records
from summarize_exp0217 import normalized
from measure_exp0218 import LEDGER,OVERVIEW
import statistics,numpy as np

def main():
 phase='formal' if (R/'full-formal.json').exists() else 'short';g=read(R/f'full-{phase}.json');n=g['rounds'];groups={};runs={}
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   for fp in [0,1,2]:
    raw=[];means=[]
    for i in range(n):
     p=R/f'full-{phase}/round{i:02d}-r{rep}-fp{fp}/stdout.jsonl';a=[v for v in records(p) if v.get('record')=='generation_profile' and v['mode']==mode];assert len(a)==rep*(1 if mode=='prefill' else 15);raw+=a;means.append(normalized(a))
    key=f'r{rep}_{mode}_fp{fp}';groups[key]=normalized(raw);runs[key]=means
 allraw=[];cli=0
 for p in R.rglob('stdout.jsonl'):
  cli+=1
  for v in records(p):
   if v.get('record') not in ['generation_profile','exp0240_profile']:continue
   a=normalized([v]);assert sum(a[k] for _,k in LEDGER)==v['invocation_ticks'];assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608;assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks']);allraw.append(v)
 perf={}
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   vals=[]
   for fp in [0,1,2]:
    a=groups[f'r{rep}_{mode}_fp{fp}'];tokens=64 if mode=='prefill' else 1;vals.append(dict(mode=fp,host_us=a['host_us'],tps=tokens*1e6/a['host_us']))
   perf[f'r{rep}_{mode}']=vals
 summary=dict(experiment='EXP-0272',phase=phase,formal_run=phase=='formal',full_gate_pass=bool(phase=='formal' and g['speed_pass']),short_gate_pass=read(R/'full-short.json')['speed_pass'],performance=perf,ratios=g['performance'],hardware_cli=cli,hardware_profiles=len(allraw),full_profiles=sum(v['record']=='generation_profile' for v in allraw),peak_vtcm_bytes=max(v['vtcm_peak_plan_bytes'] for v in allraw),tested_source_head=read(R/'runtime-l28.json')['seal']['source_head'],selection=read(R/'candidate_selection.json'),single_gate=read(R/'single_gate.json'),slice_gate=read(R/'slice_gate.json'),full_gate=read(R/'full_sanity_gate.json'),baseline_promoted=False,PPL=None)
 rng=np.random.default_rng(272);idx=rng.integers(0,n,(20000,n))
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   for den in [0,1]:
    a=np.array([[v['host_us']*1000 for v in runs[f'r{rep}_{mode}_fp{fp}']] for fp in [den,2]]);ci=np.quantile(a[1,idx].mean(1)/a[0,idx].mean(1),[.025,.975]);v=g['performance'][f'r{rep}_{mode}_ns_vs{den}'];assert np.allclose(ci,v['ci95'],rtol=1e-12)
 write(R/'SUMMARY.json',summary);write(R/'PROFILE.json',dict(groups=groups,per_run_means=runs,ratios=g['performance']))
 lines=['# EXP0272: exact FP32 Norm and full-model gate','',f"Source `{summary['tested_source_head']}`; Qwen3-0.6B28layers, nativeW4/SP2mode8, no rotation, original frozen C64/SP2 packages. Full M64+15, cache128 and offlineEOSprefix. Mode0 integer residual,mode1 priorFP32,mode2 interleaved16-row orderedNorm. Candidate stores and arithmetic are unchanged except independent-row instruction scheduling. Llama and other recipes frozen.",'',f"Result: phase={phase}; full formal gate={summary['full_gate_pass']}; short gate={summary['short_gate_pass']}. Primary repeat10, auxiliary repeat1. Fixed5short then10formal only if short passes; pairedbootstrap20000 seed27295CI upper<=1.10 vsinteger for BOTH prefill and decode. No optional resampling or automatic baseline promotion.",'','| Repeat / phase | Integer Host us / tps | Prior FP32 Host us / tps | Optimized FP32 Host us / tps |','|---|---:|---:|---:|']
 for k,vs in perf.items():lines.append('| '+k+' | '+' | '.join(f"{v['host_us']:.3f} / {v['tps']:.3f}" for v in vs)+' |')
 lines+=['','| Ratio | Point | 95% CI | Gate |','|---|---:|---|---|']
 for k,v in g['performance'].items():lines.append(f"| {k} | {v['ratio']:.9f} | {v['ci95']} | {v['gate']} |")
 lines+=['','## Attribution and bounded candidates','', 'EXP0270 singlelayer inputNorm20.25->75.24us; EXP0271 full inputNorm562.59->2119.73us (~20.09->75.70us/layer). Norm overhead accumulates almost linearly; lower fullmodel relative regression comes from amortized RPC/boundary work and unchanged head/other work in the denominator, not lossless cancellation of Norm. EXP0270 Host-DSP delta+53.47us per singlelayer; EXP0271 entirefull delta-68.16us. Separate sessions and scopes, not causal subtraction or an E2E extrapolation.','', 'C1 holds two independent8-row FP32 sums and interleaves their exact ascending-channel updates inside one16-row reduction. No reassociation, reciprocal change, extra DDR, weight/quantizer modification or approximate reduction. Singlelayer Norm75.2->71.6us in diagnosis, while complete singlelayer wall is noisy. C2 packed four repair-flag fragments into aligned128-byte writes; it is exact but slower (86-88us Norm after specialization), and was fully removed. Generic C2 branches initially also slowed mode1; dispatch isolation restored it. One compile failed because clang fp pragma followed a local constant; fixed before hardware, failed source retained. No hardware numerical failure. C2 module cost, not a raw best wall sample, drove rejection.','', f"Final selectedlayers0/14/27 independently exact399360 FP32 values plus Q/K,AV,postnorm,GateUp,SP2 and physicalKV; consecutive3 exact133120values; repeats10 deterministic. Full28 all layer ledgers/cache lengths pass,88 captured residual/finalNorm/KV files byte-exact vs sealedEXP0271. Independent finalNorm32768codes and fullvocabhead16steps exact. Integer control matches sealedEXP0268; optimized and priorFP32 token/logit codes identical. {cli}CLI/{len(allraw)}profiles;8MiB grant,peak{summary['peak_vtcm_bytes']}B; zero timed boundary/intermediate tensorDDR/spill/unattributed. New reduce16 has no vector stack; retained genericNorm stack holds uniform constants and inverse-RMS splats only. C2 assemblies remain archived, no C2 code in selected build.",'', 'Timing denominator is complete Host invocation wall including embedding/allblocks/finalNorm/head/greedy and Host-DSP boundary, with loaded weights. Prefill64tokens; decode15tokens persequence. Excludes cold load/ADB/CPU tokenizer and diagnostic exports. No full28 independent CPU transformer oracle, PPL or quality acceptance.','', '## Stable module overview','', 'Microseconds (share of complete Host wall),repeat10. F16F16/W4F16 not rerun and thereforeN/A; all three columns here are same-recipe paired controls. Exclusive module ledger sums to Host wall.']
 for mode in ['prefill','decode']:
  lines+=['',f'### {mode}','','| Module | Integer SP2 | Prior FP32 SP2 | Optimized FP32 SP2 |','|---|---:|---:|---:|']
  for name,keys in OVERVIEW:
   cells=[]
   for fp in [0,1,2]:
    a=groups[f'r10_{mode}_fp{fp}'];v=sum(a[k] if k.endswith('_us') else a[k]/19.2 for k in keys);cells.append(f"{v:.2f} ({v/a['host_us']*100:.2f}%)")
   lines.append('| '+name+' | '+' | '.join(cells)+' |')
 lines+=['','## Complete counters','', 'Per-run mean medians; tick counters converted to microseconds, other units unchanged. Overlapping work/wait/readiness counters must not be added to exclusive ledger. Means also retained in PROFILE.json.']
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   lines+=['',f'### repeat{rep} {mode}','','| Field | Integer | Prior FP32 | Optimized FP32 |','|---|---:|---:|---:|']
   for k in sorted(groups[f'r{rep}_{mode}_fp0']):
    div=19.2 if k.endswith('_ticks') else 1;vals=[statistics.median(v[k] for v in runs[f'r{rep}_{mode}_fp{fp}'])/div for fp in [0,1,2]];lines.append('| '+k+(' (us)' if div!=1 else '')+' | '+' | '.join(f'{v:.7g}' for v in vals)+' |')
 lines.append('\n## Same-session prior FP32 diagnostic\n\nPrior FP32/integer prefill ratio1.097978170,95CI[1.095060869,1.101093893]: point estimate below10percent, confidence gate fails narrowly. Optimized1.089965757,95CI[1.087257958,1.092578145]: passes. Thus do not attribute all change from historical10.718percent to optimization; matched old->new prefill wall improves0.729742percent,95CI ratio[0.989440347,0.995995624]. Old/new decode is statistically tied.\n')
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:summary[k] for k in ['phase','full_gate_pass','short_gate_pass','performance','ratios','hardware_cli','hardware_profiles']},indent=2))
if __name__=='__main__':main()
