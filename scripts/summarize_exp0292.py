"""EXP0292 complete paired speed report, precision limitations retained."""
import json,statistics,numpy as np
from common_exp0289 import Path,read,write
from measure_exp0218 import OVERVIEW
from summarize_exp0217 import normalized,TICKS
from summarize_exp0289 import LABELS,table
from device_exp0292 import records,R
def main():
 formal=read(R/'formal.json');short=read(R/'short.json')
 assert len(formal)==20 and len(short)==10
 series={}
 for z in formal:
  ps=[v for v in records(R/z['tag']/'stdout.jsonl') if v.get('record')=='generation_profile']
  assert len(ps)==430
  for mode in ['prefill','decode']:
   series.setdefault(z['case']+':'+mode,[]).append(normalized([p for p in ps if p['mode']==mode]))
 means={k:{n:statistics.mean(v[n] for v in vs) for n in vs[0]} for k,vs in series.items()}
 perf={}
 for case in ['fp16','fp32']:
  a,b=means[case+':prefill'],means[case+':decode']
  perf[case]=dict(prefill_tps=64e6/a['host_us'],decode_tps=1e6/b['host_us'],prefill_host_ms=a['host_us']/1000,decode_42_host_ms=42*b['host_us']/1000)
 comparison={};rng=np.random.default_rng(292)
 for mode in ['prefill','decode']:
  control=np.array([x['host_us'] for x in series['fp16:'+mode]])
  candidate=np.array([x['host_us'] for x in series['fp32:'+mode]])
  ix=rng.integers(0,10,(10000,10));boot=candidate[ix].mean(1)/control[ix].mean(1)
  comparison[mode]=dict(wall_ratio=float(candidate.mean()/control.mean()),wall_overhead_percent=float(100*(candidate.mean()/control.mean()-1)),paired_bootstrap95_ratio=np.quantile(boot,[.025,.975]).tolist(),throughput_change_percent=float(100*(control.mean()/candidate.mean()-1)))
 histw4=read(R.parent/'exp0289/MODULES.json')['means']
 histsp=read(R.parent/'exp0288/MODULES.json')['means']
 modules={}
 for mode in ['prefill','decode']:
  vs=[means['fp32:'+mode],histw4['w4f16:'+mode],histsp['OPT1:'+mode]]
  rows=[]
  for label,(_,fields) in zip(LABELS,OVERVIEW):
   values=[sum(v[k] if k.endswith('_us') else v[k]/TICKS for k in fields) for v in vs]
   cells=[f'{a:.2f} ({100*a/v["host_us"]:.2f}%)' for a,v in zip(values,vs)]
   rows.append([label,*cells,f'{100*(values[1]/values[2]-1):+.2f}%' if values[2] else 'N/A zero denominator'])
  modules[mode]=table(['模块','W16A16 FP32残差（本次）','W4A16（EXP0289历史）','W4A8-SP2（EXP0288历史）','SP2 相对 W4A16'],rows)
 summary=dict(experiment='EXP-0292',execution_state='completed',shape='Qwen3-0.6B full28 batch1 M64+42 cache128',measured_source=read(R/'runtime-l28.json')['seal']['source_head'],short_rounds=5,formal_rounds=10,repeat=10,formal_profiles=8600,performance=perf,comparison=comparison,local_gate='fail',gate_role='full_floating_alignment_remains_failed; new_residual_boundaries_pass',residual_adds_bit_exact=True,changed_norm_max_half_ulp=1,original_control_byte_exact_files=2451,repeat_candidate_byte_exact_files=2451,cache_files=4816,full_floating_alignment_pass=False,full_floating_nrmse_max={c:max(z['nrmse'] for z in read(R/('audit/'+c+'/validated.json'))['independent_output']) for c in ['fp16','fp32']},head_boundary_pass=read(R/'head-boundary-checks.json')['pass_all'],quality_claim=False,baseline_promoted=False)
 et=table(['配置','Prefill token/s','Decode token/s','64-token Host ms','42-step Host ms'],[[c,f'{z["prefill_tps"]:.2f}',f'{z["decode_tps"]:.2f}',f'{z["prefill_host_ms"]:.3f}',f'{z["decode_42_host_ms"]:.3f}'] for c,z in perf.items()])
 notes="""Temporary FP32-residual W16A16; original FP16-residual implementation retained. Identical original weights, FP16 HMX O/Down outputs, KV16, head, OPT4 and pipeline settings; candidate Norm/residual handles only valid rows on decode, while the unchanged control retains its original full-M64 row work. This compares complete implementations, not isolated dtype arithmetic cost. Embedding widens to FP32; O/Down widen/add once in FP32; input/post/final RMSNorm read FP32 and produce FP16. This aligns residual storage/add precision with A8, not the A8 raw projection-output contract. No deliberate slowdown, extra delay, removed control optimization, PPL or quality claim. Original historical formal F16 remains1710.94/27.68; samebinary paired control below is separately measured.
Selected0/14/27 and chain3 M64/M1 pass unchanged wholeblock thresholds. Independent actual-operand residual additions exact; norm outputs <=1 FP16 ULP. Full-model software reference remains failed for BOTH arms; see measured maxima below. Do not call this full-model numerical/quality acceptance. Final norm and independent entire LMhead checks pass all43steps, argmax43/43. FP16 full audit matches2451 original files; FP32 repeated audit matches2451 files. Both KV streams finite, prefix preserved, valid length extends once, padding untouched. Timed paths have full8MiB grant, zero intermediateDDR/spill/output audit, oneRPC/pass, exact own token/logit repeats and complete additive ledgers.
Five short/ten balanced formal rounds, repeat10; repeat1 auxiliary. Complete Host wall includes embedding, all28blocks, finalnorm, head/greedy, FastRPC; excludes coldloading, tokenizer and audit I/O. Audit-enabled times are not speed evidence. HMX/HVX/DMA/worker counters overlap and cannot be added.
"""
 (R/'RESULTS.md').write_text('# EXP0292 temporary FP32 residual result\n\n'+notes+'\n'+et+'\n## Paired speed comparison\n\n'+json.dumps(comparison,indent=2)+'\n## Full floating alignment, NOT passed\n\n'+json.dumps(summary['full_floating_nrmse_max'],indent=2)+'\n\nNative flag: QBH_FP32_RESIDUAL=1 with F16F16, QBH_F16F16_OPT=4, Qwen0.6 build ABI137. Flag0 retains original. Packages: /mnt/d/llm_exp/models/qwen3-block-htp/exp0292/f16f16; exact commands and binary hashes in each protocol.json. Other recipes and original artifacts unchanged.\n')
 module_notes='All module values are microseconds and percent of complete Host wall. W4A16 and SP2 are non-paired historical formal repeat10 references; EXP0291 faster W4 diagnostic remains archived separately, not substituted for a formal column.'
 (R/'MODULES.md').write_text('# EXP0292 modules\n\n'+notes+'\n'+module_notes+'\n## Prefill M64\n\n'+modules['prefill']+'\n## Decode per step\n\n'+modules['decode']+'\n## Paired E2E\n\n'+et)
 report=['# EXP0292 FULL PROFILING REPORT',notes,module_notes,et,json.dumps(summary,indent=2),'## Prefill overview\n\n'+modules['prefill'],'## Decode overview\n\n'+modules['decode']]
 aux={}
 for z in read(R/'repeat1.json'):
  ps=[p for p in records(R/z['tag']/'stdout.jsonl') if p.get('record')=='generation_profile']
  for mode in ['prefill','decode']:aux[z['case']+':'+mode]=normalized([p for p in ps if p['mode']==mode])
 for mode in ['prefill','decode']:
  a,b=means['fp16:'+mode],means['fp32:'+mode];x,y=aux['fp16:'+mode],aux['fp32:'+mode];rows=[]
  for k in sorted(set(a)&set(b)&set(x)&set(y)):
   am=statistics.median(z[k] for z in series['fp16:'+mode]);bm=statistics.median(z[k] for z in series['fp32:'+mode])
   rows.append([k,f'{x[k]:.6f}',f'{y[k]:.6f}',f'{100*(y[k]/x[k]-1):+.3f}%' if x[k] else 'N/A zero denominator',f'{a[k]:.6f}',f'{b[k]:.6f}',f'{am:.6f}',f'{bm:.6f}',f'{100*(bm/am-1):+.3f}%' if am else 'N/A zero denominator'])
  report+=['## '+mode,table(['field','FP16 residual R1','FP32 residual R1','R1 delta','FP16 R10 mean','FP32 R10 mean','FP16 R10 median','FP32 R10 median','median delta'],rows)]
 (R/'FULL_PROFILING_REPORT.md').write_text('\n\n'.join(report)+'\n')
 write(R/'SUMMARY.json',summary);write(R/'MODULES.json',dict(means=means,series=series))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
