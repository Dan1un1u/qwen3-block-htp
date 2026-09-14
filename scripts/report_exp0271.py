"""Reconstruct the bounded full-model diagnostic from immutable raw timing."""
from common_exp0271 import *
from device_exp0271 import records
from measure_exp0218 import LEDGER,OVERVIEW
from summarize_exp0217 import normalized
import statistics

def main():
 assert read(R/'slice_gate.json')['pass_all'] and read(R/'full_sanity_gate.json')['pass_all']
 groups={};rounds=[]
 for rep in [1,10]:
  for fp in [0,1]:
   tags=[f'timing-r1-fp{fp}'] if rep==1 else [f'timing-pair{i}-r10-fp{fp}' for i in [0,1]]
   raw=[]
   for tag in tags:
    z=read(R/tag/'validated.json');assert z['pass_all'] and not z['audit'];rounds.append(dict(tag=tag,**z))
    a=[v for v in records(R/tag/'stdout.jsonl') if v.get('record')=='generation_profile'];assert len(a)==16*rep;raw+=a
   for mode in ['prefill','decode']:
    a=[v for v in raw if v['mode']==mode];groups[f'r{rep}_{mode}_fp{fp}']=normalized(a)
 perf={}
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   a=groups[f'r{rep}_{mode}_fp0']['host_us'];b=groups[f'r{rep}_{mode}_fp1']['host_us'];tokens=64 if mode=='prefill' else 1
   perf[f'r{rep}_{mode}']=dict(control_host_us=a,candidate_host_us=b,control_tps=tokens*1e6/a,candidate_tps=tokens*1e6/b,host_increase_percent=(b/a-1)*100,tps_change_percent=(a/b-1)*100,diagnostic_only=True,formal_gate=None)
 allraw=[];cli=0
 for p in R.rglob('stdout.jsonl'):
  cli+=1
  for v in records(p):
   if v.get('record') not in ['generation_profile','exp0240_profile']:continue
   n=normalized([v]);assert sum(n[k] for _,k in LEDGER)==v['invocation_ticks'];allraw.append(v)
 summary=dict(experiment='EXP-0271',tested_source_head=read(R/'runtime-l28.json')['seal']['source_head'],source_branch='codex/exp-0271-fp32-fullmodel-speed',native_parent='645b0cdc4d6787dcde7bade9af59c3de3d6d8451',native_changed=False,full_model=True,layers=28,prompt_tokens=64,generated_steps=16,decode_tokens=15,cache_capacity=128,offline_prefix=1,SP2_mode=8,rotation=False,performance=perf,rounds=rounds,hardware_cli=cli,hardware_profiles=len(allraw),full_profiles=sum(v['record']=='generation_profile' for v in allraw),peak_vtcm_bytes=max(v['vtcm_peak_plan_bytes'] for v in allraw),requested_vtcm_bytes=8388608,baseline_promoted=False,formal_gate=None,quality_gate=None,correctness=dict(chain3=read(R/'slice_gate.json'),full=read(R/'full_sanity_gate.json')),timing_scope='Mean complete invocation Host wall: embedding + all28 blocks + finalNorm + LM head/greedy + Host-DSP. Prepared weights, token IDs in/out; excludes cold model load/ADB/WSL tokenizer and optional diagnostic exports. Two fixed repeat10 AB/BA pairs, no statistical promotion; per-arm generated tokens may differ. Repeat1 auxiliary.')
 write(R/'SUMMARY.json',summary);write(R/'PROFILE.json',dict(groups=groups,performance=perf))
 lines=['# EXP0271: Qwen full-model SP2 FP32 residual speed','',summary['timing_scope'],'',f"Measured source `{summary['tested_source_head']}`. Native code unchanged from EXP0270 C1+C3. Same frozen original C64/SP2 weights, scales and offline prefix; original FP16 embedding added to candidate package. Integer control uses original U8 embedding. Llama unchanged.",'','| Repeat / phase | Integer Host us | FP32 Host us | Integer token/s | FP32 token/s | Host increase | TPS change |','|---|---:|---:|---:|---:|---:|---:|']
 for k,v in perf.items():lines.append(f"| {k} | {v['control_host_us']:.3f} | {v['candidate_host_us']:.3f} | {v['control_tps']:.3f} | {v['candidate_tps']:.3f} | {v['host_increase_percent']:+.2f}% | {v['tps_change_percent']:+.2f}% |")
 lines+=['','## Correctness and scope','',f"Consecutive3 exact FP32 output:133120 values; last-layer Q/K, AV, postnorm, Gate/Up and SP2 live bytes exact; repeat10 deterministic. Full28 all layers/ledgers/cache lengths pass, no hidden tensor DDR or spill/fill. FinalNorm32768 actual A8 codes exact, full151936-vocabulary LM-head CPU reference selects identical token/logit code for all16 steps; head bias exact. Original integer control matches sealed EXP0268 token/code sequence. Candidate audit and timing agree, every repeated sequence deterministic. {cli} CLI, {len(allraw)} profiles; VTCM requested/acquired8MiB, maxplan{summary['peak_vtcm_bytes']}B. Timed boundary writes0; audit-only FP32 hidden/nativeNorm exports139264B perstep. No whole28 CPU transformer oracle, independent PPL or text-quality acceptance is claimed.",'', 'One validator repair: separate offline prefix metadata was incorrectly counted in logical cache_valid. Device returned0/all_steps_pass; actual lengths0->64 then64->65 agree with sealedEXP0268. Original audit raw retained and revalidated; native arithmetic unchanged.','', 'Generated affected assembly: reduce8, epilogue, HMX worker and streamed Down have no vector stack payload stores. Generic Norm uses stack for uniform constants and scalar inverse-RMS splats, matching the already-audited EXP0270 behavior. Assembly extracts archived.','', 'Diagnostic fullmodel prefill remains slightly above10percent Host overhead; decode stays below. Two pairs are insufficient for the formal confidence-interval gate. Prior EXP0270 singlelayer gate remains failed and neither paper baseline is replaced. Historical Qwen integer2030.238/48.173tok/s is a separate session; use current paired control for overhead.','', '## Stable module overview','', 'Microseconds and share of complete Host wall; repeat10 pooled means,20 prefill and300 decode invocations per arm. W16A16 and W4A16 were not rerun; no substituted historical comparisons. Exclusive ledger sums to Host wall. Overlapping work/wait counters below must not be summed.']
 for mode in ['prefill','decode']:
  lines+=['',f'### {mode}','','| Module | W4A8-SP2 integer residual | W4A8-SP2 FP32 residual |','|---|---:|---:|']
  for name,keys in OVERVIEW:
   cells=[]
   for fp in [0,1]:
    st=groups[f'r10_{mode}_fp{fp}'];v=sum(st[k] if k.endswith('_us') else st[k]/19.2 for k in keys);cells.append(f"{v:.2f} ({v/st['host_us']*100:.2f}%)")
   lines.append('| '+name+' | '+' | '.join(cells)+' |')
 lines+=['','## Raw counters','', 'Means reconstructed from raw JSON. Tick columns converted to microseconds; count/byte units unchanged. No resampling.']
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   lines+=['',f'### repeat{rep} {mode}','','| Field | Integer | FP32 | Change |','|---|---:|---:|---:|'];a=groups[f'r{rep}_{mode}_fp0'];b=groups[f'r{rep}_{mode}_fp1']
   for k in sorted(a):
    av=a[k];bv=b[k];div=19.2 if k.endswith('_ticks') else 1;delta=f'{(bv/av-1)*100:+.3f}%' if av else 'N/A: zero denominator';lines.append(f'| {k}'+(' (us)' if div!=1 else '')+f' | {av/div:.7g} | {bv/div:.7g} | {delta} |')
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n');print(perf)
if __name__=='__main__':main()
