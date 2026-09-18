import json,sys,statistics
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
from report_llama32_pipeline_profile import MODULES
from llama_reference import sha256
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0047')
M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0047')
def read(p):return json.loads(p.read_text())
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
 result={};raw=['# Full counters: repeat1 auxiliary, repeat10 formal'];mods={};rng=np.random.default_rng(320047)
 for phase in ['short','formal']:
  runs=read(R/(phase+'.json'))['runs'];result[phase]={}
  for mode in ['prefill','decode']:
   sets=[[v for v in read(R/phase/f'{i:02d}/records.json') if v.get('record')=='generation_profile' and v['mode']==mode] for i in range(len(runs))]
   walls=np.array([statistics.mean(v['host_wall_ns'] for v in a) for a in sets])
   avg={k:statistics.mean(statistics.mean(v[k] for v in a) for a in sets) for k,x in sets[0][0].items() if isinstance(x,(int,float))}
   ix=rng.integers(0,len(walls),(20000,len(walls)));tps=(64 if mode=='prefill' else 1)*1e9/walls.mean()
   ci=np.quantile((64 if mode=='prefill' else 1)*1e9/walls[ix].mean(1),[.025,.975]).tolist()
   rows=[]
   for label,keys in MODULES:
    us=(sum(avg[k] for k in keys)-(avg['generation_final_norm_ticks'] if keys==['generation_lm_head_ticks'] else 0))/19.2
    rows.append(dict(module=label,us=us,share=100*us/(walls.mean()/1000)))
   boundary=walls.mean()/1000-sum(v['us'] for v in rows);assert boundary>=0
   rows += [dict(module='Host–DSP 边界',us=boundary,share=100*boundary/(walls.mean()/1000)),dict(module='完整 Host wall',us=walls.mean()/1000,share=100)]
   result[phase][mode]=dict(tps=tps,ci95_tps=ci,host_ns=float(walls.mean()),rounds=walls.tolist(),modules=rows)
   if phase=='formal':
    mods[mode]=avg;aux=[v for v in read(R/'aux-repeat1/records.json') if v.get('record')=='generation_profile' and v['mode']==mode]
    raw+=['## '+mode,'| Counter | Repeat1 | Repeat10 mean | Repeat10 median of round means |','|---|---:|---:|---:|']
    for k,v in avg.items():raw.append(f"| {k} | {statistics.mean(x[k] for x in aux):.6f} | {v:.6f} | {statistics.median(statistics.mean(x[k] for x in a) for a in sets):.6f} |")
 math=read(R/'full-audit-c/mathematical-audit.json');native=read(R/'f16-l28c/runtime.json')['seal']['source_head']
 assert read(R/'rowcache-audit-b/validated.json')['pass_all']
 for arm in ['A8','SP2']:assert read(R/('regression-3B-'+arm)/'validated.json')['pass_all']
 # Original quantized payload identity, including channel scales and model front/back.
 mf=read(M/'frontend64-fixed/manifest.json');old=read(M.parent/'l32-0044/frontend64-a01/manifest.json');same=[]
 for n,h in old['files'].items():
  if 'weight_' in n and not n.endswith('.npy'):
   assert mf['files'][n]['sha256']==h['sha256'],n;same.append(n)
 save(R/'CONTRACT_AUDIT.json',dict(pass_all=True,identical_weight_scale_norm_files=len(same),files=same,
  recipe='Per-output-channel W4, original0041 payload; FP16 ordinary baseline nonlinear/residual/KV/embedding/finalnorm, W4 coarse head; no SP2/R3/R4',
  math_alignment_gate_pass=math['pass_all'],quality_accepted=False))
 z=dict(experiment='L32-0047',shape='M64+42/cache128/full28',measured_source=native,performance=result,
  implementation_crosscheck_pass=True,selected_composition_gate_pass=True,full_mathematical_alignment_pass=math['pass_all'],
  max_full_boundary_nrmse=max(v['nrmse'] for v in math['boundaries']),min_prefill_cache_cosine=min(v['cosine'] for v in math['prefill_caches']),
  peak_vtcm_bytes=7964160,quality_accepted=False,baseline_promoted=False)
 save(R/'SUMMARY.json',z);save(R/'MODULES.json',mods)
 lines=['# L32-0047 generic Llama3B W4A16 completion','',
 'Reuses sealed0041 per-channel W4 codes/scales. Existing FP16 embedding, residual, nonlinear/attention/KV and final norm; existing W4 coarse LM head. No SP2 or rotation. No shape-specific performance search.','',
 'Generic head128 HVX RoPE; separate capacity-correct compressed/expanded DMA slots; existing Gate4 pipeline with bounded16-slot producer/consumer ring and dead-phase reuse. Largest expanded slots cover the unchanged8-tile coarse head. Peak VTCM7964160/8388608 bytes, one HMX owner, one RPC/token, zero timed intermediate DDR/spill.','',
 'A genuine inherited native-FP16 KV bug was found: only one32-token delta tile was packed. At decode33 (KV97), additional tokens were omitted. The corrected path packs all padded tail tiles. Failed attempt retained in full-audit-b; corrected full-audit-c agrees bitwise with independent row-major cache reconstruction for264192 hidden/norm elements and all43 selected IDs/logit codes.','',
 'Selected0/13/27 and continuous3-layer M64+M1 pass unchanged composition_v2. Head128 RoPE196608 elements and W4 decode for K3072/K8192 (360448 elements) exact against independent ISA-simulator references. 1B floating bounded regression and3B A8/SP2 full audits retained.','',
 f"Full independent FP16 mathematical alignment remains FAILED: maximum hidden/norm relative L2 {z['max_full_boundary_nrmse']:.9f}, existing limit0.003; minimum prefill cache cosine {z['min_prefill_cache_cosine']:.9f}, existing limit0.99999. This is not relabeled a pass and thresholds were not relaxed. Numerical implementation crosschecks are separate from this mathematical alignment and from model-quality/PPL acceptance. No PPL or quality claim.",'',
 '| Configuration | Prefill TPS | Decode TPS |','|---|---:|---:|',
 f"| Llama3B W4A16 | {result['formal']['prefill']['tps']:.6f} | {result['formal']['decode']['tps']:.6f} |",'',
 'Five short and ten formal repeat10 rounds. Repeat1 auxiliary only. Complete Host wall includes embedding/all28layers/final norm/LM head/greedy/FastRPC; excludes cold loading, tokenizer and audit collection. Project-owned prompt/trajectory, not named-dataset results. No automatic baseline promotion. Llama3B W16A16 remains user-deferred (5.25GiB FP16 backbone exceeds uint32 shared-offset ABI).']
 (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 tables=['# L32-0047 additive modules','Units microseconds (% complete Host wall).']
 tables+=['| Module | M64 prefill | Decode per token |','|---|---:|---:|']
 for a,b in zip(result['formal']['prefill']['modules'],result['formal']['decode']['modules']):tables.append(f"| {a['module']} | {a['us']:.3f} ({a['share']:.2f}%) | {b['us']:.3f} ({b['share']:.2f}%) |")
 (R/'MODULES.md').write_text('\n'.join(tables)+'\n')
 (R/'FULL_PROFILING_REPORT.md').write_text('\n'.join(lines+tables+raw)+'\n')
 print({m:result['formal'][m]['tps'] for m in ['prefill','decode']},flush=True)
if __name__=='__main__':main()
