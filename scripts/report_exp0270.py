"""Reconstruct EXP0270 from raw fixed-round evidence; never execute devices."""
from common_exp0270 import *
from device_exp0270 import records
from measure_exp0218 import LEDGER,OVERVIEW
from summarize_exp0217 import normalized
import statistics,subprocess,numpy as np

def main():
 gate=read(R/'layer-short.json');single=read(R/'single_gate.json');assert single['pass_all']
 groups={};profiles=0;clis=0
 for p in R.rglob('stdout.jsonl'):
  a=[x for x in records(p) if x.get('record')=='exp0240_profile'];clis+=1;profiles+=len(a)
  for v in a:
   n=normalized([v]);assert sum(n[k] for _,k in LEDGER)==v['invocation_ticks']
   assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608
   assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks'])
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   for fp in [0,1]:
    z=[]
    for i in range(5):
     p=R/f'layer-short/round{i:02d}-r{rep}-fp{fp}/stdout.jsonl';a=[v for v in records(p) if v.get('record')=='exp0240_profile' and v['mode']==mode];assert len(a)==rep;z.append(normalized(a))
    groups[f'r{rep}_{mode}_fp{fp}']=z
 rng=np.random.default_rng(270);idx=rng.integers(0,5,(20000,5))
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   a=np.array([[x['host_us']*1000 for x in groups[f'r{rep}_{mode}_fp{fp}']] for fp in [0,1]])
   ci=np.quantile(a[1,idx].mean(1)/a[0,idx].mean(1),[.025,.975]);v=gate['performance'][f'r{rep}_{mode}_ns'];assert np.allclose(ci,v['ci95'],rtol=1e-12);assert abs(a[1].mean()/a[0].mean()-v['ratio'])<1e-12
 stats={key:{k:dict(mean=statistics.mean(x[k] for x in vals),median=statistics.median(x[k] for x in vals)) for k in vals[0]} for key,vals in groups.items()}
 runtime=read(R/'runtime-l1.json');head=runtime['seal']['source_head']
 summary=dict(experiment='EXP-0270',source_branch='codex/exp-0270-fp32-norm-pipeline',tested_source_head=head,selected='C1+C3',single_layer_gate=single,short_performance=gate['performance'],speed_pass=gate['speed_pass'],formal_run=False,chain3_run=False,fullmodel_run=False,e2e_run=False,PPL=None,hardware_cli=clis,hardware_profiles=profiles,unique_final_fp32_values=3*65*2048,peak_vtcm_bytes=6682752,reference_ledger=read(R/'sealed_reference_verification.json'),candidates=[dict(name='C1 eight-row ordered register Norm reduction',decision='retained, exact, no payload stack'),dict(name='C2 native paired-row quantized output',decision='rejected: exact but slower; original four-row variants rejected before device for spills'),dict(name='C3 invariant retained-output bias tables',decision='retained, exact, existing VTCM only')],baseline_promoted=False,next_direction='Discuss FP32 residual layout shared by O/Down producers and RMSNorm consumers; current row-major residual requires costly transposition. No extra experiment started.')
 write(R/'SUMMARY.json',summary);write(R/'PROFILE.json',dict(groups=stats,performance=gate['performance']))
 lines=['# EXP0270: exact Qwen SP2 FP32 residual optimization','',f'Tested source `{head}`, branch codex/exp-0270-fp32-norm-pipeline. Evidence `{R}`; frozen models `{O}`. Single-layer M64 then M1/past64/cache128, not complete-model token generation. Five fixed alternating pairs; repeat10 primary and repeat1 auxiliary, paired bootstrap20000 seed270.','', 'Independent implementation gate passes. Prefill short speed gate fails; decode passes. Formal10, chain3/full28/E2E/PPL are N/A because short prefill did not establish eligibility. Original user-selected paper speed baselines remain unchanged.','', '| Repeat / phase | Control Host us | FP32 Host us | Increase | 95% ratio CI | Gate |','|---|---:|---:|---:|---|---|']
 for k,v in gate['performance'].items():lines.append(f"| {k} | {v['control_mean_ns']/1000:.3f} | {v['candidate_mean_ns']/1000:.3f} | {(v['ratio']-1)*100:+.2f}% | {v['ci95']} | {'pass' if v['gate'] else 'fail'}{' (auxiliary)' if v['auxiliary_only'] else ''} |")
 lines+=['','## Changes and evidence','', 'C1 uses an eight-row register transpose with four ordered columns per vector. It keeps ascending-channel arithmetic, stores sums in VTCM, and has no stack in generated reduction assembly. C2 tried four-row native output, rejected two compiler-spilling attempts before device; a corrected two-row version was exact but slower (diagnostic input Norm101.26us vs C1 74.99us) and was removed. C3 prebuilds three invariant HMX retained-output bias tables (768B within the existing2KiB reserve), replacing repeated scalar rewrites. Control integer SP2 keeps its original path.','', 'Nonpaired selection diagnostics: C1->C1+C3 Down prefill162.81->142.12us, O decode51.82->46.39us. O prefill74.37->75.42us, so no O prefill gain established. These are module diagnoses, not formal or E2E speed improvements. Do not infer whole-model throughput or statistically significant gains versus EXP0269 from separate runs.','', f'Final layers0/14/27 exact FP32 outputs ({3*65*2048} values), actual Q/K, AV, postnorm, Gate/Up, SP2 live planes and prefill physical KV all match sealed independent references; repeat10 deterministic. Signed24 raw and signed32 merged bounds remain valid. All{clis} CLI/{profiles} profiles satisfy8MiB, peak6682752B, zero tensorDDR/spill/unattributed and exact additive ledgers. No new numerical failure. All546 EXP0269 sealed evidence files and three reused model manifests reverified.','', 'Assembly: reduce8, O/Down epilogue, HMX worker and streamed raw-output code have no activation stack stores. Generic Norm stack slots contain uniform constants and inverse-RMS coefficient splats, not tensor tiles or partial-sum arrays; all activation tiles/sums stay registers or VTCM. Failed C2 assemblies preserved separately. One C3 compiler API const-argument error repaired before device, recorded in c3-build-attempt.json.','', '## Stable repeat10 module overview','', 'Current control/candidate means; microseconds (percent of complete Host wall). F16F16 and W4F16 have no equivalent-scope run in this experiment and are N/A (unchanged); no cross-recipe speed ratio is claimed. Historical complete-model measurements cannot fill these single-layer cells.']
 for mode in ['prefill','decode']:
  lines+=['',f'### {mode}','','| Module | F16F16 | W4F16 | W4U8-SP2 control | W4U8-SP2 FP32 | W4F16/FP32 speed delta |','|---|---|---|---:|---:|---|']
  for name,keys in OVERVIEW:
   cells=[]
   for fp in [0,1]:
    st=stats[f'r10_{mode}_fp{fp}'];v=sum(st[k]['mean'] if k.endswith('_us') else st[k]['mean']/19.2 for k in keys)
    cells.append('N/A: outside replay' if name in ['Token embedding','Final model RMSNorm','LM head + greedy selection (excluding final norm)'] else f"{v:.2f} ({v/st['host_us']['mean']*100:.2f}%)")
   lines.append(f'| {name} | N/A | N/A | {cells[0]} | {cells[1]} | N/A |')
 lines+=['','## Complete raw-counter comparisons','', 'Each cell uses the median of five per-run means. Tick fields are converted to microseconds; other fields retain their stated units. Engine work/readiness/wait counters overlap and must not be added to the exclusive ledger. PROFILE.json also retains all means. Missing nonnumeric metadata remain in protocol.json/raw JSON. No absent hardware measurement is replaced by zero.']
 for rep in [1,10]:
  for mode in ['prefill','decode']:
   lines+=['',f'### repeat{rep} {mode}','','| Field | Control median | FP32 median | Change |','|---|---:|---:|---:|']
   a=stats[f'r{rep}_{mode}_fp0'];b=stats[f'r{rep}_{mode}_fp1']
   order=['host_us','host_boundary_us']+[k for _,k in LEDGER]+sorted(set(a)-{'host_us','host_boundary_us'}-{k for _,k in LEDGER})
   for k in order:
    av=a[k]['median'];bv=b[k]['median'];delta=f'{(bv/av-1)*100:+.3f}%' if av else 'N/A: zero denominator';div=19.2 if k.endswith('_ticks') else 1
    lines.append(f'| {k}'+(' (us)' if div!=1 else '')+f' | {av/div:.6g} | {bv/div:.6g} | {delta} |')
 lines+=['','## E2E and next direction','', 'New Qwen FP32 residual complete-model prefill/decode: N/A, not run. Selected historical Qwen SP2 integer residual EXP0268 is2030.238/48.173tok/s; selected Llama SP2 FP32 residual L32-0018 is2069.703/42.510tok/s. These separate historical full-model results are not this candidate performance.','', 'The remaining strict FP32 Norm overhead dominates the prefill difference. A future structural candidate could align FP32 O/Down residual production and RMSNorm consumption in one native physical format, avoiding row-major transposition while preserving exact ascending-channel sums. Fullmodel cost also requires actual scope measurement; no inference or gate bypass is made here. Llama remains frozen, new Qwen C1/C3 have not been timed on Llama.']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['single_layer_gate','reference_ledger']},indent=2))
if __name__=='__main__':main()
