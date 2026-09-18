"""Reconstruct EXP0294 paired metrics, ledgers and exact contract differences."""
import os
os.environ['QBH_REFERENCE_REVISION']='-sdk'
import experiment_exp0294 as e
import numpy as np
from measure_exp0218 import OVERVIEW,LEDGER
from summarize_exp0217 import normalized,TICKS
def main():
 rng=np.random.default_rng(294);summary={};modules={}
 for phase in ['short','formal']:
  runs=e.read(e.R/(phase+'.json'))['runs'];out={}
  for arm in ['SP2','A8']:
   a=[r for r in runs if r['arm']==arm]
   pre=float(np.mean([x['prefill_ns'] for x in a]));dec=float(np.mean([x['decode_ns'] for x in a]))
   out[arm]=dict(prefill_tps=64e9/pre,decode_tps=1e9/dec,prefill_host_ms=pre/1e6,decode42_host_ms=42*dec/1e6)
  for key in ['prefill_ns','decode_ns']:
   a=np.array([next(r[key] for r in runs if r['cycle']==i and r['arm']=='A8') for i in range(len(runs)//2)])
   b=np.array([next(r[key] for r in runs if r['cycle']==i and r['arm']=='SP2') for i in range(len(runs)//2)])
   idx=rng.integers(0,len(a),(10000,len(a)));boot=(a[idx].mean(1)/b[idx].mean(1))
   out[key]=dict(a8_over_sp2_wall_ratio=float(a.mean()/b.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
  summary[phase]=out
  if phase=='formal':
   for arm in ['SP2','A8']:
    ps=[x for r in runs if r['arm']==arm for x in e.d.records(e.R/r['tag']/'stdout.jsonl') if x.get('record')=='generation_profile']
    for mode in ['prefill','decode']:
     v=normalized([x for x in ps if x['mode']==mode]);assert abs(sum(v[k] for _,k in LEDGER)-v['invocation_ticks'])<1e-5
     modules[arm+'-'+mode]=dict(raw=v,rows={name:dict(us=sum(v[k] if k in ['host_us','host_boundary_us'] else v[k]/TICKS for k in keys),host_percent=100*sum(v[k] if k in ['host_us','host_boundary_us'] else v[k]/TICKS for k in keys)/v['host_us']) for name,keys in OVERVIEW})
 parent=e.OLD_O/'frontend64-a03';new=e.O/'full-a8-fixed-sdk'
 pm=e.read(parent/'manifest.json');nm=e.read(new/'manifest.json');frozen=[]
 for n,v in pm['files'].items():
  if 'weight_' in n or 'attention_config' in n or n.startswith('generation_') and not any(k in n for k in ['expected_token','audit']):
   assert nm['files'][n]['sha256']==v['sha256'],n;frozen.append(n)
 for i in range(28):
  a=e.load_qparams_bin(parent/f'layer{i}/qparams_u8.bin');b=e.load_qparams_bin(new/f'layer{i}/qparams_u8.bin')
  assert set(a)==set(b)
  for n in a:
   if n!='middle':assert a[n]==b[n],(i,n)
 audit=dict(pass_all=True,frozen_payload_files=len(frozen),frozen_nonmiddle_qparams=28*(len(a)-1),native_change='Qwen0.6 header accepts ordinary A8; all existing kernels unchanged',configuration_difference='SP2=8,U8_PREFILL_OPT=0 vs SP2=0,U8_PREFILL_OPT=3 (existing equivalent Gate/Up stream)',reference_recovery='SDK v79 inverse square root via independent ISA simulator; original failures retained',quality_accepted=False,baseline_promoted=False)
 e.write(e.R/'CONTRACT_AUDIT.json',audit)
 e.write(e.R/'SUMMARY.json',dict(experiment='EXP-0294',shape='M64+42/cache128/full28',short_rounds=5,formal_rounds=10,repeat=10,performance=summary,numerical_physical_pass=True,quality_claim=False,baseline_promoted=False))
 e.write(e.R/'MODULES.json',modules)
 lines=['# EXP0294 complete: Qwen3-0.6B ordinary A8','', 'Same native binary, frozen W4 weights/scales, FP32 residual mode2 and all non-Down boundaries. No rotation, new calibration, quality acceptance or baseline promotion. Only Down input representation and required one-pass/two-plane execution differ. Existing equivalent Gate/Up streams enabled on both.','', '| Recipe | Prefill TPS | Decode TPS | Prefill ms | Decode42 ms |','|---|---:|---:|---:|---:|']
 for arm,z in summary['formal'].items():
  if arm in ['SP2','A8']:lines.append(f"| {arm} | {z['prefill_tps']:.6f} | {z['decode_tps']:.6f} | {z['prefill_host_ms']:.6f} | {z['decode42_host_ms']:.6f} |")
 lines+=['','Five short + ten balanced paired formal repeat10 rounds; repeat1 auxiliary. Complete embedding/all28/finalNorm/head/greedy/FastRPC wall, excluding cold load/tokenizer. Fixed same SP2 input trajectory; not named-dataset performance.','', 'Independent selected0/14/27 and chain3 exact; full fixed and free-greedy A8, plus SP2 control, exact hidden/norm/KV/IDs/codes after reference correction. Prior reference failed at one layer24 K code: SDK rsqrt is a three-step approximation, not mathematically rounded rsqrt. Vendor ISA simulator supplies that operation only from reference-derived inputs. Hardware mathematics unchanged; no threshold relaxation. See reference_recovery.md and sdk-reference-provenance.json.']
 (e.R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 ml=['# EXP0294 additive module comparison','']
 for mode in ['prefill','decode']:
  ml+=['## '+mode,'','| Module | SP2 us (%Host) | A8 us (%Host) |','|---|---:|---:|']
  for name,_ in OVERVIEW:
   a=modules['SP2-'+mode]['rows'][name];b=modules['A8-'+mode]['rows'][name]
   ml.append(f"| {name} | {a['us']:.3f} ({a['host_percent']:.2f}%) | {b['us']:.3f} ({b['host_percent']:.2f}%) |")
 (e.R/'MODULES.md').write_text('\n'.join(ml)+'\n')
 print(summary['formal'],flush=True)
if __name__=='__main__':main()
