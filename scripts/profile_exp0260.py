#!/usr/bin/env python3
from device_exp0260 import *
import numpy as np
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized

def physical(ps,count,arm):
 for q in ps:
  assert q['w4f16_decode_opt']==arm and q['w4f16_decode_audit']==0
  assert q['block_invocation_count']==count and q['vtcm_acquired_bytes']==8388608 and q['vtcm_requested_bytes']==8388608
  assert q['intermediate_ddr_read_bytes']==q['intermediate_ddr_write_bytes']==q['intermediate_spill_fill_count']==0
  assert sum(normalized([q])[k] for _,k in LEDGER)==q['invocation_ticks']
  assert q['w4f16_decode_conversion_audit_mismatches']==0
  assert q['w4f16_decode_opt_calls']==(count*8 if arm and q['mode']=='decode' else 0)
  for i in range(count):
   l=q[f'slice_layer_{i}'];assert l['status']==3 and l['layer_unattributed_ticks']==0
   fields=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','layer_unattributed_ticks']
   assert sum(l[k] for k in fields)==l['layer_ticks'] and l['layer_index']==i
   step=q.get('generation_step',q.get('replay_step'));assert l['cache_valid_before']==(0 if step==0 else 63+step) and l['cache_valid_after']==64+step
   assert l['hidden_ddr_read_bytes']==(262144 if count!=28 and i==0 else 0)
   assert l['hidden_ddr_write_bytes']==(262144 if count!=28 and i==count-1 else 0)
  assert q['boundary_ddr_write_bytes']==(0 if count==28 else 262144)

def numerical():
 preflight();checks=[]
 for name in ['layer_control_final','layer_batch_audit','layer_vector_exact']:
  for p in sorted((R/'layer_control').glob('*.bin')):
   q=R/name/p.name;assert q.read_bytes()==p.read_bytes(),str(q);checks.append(str(q.relative_to(R)))
 for name in ['layer_vector_exact']:
  ps=[q for q in records(R/name/'stdout.jsonl') if q.get('record')=='exp0240_profile']
  assert len(ps)==9 and all(q['w4f16_decode_conversion_audit_mismatches']==0 for q in ps)
 write(R/'layer_numerical_gate.json',dict(pass_all=True,files=checks,comparisons=len(checks),sentinels='63488 finite half patterns, 46080 [0,1] midpoint and neighbouring float values; negative zero preserved',all_live_hidden_and_complete_KV_exact=True,real_softmax_probability_all_physical_rows_exact=True))

def profile(phase,count):
 preflight();assert read(R/'layer_numerical_gate.json')['pass_all']
 if count==28:assert read(R/'slice_gate.json')['pass_all']
 prefix='layer' if count==1 else 'full';n=5 if phase=='short' else 10
 if phase=='formal':assert read(R/f'{prefix}_short_gate.json')['integrity_pass']
 rows=[]
 gold=read(R/('layer_control/validated.json' if count==1 else 'full_control/validated.json'))
 for i in range(n):
  for rep in [1,10]:
   for arm in ([0,2] if i%2==0 else [2,0]):
    tag=f'{prefix}_{phase}/round{i+1:02d}_r{rep}_a{arm}';p=R/tag/'validated.json'
    if not p.exists():run(count,arm,rep,tag)
    z=read(p);ps=[q for q in records(p.parent/'stdout.jsonl') if q.get('record') in ['exp0240_profile','generation_profile']];physical(ps,count,arm)
    if count==1:assert z['output_hashes']==gold['output_hashes']*rep
    else:assert z['token_sequences']==gold['token_sequences']*rep and z['selected_codes']==gold['selected_codes']*rep
    rows.append(dict(round=i+1,**z))
 rng=np.random.default_rng(260);perf={}
 for rep in [1,10]:
  for mode in ['prefill_ns','decode_ns']:
   pairs=np.array([[next(r[mode] for r in rows if r['round']==i+1 and r['repeat']==rep and r['arm']==a) for a in [0,2]] for i in range(n)]);ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,n,(10000,n))],axis=1),[.025,.975]);perf[f'r{rep}_{mode}']=dict(control_ns=float(np.median(pairs[:,0])),candidate_ns=float(np.median(pairs[:,1])),paired_ratio=float(np.median(ratios)),ci95=ci.tolist())
 eligible=perf['r10_decode_ns']['ci95'][1]<1.0 and all(perf[f'r{r}_prefill_ns']['ci95'][1]<=1.10 for r in [1,10])
 write(R/f'{prefix}_{phase}_gate.json',dict(integrity_pass=True,performance=perf,rows=rows,speed_eligible=eligible));print(prefix,phase,json.dumps(perf),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--count',type=int,default=1);a=p.parse_args()
 if a.action=='numerical':numerical()
 else:profile(a.action,a.count)
