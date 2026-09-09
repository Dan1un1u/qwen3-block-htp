#!/usr/bin/env python3
"""Five short / ten formal rotated native-A8 token-boundary rounds."""
from device_exp0257 import *
from measure_exp0218 import LEDGER,OVERVIEW
from summarize_exp0217 import normalized
import struct,statistics

def validate(path,wide,repeats):
 records=[json.loads(l) for l in (path/'stdout.jsonl').read_text().splitlines() if l.startswith('{')]
 profiles=[x for x in records if x.get('record')=='generation_profile'];final=[x for x in records if x.get('generation_sequence_complete')]
 assert len(profiles)==repeats*16 and len(final)==repeats
 ref=json.loads((R/('full_nr64' if wide==4 else 'full_sole')/'validated.json').read_text())['token_ids'][:16]
 for f in final:assert f['all_steps_pass'] and f['token_ids']==ref
 fields=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','layer_unattributed_ticks']
 for i,p in enumerate(profiles):
  step=i%16;assert p['generation_step']==step and p['repeat_count']==1 and p['wide_score_mode']==wide
  assert p['variant']=='W4U8' and p['backend']=='standalone_fastrpc_dsp' and p['qnn']=='none'
  assert p['vtcm_acquired_bytes']==p['vtcm_requested_bytes']==8388608 and p['block_invocation_count']==28
  assert p['boundary_ddr_write_bytes']==p['intermediate_ddr_read_bytes']==p['intermediate_ddr_write_bytes']==p['intermediate_spill_fill_count']==0
  n=normalized([p]);assert sum(n[k] for _,k in LEDGER)==p['invocation_ticks'],('ledger',i)
  for j in range(28):
   q=p[f'slice_layer_{j}'];assert q['status']==3 and q['layer_index']==j and sum(q[k] for k in fields)==q['layer_ticks']
   assert q['layer_unattributed_ticks']==q['hidden_ddr_read_bytes']==q['hidden_ddr_write_bytes']==0
   assert q['cache_valid_before']==(0 if step==0 else 63+step) and q['cache_valid_after']==64+step
 result=dict(pass_all=True,wide=wide,repeats=repeats,profiles=len(profiles),prefill_ns=sum(p['host_wall_ns'] for p in profiles[::16])/repeats,decode_ns=sum(p['host_wall_ns'] for i,p in enumerate(profiles) if i%16)/repeats)
 write(path/'validated.json',result);return result

def suite(wide,repeat,tag):
 root,e=env(28,wide);e.update(QBH_GENERATION_SEQUENCE='9',QBH_GENERATION_STEPS='16')
 ids=json.loads((R/'prompts.json').read_text())['samples'][0]['token_ids'];row=[0,2,16]+ids+[0]*16
 file=R/(tag.replace('/','_')+'.bin');file.write_bytes(struct.pack('<4I',0x51424556,1,repeat,83)+b''.join(struct.pack('<83I',i,*row[1:]) for i in range(repeat)))
 remote=root+'/'+file.name;adb('push',win(file),remote);e['QBH_EVAL_FILE']=remote
 p=execute(28,e,REMOTE+'-package-v2',1,tag);return validate(p,wide,repeat)

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['short','formal']);a=parser.parse_args()
 preflight();assert json.loads((R/'full_model_gate.json').read_text())['pass_all']
 if a.phase=='formal':assert json.loads((R/'short_gate.json').read_text())['pass_all']
 rows=[]
 for i in range(5 if a.phase=='short' else 10):
  for repeat in [1,10]:
   for wide in ([1,4] if i%2==0 else [4,1]):
    tag=f'{a.phase}/round{i+1:02d}_r{repeat}_w{wide}'
    p=R/tag/'validated.json'
    d=json.loads(p.read_text()) if p.exists() else suite(wide,repeat,tag)
    rows.append(dict(round=i+1,repeat=repeat,**d));print('PROFILE',a.phase,i+1,repeat,wide,round(d['prefill_ns']/1e6,3),round(d['decode_ns']/15e6,3),flush=True)
 # Gate stable speed using stratified paired-round bootstrap, fixed10000seed257.
 import numpy as np
 perf={};ok=True;rng=np.random.default_rng(257)
 for repeat in [1,10]:
  for mode in ['prefill_ns','decode_ns']:
   pairs=np.array([[next(r[mode] for r in rows if r['round']==i+1 and r['repeat']==repeat and r['wide']==w) for w in [1,4]] for i in range(5 if a.phase=='short' else 10)])
   ratios=pairs[:,1]/pairs[:,0];boot=np.median(ratios[rng.integers(0,len(ratios),(10000,len(ratios)))] ,axis=1);ci=np.quantile(boot,[.025,.975]);stable=bool(ci[0]>1.1);ok&=not stable
   perf[f'r{repeat}_{mode}']=dict(control_ns=float(np.median(pairs[:,0])),candidate_ns=float(np.median(pairs[:,1])),paired_ratio=float(np.median(ratios)),ci95=ci.tolist(),stable_over10percent=stable)
 write(R/(a.phase+'_gate.json'),dict(pass_all=ok,physical_correctness_pass=True,rounds=rows,performance=perf))
 if not ok:raise SystemExit('Stable>10percent slowdown: stop dependent escalation')
 print(a.phase.upper()+'_GATE_PASS',flush=True)
if __name__=='__main__':main()
