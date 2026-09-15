#!/usr/bin/env python3
import sys,json,statistics,subprocess,hashlib
from pathlib import Path
import numpy as np
S=Path(sys.argv[1]);sys.path.insert(0,str(S/'scripts'))
import f16_baseline_refresh as f
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized
F=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','layer_unattributed_ticks']
def check(z):
 ps=[q for q in f.records(f.R/z['tag']/'stdout.jsonl') if q.get('record')=='generation_profile']
 gold=f.read(f.R/'full-control-a01/validated.json');rep=z['repeat']
 assert 'QBH_GENERATION_BOUNDARY_AUDIT' not in f.read(f.R/z['tag']/'protocol.json')['command']
 assert z['selected_codes']==gold['selected_codes']*rep and z['token_sequences']==gold['token_sequences']*rep
 assert len(ps)==16*rep
 for step,q in enumerate(ps):
  assert q['boundary_ddr_write_bytes']==0 and q['w4f16_decode_audit']==0
  assert q['backend']=='standalone_fastrpc_dsp' and q['qnn']=='none'
  assert sum(normalized([q])[k] for _,k in LEDGER)==q['invocation_ticks']
  assert q['w4f16_decode_opt_calls']==(f.COUNT*(8 if f.LLAMA else 8) if z['opt'] and step%16 else 0)
  for i in range(f.COUNT):
   l=q[f'slice_layer_{i}'];assert l['status']==3 and l['layer_index']==i and l['layer_unattributed_ticks']==0
   assert sum(l[k] for k in F)==l['layer_ticks']
   assert l['cache_valid_before']==(0 if step%16==0 else 63+step%16) and l['cache_valid_after']==64+step%16
   assert l['hidden_ddr_read_bytes']==l['hidden_ddr_write_bytes']==0
 return ps
f.preflight();f.R.mkdir(parents=True,exist_ok=True)
assert f.read(f.R/'full-control-a01-paired-gate.json')['pass_all']
baseboot=f.read(f.R/f'runtime-l{f.COUNT}.json')['boot_id'];assert f.adb('shell','cat /proc/sys/kernel/random/boot_id').stdout.strip()==baseboot
for phase,n,rep in [('warmup',1,1),('auxiliary',1,1),('short',5,10),('formal',10,10)]:
 rows=[]
 for i in range(n):
  for opt in ([0,3] if i%2==0 else [3,0]):
   tag=f'{phase}/round{i+1:02d}-opt{opt}-r{rep}'
   path=f.R/tag/'validated.json'
   if path.exists():z=f.read(path)
   else:z=f.run(f.COUNT,opt,rep,tag)
   check(z);rows.append(dict(round=i+1,**z))
 if not (f.R/f'{phase}-validated.json').exists():f.write(f.R/f'{phase}-validated.json',dict(integrity_pass=True,rows=rows))
 print('PHASE_COMPLETE',phase,flush=True)
rows=f.read(f.R/'formal-validated.json')['rows'];rng=np.random.default_rng(283 if not f.LLAMA else 39);result={}
for mode in ['prefill_ns','decode_ns']:
 pairs=np.array([[next(r[mode] for r in rows if r['round']==i+1 and r['opt']==a) for a in [0,3]] for i in range(10)])
 ratio=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratio[rng.integers(0,10,(10000,10))],axis=1),[.025,.975])
 result[mode]=dict(control_ns=float(np.median(pairs[:,0])),candidate_ns=float(np.median(pairs[:,1])),wall_ratio=float(np.median(ratio)),ci95=ci.tolist())
for mode,tokens in [('prefill_ns',64),('decode_ns',1)]:
 for opt in ['control','candidate']:result[mode][opt+'_tokens_per_second']=tokens*1e9/result[mode][opt+'_ns']
f.write(f.R/'formal-summary.json',dict(experiment=f.EXP,protocol='warmup, auxiliary r1, five short r10, ten balanced formal r10; medians of round means; no optional stopping',counts=dict(prefill=64,decode=15),performance=result,baseline_promoted=False,model_quality='unchanged arithmetic; PPL not rerun'))
print(json.dumps(result,indent=2),flush=True)
