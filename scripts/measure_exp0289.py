"""User-authorized provisional speed campaign; floating alignment failures retained."""
import sys,statistics,struct
import numpy as np
import torch
import device_exp0289 as d
from common_exp0289 import *
ARM={'f16f16':('f16f16','full-f16-a01'),
     'w4f16':('reference-a03/w4f16','full-w4-a03')}
read_original=d.read
FAST=read(R/'full-f16-a01/protocol.json')['runtime']
SLOW=read(R/'vector-sf-full-f16f16/protocol.json')['runtime']
def runtime(which):
 state=FAST if which=='fast' else SLOW
 d.read=lambda p:state if Path(p)==R/'runtime-l28.json' else read_original(p)
def signatures(tag):
 rs=d.records(R/tag/'stdout.jsonl')
 return [(v['selected_token_id'],v['selected_logit_half_bits']) for v in rs
    if 'generation_step' in v and v.get('record')!='generation_profile']
def check_profiles(tag,repeat,gold=None):
 rs=d.records(R/tag/'stdout.jsonl')
 ps=[v for v in rs if v.get('record')=='generation_profile']
 finals=[v for v in rs if v.get('generation_sequence_complete')]
 assert len(ps)==43*repeat and len(finals)==repeat
 sig=signatures(tag)
 if gold is not None:assert sig==gold*repeat,(tag,'tokens/logits changed')
 fields=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks',
 'qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks',
 'post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks',
 'cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks',
 'layer_bookkeeping_ticks','layer_unattributed_ticks']
 for i,p in enumerate(ps):
  step=i%43
  assert p['generation_step']==step
  assert p['boundary_ddr_write_bytes']==0
  assert p['generation_lm_head_prefetch_count']==593
  for layer in range(28):
   z=p[f'slice_layer_{layer}']
   assert z['status']==3 and z['layer_index']==layer
   assert z['cache_valid_before']==(0 if step==0 else 63+step)
   assert z['cache_valid_after']==64+step
   assert z['hidden_ddr_read_bytes']==z['hidden_ddr_write_bytes']==z['layer_unattributed_ticks']==0
   assert sum(z[k] for k in fields)==z['layer_ticks']
 return ps
def hardware():
 preflight()
 # Both memory namespaces agree no other active device experiment.
 import yaml
 other=yaml.safe_load(Path('/home/daniuniu/work/llama32-htp-project-memory/PROJECT_STATUS.yaml').read_text())
 assert other['governance']['active_experiment'] is None
 assert not d.adb('shell','pidof qwen3_block_cli llama_sp2_cli',check=False).stdout.strip()
 for state in [FAST,SLOW]:
  for name,h in state['seal']['files'].items():
   f=Path(state['archive'])/Path(name).name
   assert sha(f)==h
   assert d.adb('shell','sha256sum '+state['remote']+'/'+f.name).stdout.split()[0]==h
 for recipe,(package,audit) in ARM.items():
  m=read(O/package/'manifest.json');remote=read(R/('deployment-'+package+'.json'))['remote']
  for n,v in m['files'].items():assert sha(O/package/n)==v['sha256'],n
  names=list(m['files'])
  for start in range(0,len(names),32):
   txt=d.adb('shell','sha256sum '+' '.join(remote+'/'+n for n in names[start:start+32])).stdout
   for line in txt.splitlines():
    h,n=line.split(None,1);assert h==m['files'][n.removeprefix(remote+'/')]['sha256']
  assert read(R/audit/'validated.json')['physical_pass']
  assert read(R/('slice-f16f16-a01' if recipe=='f16f16' else 'slice-w4-reference-a03')/'validated.json')['numerical_pass']
  checks=0
  for layer in range(28):
   for kind in ['k','v']:
    prev=None
    for step in range(43):
     x=np.fromfile(R/audit/f'generation_step{step:02d}_layer{layer:02d}_{kind}_f16.bin','<f2').reshape(8,128*128)
     assert np.isfinite(x).all(),(recipe,layer,kind,step)
     live=(64+step)*128
     if prev is not None:assert np.array_equal(x[:,:live-128],prev[:,:live-128]),(recipe,layer,kind,step,'prefix changed')
     assert np.all(x[:,live:]==0),(recipe,layer,kind,step,'write beyond live cache')
     assert np.any(x[:,:live]!=0)
     prev=x;checks+=1
  write(R/('cache-invariants-'+recipe+'.json'),dict(pass_all=True,files=checks,append_steps=42,layers=28,cache_capacity=128))
  print('HARDWARE_CACHE_PASS',recipe,checks,flush=True)
 write(R/'hardware-invariants.json',dict(pass_all=True,full_floating_alignment_pass=False,
  original_failures_retained=True,user_authorized_provisional_timing=True))
def one(recipe,tag,repeat=10,which='fast'):
 runtime(which);pkg,audit=ARM[recipe]
 out=d.run(pkg,tag,count=28,repeat=repeat,audit=False,full=True)
 gold=signatures(audit if which=='fast' else 'vector-sf-full-'+recipe)
 ps=check_profiles(tag,repeat,gold)
 return dict(recipe=recipe,tag=tag,repeat=repeat,runtime=which,
  prefill_ns=statistics.mean(p['host_wall_ns'] for p in ps if p['mode']=='prefill'),
  decode_ns=statistics.mean(p['host_wall_ns'] for p in ps if p['mode']=='decode'),
  profiles=len(ps),deterministic_token_logit_pass=True)
def campaign():
 assert read(R/'hardware-invariants.json')['pass_all']
 assert read(R/'head-boundary-checks.json')['pass_all']
 for recipe in ARM:one(recipe,'speed-warmup-'+recipe,1)
 write(R/'speed-repeat1.json',[one(recipe,'speed-repeat1-'+recipe,1) for recipe in ARM])
 for phase,n in [('short',5),('formal',10)]:
  rows=[]
  for i in range(n):
   order=list(ARM) if i%2==0 else list(ARM)[::-1]
   block=[]
   for recipe in order:block.append(dict(round=i,**one(recipe,f'speed-{phase}/{i:02d}-{recipe}')))
   rows+=block;write(R/f'speed-{phase}-round-{i:02d}.json',block)
   print('ROUND_DONE',phase,i,flush=True)
  write(R/f'speed-{phase}.json',dict(pass_all=True,rounds=n,repeat=10,runs=rows,full_floating_alignment_pass=False))
 write(R/'fp32-intermediate-repeat1.json',[one(recipe,'sf-intermediate-repeat1-'+recipe,1,'sf') for recipe in ARM])
if __name__=='__main__':
 if sys.argv[1]=='hardware':hardware()
 elif sys.argv[1]=='campaign':campaign()
