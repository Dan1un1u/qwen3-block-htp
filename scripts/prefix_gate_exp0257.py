#!/usr/bin/env python3
"""Prefix publication: independent full repack and actual integer attention oracle."""
from device_exp0257 import *
import numpy as np
import audit_exp0248 as au
import integer_attention_exp0252 as ia
from prepare_exp0042_attention import CONFIG
from prepare_exp0161_segmented_cache import pack_k_segment

def main():
 preflight();seed=O/'prefix/prefix_kv_u8.bin';proof=json.loads((R/'prefix_export.json').read_text());assert proof['pass_all'] and sha(seed)==proof['sha256']
 remote=REMOTE+'-prefix';adb('shell',f'mkdir {remote}');adb('push',win(seed),remote+'/prefix_kv_u8.bin');assert adb('shell',f'sha256sum {remote}/prefix_kv_u8.bin').split()[0]==proof['sha256']
 write(R/'prefix_device.json',dict(remote=remote+'/prefix_kv_u8.bin',sha256=proof['sha256']))
 base=REMOTE+'-package-v2';a=replay(3,4,'seed_slice_fast',base,prefix_mode=1);b=replay(3,4,'seed_slice_repack',base,prefix_mode=2)
 data=np.fromfile(seed,np.uint8).reshape(28,2,8,128);au.R=R;ks=[];vs=[];rows=[]
 cfg=list(CONFIG.iter_unpack((O/'package/layer2/attention_config_all_groups.bin').read_bytes()))
 for step in range(9):
  name=f'step{step:02d}_output.bin';assert (a/name).read_bytes()==(b/name).read_bytes()
  _,_,codes,slots=au.capture('seed_slice_fast',step);nr=64 if step==0 else 1
  q=codes[:16];k=codes[16:].copy();v=au.feat(slots[10],8)[:,:nr].copy()
  if step==0:k[:,0]=data[2,0];v[:,0]=data[2,1]
  ks.append(k);vs.append(v);kk=np.concatenate(ks,axis=1);vv=np.concatenate(vs,axis=1)
  valid=np.tril(np.ones((64,64),bool)) if step==0 else np.ones((1,64+step),bool)
  refs={n:[] for n in ['raw','probability','av']}
  for g,c in enumerate(cfg):
   out=ia.numpy_oracle(q[2*g:2*g+2],np.repeat(kk[g:g+1],2,0),np.repeat(vv[g:g+1],2,0),valid,ia.config(c),'wide_nr64')
   for n in refs:refs[n].extend(out[n])
  got=dict(raw=au.scores(slots[0]) if step==0 else slots[0][:2048].reshape(16,128)[:,:64+step,None].transpose(0,2,1),probability=au.scores(slots[1]) if step==0 else slots[1][:2048].reshape(16,128)[:,:64+step,None].transpose(0,2,1),av=au.feat(slots[2],16)[:,:nr])
  checks={n:au.diff(got[n],np.array(refs[n])) for n in refs};assert all(v['changed']==0 for v in checks.values()),(step,checks)
  rows.append(dict(step=step,output_exact_repack=True,integer=checks))
 # First layer's actual stored K carrier contains the sealed offline seed at token0.
 cfg0=list(CONFIG.iter_unpack((O/'package/layer0/attention_config_all_groups.bin').read_bytes()));cache=(a/'prefill_k_cache.bin').read_bytes();per_head=len(cache)//8
 for g,c in enumerate(cfg0):
  # Independent K packing embeds token0 into first segment; compare its column and bias.
  matrix=np.full((32,128),c[5],np.uint8);matrix[0]=data[0,0,g];packed=pack_k_segment(matrix,c)
  actual=cache[g*per_head:g*per_head+len(packed)]
  for ch in range(128):
   off=(ch//32)*1024+(ch%32//4)*128+(ch%4);assert actual[off]==packed[off]
  assert actual[4096:4100]==packed[4096:4100] and actual[4224:4228]==packed[4224:4228]
 write(R/'seed_slice_gate.json',dict(pass_all=True,steps=rows,seed_cache_verified=True,scope='layers0..2 M64+8M1; independent scalar int64 attention and complete K repack',prefix_bytes=57344))
 print('SEED_SLICE_GATE_PASS',flush=True)
if __name__=='__main__':main()
