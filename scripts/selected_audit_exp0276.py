"""Independent physical boundary, KV and fixed paired timing gates."""
import argparse,json
import numpy as np
from common_exp0276 import *
from device_exp0276 import run,records
from reference_w4u8_hmx import unpack_u8_hmx_activation,unpack_w4_codes
from prepare_exp0161_segmented_cache import pack_k_segment,pack_v_segment
BASE=983040;CAP=393216

def audit(package,tag):
 p=package_path(package);d=R/tag;z=read(d/'validated.json');assert z['numerical_pass']
 for step in range(2):
  rows=64 if step==0 else 1;b=np.fromfile(d/f'step{step:02d}_r3.bin','u1');assert len(b)==BASE+11*CAP
  for slot,name,k in [(2,'attention',2048),(5,'post',2048),(6,'gate',6144),(7,'up',6144)]:
   actual=unpack_u8_hmx_activation(b[BASE+slot*CAP:BASE+slot*CAP+64*k],k)[:rows];expect=np.load(p/f'fp32_step{step:02d}_{name}.npy');assert np.array_equal(actual,expect),(tag,step,name)
  codes=b[2*CAP:BASE].reshape(24,4,64,32).transpose(0,2,1,3).reshape(24,64,128)[:,:rows]
  for sl,name in [(slice(0,16),'q'),(slice(16,24),'k')]:assert np.array_equal(codes[sl],np.load(p/f'fp32_step{step:02d}_{name}.npy').transpose(1,0,2)),(tag,step,name)
  mid=np.load(p/f'fp32_step{step:02d}_middle.npy');a=unpack_u8_hmx_activation(b[BASE+8*CAP:BASE+9*CAP],6144);assert np.array_equal(a[:rows],((mid+32768)&255).astype('u1'))
  if read(d/'protocol.json')['command'].find('QBH_PAPER_FORMAT_DISABLE=4 ')>=0:
   high=unpack_u8_hmx_activation(b[BASE+9*CAP:BASE+10*CAP],6144);assert np.array_equal(high[:rows],((mid+32768)>>8).astype('u1'))
  elif step:assert np.array_equal(a[4:5],((mid+32768)>>8).astype('u1'))
 cfg=np.fromfile(p/'layer0/attention_config_all_groups.bin','i4').reshape(8,15).tolist();k=np.load(p/'fp32_step00_cache_k.npy');v=np.load(p/'fp32_step00_cache_v.npy');ks=[];vs=[]
 for g in range(8):
  ks.append(b''.join(pack_k_segment(k[g,i:i+32],cfg[g]) for i in [0,32])+bytes(4352+4096))
  seg=[pack_v_segment(v[g,i:i+32],cfg[g]) for i in [0,32]]
  vs.append(b''.join(seg[0][0][nt*1024:(nt+1)*1024]+seg[1][0][nt*1024:(nt+1)*1024]+bytes(1024) for nt in range(4))+seg[0][1]+bytes(4096))
 assert (d/'prefill_k_cache.bin').read_bytes()==b''.join(ks),(tag,'K physical')
 assert (d/'prefill_v_cache.bin').read_bytes()==b''.join(vs),(tag,'V physical')
 bounds={}
 for name,kdim in [('o',2048),('down',6144)]:
  w=unpack_w4_codes(p/'layer0',name,2048,kdim).astype('i8');pos=np.maximum(w,0).sum(1);neg=-np.minimum(w,0).sum(1)
  bound=int(max(pos.max(),neg.max())*255);assert bound<2**23
  merged=int(np.abs(w).sum(1).max()*32768);assert merged<2**31
  bounds[name]=dict(raw_signed24_bound=bound,reconstructed_signed32_bound=merged)
 z=dict(pass_all=True,package=package,tag=tag,fp32_output_exact=True,QK_AV_postnorm_gate_up_SP2_exact=True,KV_physical_exact=True,bounds=bounds)
 write(d/'independent_gate.json',z);print('AUDIT_PASS',tag,flush=True);return z

if __name__=="__main__":
 import sys
 audit(sys.argv[1],sys.argv[2])
