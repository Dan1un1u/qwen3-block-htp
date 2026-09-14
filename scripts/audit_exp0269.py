"""Independent physical boundary, KV and fixed paired timing gates."""
import argparse,json
import numpy as np
from common_exp0269 import *
from device_exp0269 import run,records
from reference_w4u8_hmx import unpack_u8_hmx_activation,unpack_w4_codes
from prepare_exp0161_segmented_cache import pack_k_segment,pack_v_segment
BASE=983040;CAP=393216

def audit(package,tag):
 p=O/package;d=R/tag;z=read(d/'validated.json');assert z['numerical_pass']
 for step in range(2):
  rows=64 if step==0 else 1;b=np.fromfile(d/f'step{step:02d}_r3.bin','u1');assert len(b)==BASE+11*CAP
  for slot,name,k in [(2,'attention',2048),(5,'post',2048),(6,'gate',6144),(7,'up',6144)]:
   actual=unpack_u8_hmx_activation(b[BASE+slot*CAP:BASE+slot*CAP+64*k],k)[:rows];expect=np.load(p/f'fp32_step{step:02d}_{name}.npy');assert np.array_equal(actual,expect),(tag,step,name)
  codes=b[2*CAP:BASE].reshape(24,4,64,32).transpose(0,2,1,3).reshape(24,64,128)[:,:rows]
  for sl,name in [(slice(0,16),'q'),(slice(16,24),'k')]:assert np.array_equal(codes[sl],np.load(p/f'fp32_step{step:02d}_{name}.npy').transpose(1,0,2)),(tag,step,name)
  mid=np.load(p/f'fp32_step{step:02d}_middle.npy');a=unpack_u8_hmx_activation(b[BASE+8*CAP:BASE+9*CAP],6144);assert np.array_equal(a[:rows],((mid+32768)&255).astype('u1'))
  if step:assert np.array_equal(a[4:5],((mid+32768)>>8).astype('u1'))
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

def single(prefix=""):
 preflight();proof=[]
 for l in [0,14,27]:
  p=f'layer{l}-fp32-a01';tag=prefix+f'final-audit-layer{l}';f=R/tag/'validated.json'
  if not f.exists():run(p,tag,dump=True)
  af=R/tag/'independent_gate.json';proof.append(read(af) if af.exists() else audit(p,tag))
  rz=run(p,prefix+f'final-repeat-layer{l}',repeat=10);assert rz['output_hashes']==read(f)['output_hashes']*10
 write(R/(prefix+'single_gate.json'),dict(pass_all=True,layers=proof,repeat10_exact=True))

def profile(phase,prefix=""):
 preflight();assert read(R/(prefix+'single_gate.json'))['pass_all']
 if phase=='formal':assert read(R/(prefix+'layer-short.json'))['speed_pass']
 n=5 if phase=='short' else 10;rows=[]
 for i in range(n):
  for rep in [1,10]:
   for fp in ([0,1] if i%2==0 else [1,0]):
    tag=prefix+f'layer-{phase}/round{i:02d}-r{rep}-fp{fp}';f=R/tag/'validated.json';z=read(f) if f.exists() else run('layer0-fp32-a01',tag,repeat=rep,fp32=fp)
    gold=read(R/(prefix+'final-audit-layer0' if fp else 'diag-control-a01')/'validated.json')['output_hashes'][:2];assert z['output_hashes']==gold*rep;rows.append(dict(round=i,repeat=rep,fp32=fp,**z))
 rng=np.random.default_rng(269);idx=rng.integers(0,n,(20000,n));perf={}
 for rep in [1,10]:
  for key in ['prefill_ns','decode_ns']:
   pair=np.array([[next(z[key] for z in rows if z['round']==i and z['repeat']==rep and z['fp32']==m) for m in [0,1]] for i in range(n)]);ci=np.quantile(pair[idx,1].mean(1)/pair[idx,0].mean(1),[.025,.975]);perf[f'r{rep}_{key}']=dict(control_mean_ns=float(pair[:,0].mean()),candidate_mean_ns=float(pair[:,1].mean()),ratio=float(pair[:,1].mean()/pair[:,0].mean()),ci95=ci.tolist(),gate=bool(ci[1]<=1.1),auxiliary_only=rep==1)
 write(R/(prefix+'layer-'+phase+'.json'),dict(pass_all=True,rows=rows,performance=perf,speed_pass=all(v['gate'] for k,v in perf.items() if k.startswith('r10_'))));print('PROFILE_DONE',phase,json.dumps(perf),flush=True)
if __name__=='__main__':
 import sys
 if sys.argv[1]=='single':single(sys.argv[2] if len(sys.argv)>2 else '')
 elif sys.argv[1] in ['short','formal']:profile(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else '')
 else:audit(sys.argv[1],sys.argv[2])
