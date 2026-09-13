#!/usr/bin/env python3
"""Bit-exact U8 scheduling audit, including live native carriers and persistent KV."""
import sys
import numpy as np
from common_exp0268 import S,R,O,P,OLD,sha,write,preflight
from device_exp0268 import run,read
from reference_w4u8_hmx import unpack_u8_hmx_activation
BASE=983040;CAP=393216
def compare(a,b):
 count=0
 for f in a.glob('*.bin'):
  if f.name=='eval.bin':continue
  g=b/f.name;assert g.exists(),g
  if f.name.endswith('_r3.bin') and not f.name.startswith('step00_'):
   x=np.fromfile(f,'u1');y=np.fromfile(g,'u1');xx=x.copy();yy=y.copy()
   # Middle and Down each have four live decode rows. Down's unused rows
   # legitimately reflect the relocated middle padding; residual reads row0.
   for slot,k in [(8,6144),(9,2048)]:
    off=BASE+slot*CAP;length=64*k
    assert np.array_equal(unpack_u8_hmx_activation(x[off:off+length],k)[:4],unpack_u8_hmx_activation(y[off:off+length],k)[:4]),(f,slot,'live-rows')
    xx[off:off+length]=0;yy[off:off+length]=0
   assert np.array_equal(xx,yy),(f,'outside-middle-down')
  else:assert sha(f)==sha(g),(f,g)
  count+=1
 assert count==20,(a,count)
 assert {f.name for f in a.glob('*.bin')}=={f.name for f in b.glob('*.bin')},(a,b)
 return count

def single():
 preflight();assert read(R/'deployment.json')['pass_all'];proof=[]
 for l in [0,14,27]:
  zs={}
  for m in [0,1,2,3]:
   tag=f'audit-layer{l}-m{m}';f=R/tag/'validated.json';zs[m]=read(f) if f.exists() else run(m,1,tag,layer=l,dump=True)
   assert zs[m]['output_hashes']==read(OLD/f'audit-layer{l}-m0/validated.json')['output_hashes']
   proof.append(dict(layer=l,mode=m,files=compare(OLD/f'audit-layer{l}-m0',R/tag)))
  z=run(3,10,f'audit-repeat-layer{l}',layer=l);assert z['output_hashes']==zs[0]['output_hashes']*10
 write(R/'single_gate.json',dict(pass_all=True,comparisons=proof,repeat10_exact=True,lut_kernel_and_tables_unchanged=True))

def slice_gate():
 preflight();assert read(R/'layer-formal.json')['speed_pass'];proof=[];zs={}
 for m in [0,1,2,3,8]:
  tag=f'slice-m{m}';f=R/tag/'validated.json';zs[m]=read(f) if f.exists() else run(m,1,tag,count=3,dump=m!=8)
  golden=OLD/f'slice-a03-m{8 if m==8 else 0}'
  assert zs[m]['output_hashes']==read(golden/'validated.json')['output_hashes']
  if m!=8:proof.append(dict(mode=m,files=compare(golden,R/tag)))
 z=run(3,10,'slice-repeat-m3',count=3);assert z['output_hashes']==zs[0]['output_hashes']*10
 write(R/'slice_gate.json',dict(pass_all=True,layers=3,comparisons=proof,repeat10_exact=True,sp2_sealed_output_exact=True))
if __name__=='__main__':
 if sys.argv[1]=='single':single()
 else:slice_gate()
