#!/usr/bin/env python3
"""Independent real-carrier integer attention and dense refinement audit."""
import sys,json,hashlib,math
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from prepare_exp0042_attention import CONFIG,centered_hmx_requant,log2_softmax
from prepare_exp0161_segmented_cache import pack_k_segment
from audit_exp0247 import ulp
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0248')
P=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0247/r3')
BASE=983040;CAP=393216
names=['raw_score','probability','AV','O','residual_mid','post_norm','gate','up','SwiGLU','down','V']
sizes=[65536,65536,131072,131072,131072,131072,393216,393216,393216,131072,65536]
def feat(x,h):return x.reshape(h,4,64,32).transpose(0,2,1,3).reshape(h,64,128)
def scores(x):return x.reshape(16,2,64,32).transpose(0,2,1,3).reshape(16,64,64)
def capture(tag,step):
 b=np.fromfile(R/tag/f'step{step:02d}_r3.bin',np.uint8);assert len(b)==BASE+11*CAP
 nr=64 if step==0 else 1
 raw=b[:24*nr*256].view('<f2').reshape(24*nr,128)
 out=b[CAP:CAP+24*nr*256].view('<f2').reshape(24*nr,128)
 codes=feat(b[2*CAP:BASE],24)[:,:nr]
 slots=[b[BASE+i*CAP:BASE+i*CAP+n] for i,n in enumerate(sizes)]
 return raw,out,codes,slots
configs=list(CONFIG.iter_unpack((P/'attention_config_all_groups.bin').read_bytes()))
qp=json.loads((P/'manifest.json').read_text())['qparams']
def diff(a,b):
 d=a.astype(np.int32)-b.astype(np.int32)
 return dict(max_lsb=int(abs(d).max()),changed=int(np.count_nonzero(d)),elements=int(d.size),mean_abs_lsb=float(abs(d).mean()))
def integer_core(tag):
 _,_,c,slots=capture(tag,0);q,k=c[:16],c[16:];v=feat(slots[10],8)
 actual=[scores(slots[0]),scores(slots[1]),feat(slots[2],16)];expected=[[],[],[]];exponents=[]
 for g,fields in enumerate(configs):
  abi,group,fb,dm,qz,kz,vz,pz,oz,vn,vd,ss,sm,avs,avm=fields
  assert abi==1 and group==g
  acc=(q[g*2:g*2+2].astype(np.int32)-qz)@np.clip(k[g].astype(np.int32)-kz,-128,127).T
  # The optimized template path retains raw HMX scores in the score buffer.
  # Requantized scores are consumed in registers, not stored in-place.
  raw=np.clip((acc.astype(np.int64)+128*(1<<ss)+(1<<(ss-1)))//(1<<ss),0,255).astype(np.uint8)
  scaled,_=centered_hmx_requant(acc,sm,ss,128)
  pp,ee=log2_softmax(scaled,fb,{1:'exact',2:'sole',3:'endpoint'}[dm]);exponents.extend(ee)
  vv=v[g].astype(np.int32)-vz;vv=np.clip(np.sign(vv)*((abs(vv)*vn+vd//2)//vd),-128,127)
  aa,_=centered_hmx_requant(pp.astype(np.int32)@vv,avm,avs,oz)
  for dest,values in zip(expected,[raw,pp,aa]):dest.extend(values)
 exact={n:diff(a,np.array(b)) for n,a,b in zip(['raw_QK','integer_softmax','AV'],actual,expected)}
 assert all(v['changed']==0 for v in exact.values()),exact
 mask=np.triu(np.ones((64,64),bool),1);assert np.count_nonzero(actual[1][:,mask])==0
 return exact,actual,np.array(exponents)
def main():
 tags=['chain_r3_01','chain_scalar_01','chain_refined_01','chain_refined_02']
 proofs={};arrays={};exponents={}
 for tag in tags:proofs[tag],arrays[tag],exponents[tag]=integer_core(tag)
 sign=np.array([[1-2*((i&j).bit_count()%2) for j in range(128)] for i in range(128)],float)
 assert np.array_equal(sign@sign.T,np.eye(128)*128)
 norm=float(np.float16(1/math.sqrt(128)));records=[];chain=[]
 for step in range(9):
  raw,orig,oc,os=capture('chain_r3_01',step)
  sr,scalar,sc,ss=capture('chain_scalar_01',step)
  rr,refined,rc,rs=capture('chain_refined_02',step)
  ir,identity,_,_=capture('chain_identity_02',step)
  assert np.array_equal(raw,sr) and np.array_equal(raw,rr) and np.array_equal(raw,ir)
  assert np.array_equal(identity,raw)
  expected=raw.astype(float)@(sign*norm)
  assert np.array_equal(scalar,expected.astype('<f2'))
  err=abs(refined.astype(float)-expected);assert np.all(err<=ulp(expected)+2**-14)
  assert np.array_equal(rc,sc),'refined Q/K must exactly match scalar in this replay'
  for h in range(24):
   p=qp['q_rope' if h<16 else 'k_rope'];nr=64 if step==0 else 1
   y=refined.reshape(24,nr,128)[h].astype(np.float32)
   q=np.clip(np.floor(y*np.float32(1/p['scale'])+np.float32(p['zero_point'])+np.float32(.5)),0,255)
   assert np.array_equal(q,rc[h])
  a=np.fromfile(R/'chain_refined_02'/f'step{step:02d}_output.bin',np.uint8)
  b=np.fromfile(R/'chain_scalar_01'/f'step{step:02d}_output.bin',np.uint8)
  previous=np.fromfile(R/'chain_refined_01'/f'step{step:02d}_output.bin',np.uint8)
  assert np.array_equal(a,previous),'cross-build refinement reproducibility'
  z=qp['block_output']['zero_point'];af=a.astype(float)-z;bf=b.astype(float)-z
  cosine=float(af@bf/np.linalg.norm(af)/np.linalg.norm(bf));d=diff(a,b)
  assert d['max_lsb']<=2 and cosine>=.999
  records.append(dict(step=step,whole=d,cosine=cosine,qk_exact=True,dense_max_abs=float(err.max()),dense_max_ulp=float((err/np.maximum(ulp(expected),2**-24)).max())))
  if step==0:
   chain=[dict(boundary=n,before=diff(x,y),after=diff(z,y)) for n,x,y,z in zip(names,os,ss,rs)]
  # Only live rows for native outputs in decode; padding is not a tensor value.
  for i in [2,3,5,6,7,8,9]:
   n=sizes[i]//64;nr=64 if step==0 else 1
   assert np.array_equal(feat(rs[i],n//128)[:,:nr],feat(ss[i],n//128)[:,:nr]),(step,names[i])
 _,_,codes,_=capture('chain_refined_02',0)
 expected_cache=b''.join(b''.join(pack_k_segment(codes[16+g,i:i+32],configs[g]) for i in [0,32])+bytes(4096) for g in range(8))
 assert (R/'chain_refined_02/prefill_k_cache.bin').read_bytes()==expected_cache
 assert (R/'chain_refined_02/prefill_v_cache.bin').read_bytes()==(R/'chain_scalar_01/prefill_v_cache.bin').read_bytes()
 pp=arrays['chain_r3_01'][1];sp=arrays['chain_scalar_01'][1];ix=tuple(int(v) for v in np.unravel_index(np.argmax(abs(pp.astype(int)-sp.astype(int))),pp.shape))
 h,row,col=ix;g=h//2;sm=configs[g][12]
 example=dict(head=h,row=row,key=col,hmx_raw_score=int(arrays['chain_r3_01'][0][ix]),scalar_raw_score=int(arrays['chain_scalar_01'][0][ix]),hmx_probability=int(pp[ix]),scalar_probability=int(sp[ix]),hmx_exponent=int(exponents['chain_r3_01'][ix]),scalar_exponent=int(exponents['chain_scalar_01'][ix]),score_multiplier=int(sm))
 result=dict(pass_all=True,gate_role='single_layer_repair_numerical_only',independent_integer_core=proofs,whole_layer=records,chain_prefill=chain,worst_probability_example=example,identity_exact=True,cross_build_exact=True,prefill_cache_exact=True,limits='Fixed layer0 EOS+71-token replay only; not a universal HMX error bound, model PPL, or deployment acceptance. Original failed evidence preserved.')
 with (R/'numerical_audit.json').open('x') as f:json.dump(result,f,indent=2)
 print('NUMERICAL PASS',example,flush=True)
if __name__=='__main__':main()
