#!/usr/bin/env python3
import sys,math,json
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from device_exp0258 import *
import audit_exp0248 as au
from audit_exp0247 import ulp
import integer_attention_exp0252 as ia
from prepare_exp0042_attention import CONFIG
from prepare_exp0161_segmented_cache import pack_k_segment
preflight();au.R=R
for a,tag in [(0,'slice_a0'),(1,'slice_a1'),(1,'slice_a1_repeat')]:
 if not (R/tag/'validated.json').exists():run(a,1,tag,3,True)
seed=np.fromfile(O/'prefix/prefix_kv_u8.bin','u1').reshape(28,2,8,128);cfg=list(CONFIG.iter_unpack((O/'r3/layer2/attention_config_all_groups.bin').read_bytes()));qs=read(O/'r3/manifest.json')['layer_qparams'][2];sgn=np.array([[1-2*((i&j).bit_count()%2) for j in range(128)] for i in range(128)],float);scale=float(np.float16(1/math.sqrt(128)));ks=[];vs=[];rows=[]
for step in range(9):
 n=f'step{step:02d}_output.bin';assert (R/'slice_a1'/n).read_bytes()==(R/'slice_a1_repeat'/n).read_bytes();assert (R/'slice_a0'/n).read_bytes()==(R.parent/'exp0257/seed_slice_fast'/n).read_bytes(),'parent noR3 exact'
 raw,out,codes,slots=au.capture('slice_a1',step);nr=64 if step==0 else 1;expected=raw.astype(float)@sgn*scale;err=np.abs(out.astype(float)-expected);assert np.isfinite(out).all() and np.all(err<=ulp(expected)+2**-14)
 qerr=0
 for head in range(24):
  qp_=qs['q_rope' if head<16 else 'k_rope'];oracle=qdqcode(out.reshape(24,nr,128)[head],qp_);qerr=max(qerr,int(np.max(np.abs(oracle.astype(int)-codes[head].astype(int)))))
 assert qerr<=1
 q=codes[:16];k=codes[16:].copy();v=au.feat(slots[10],8)[:,:nr].copy()
 if step==0:k[:,0]=seed[2,0];v[:,0]=seed[2,1]
 ks.append(k);vs.append(v);kk=np.concatenate(ks,axis=1);vv=np.concatenate(vs,axis=1);valid=np.tril(np.ones((64,64),bool)) if step==0 else np.ones((1,64+step),bool);refs={n:[] for n in ['raw','probability','av']}
 for g,c in enumerate(cfg):
  z=ia.numpy_oracle(q[2*g:2*g+2],np.repeat(kk[g:g+1],2,0),np.repeat(vv[g:g+1],2,0),valid,ia.config(c),'wide_nr64')
  for n in refs:refs[n].extend(z[n])
 got=dict(raw=au.scores(slots[0]) if step==0 else slots[0][:2048].reshape(16,128)[:,:64+step,None].transpose(0,2,1),probability=au.scores(slots[1]) if step==0 else slots[1][:2048].reshape(16,128)[:,:64+step,None].transpose(0,2,1),av=au.feat(slots[2],16)[:,:nr]);checks={n:au.diff(got[n],np.array(refs[n])) for n in refs};assert all(v['changed']==0 for v in checks.values()),(step,checks)
 rows.append(dict(step=step,dense_max_abs=float(err.max()),dense_original_component_gate=True,U8_max_LSB=qerr,integer=checks,repeat_exact=True,parent_noR3_exact=True))
# Independently inspect stored first-layer K seed column and affine bias.
cfg0=list(CONFIG.iter_unpack((O/'r3/layer0/attention_config_all_groups.bin').read_bytes()));cache=(R/'slice_a1/prefill_k_cache.bin').read_bytes();per=len(cache)//8
for g,c in enumerate(cfg0):
 matrix=np.full((32,128),c[5],np.uint8);matrix[0]=seed[0,0,g];p=pack_k_segment(matrix,c);actual=cache[g*per:g*per+len(p)]
 for ch in range(128):
  off=ch//32*1024+(ch%32//4)*128+ch%4;assert actual[off]==p[off]
 assert actual[4096:4100]==p[4096:4100] and actual[4224:4228]==p[4224:4228]
write(R/'slice_gate.json',dict(pass_all=True,scope='3layers M64+8decode component/determinism/physical gate under explicit ideal-R3 exception',rows=rows,rotated_prefix_cache_verified=True,ideal_whole_layer_gate='known_failed_not_reclassified',R3_calls_per_step=3))
print('SLICE_GATE_PASS',flush=True)
