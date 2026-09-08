import sys,json,math
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
import audit_exp0248 as a
import integer_attention_exp0250 as ia
from prepare_exp0042_attention import CONFIG
from prepare_exp0161_segmented_cache import pack_k_segment
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0251');a.R=R;P=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0247')
def read(p):return json.loads(p.read_text())
def output(tag,step):return np.fromfile(R/tag/f'step{step:02d}_output.bin',np.uint8)
def exact(x,y):return a.diff(x,y)
proof={};integer_pass=True
for tag,package,mode in [('audit_control','control','sole'),('audit_wide','control','wide_sole'),('audit_wide_scalar','control','wide_sole'),('audit_r3_wide','r3','wide_sole'),('audit_scalar_wide','r3','wide_sole')]:
 cfg=list(CONFIG.iter_unpack((P/package/'attention_config_all_groups.bin').read_bytes()));ks=[];vs=[];steps=[]
 for step in range(9):
  _,_,codes,slots=a.capture(tag,step);nr=64 if step==0 else 1
  q=codes[:16];ks.append(codes[16:]);vs.append(a.feat(slots[10],8)[:,:nr]);k=np.concatenate(ks,axis=1);v=np.concatenate(vs,axis=1)
  ref={'raw':[],'probability':[],'av':[]}
  valid=np.tril(np.ones((64,64),bool)) if step==0 else np.ones((1,64+step),bool)
  for g,c in enumerate(cfg):
   z=ia.numpy_oracle(q[2*g:2*g+2],np.repeat(k[g:g+1],2,0),np.repeat(v[g:g+1],2,0),valid,ia.config(c),mode)
   for n in ref:ref[n].extend(z[n])
  got=dict(raw=a.scores(slots[0]) if step==0 else slots[0][:2048].reshape(16,128)[:,:64+step,None].transpose(0,2,1),probability=a.scores(slots[1]) if step==0 else slots[1][:2048].reshape(16,128)[:,:64+step,None].transpose(0,2,1),av=a.feat(slots[2],16)[:,:nr])
  checks={n:exact(got[n],np.array(ref[n])) for n in ref};ok=all(x['changed']==0 for x in checks.values());integer_pass &=ok;steps.append(dict(step=step,checks=checks,exact=ok))
  print('BOUNDARY',tag,step,{n:x['max_lsb'] for n,x in checks.items()},flush=True)
 proof[tag]=steps
limits={};sign=np.array([[1-2*((i&j).bit_count()%2) for j in range(128)] for i in range(128)],float);norm=float(np.float16(1/math.sqrt(128)));qp=read(P/'r3/manifest.json')['qparams'];assert np.array_equal(sign@sign.T,128*np.eye(128))
wr_pass=True;wr=[]
for step in range(9):
 raw,out,codes,_=a.capture('audit_r3_wide',step);sr,so,sc,_=a.capture('audit_scalar_wide',step);assert np.array_equal(raw,sr)
 expected=raw.astype(float)@(sign*norm);assert np.array_equal(so,expected.astype('<f2'))
 err=abs(out.astype(float)-expected);dense=bool(np.all(err<=a.ulp(expected)+2**-14));qk=exact(codes,sc);actual=output('audit_r3_wide',step);ref=output('audit_scalar_wide',step);d=exact(actual,ref);zero=qp['block_output']['zero_point'];x=actual.astype(float)-zero;y=ref.astype(float)-zero;cos=float(x@y/np.linalg.norm(x)/np.linalg.norm(y));ok=dense and qk['max_lsb']<=1 and d['max_lsb']<=2 and cos>=.999;wr_pass &=ok
 wr.append(dict(step=step,dense_pass=dense,dense_max_abs=float(err.max()),qk=qk,whole=d,cosine=cos,gate_pass=ok))
 print('WR',step,d['max_lsb'],cos,ok,flush=True)
w0=[];control=[]
for step in range(9):
 x=output('audit_wide',step);d=exact(x,output('audit_wide_scalar',step));repeat=np.array_equal(x,output('audit_wide_repeat',step));w0.append(dict(step=step,whole=d,repeat_exact=repeat));assert d['changed']==0 and repeat
 old=np.fromfile(R.parent/'exp0248/chain_control_02'/f'step{step:02d}_output.bin',np.uint8);d=exact(output('audit_control',step),old);control.append(dict(step=step,whole=d));assert d['changed']==0
cache={}
for tag,pkg in [('audit_wide','control'),('audit_r3_wide','r3')]:
 cfg=list(CONFIG.iter_unpack((P/pkg/'attention_config_all_groups.bin').read_bytes()));_,_,c,_=a.capture(tag,0)
 expected=b''.join(b''.join(pack_k_segment(c[16+g,i:i+32],cfg[g]) for i in [0,32])+bytes(4096) for g in range(8));assert (R/tag/'prefill_k_cache.bin').read_bytes()==expected
 oracle='audit_wide_scalar' if pkg=='control' else 'audit_scalar_wide';assert (R/tag/'prefill_v_cache.bin').read_bytes()==(R/oracle/'prefill_v_cache.bin').read_bytes();cache[tag]='independent_K_exact_V_oracle_exact'
out=dict(pass_all=integer_pass,integer_boundaries=proof,control_reproduction=control,wide_whole=w0,r3_wide_whole=wr,eligible=dict(control=True,wide=integer_pass,r3_wide=integer_pass and wr_pass),prefill_cache=cache,limits='Original2LSB/cosine0.999 dense oracle; fixed one-layer replay, not full DSP PPL or model quality acceptance.')
with (R/'numerical_audit.json').open('x') as f:json.dump(out,f,indent=2)
assert integer_pass,'integer boundaries fail'
print('NUMERICAL_DONE',out['eligible'],flush=True)
