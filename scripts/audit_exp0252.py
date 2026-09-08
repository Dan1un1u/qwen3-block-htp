import sys,json,math
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
import audit_exp0248 as a
import integer_attention_exp0252 as ia
from prepare_exp0042_attention import CONFIG
from prepare_exp0161_segmented_cache import pack_k_segment
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0252');a.R=R;P=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0247')
def output(tag,step):return np.fromfile(R/tag/f'step{step:02d}_output.bin',np.uint8)
def diff(x,y,zero=0):
 d=a.diff(x,y);xx=x.astype(float).reshape(-1)-zero;yy=y.astype(float).reshape(-1)-zero
 d['cosine']=float(xx@yy/np.linalg.norm(xx)/np.linalg.norm(yy)) if np.linalg.norm(xx)*np.linalg.norm(yy) else float(np.array_equal(xx,yy));return d
modes=['control','exact','nr64'];cells=[p+m for p in ['', 'r3_', 'scalar_'] for m in modes]+[m+'_scalar' for m in modes]
proof={};captured={};outputs={}
for cell in cells:
 tag='audit_'+cell;pkg='r3' if cell.startswith(('r3_','scalar_')) else 'control';mode=next(m for m in modes if m in cell);mode={'control':'wide_sole','exact':'wide_exact','nr64':'wide_nr64'}[mode]
 cfg=list(CONFIG.iter_unpack((P/pkg/'attention_config_all_groups.bin').read_bytes()));ks=[];vs=[];steps=[]
 captured[cell]=[];outputs[cell]=[]
 for step in range(9):
  cap=a.capture(tag,step);captured[cell].append(cap);_,_,codes,slots=cap;nr=64 if step==0 else 1
  q=codes[:16];ks.append(codes[16:]);vs.append(a.feat(slots[10],8)[:,:nr]);k=np.concatenate(ks,axis=1);v=np.concatenate(vs,axis=1)
  ref={'raw':[],'probability':[],'av':[]};valid=np.tril(np.ones((64,64),bool)) if step==0 else np.ones((1,64+step),bool)
  for g,c in enumerate(cfg):
   z=ia.numpy_oracle(q[2*g:2*g+2],np.repeat(k[g:g+1],2,0),np.repeat(v[g:g+1],2,0),valid,ia.config(c),mode)
   for n in ref:ref[n].extend(z[n])
  got=dict(raw=a.scores(slots[0]) if step==0 else slots[0][:2048].reshape(16,128)[:,:64+step,None].transpose(0,2,1),probability=a.scores(slots[1]) if step==0 else slots[1][:2048].reshape(16,128)[:,:64+step,None].transpose(0,2,1),av=a.feat(slots[2],16)[:,:nr])
  checks={n:a.diff(got[n],np.array(ref[n])) for n in ref};assert all(x['changed']==0 for x in checks.values()),(cell,step,checks);steps.append(dict(step=step,checks=checks,exact=True));outputs[cell].append(output(tag,step))
 proof[cell]=steps;print('INTEGER_EXACT',cell,flush=True)
# Frozen prior control replay is evidence of unchanged arithmetic and inputs.
for cell,parent in [('control','audit_wide'),('r3_control','audit_r3_wide'),('scalar_control','audit_scalar_wide')]:
 for step in range(9):assert np.array_equal(outputs[cell][step],np.fromfile(R.parent/'exp0251'/parent/f'step{step:02d}_output.bin',np.uint8))
# Scalar and HVX normalization operate on identical codes, independent downstream checks above.
for mode in modes:
 for step in range(9):assert np.array_equal(outputs[mode][step],outputs[mode+'_scalar'][step]),(mode,step)
for step in range(9):assert np.array_equal(outputs['nr64'][step],output('audit_nr64_repeat',step))
qp={pkg:json.loads((P/pkg/'manifest.json').read_text())['qparams'] for pkg in ['control','r3']}
sign=np.array([[1-2*((i&j).bit_count()%2) for j in range(128)] for i in range(128)],float);mat=sign*float(np.float16(1/math.sqrt(128)));assert np.array_equal(sign@sign.T,128*np.eye(128))
def boundaries(cell,step):
 slots=captured[cell][step][3];nr=64 if step==0 else 1
 return dict(probability=a.scores(slots[1]) if step==0 else slots[1][:2048].reshape(16,128)[:,:64+step],AV=a.feat(slots[2],16)[:,:nr],O=a.feat(slots[3],16)[:,:nr],residual=a.feat(slots[4],16)[:,:nr],final=outputs[cell][step])
zeros=dict(probability='attention_probability',AV='attention_concat',O='attention_projection',residual='post_attention_residual',final='block_output')
amplification={};eligible={'control':True,'nr64':True}
for mode in modes:
 arm='r3_'+mode;ref='scalar_'+mode;rows=[]
 for step in range(9):
  raw,out,codes,_=captured[arm][step];sr,so,sc,_=captured[ref][step];assert np.array_equal(raw,sr)
  expected=raw.astype(float)@mat;assert np.array_equal(so,expected.astype('<f2'))
  dense=bool(np.all(abs(out.astype(float)-expected)<=a.ulp(expected)+2**-14));qk=a.diff(codes,sc)
  aa=boundaries(arm,step);bb=boundaries(ref,step);checks={k:diff(aa[k],bb[k],qp['r3'][zeros[k]]['zero_point']) for k in aa}
  ok=dense and qk['max_lsb']<=1 and checks['final']['max_lsb']<=2 and checks['final']['cosine']>=.999
  rows.append(dict(step=step,dense_pass=dense,qk=qk,boundaries=checks,gate_pass=ok))
 amplification[mode]=rows
 if mode in ['control','nr64']:eligible[arm]=all(x['gate_pass'] for x in rows)
 print('AMPLIFICATION',mode,[(x['step'],x['boundaries']['final']['max_lsb'],x['boundaries']['final']['cosine']) for x in rows],flush=True)
# Same-input accuracy against exact division; includes actual downstream O/residual/final.
normalization={}
for prefix in ['', 'r3_', 'scalar_']:
 for mode in ['control','nr64']:
  rows=[]
  for step in range(9):
   cell=prefix+mode;ref=prefix+'exact';assert np.array_equal(captured[cell][step][2],captured[ref][step][2])
   aa=boundaries(cell,step);bb=boundaries(ref,step);pkg='r3' if prefix else 'control';rows.append(dict(step=step,boundaries={k:diff(aa[k],bb[k],qp[pkg][zeros[k]]['zero_point']) for k in aa}))
  normalization[prefix+mode]=rows
cache={}
for cell in ['control','nr64','r3_control','r3_nr64']:
 pkg='r3' if cell.startswith('r3_') else 'control';cfg=list(CONFIG.iter_unpack((P/pkg/'attention_config_all_groups.bin').read_bytes()));c=captured[cell][0][2]
 expected=b''.join(b''.join(pack_k_segment(c[16+g,i:i+32],cfg[g]) for i in [0,32])+bytes(4096) for g in range(8));assert (R/('audit_'+cell)/'prefill_k_cache.bin').read_bytes()==expected
 oracle='scalar_'+cell[3:] if pkg=='r3' else cell+'_scalar';assert (R/('audit_'+cell)/'prefill_v_cache.bin').read_bytes()==(R/('audit_'+oracle)/'prefill_v_cache.bin').read_bytes();cache[cell]=True
z=dict(pass_all=True,integer_boundaries=proof,eligible=eligible,r3_amplification=amplification,same_QK_normalization=normalization,prefill_cache=cache,control_reproduction=True,scalar_hvx_exact=True,repeated_nr64_exact=True,scope='real layer0 M64+eight cached M1, no full-model PPL')
with (R/'numerical_audit.json').open('x') as f:json.dump(z,f,indent=2)
print('AUDIT_COMPLETE',eligible)
