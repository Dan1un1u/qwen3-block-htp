"""Exact host-reference acceleration; compare all completed original CPU checkpoints."""
import sys,json,functools,subprocess,os,signal,time
from pathlib import Path
import numpy as np,torch
import llama32_fp32_residual as ref
import llama_u8_reference as u8
from llama_reference import sha256 as sha
S=Path('/home/daniuniu/work/llama32-htp');R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0037');P=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0037/full-a01');O=P.parent.parent/'l32-0034/full-a01';D=R/'accelerated-oracle-a2'
def read(p):return json.loads(p.read_text())
def write(p,z):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,indent=2);f.write('\n')
def preflight():
 z=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'ACTIVE_EXPERIMENT=L32-0037' in z
preflight();D.mkdir(exist_ok=False);torch.set_num_threads(4)
cache={};rawcache={};ref.unpack_w4_codes=functools.lru_cache(None)(ref.unpack_w4_codes)
def projection(a,p,name,n,k,iq,oq,cv):
 key=(str(p),name,n,k,tuple(sorted(iq.items())),tuple(sorted(oq.items())))
 if key not in cache:
  w,ws=u8._cached_w4_projection(str(p.resolve()),name,n,k);lo,hi=u8.projection_bias_words(w,ws,iq,oq);sums=w.astype('i4').sum(1,dtype='i8');cache[key]=(torch.from_numpy(w.T.copy()).cuda(),sums,lo,hi)
 w,sums,lo,hi=cache[key];a=np.ascontiguousarray(a,dtype='u1');rows=len(a);m=max(32,(rows+15)//16*16);x=np.zeros((m,k),'i1');x[:rows]=(a.astype('i2')-128).astype('i1');v=torch._int_mm(torch.from_numpy(x).cuda(),w).cpu().numpy()[:rows].astype('i8')+128*sums[None];assert np.max(np.abs(v))<2**31
 return cv.convert(v,lo,hi)
def rawdot(a,w):
 key=id(w)
 if key not in rawcache:rawcache[key]=(w,torch.from_numpy(w.astype('f8')).cuda().T.contiguous())
 # Integer products and partial sums bounded by2^31 for this recipe; Float64 exact.
 assert float(np.max(np.abs(a.astype('f8'))))*8*a.shape[1]<2**53
 y=(torch.from_numpy(a.astype('f8')).cuda()@rawcache[key][1]).cpu().numpy().astype('i8')
 assert np.array_equal(y[:min(3,len(a)),::max(1,len(w)//16)],a[:3].astype('i8')@w[::max(1,len(w)//16)].astype('i8').T)
 return y
ref.project_w4u8=projection;ref.oracle=rawdot
embed=np.memmap(P/'generation_embedding_weight_f16.bin','<f2','r',shape=(128256,2048));qs=[u8.load_qparams_bin(P/f'layer{i}/qparams_u8.bin') for i in range(16)];gq=u8.load_qparams_bin(P/'generation_qparams_u8.bin');gamma=np.fromfile(P/'generation_final_norm_weight_f16.bin','<f2');cv=u8.HmxU8Converter(S/'build/l32-0003/qbh_hmx_u8_reference.so');matches=[]
for label,prompt in read(R/'prompts.json').items():
 caches=[None]*16;ids=[];codes=[]
 for step in range(64):
  x=np.array(embed[prompt['ids'] if not step else [ids[-1]]],dtype='f4');n='rope_cos_f16.bin' if not step else f'generation_decode_rope_cos_{step-1:02d}_f16.bin';c=np.fromfile(P/n,'<f2').reshape(64,64);sn=np.fromfile(P/n.replace('cos','sin'),'<f2').reshape(64,64)
  for i in range(16):x,caches[i],_=ref.layer(x,P/f'layer{i}',qs[i],c,sn,caches[i])
  d=D/label;d.mkdir(exist_ok=True);np.save(d/f'hidden-{step:02d}.npy',x)
  a=ref.norm(x[-1:],gamma,gq['generation_final_norm_output']);logits=projection(a,P,'generation_lm_head',128256,2048,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)[0];tok=int(logits.argmax());ids.append(tok);codes.append(int(logits[tok]));old=R/'oracle'/label/f'hidden-{step:02d}.npy'
  if old.exists():assert np.array_equal(np.load(old),x),(label,step,'original CPU oracle');matches.append([label,step])
  print('EXACT_ACCELERATED_ORACLE',label,step,tok,codes[-1],flush=True)
 write(D/f'teacher-{label}.json',dict(ids=ids,codes=codes,steps=64,contract='independent exact GPU int8 dot/Float64 integer dot plus original SDK CPU converter and unchanged host nonlinear reference; original CPU checkpoints exact; no PPL'))
# Stop only the known slow host oracle, after fast results and all completed outputs agree.
cmds=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();pids=[]
for line in cmds:
 fields=line.strip().split(None,1)
 if len(fields)==2 and fields[1]=='/home/daniuniu/work/rotation-quant/.venv/bin/python tools/prepare_l32_0037_generalization.py':pids.append(int(fields[0]))
assert len(pids)<=1
for pid in pids:os.kill(pid,signal.SIGINT)
for _ in range(100):
 if not any(Path(f'/proc/{pid}').exists() for pid in pids):break
 time.sleep(.1)
assert not any(Path(f'/proc/{pid}').exists() for pid in pids)
all_old=[]
for label in ['A','B']:
 for p in sorted((R/'oracle'/label).glob('hidden-*.npy')):
  assert np.array_equal(np.load(p),np.load(D/label/p.name)),str(p);all_old.append(str(p.relative_to(R)))
 assert len(read(D/f'teacher-{label}.json')['ids'])==64
 dst=R/f'teacher-{label}.json'
 if dst.exists():
  old=read(dst);new=read(D/f'teacher-{label}.json');assert old['ids']==new['ids'] and old['codes']==new['codes']
 else:write(dst,read(D/f'teacher-{label}.json'))
write(R/'oracle_acceleration_gate.json',dict(pass_all=True,completed_original_cpu_checkpoints_exact=all_old,accelerated_host_reference=str(D),stopped_host_oracle_pids=pids,integer_matmul_contract='GPU int8 int32 accumulator, exact64-bit correction; FP64 SP2 dot <2^53 and sampled int64 equality',device_arithmetic_changed=False))
prior=read(R.parent/'l32-0034/deployment.json');pm=read(O/'manifest.json');files={str(p.relative_to(P)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='manifest.json'}
for n,v in files.items():
 if 'weight' in Path(n).name:assert v['sha256']==pm['files'][n]['sha256']
write(P/'manifest.json',dict(experiment='L32-0037',layers=16,cache_capacity=128,generation_steps=64,parent=str(O),parent_manifest_sha256=sha(O/'manifest.json'),teacher_hashes={k:sha(R/f'teacher-{k}.json') for k in ['A','B']},files=files))
write(R/'package.json',dict(package=str(P),manifest_sha256=sha(P/'manifest.json'),original_command=prior['original_command'],parent_remote=prior['remote'],parent_package=str(O),teacher_hashes={k:sha(R/f'teacher-{k}.json') for k in ['A','B']},oracle_directory=str(D)))
print('PREPARED_ACCELERATED_REFERENCE',len(all_old),flush=True)
