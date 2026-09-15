"""A9 immutable component fixtures and independent Float64/raw-code gates."""
import os,sys,struct,json,math
import numpy as np
from pathlib import Path
import hashlib,subprocess,shutil
S=Path('/home/daniuniu/work/llama32-htp');R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0036/opt-a2');F=R.parents[1]/'l32-0033/component-fixtures'
from run_llama32_layer import adb,windows as win
from llama_reference import sha256 as sha
from export_llama32_u8 import divide
CFG=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0016/layer7-a02/layer0/attention_config_all_groups.bin')
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,indent=2)
def preflight():
 x=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'ACTIVE_EXPERIMENT=L32-0036' in x

def stage(count):
 preflight();seal=read(S/'build/llama-build-seal.json');assert seal['paper_trace'] is False and seal['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 assert 'QBH_LLAMA_LAYER_COUNT:STRING=1\n' in (S/'hexagon_ReleaseG_toolv19_v79/CMakeCache.txt').read_text()
 attempt=1
 while (R/f'binaries-l1-a{attempt}').exists():attempt+=1
 archive=R/f'binaries-l1-a{attempt}';archive.mkdir();remote=f'/data/local/tmp/llama32-htp/l32-0036-opt-a2-l1-a{attempt}';assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for f,h in seal['files'].items():
  p=Path(f);assert sha(p)==h;shutil.copy2(p,archive/p.name);adb('push',win(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/llama_sp2_cli '+remote+'/qwen3_block_cli');write(archive/'seal.json',seal);(R/'runtime-l1.json').write_text(json.dumps(dict(remote=remote,archive=str(archive),seal=seal),indent=2));print('STAGED',flush=True)
def native(x):
 h,rows,k=x.shape;out=np.zeros((h,k//32,64,32),'u1');out[:,:,:rows]=x.reshape(h,rows,k//32,32).transpose(0,2,1,3);return out.tobytes()
def unpack(x,h,k):return np.frombuffer(x,'u1').reshape(h,k//32,64,32).transpose(0,2,1,3).reshape(h,64,k)
def prepare():
 preflight();F.mkdir(exist_ok=False);rng=np.random.default_rng(30033);cfg=np.fromfile(CFG,'<i4').reshape(8,15);cases=[]
 def add(name,raw,rows,past,c):
  h,_,k=raw.shape;raw=raw.copy();off=256;sz=h*k*64;prob=off+sz;fp=prob+sz;size=fp+4*sz
  z=bytearray(size);struct.pack_into('<16I',z,0,0x3250534c,1,size,5,rows,k,h,off,128,prob,fp,0,0,0,0,0)
  z[128:188]=np.asarray(c,dtype='<i4').tobytes();struct.pack_into('<2I',z,188,past,100);z[off:off+sz]=native(raw)
  p=F/(name+'.bin');p.write_bytes(z);cases.append(dict(name=name,rows=rows,past=past,padded=k,heads=h,config=[int(v) for v in c],sha256=sha(p)))
 for rows,past,k in [(1,0,32),(1,31,32),(1,63,64),(1,64,96),(1,95,96),(1,127,128),(64,0,64),(64,64,128)]:
  for kind,mult in [('equal',1),('spike',18),('random',3),('ramp',7)]:
   x=rng.integers(0,256,(4,64,k),dtype='u1')
   if kind=='equal':x[:]=128
   elif kind=='spike':x[:]=0;x[:,:,0]=255
   elif kind=='ramp':x[:]=np.arange(k,dtype='u1')
   c=cfg[0].copy();c[12]=mult;add(f'm{rows}-p{past}-{kind}',x,rows,past,c)
 write(F/'manifest.json',dict(seed=30033,cases=cases,config_source=str(CFG),config_sha256=sha(CFG),timing_cases=['m64-p0-random','m1-p64-random'],probability_gate=dict(max_abs=2e-6,row_sum=2e-6),rounding='nonnegative round-half-up',reference_scope='synthetic raw HMX carrier values with frozen actual Llama layer7 config; actual layer inputs checked in integration'))
 print('FIXTURES',len(cases),flush=True)
def integer(raw,c,past,rows):
 out=np.zeros_like(raw);score=np.clip((raw.astype('i8')-128)*int(c[12])+128,0,255)
 for h in range(raw.shape[0]):
  for row in range(rows):
   n=past+row+1;v=score[h,row,:n];es=np.minimum(15,(int(v.max())-v+(1<<(int(c[2])-1)))>>int(c[2]));total=sum(1<<(15-int(e)) for e in es)
   out[h,row,:n]=[divide(int(e),total,{1:'exact',2:'sole',3:'endpoint'}[int(c[3])],n) for e in es]
 return out
def one(case,arm,tag,audit=True):
 preflight();cfg=read(R/'runtime-l1.json');root=cfg['remote'];d=R/tag;d.mkdir(parents=True,exist_ok=False)
 p=F/(case['name']+'.bin');assert sha(p)==case['sha256'];data=bytearray(p.read_bytes());struct.pack_into('<I',data,12,5 if arm=='FP' else 10 if arm=='FAST' else 6);inp=d/'input.bin';inp.write_bytes(data)
 remote=root+'/'+tag.replace('/','_');adb('shell','mkdir '+remote);adb('push',win(inp),remote+'/input.bin')
 cmd='cd '+root+' && LD_LIBRARY_PATH='+root+' DSP_LIBRARY_PATH='+root+' ADSP_LIBRARY_PATH='+root+' ./llama_sp2_cli '+remote+'/input.bin '+remote+'/output.bin'
 write(d/'protocol.json',dict(case=case,arm=arm,command=cmd,runtime=cfg));z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);write(d/'exit.json',dict(returncode=z.returncode));assert z.returncode==0,(tag,z.stdout,z.stderr)
 adb('pull',remote+'/output.bin',win(d/'output.bin'));out=(d/'output.bin').read_bytes();h=np.frombuffer(out[:64],'<u4');assert h[11]==0 and h[12]==8388608 and h[13]<=8388608
 raw=unpack(data[int(h[7]):int(h[7])+case['heads']*case['padded']*64],case['heads'],case['padded']);a=unpack(out[int(h[9]):int(h[9])+case['heads']*case['padded']*64],case['heads'],case['padded'])
 checks={}
 if arm=='FP':
  f=np.frombuffer(out[int(h[10]):],'<f4').reshape(case['heads'],64,case['padded']);expected=np.zeros(f.shape,'f8');mask=np.arange(case['padded'])[None,:]<=case['past']+np.arange(64)[:,None];mask[case['rows']:]=False
  for head in range(case['heads']):
   for row in range(case['rows']):
    v=raw[head,row,mask[row]].astype('f8');v=(v-v.max())*case['config'][12]/2**case['config'][2];e=np.exp2(v);expected[head,row,mask[row]]=e/e.sum()
  error=float(np.max(np.abs(f-expected)));mass=float(np.max(np.abs(f[:,:case['rows']].sum(2,dtype='f8')-1)));assert np.isfinite(f).all() and error<=2e-6 and mass<=2e-6,(tag,error,mass)
  assert np.count_nonzero(f[:,~mask])==0 and np.count_nonzero(a[:,~mask])==0
  own=np.clip(np.floor(f*np.float32(255)+np.float32(.5)),0,255).astype('u1');assert np.array_equal(a,own),(tag,'own FP32 U8 rounding',int(np.count_nonzero(a!=own)))
  ideal=np.clip(np.floor(expected*255+.5),0,255).astype('u1');checks=dict(max_abs=error,row_sum_error=mass,own_u8_exact=True,float64_u8_threshold_differences=int(np.count_nonzero(a!=ideal)),float64_u8_max_lsb=int(np.max(np.abs(a.astype('i4')-ideal.astype('i4')))))
 else:
  ideal=integer(raw,case['config'],case['past'],case['rows']);assert np.array_equal(a,ideal),(tag,'integer',int(np.count_nonzero(a!=ideal)))
  checks=dict(integer_exact=True)
 ticks=int(np.frombuffer(out[64:72],'<u8')[0]);write(d/'validated.json',dict(pass_all=True,case=case['name'],arm=arm,ticks=ticks,repeat=100,**checks));print('COMPONENT_PASS',tag,ticks/100/19.2,checks,flush=True);return read(d/'validated.json')
def gates():
 manifest=read(F/'manifest.json');z=[]
 for c in manifest['cases']:
  for arm in ['FP','LOG2','FAST']:z.append(one(c,arm,'component-gate/'+c['name']+'-'+arm))
 write(R/'component_gate.json',dict(pass_all=True,runs=z))
def timing():
 assert read(R/'component_gate.json')['pass_all'];m=read(F/'manifest.json')
 for case in [c for c in m['cases'] if c['name'] in m['timing_cases']]:
  for phase,n in [('short',5),('formal',10)]:
   z=[]
   for c in range(n):
    for a in (['FP','LOG2','FAST'] if c%2==0 else ['FAST','LOG2','FP']):z.append(dict(cycle=c,**one(case,a,f'component-{phase}/{case["name"]}/{c:02d}-{a}')))
   write(R/f'component-{case["name"]}-{phase}.json',dict(pass_all=True,runs=z))
if __name__=='__main__':
 a=sys.argv[1]
 if a=='stage':stage(1)
 else:{'prepare':prepare,'gate':gates,'timing':timing}[a]()
