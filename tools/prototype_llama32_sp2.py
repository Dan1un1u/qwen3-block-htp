#!/usr/bin/env python3
"""L32-0008 immutable native W4 SP2 component fixtures and device probes."""
import argparse,ctypes,json,os,struct,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
from llama_u8_reference import unpack_w4_codes,load_qparams_bin
from run_llama32_layer import adb,windows,ROOT
OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0008')
H=struct.Struct('<16I8Q')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def preflight():subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
def pack_w(w):
 n,k=w.shape
 a=(w.reshape(n//32,32,k//32,4,8).transpose(0,2,3,1,4).astype('u1')&15)
 return np.ascontiguousarray(a[...,:4]|(a[...,4:]<<4)).tobytes()
def oracle(v,w):
 # Every product and intermediate integer sum is <2**53, so FP64 BLAS is an
 # exact integer dot-product accelerator. Audit sampled entries using int64.
 y=np.rint(v.astype('f8')@w.astype('f8').T).astype('i8')
 assert np.array_equal(y[:min(3,len(v)),::max(1,len(w)//16)],v[:3].astype('i8')@w[::max(1,len(w)//16)].astype('i8').T)
 return y

def fixture(name,v,w,mode,metadata=None):
 d=OUT/'fixtures'/name;d.mkdir(parents=True,exist_ok=False)
 v=np.asarray(v,dtype='<i2');rows,k=v.shape;n=len(w)
 x=v.tobytes();weights=pack_w(w);sums=w.astype('i4').sum(1).astype('<i4').tobytes()
 xo=2048;wo=(xo+len(x)+2047)&~2047;so=wo+len(weights);oo=(so+len(sums)+2047)&~2047;size=oo+rows*n*4
 blob=bytearray(size);blob[:128]=H.pack(0x3250534c,1,size,mode,rows,k,n,xo,wo,so,oo,0xffffffff,0,0,0,0,*([0]*8));blob[xo:xo+len(x)]=x;blob[wo:wo+len(weights)]=weights;blob[so:so+len(sums)]=sums
 (d/'input.bin').write_bytes(blob)
 y=oracle(v,w)
 if mode==3:y=np.clip(y,0,255)
 assert np.max(np.abs(y))<2**31
 y.astype('<i4').tofile(d/'reference.bin')
 put(d/'manifest.json',dict(name=name,mode=mode,rows=rows,k=k,n=n,input_sha256=sha(d/'input.bin'),reference_sha256=sha(d/'reference.bin'),metadata=metadata))
 print('fixture',name,flush=True)

def emulator():
 lib=ROOT/'build/l32-0003/qbh_hmx_u8_reference.so'
 f=ctypes.CDLL(str(lib)).hmx_u8_cvt
 f.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_int32,ctypes.c_int16,ctypes.c_int16,ctypes.c_int16,ctypes.c_uint16,ctypes.c_int32,ctypes.c_int16];f.restype=ctypes.c_uint32
 rng=np.random.default_rng(8008);xs=list(range(-1024,1025))+[int(x) for x in rng.integers(-14622720,14622721,20000)]
 bad=[]
 for x in xs:
  b=[(f(None,x,0,24-8*j,0,2048,0,0,1)>>4)&255 for j in range(4)]
  if sum(t<<(8*j) for j,t in enumerate(b))!=(x&0xffffffff):bad.append(x)
 put(OUT/'emulator.json',dict(library=str(lib),sha256=sha(lib),samples=len(xs),mismatches=len(bad),first=bad[:16],rounding_bias=0,saturate=False,retain=True))
 assert not bad

def prepare():
 preflight();OUT.mkdir(exist_ok=False,parents=True);emulator()
 rng=np.random.default_rng(8008)
 w=rng.integers(-7,8,(32,8192),dtype='i1');w[0]=7;w[1]=-7;w[2]=0;w[3]=np.tile([7,-7],4096)
 for label,rows,k in [('small',16,32),('long',16,8192)]:
  v=rng.integers(0,256,(rows,k),dtype='i2');v[0]=255;v[1]=0
  fixture('raw-'+label,v,w[:,:k],0)
  v=rng.integers(-32768,32768,(rows,k),dtype='i2');v[0]=-32768;v[1]=32767;v[2]=np.resize(np.array([-257,-256,-255,-1,0,1,255,256,257],dtype='i2'),k)
  fixture('signed-'+label+'-rows',v,w[:,:k],1)
  fixture('signed-'+label+'-passes',v,w[:,:k],2)
 levels=np.array(sorted({0}|{s*(2**p) for s in [-1,1] for p in range(15)}|{s*(2**p+2**q) for s in [-1,1] for p in range(15) for q in range(p)}),dtype='i2')
 assert len(levels)==241
 put(OUT/'codebook.json',dict(levels=levels.tolist(),code_bits=8,scale='max(abs(frozen calibration middle minimum/maximum))/24576',selection='nearest, ties choose lower numeric level',source='pre-middle-rounding SwiGLU of reconstructed frozen Gate/Up codes',frozen_before_scoring=True))

def llama():
 preflight();levels=np.array(json.loads((OUT/'codebook.json').read_text())['levels'],dtype='i2');metrics=[]
 for layer in [0,7,15]:
  for phase in ['prefill','decode']:
   old=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0003/layers-a01')/f'layer{layer}-{phase}'
   m=json.loads((old/'manifest.json').read_text())
   inputs=['down_weight_w4_hmx.bin','down_weight_w4_scale_f32.bin','qparams_u8.bin','reference_gate.npy','reference_up.npy','reference_middle.npy']
   for f in inputs:assert sha(old/f)==m['files'][f]['sha256'],f
   q=load_qparams_bin(old/'qparams_u8.bin');w=unpack_w4_codes(old,'down',2048,8192);ws=np.fromfile(old/'down_weight_w4_scale_f32.bin',dtype='<f4')
   g=(np.load(old/'reference_gate.npy').astype('f8')-q['gate']['zero_point'])*q['gate']['scale'];u=(np.load(old/'reference_up.npy').astype('f8')-q['up']['zero_point'])*q['up']['scale']
   x=(g/(1+np.exp(-np.clip(g,-700,700))))*u
   scale=max(abs(q['middle']['minimum']),abs(q['middle']['maximum']))/24576
   grid=levels.astype('f8')*scale;i=np.searchsorted(grid,x).clip(1,len(grid)-1);i-=np.abs(x-grid[i-1])<=np.abs(x-grid[i]);v=levels[i]
   uq=np.load(old/'reference_middle.npy');ux=(uq.astype('f8')-q['middle']['zero_point'])*q['middle']['scale']
   y=x@w.astype('f8').T*ws;sy=(v.astype('f8')*scale)@w.astype('f8').T*ws;uy=ux@w.astype('f8').T*ws
   def error(a,b):return dict(nrmse=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)),max_abs=float(np.max(np.abs(a-b))),mse=float(np.mean((a-b)**2)))
   meta=dict(layer=layer,phase=phase,scale=scale,source_manifest_sha256=sha(old/'manifest.json'),inputs={f:sha(old/f) for f in inputs},reference_scope='local Down error on frozen A8 Gate/Up trajectory, not FP16 whole-model quality')
   metrics.append(dict(**meta,shape=list(x.shape),u8_activation=error(ux,x),sp2_activation=error(v*scale,x),u8_down=error(uy,y),sp2_down=error(sy,y),sp2_clipped=int(np.count_nonzero(np.abs(x)>grid[-1]))))
   fixture(f'llama{layer}-{phase}-sp2',v,w,1 if phase=='decode' else 2,meta)
   fixture(f'llama{layer}-{phase}-u8wide',uq,w,0,meta)
   if layer==0:fixture(f'llama{layer}-{phase}-u8store',uq,w,3,meta)
 put(OUT/'local_quality.json',metrics)

def deploy():
 preflight();head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();remote='/data/local/tmp/llama32-htp/l32-0008/prototype-'+head[:8]
 assert adb('shell',f'test ! -e {remote}',check=False).returncode==0
 adb('shell',f'mkdir -p {remote}')
 artifacts={}
 for name,build in [('llama_sp2_cli','android_ReleaseG_aarch64'),('libqwen3_probe.so','android_ReleaseG_aarch64'),('libqwen3_probe_skel.so','hexagon_ReleaseG_toolv19_v79')]:
  p=ROOT/build/'ship'/name;artifacts[name]=dict(path=str(p),sha256=sha(p));adb('push',windows(p),remote+'/'+name)
 adb('shell',f'chmod 755 {remote}/llama_sp2_cli')
 record=dict(remote=remote,artifacts=artifacts,source_head=head);put(OUT/('deployment-'+head[:8]+'.json'),record);put(OUT/'deployment.json',record)

def run(name,attempt):
 preflight();d=OUT/'fixtures'/name;m=json.loads((d/'manifest.json').read_text());assert sha(d/'input.bin')==m['input_sha256'];assert sha(d/'reference.bin')==m['reference_sha256']
 deployment=json.loads((OUT/'deployment.json').read_text());remote=deployment['remote'];dest=OUT/'device'/attempt;dest.mkdir(parents=True,exist_ok=False)
 rem=remote+'/'+attempt;assert adb('shell',f'test ! -e {rem}',check=False).returncode==0;adb('shell',f'mkdir -p {rem}');adb('push',windows(d/'input.bin'),rem+'/input.bin')
 cmd=f'cd {remote} && LD_LIBRARY_PATH={remote} DSP_LIBRARY_PATH={remote} ADSP_LIBRARY_PATH={remote} ./llama_sp2_cli {rem}/input.bin {rem}/output.bin'
 r=adb('shell',cmd,check=False);(dest/'stdout.txt').write_text(r.stdout);(dest/'stderr.txt').write_text(r.stderr)
 records=[]
 for line in r.stdout.splitlines():
  try:records.append(json.loads(line))
  except ValueError:pass
 adb('pull',rem+'/output.bin',windows(dest/'output.bin'),check=False)
 result=dict(fixture=name,command=cmd,exit=r.returncode,records=records,pass_exact=False)
 if (dest/'output.bin').exists():
  raw=(dest/'output.bin').read_bytes();h=H.unpack_from(raw);out=np.frombuffer(raw,dtype='<i4',count=h[4]*h[6],offset=h[10]);ref=np.fromfile(d/'reference.bin',dtype='<i4')
  diff=out.astype('i8')-ref;result.update(mismatches=int(np.count_nonzero(diff)),max_abs=int(np.max(np.abs(diff))),first_actual=out[:16].tolist(),first_reference=ref[:16].tolist(),pass_exact=bool(not r.returncode and not np.any(diff) and h[11]==0 and h[12]==8388608))
 put(dest/'result.json',result);print(json.dumps(result),flush=True)
 if not result['pass_exact']:raise SystemExit(1)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','llama','deploy','run']);a.add_argument('--fixture');a.add_argument('--attempt');args=a.parse_args()
 if args.action=='run':run(args.fixture,args.attempt)
 else:globals()[args.action]()
