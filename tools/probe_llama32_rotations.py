#!/usr/bin/env python3
"""L32-0015 isolated real-HMX dense rotation contracts; never full-block timing."""
import argparse,json,struct,subprocess,shutil
from pathlib import Path
import numpy as np
from run_llama32_layer import ROOT,adb,windows
from llama_reference import sha256
from llama_u8_reference import load_qparams_bin
from prototype_llama32_sp2 import preflight
OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0015')
H=struct.Struct('<16I8Q')
def save(p,x):
 assert not p.exists(),p
 p.write_text(json.dumps(x,indent=2)+'\n')
def had(n):
 a=np.arange(n,dtype=np.uint32)
 return np.array([[1-2*((int(i)&int(j)).bit_count()%2) for j in a] for i in a],dtype='f8')
def r4(x):
 a=(x.astype('f8').reshape(-1,16,512)@had(512)*float(np.float16(1/np.sqrt(512)))).astype('f2')
 b=np.einsum('ag,...gc->...ac',had(16),a.astype('f8'))*.25
 return a.reshape(x.shape),b.astype('f2').reshape(x.shape)
def encode(x,levels,scale):
 grid=levels.astype('f8')*scale;ix=np.searchsorted(grid,x).clip(1,len(grid)-1)
 ix-=np.abs(x-grid[ix-1])<=np.abs(x-grid[ix]);return (levels[ix].astype('i4')+32768).astype('<u2')
def fixture(name,x,mode,scale=1.,meta=None):
 d=OUT/'fixtures'/name;d.mkdir(parents=True,exist_ok=False);x=np.asarray(x,dtype='<f2');rows,n=x.shape
 xo=2048;wo=xo+x.nbytes;oo=wo+(131072 if mode==7 else 0);size=oo+x.nbytes*(3 if mode==7 else 1)
 blob=bytearray(size);blob[:128]=H.pack(0x3250534c,1,size,mode,rows,n,n,xo,wo,0,oo,0xffffffff,0,0,0,0,*([0]*8));blob[xo:xo+x.nbytes]=x.tobytes()
 if mode==7:
  levels=np.array(json.loads(Path('/mnt/d/llm_exp/results/llama32-htp/l32-0008/codebook.json').read_text())['levels'],dtype='i2')
  bits=np.arange(65536,dtype='<u2');vals=bits.view('<f2').astype('f8');vals[~np.isfinite(vals)]=0
  lut=encode(vals,levels,scale);blob[wo:oo]=lut.tobytes()
  s,y=r4(x);s.tofile(d/'stage1.bin');y.tofile(d/'ideal.bin');encode(y.astype('f8'),levels,scale).tofile(d/'ideal-sp2.bin')
 else:
  y=(x.astype('f8')@had(64)*.125).astype('f2');y.tofile(d/'ideal.bin')
 (d/'input.bin').write_bytes(blob)
 save(d/'manifest.json',dict(mode=mode,rows=rows,n=n,alpha=scale,metadata=meta,scope='component inputs only; frozen old SP2 alpha is not rotated-model calibration',files={p.name:sha256(p) for p in d.iterdir() if p.is_file()}))
 print('FIXTURE',name,flush=True)
def prepare():
 OUT.mkdir(exist_ok=True,parents=True)
 rng=np.random.default_rng(15015)
 for layer in [0,7,15]:
  for phase in ['prefill','decode']:
   root=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0003/layers-a01')/f'layer{layer}-{phase}'
   manifest=json.loads((root/'manifest.json').read_text())
   names=['qparams_u8.bin','reference_gate.npy','reference_up.npy','reference_q.npy','reference_k.npy']
   for name in names:assert sha256(root/name)==manifest['files'][name]['sha256'],root/name
   q=load_qparams_bin(root/'qparams_u8.bin')
   g=(np.load(root/'reference_gate.npy').astype('f8')-q['gate']['zero_point'])*q['gate']['scale'];u=(np.load(root/'reference_up.npy').astype('f8')-q['up']['zero_point'])*q['up']['scale']
   z=g/(1+np.exp(-np.clip(g,-700,700)))*u
   alpha=float(np.float32(max(abs(q['middle']['minimum']),abs(q['middle']['maximum']))/24576))
   meta=dict(layer=layer,phase=phase,source_manifest_sha256=sha256(root/'manifest.json'),files={n:sha256(root/n) for n in names})
   fixture(f'r4-layer{layer}-{phase}',z,7,alpha,meta)
   parts=[]
   for name in ['q','k']:
    v=np.load(root/f'reference_{name}.npy');v=(v.astype('f8')-q[name+'_rope']['zero_point'])*q[name+'_rope']['scale'];parts.append(v.reshape(-1,64))
   fixture(f'r3-layer{layer}-{phase}',np.concatenate(parts),8,meta=meta)
 for mode,n in [(7,8192),(8,64)]:
  x=np.zeros((32,n),dtype='f2');x[0,0]=64;x[1,-1]=-64;x[2]=1;x[3]=np.resize([1,-1],n);x[4:]=rng.normal(0,.1,(28,n))
  fixture(f'stress-{mode}',x,mode,.001)
def deploy():
 head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();seal=json.loads((ROOT/'build/llama-build-seal.json').read_text());assert seal['source_head']==head
 remote='/data/local/tmp/llama32-htp/l32-0015/probe-'+head[:8];assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 d=OUT/('probe-'+head[:8]);d.mkdir(exist_ok=False);files={}
 for name,b in [('llama_sp2_cli','android_ReleaseG_aarch64'),('libqwen3_probe.so','android_ReleaseG_aarch64'),('libqwen3_probe_skel.so','hexagon_ReleaseG_toolv19_v79')]:
  p=ROOT/b/'ship'/name;assert sha256(p)==seal['files'][str(p)];shutil.copy2(p,d/name);adb('push',windows(p),remote+'/'+name);assert adb('shell','sha256sum '+remote+'/'+name).stdout.split()[0]==sha256(p);files[name]=sha256(p)
 adb('shell','chmod 755 '+remote+'/llama_sp2_cli');save(d/'deployment.json',dict(source_head=head,remote=remote,files=files,build_seal=seal));print(d,flush=True)
def err(a,b):
 a=a.astype('f8');b=b.astype('f8');bound=np.spacing(np.abs(b).astype('f2')).astype('f8')+2**-14
 return dict(finite=bool(np.isfinite(a).all()),max_abs=float(np.max(np.abs(a-b))),nrmse=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)),outside_1ulp_plus_minnormal=int(np.count_nonzero(np.abs(a-b)>bound)))
def run(deployment):
 d=Path(deployment);remote=json.loads((d/'deployment.json').read_text())['remote'];allresults=[]
 for f in sorted((OUT/'fixtures').iterdir()):
  m=json.loads((f/'manifest.json').read_text())
  for name,h in m['files'].items():assert sha256(f/name)==h
  dest=d/f.name;dest.mkdir(exist_ok=False);rd=remote+'/'+f.name;adb('shell','mkdir '+rd);adb('push',windows(f/'input.bin'),rd+'/input.bin')
  cmd=f'cd {remote} && LD_LIBRARY_PATH={remote} DSP_LIBRARY_PATH={remote} ADSP_LIBRARY_PATH={remote} ./llama_sp2_cli {rd}/input.bin {rd}/output.bin'
  r=adb('shell',cmd,check=False);(dest/'stdout.txt').write_text(r.stdout);(dest/'stderr.txt').write_text(r.stderr);save(dest/'command.json',dict(command=cmd,exit_code=r.returncode))
  adb('pull',rd+'/output.bin',windows(dest/'output.bin'),check=False)
  assert r.returncode==0,(f.name,r.stdout,r.stderr)
  raw=(dest/'output.bin').read_bytes();h=H.unpack_from(raw);assert h[11]==0 and h[12]==8388608 and h[13]<=8388608
  count=m['rows']*m['n'];y=np.frombuffer(raw,dtype='<f2',count=count,offset=h[10]).copy().reshape(m['rows'],m['n'])
  if m['mode']==7:
   stage=y;idealstage=np.fromfile(f/'stage1.bin',dtype='<f2').reshape(stage.shape)
   y=np.frombuffer(raw,dtype='<f2',count=count,offset=h[10]+count*2).copy().reshape(stage.shape)
   # Stage2 reference conditioned on actual first-stage results, not ideal Stage1.
   conditional=(np.einsum('ag,...gc->...ac',had(16),stage.astype('f8').reshape(-1,16,512))*.25).astype('f2').reshape(stage.shape)
   sp=np.frombuffer(raw,dtype='<u2',count=count,offset=h[10]+count*4)
   inp=(f/'input.bin').read_bytes();lut=np.frombuffer(inp,dtype='<u2',count=65536,offset=h[8]);expected=lut[y.view('u2').reshape(-1)]
   result=dict(fixture=f.name,stage1=err(stage,idealstage),stage2=err(y,conditional),ideal=err(y,np.fromfile(f/'ideal.bin',dtype='<f2').reshape(y.shape)),sp2_encoding_mismatches=int(np.count_nonzero(sp!=expected)),ideal_sp2_changes=int(np.count_nonzero(sp!=np.fromfile(f/'ideal-sp2.bin',dtype='<u2'))))
   result['pass_component']=all(result[k]['finite'] and result[k]['outside_1ulp_plus_minnormal']==0 for k in ['stage1','stage2']) and result['sp2_encoding_mismatches']==0
  else:
   result=dict(fixture=f.name,rotation=err(y,np.fromfile(f/'ideal.bin',dtype='<f2').reshape(y.shape)))
   result['pass_component']=result['rotation']['finite'] and result['rotation']['outside_1ulp_plus_minnormal']==0
  result['probe_rpc_calls']=2;result['physical_scope']='diagnostic DDR capture; no full-block/performance gate';save(dest/'result.json',result);allresults.append(result);print(json.dumps(result),flush=True)
 save(d/'component-summary.json',dict(pass_all=all(r['pass_component'] for r in allresults),results=allresults,device_cli_runs=len(allresults),probe_rpc_calls=2*len(allresults),full_block_tested=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','deploy','run']);a.add_argument('--deployment');v=a.parse_args();preflight()
 if v.stage=='run':run(v.deployment)
 else:globals()[v.stage]()
