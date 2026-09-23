#!/usr/bin/env python3
"""L32-0069 reproducible component experiment; no model inference."""
import argparse,hashlib,json,os,struct,subprocess,sys
from pathlib import Path
import numpy as np
from run_llama32_layer import ROOT,adb,windows
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0069')
H=struct.Struct('<16I8Q')
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def preflight():subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
def fwht(x,layers=1,dtype='f4'):
 a=x.astype(dtype).copy();rows=len(a)
 for _ in range(layers):
  for h in [1<<i for i in range(13)]:
   v=a.reshape(rows,-1,h*2);lo=v[:,:,:h].copy();hi=v[:,:,h:].copy();v[:,:,:h]=lo+hi;v[:,:,h:]=lo-hi
  a*=np.array(1/np.sqrt(8192),dtype=dtype)
 return a

def prepare():
 preflight();rng=np.random.default_rng(690069)
 p=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0068/A8/layer0/qparams_u8.bin')
 payload=p.read_bytes();rec=struct.Struct('<32sfi2f');qp={}
 for off in range(0,len(payload),rec.size):
  name,s,z,lo,hi=rec.unpack_from(payload,off);qp[name.split(b'\0')[0].decode()]=dict(scale=s,zero=z)
 g=rng.integers(0,256,(64,8192));u=rng.integers(0,256,(64,8192))
 gf=(g-qp['gate']['zero'])*qp['gate']['scale'];uf=(u-qp['up']['zero'])*qp['up']['scale']
 x=(gf/(1+np.exp(-np.clip(gf,-700,700)))*uf).astype('<f2');assert np.isfinite(x).all()
 # Timing tensor: deterministic A8 gate/up codes, original qparams, pre-A8 SwiGLU.
 # Numerical sentinel batch is separate from the timing tensor.
 stress=x.copy();stress[0]=0;stress[1]=.125;stress[2]=0;stress[2,0]=1000;stress[3]=np.resize([1.,-1.],8192);stress[4]=0;stress[4,8191]=1000;stress[5]=rng.normal(0,.1,8192);stress[5,1234]=1000
 inv=float(np.float32(1/qp['middle']['scale']));zp=qp['middle']['zero']
 np.save(R/'timing_input_f16.npy',x);np.save(R/'stress_input_f16.npy',stress)
 meta=dict(source='synthetic deterministic Gate/Up U8 codes with original layer0 qparams; not a model trajectory',seed=690069,qparams_path=str(p),qparams_sha256=sha(p),qparams=qp,position='post-SwiGLU/pre-middle-A8',working_dtype='FP32',output='ordinary U8, original fixed middle scale, native K32/M64 layout',inverse_scale=inv,zero=zp,rotation='normalized Sylvester H8192,13 butterflies, one final FP32 normalization',repeats=10)
 put(R/'FIXTURES.json',meta)
 for name,arr in [('prefill',x),('decode',x[:1]),('stress',stress),('impulse',stress[2:3])]:
  for mode in [16,17,18]:
   for layers in [1,16]:
    rows=len(arr);io=2048;po=io+arr.nbytes;oo=(po+12+2047)&~2047;size=oo+arr.size*4+524288
    blob=bytearray(size);blob[:128]=H.pack(0x3250534c,1,size,mode,rows,8192,layers,io,po,0,oo,0xffffffff,0,0,0,0,*([0]*8));blob[io:io+arr.nbytes]=arr.tobytes();struct.pack_into('<3f',blob,po,inv,zp,10)
    out=R/'fixtures'/f'{name}-m{mode}-l{layers}.bin';out.parent.mkdir(exist_ok=True);out.write_bytes(blob)
 # Independent direct Walsh dot products, not another butterfly implementation.
 columns=np.array([0,1,2,3,31,32,127,512,1234,4095,8191]);k=np.arange(8192)
 parity=np.array([[int(j&i).bit_count()&1 for i in k] for j in columns]);sign=1-2*parity
 exact=stress.astype('f8')@sign.T/np.sqrt(8192)
 recursive=fwht(stress,dtype='f8')[:,columns];assert np.allclose(exact,recursive,atol=1e-10,rtol=1e-12)
 put(R/'HOST_REFERENCE.json',dict(direct_Walsh_columns=columns.tolist(),rows=64,max_abs=float(abs(exact-recursive).max()),fixtures={p.name:sha(p) for p in (R/'fixtures').glob('*.bin')}))

def deploy():
 preflight();seal=json.loads((ROOT/'build/llama-build-seal.json').read_text());head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();assert seal['source_head']==head
 remote='/data/local/tmp/llama32-htp/l32-0069/'+head[:12]
 adb('shell',f'mkdir -p {remote}')
 for p,h in seal['files'].items():
  p=Path(p);assert sha(p)==h
  if p.name=='qwen3_block_cli':continue
  adb('push',windows(p),remote+'/'+p.name)
  got=adb('shell',f'sha256sum {remote}/{p.name}').stdout.split()[0];assert got==h
 adb('shell',f'chmod 755 {remote}/llama_sp2_cli');adb('push',windows(R/'fixtures'),remote+'/')
 put(R/'DEPLOYMENT.json',dict(remote=remote,seal=seal));put(R/'DEVICE_START.json',dict(battery=adb('shell','dumpsys battery').stdout,processes=adb('shell','ps -A | grep -E "qwen3_block|llama_sp2|llama-bench"',check=False).stdout))

def run(name,mode,layers,tag,audit=False):
 dep=json.loads((R/'DEPLOYMENT.json').read_text());remote=dep['remote'];d=R/'device'/tag;d.mkdir(parents=True,exist_ok=False)
 filename=f'{name}-m{mode}-l{layers}.bin'
 cmd=f'cd {remote} && LD_LIBRARY_PATH={remote} DSP_LIBRARY_PATH={remote} ADSP_LIBRARY_PATH={remote} ./llama_sp2_cli {remote}/fixtures/{filename} {remote}/output.bin'
 proc=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(proc.stdout);(d/'stderr.txt').write_text(proc.stderr)
 records=[]
 for line in proc.stdout.splitlines():
  try:
   j=json.loads(line)
   if 'rpc_status' in j:records.append(j)
  except ValueError:pass
 put(d/'records.json',records);put(d/'exit.json',dict(returncode=proc.returncode,command=cmd))
 assert proc.returncode==0 and len(records)==1,(tag,proc.stdout,proc.stderr)
 z=records[0];assert z['rpc_status']==z['dsp_status']==z['streams']==0 and z['vtcm_bytes']==8388608 and z['peak_bytes']<=8388608
 if audit:
  adb('pull',remote+'/output.bin',windows(d/'output.bin'));blob=(d/'output.bin').read_bytes();h=H.unpack_from(blob);m=h[4];oo=h[10]
  actual=np.frombuffer(blob,dtype='<f4',count=m*8192,offset=oo).reshape(m,8192)
  inp=np.frombuffer((R/'fixtures'/filename).read_bytes(),dtype='<f2',count=m*8192,offset=h[7]).reshape(m,8192)
  ref=fwht(inp,layers,'f4') if mode!=18 else inp.astype('f4');ideal=fwht(inp,layers,'f8') if mode!=18 else inp.astype('f8')
  nr=float(np.linalg.norm(actual.astype('f8')-ideal)/max(np.linalg.norm(ideal),1e-30));mx=float(abs(actual-ideal).max());exact=np.array_equal(actual.view('u4'),ref.view('u4'))
  q=np.frombuffer(blob,dtype='u1',count=524288,offset=oo+m*8192*4).reshape(256,64,32).transpose(1,0,2).reshape(64,8192)
  meta=json.loads((R/'FIXTURES.json').read_text());code=np.float32(np.float32(ref*np.float32(meta['inverse_scale']))+np.float32(meta['zero']));qref=np.clip(np.trunc(np.float32(code+np.float32(.5))),0,255).astype('u1')
  qbad=int(np.count_nonzero(q[:m]!=qref));pad=bool(np.all(q[m:]==meta['zero']))
  result=dict(relative_l2=nr,max_abs=mx,fp32_reference_bit_exact=exact,u8_mismatches=qbad,padding_pass=pad,finite=bool(np.isfinite(actual).all()),output_sha256=sha(d/'output.bin'))
  put(d/'validated.json',result);assert nr<=2e-6 and mx<=2e-5*max(1,float(abs(ideal).max())) and qbad==0 and pad and result['finite'],(tag,result)
 return z

def audit():
 preflight()
 for name in ['stress','impulse','prefill','decode']:
  for mode in [16,17,18]:
   for layers in [1,16]:run(name,mode,layers,f'audit/{name}-m{mode}-l{layers}',True)
 put(R/'NUMERICAL_PASS.json',dict(pass_all=True,cases=24,no_model_quality_claim=True))

def campaign():
 preflight();assert (R/'NUMERICAL_PASS.json').exists()
 for phase,rounds in [('short',5),('formal',10)]:
  for i in range(rounds):
   for name in ['prefill','decode']:
    for layers in [1,16]:
     modes=[16,17] if name=='prefill' else [16]
     if layers==1:modes+=[18]
     if i%2:modes.reverse()
     for mode in modes:run(name,mode,layers,f'{phase}/{i+1:02d}-{name}-m{mode}-l{layers}')
   put(R/'PROGRESS.json',dict(phase=phase,round=i+1));print(phase,i+1,flush=True)
 put(R/'CAMPAIGN_COMPLETE.json',dict(short_rounds=5,formal_rounds=10,in_rpc_repeats=10,e2e=False))

def report():
 allrows=[];rng=np.random.default_rng(690069)
 for name in ['prefill','decode']:
  for layers in [1,16]:
   for mode in ([16,17,18] if layers==1 and name=='prefill' else [16,18] if layers==1 else [16,17] if name=='prefill' else [16]):
    samples=[json.loads((R/'device'/f'formal/{i+1:02d}-{name}-m{mode}-l{layers}'/'records.json').read_text())[0] for i in range(10)]
    # qtimer19.2MHz. Ten actual core windows per RPC; pooled wall includes waits.
    core=np.array([s['mac_ticks']/19.2/10 for s in samples]);means=core[rng.integers(0,10,(20000,10))].mean(1)
    row=dict(phase=name,transforms=layers,workers=4 if mode==17 else 1,mode=mode,core_us=float(core.mean()),core_ci95_us=np.quantile(means,[.025,.975]).tolist(),prepare_us=float(np.mean([s['pack_ticks']/19.2/10 for s in samples])),quantize_pack_us=float(np.mean([s['convert_ticks']/19.2/10 for s in samples])),core_samples_us=core.tolist())
    allrows.append(row)
 put(R/'SUMMARY.json',dict(rows=allrows,historical_A8_L32_0068=dict(prefill64_ms=26.67729899,decode42_ms=907.09569526,source='399cc083f6c586e91695edef2d64c4105e3c4da6',nonpaired=True),e2e_measured=False))
 print(json.dumps(allrows,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','deploy','audit','campaign','report']);args=ap.parse_args();globals()[args.action]()
