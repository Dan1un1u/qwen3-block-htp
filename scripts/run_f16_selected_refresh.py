#!/usr/bin/env python3
import sys,json,shlex,re
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/llama32-htp');sys.path.insert(0,str(S/'scripts'));import f16_baseline_refresh as f
f.preflight();runtime=f.read(f.R/'runtime-l3.json');remote=runtime['remote'];proofs=f.read(f.R/'selected-local-provenance.json');results=[]
for proof in proofs:
 if 'stack3' in proof['package']:continue
 p=Path(proof['package']);m=f.read(p/'manifest.json');prior=f.read(proof['prior_protocols'][-1]);tokens=shlex.split(prior['command']);j=tokens.index('./qwen3_block_cli');rp=tokens[j+1]
 assert f.adb('shell','sha256sum '+rp+'/manifest.json').stdout.split()[0]==proof['manifest_sha256']
 names=list(m['files'])
 for i in range(0,len(names),32):
  lines=f.adb('shell','sha256sum '+' '.join(shlex.quote(rp+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(lines)==len(names[i:i+32])
  for l in lines:
   h,n=l.split(None,1);assert h==m['files'][n.removeprefix(rp+'/')]['sha256']
 arms=[]
 for opt in [0,2]:
  tag=f'selected/{p.name}-opt{opt}';d=f.R/tag
  if (d/'validated.json').exists():arms.append(f.read(d/'validated.json'));continue
  d.mkdir(parents=True,exist_ok=False);target=remote+'/'+p.name+f'-opt{opt}';f.adb('shell','mkdir '+target)
  e=dict(LD_LIBRARY_PATH=remote,DSP_LIBRARY_PATH=remote,ADSP_LIBRARY_PATH=remote,QBH_F16F16_OPT=str(opt),QBH_DUMP_OUTPUT_PATH=target+'/actual.bin',QBH_SCAN_MODE=m['phase'],QBH_LOGICAL_M=str(m['logical_rows']),QBH_KV_CACHE_LENGTH=str(m['past_tokens']),QBH_KV_CACHE_CAPACITY='80',QBH_DUMP_CACHE_DIR=target,QBH_DUMP_ATTENTION_DIR=target)
  if opt:e['QBH_W4F16_DECODE_AUDIT']='1'
  args=f.ARGS.split();args[4]='on'
  cmd='cd '+remote+' && '+' '.join(k+'='+shlex.quote(v) for k,v in e.items())+f' ./qwen3_block_cli {rp} F16F16 1 '+shlex.join(args)
  f.write(d/'protocol.json',dict(command=cmd,runtime=runtime,prior_proof=proof))
  z=f.adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);f.write(d/'exit.json',dict(returncode=z.returncode));f.adb('pull',target+'/.',f.win(d),check=False)
  assert z.returncode==0,(tag,z.stderr[-2000:],z.stdout[-1800:])
  n=m['logical_rows']*2048;a=np.fromfile(d/'actual.bin','<f2')[:n].astype('f8');b=np.fromfile(p/'reference_f16f16_block_output_f16.bin','<f2')[:n].astype('f8')
  err=float(np.linalg.norm(a-b)/np.linalg.norm(b));cos=float(a@b/np.linalg.norm(a)/np.linalg.norm(b));assert np.isfinite(a).all() and err<=.003 and cos>=.99999
  cache=[]
  for kind in ['k','v']:
   a=np.fromfile(d/f'actual_kv_cache_{kind}_f16.bin','<f2').astype('f8');b=np.fromfile(p/f'reference_kv_cache_{kind}_f16.bin','<f2').astype('f8');assert a.shape==b.shape
   bad=float(np.mean(np.abs(a-b)>.0625+.002*np.abs(b)));assert np.isfinite(a).all() and bad<=.01;cache.append(dict(kind=kind,bad_fraction=bad))
  result=dict(tag=tag,output_nrmse=err,output_cosine=cos,cache=cache,dump_hashes={x.name:f.sha(x) for x in d.glob('*.bin')},pass_all=True)
  f.write(d/'validated.json',result);arms.append(result);print('SELECTED_PASS',tag,err,flush=True)
 assert arms[0]['dump_hashes']==arms[1]['dump_hashes'],p
 results.append(dict(package=str(p),exact=True,runs=arms))
f.write(f.R/'selected-independent-gate.json',dict(pass_all=True,cases=results));print('ALL_SELECTED_PASS',flush=True)
