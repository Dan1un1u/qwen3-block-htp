#!/usr/bin/env python3
"""EXP0305 Qwen1.7 vector floating intermediates, own-reference profiling."""
import json,os,shlex,struct,subprocess,hashlib,shutil,argparse
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1]
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0305')
P=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0305/recovered-int16')
REMOTE='/data/local/tmp/qwen3-block-htp/exp0305'
ADB='/mnt/c/adb/adb.exe'
H=struct.Struct('<16I8Q')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p;p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def adb(*args,check=True):return subprocess.run([ADB,'-s','3B15C8007Z300000',*args],capture_output=True,text=True,check=check,timeout=600)
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def preflight():
 z=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'EXPERIMENT=EXP-0305' in z
 print(z,flush=True)
def fnv(x):
 h=0xcbf29ce484222325
 for a in np.asarray(x,dtype='<f4').tobytes():h=((h^a)*0x100000001b3)&0xffffffffffffffff
 return h
def fl(x):return np.asarray(x,dtype='f4')

def fp_table(q):
 g=fl(fl(np.arange(256)-q['gate']['zero_point'])*fl(q['gate']['scale']))[:,None]
 u=fl(fl(np.arange(256)-q['up']['zero_point'])*fl(q['up']['scale']))[None,:]
 x=fl(-np.minimum(np.abs(g),fl(80)))
 n=np.floor(fl(x*fl(1.4426950408889634)));rr=fl(x-fl(n*fl(.6931471805599453)))
 z=fl(1/5040)
 for c in [1/720,1/120,1/24,1/6,.5,1,1]:z=fl(fl(z*rr)+fl(c))
 e=fl(z*fl(np.exp2(n)));den=fl(fl(1)+e);iv=fl(.75)
 for _ in range(5):iv=fl(iv*fl(fl(2)-fl(den*iv)))
 sig=np.where(g<0,fl(e*iv),iv);v=fl(fl(fl(g*sig)*u)*fl(1/fl(q['middle']['scale'])))
 result=np.rint(np.clip(v,-32767,32767)).astype('i4')
 gg=(np.arange(256,dtype='f8')-q['gate']['zero_point'])*q['gate']['scale']
 uu=(np.arange(256,dtype='f8')-q['up']['zero_point'])*q['up']['scale']
 truth=np.rint(np.clip((gg/(1+np.exp(-np.clip(gg,-700,700))))[:,None]*uu[None,:]/q['middle']['scale'],-32767,32767)).astype('i4')
 assert np.abs(result-truth).max()<=1,('approximation',np.abs(result-truth).max())
 return result,truth
def stage(nl):
 preflight();seal=read(S/'build/qwen3-sp2-build-seal.json');assert seal['fp_islands']
 assert seal['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 d=R/f'binaries-{nl}-a03';d.mkdir(exist_ok=False);remote=REMOTE+'/'+d.name;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha(p)==h;shutil.copy2(p,d/p.name);adb('push',win(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli '+remote+'/llama_sp2_cli');save(d/'seal.json',seal)
def component():
 from reference_w4u8_hmx import load_qparams_bin
 preflight();binary=REMOTE+'/binaries-28';results=[]
 for i in range(28):
  d=R/'component'/f'layer{i}';d.mkdir(parents=True,exist_ok=True)
  if (d/'summary.json').exists():results.append(read(d/'summary.json'));continue
  q=load_qparams_bin(P/f'layer{i}/qparams_u8.bin');gold,truth=fp_table(q)
  params=[q['gate']['scale'],q['gate']['zero_point'],q['up']['scale'],q['up']['zero_point'],q['middle']['scale']]
  size=4096+131072;blob=bytearray(size);blob[:128]=H.pack(0x3250534c,1,size,15,1,32,32,2048,0,0,4096,0xffffffff,0,0,0,0,*([0]*8));blob[2048:2068]=struct.pack('<5f',*params)
  (d/'input.bin').write_bytes(blob);adb('push',win(d/'input.bin'),binary+'/component-input.bin')
  z=adb('shell',f'cd {binary} && LD_LIBRARY_PATH={binary} DSP_LIBRARY_PATH={binary} ADSP_LIBRARY_PATH={binary} ./llama_sp2_cli component-input.bin component-output.bin',check=False)
  (d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);assert z.returncode==0
  adb('pull',binary+'/component-output.bin',win(d/'output.bin'))
  raw=np.frombuffer((d/'output.bin').read_bytes(),dtype='u1',offset=4096).astype('i4');got=(raw[:65536]+256*raw[65536:]-32768).reshape(256,256)
  np.save(d/'device.npy',got);np.save(d/'reference.npy',gold)
  s=dict(layer=i,elements=65536,exact=bool(np.array_equal(got,gold)),max_code_error=int(np.abs(got-gold).max()),vs_float64_max_codes=int(np.abs(got-truth).max()),params=params)
  save(d/'summary.json',s);print('COMPONENT',s,flush=True);assert s['exact'];results.append(s)
 save(R/'component-summary.json',results)
def reference():
 import reference_fp_islands as ref
 from verify_exp0167_generation import load_generation_qparams
 preflight();assert all(x['exact'] for x in read(R/'component-summary.json'))
 seedfile=R/'prefix_kv_u8.bin'
 if not seedfile.exists():adb('pull','/data/local/tmp/qwen3-block-htp/exp0257-prefix/prefix_kv_u8.bin',win(seedfile))
 assert sha(seedfile)=='7683237318d42ac5cc80052fb53619205a3d82a0d1377bcbbaed78d7c7683b91'
 seeds=np.fromfile(seedfile,'u1').reshape(28,2,8,128)
 prompt=np.fromfile(P/'generation_prompt_token_ids_u32.bin','<u4').tolist();fixed=read(R.parent/'exp0284/fixed_tokens.json')['ids'];assert len(prompt)==64
 emb=np.memmap(P/'generation_embedding_weight_f16.bin','<f2','r',shape=(151936,2048));x=np.array(emb[prompt+fixed[:1]],dtype='f4')
 co=np.concatenate([np.fromfile(P/'rope_cos_f16.bin','<f2').reshape(64,128),np.fromfile(P/'generation_decode_rope_cos_00_f16.bin','<f2').reshape(64,128)[:1]])
 si=np.concatenate([np.fromfile(P/'rope_sin_f16.bin','<f2').reshape(64,128),np.fromfile(P/'generation_decode_rope_sin_00_f16.bin','<f2').reshape(64,128)[:1]])
 out=R/'reference';out.mkdir(exist_ok=True);hashes=[];heads={}
 for i in range(28):
  f=out/f'l{i:02d}.npy'
  if f.exists():x=np.load(f)
  else:
   x,kv,diag=ref.layer(x,P/f'layer{i}',co,si,seed=seeds[i]);np.save(f,x)
   if i==0:
    for k,v in diag.items():np.save(out/(k+'.npy'),v)
   for k,v in zip(['k','v'],kv):np.save(out/f'l{i:02d}-{k}.npy',v)
  hashes.append([fnv(x[:64]),fnv(x[64:])]);ref.weight.cache_clear()
  if i+1 in [1,3,28]:
   hfile=out/f'heads{i+1}.json'
   if hfile.exists():heads[str(i+1)]=read(hfile)
   else:
    q=load_generation_qparams(P/'generation_qparams_u8.bin');a=ref.norm(x[63:],np.fromfile(P/'generation_final_norm_weight_f16.bin','<f2'),q['generation_final_norm_output'])
    logits=ref.project(a,P,'generation_lm_head',151936,q['generation_final_norm_output'],q['generation_lm_head_output']);ids=logits.argmax(1)
    heads[str(i+1)]=[dict(token=int(t),code=int(logits[j,t])) for j,t in enumerate(ids)];save(hfile,heads[str(i+1)]);ref.weight.cache_clear()
  print('REFERENCE',i,flush=True)
 save(out/'summary.json',dict(hashes=hashes,heads=heads));save(R/'fixture.json',dict(prompt=prompt,fixed=fixed))
def execute(tag,repeat=1,audit=False,nl=28):
 from device_exp0284 import records
 from summarize_exp0217 import normalized
 from measure_exp0218 import LEDGER
 d=R/tag;d.mkdir(exist_ok=True)
 if (d/'validated.json').exists():return read(d/'validated.json')
 base=read(R.parent/'exp0284/fixed-aux-INT16-r1/protocol.json');prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
 env=dict(t.split('=',1) for t in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);binary=REMOTE+f'/binaries-{nl}'
 # Retain vector operators, four attention contexts and ordinary group preparation.
 argv[8]='hvx_fused_post_norm_pool4';argv[15]='u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch';argv[16]='4';argv[18]='qkvo_batch4';argv[19]='hvx_tree';argv[24]='0'
 for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT']:env.pop(k,None)
 env.update(LD_LIBRARY_PATH=binary,DSP_LIBRARY_PATH=binary,ADSP_LIBRARY_PATH=binary,QBH_GENERATION_STEPS='2',QBH_GENERATION_EXPECTED_TOKENS='16',QBH_GENERATION_SEQUENCE='9',QBH_SP2='8',QBH_WIDE_SCORE='7',QBH_PAPER_PIPELINE_DISABLE='3',QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_W4U8_DECODE_AV_REQUANT_ROWS='4',QBH_W4U8_QKV_RING_EXPAND_WORKERS='0')
 fixture=read(R/'fixture.json');words=[0x51424556,2,repeat,69]
 for j in range(repeat):words += [j,3,2]+fixture['prompt']+fixture['fixed'][:2]
 ef=d/'trajectory.bin';ef.write_bytes(struct.pack('<'+'I'*len(words),*words));er=binary+'/'+tag+'-trajectory.bin';env['QBH_EVAL_FILE']=er
 if audit:env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=REMOTE+'/'+tag)
 cmd='cd '+binary+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+' '.join(shlex.quote(v) for v in argv)
 if not (d/'protocol.json').exists():save(d/'protocol.json',dict(command=cmd,layers=nl,repeat=repeat,audit=audit,seal_sha256=sha(R/f'binaries-{nl}-a03/seal.json'),package_manifest_sha256=sha(P/'manifest.json')))
 if not (d/'exit.json').exists():
  adb('push',win(ef),er)
  if audit:adb('shell','mkdir -p '+env['QBH_GENERATION_AUDIT_DIR'])
  z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);save(d/'exit.json',dict(returncode=z.returncode))
  if audit:adb('pull',env['QBH_GENERATION_AUDIT_DIR']+'/.',win(d/'audit'))
 assert read(d/'exit.json')['returncode']==0,(tag,(d/'stderr.txt').read_text()[-1500:])
 rr=records(d/'stdout.txt');pp=[x for x in rr if x.get('record')=='generation_profile'];ss=[x for x in rr if 'selected_logit_half_bits' in x and 'generation_step' in x];assert len(pp)==len(ss)==2*repeat,(tag,len(pp),len(ss))
 gold=read(R/'reference/summary.json');checks=[]
 for j,(x,t) in enumerate(zip(pp,ss)):
  step=j%2;g=gold['heads'][str(nl)][step];assert x['dsp_status']==3 and x['numerical_status']==1
  assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608 and x['vtcm_peak_plan_bytes']<=8388608
  assert x['intermediate_spill_fill_count']==x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==0
  z=normalized([x]);assert sum(z[k] for _,k in LEDGER)==x['invocation_ticks']
  for i in range(nl):
   layer=x[f'slice_layer_{i}'];assert layer['status']==3 and layer['layer_unattributed_ticks']==0
   if audit:checks.append(dict(step=step,layer=i,exact=(int(layer['output_hash'],16) if isinstance(layer['output_hash'],str) else layer['output_hash'])==gold['hashes'][i][step],got=layer['output_hash'],expected=gold['hashes'][i][step]))
  assert (t['selected_token_id'],t['selected_logit_half_bits'])==(g['token'],g['code']),(tag,j,'head',t['selected_token_id'],g)
 if audit:
  save(d/'hash-checks.json',checks);assert all(x['exact'] for x in checks),[x for x in checks if not x['exact']][:5]
 save(d/'records.json',rr);save(d/'validated.json',dict(pass_checks=True,exact_layers=len(checks),profiles=pp));print('PASS',tag,flush=True)
 return read(d/'validated.json')
def run():
 preflight();execute('audit28',audit=True);execute('warmup')
 for kind,n in [('short',5),('formal',10)]:
  for i in range(n):execute(f'{kind}-{i:02d}',repeat=10)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action');a.add_argument('--layers',type=int,default=28);a.add_argument('--tag',default='audit');args=a.parse_args()
 if args.action=='stage':stage(args.layers)
 elif args.action=='audit':preflight();execute(args.tag,nl=args.layers,audit=True)
 else:globals()[args.action]()
