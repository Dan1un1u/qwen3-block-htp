#!/usr/bin/env python3
"""L32-0064 3B vector FP intermediates: component/chain correctness and profiling."""
import argparse,json,os,shlex,shutil,struct,subprocess
from pathlib import Path
import numpy as np
from run_llama32_layer import ROOT,adb,windows
from llama_reference import sha256
from llama_u8_reference import load_qparams_bin,project_w4u8,HmxU8Converter,_cached_w4_projection
from prototype_llama32_sp2 import preflight,H
from probe_llama32_av_o_fullmodel import fnv
from report_llama32_pipeline_profile import MODULES
from run_llama32_frontend import records
import llama32_3b_reference as ref
from fp_islands_3b_attention import floating_attention

R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0064')
P=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0064/int16')
REMOTE='/data/local/tmp/llama32-htp/l32-0064'
FIXTURE=R/'fixture.json'
BASE=R/'base.json'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 assert not p.exists(),p
 p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
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
def stage():
 preflight();seal=read(ROOT/'build/llama-build-seal.json');assert seal['fp_islands'] and seal['model_size']=='3B'
 assert seal['source_head']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 d=R/f'binaries-{seal["layer_count"]}-phases';d.mkdir(exist_ok=False);remote=REMOTE+'/'+d.name
 adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha256(p)==h;shutil.copy2(p,d/p.name);adb('push',windows(p),remote+'/'+p.name)
  assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli '+remote+'/llama_sp2_cli');save(d/'seal.json',seal)
def component():
 preflight();binary=REMOTE+'/binaries-28-phases';results=[]
 for i in range(28):
  d=R/'component'/f'layer{i}';d.mkdir(parents=True,exist_ok=True)
  if (d/'summary.json').exists():results.append(read(d/'summary.json'));continue
  q=load_qparams_bin(P/f'layer{i}/qparams_u8.bin');gold,truth=fp_table(q)
  params=[q['gate']['scale'],q['gate']['zero_point'],q['up']['scale'],q['up']['zero_point'],q['middle']['scale']]
  size=4096+131072;blob=bytearray(size)
  blob[:128]=H.pack(0x3250534c,1,size,15,1,32,32,2048,0,0,4096,0xffffffff,0,0,0,0,*([0]*8));blob[2048:2068]=struct.pack('<5f',*params)
  (d/'input.bin').write_bytes(blob);adb('push',windows(d/'input.bin'),binary+'/component-input.bin')
  z=adb('shell',f'cd {binary} && LD_LIBRARY_PATH={binary} DSP_LIBRARY_PATH={binary} ADSP_LIBRARY_PATH={binary} ./llama_sp2_cli component-input.bin component-output.bin',check=False)
  (d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);assert z.returncode==0,z.stderr
  adb('pull',binary+'/component-output.bin',windows(d/'output.bin'))
  raw=np.frombuffer((d/'output.bin').read_bytes(),dtype='u1',offset=4096).astype('i4');got=(raw[:65536]+256*raw[65536:]-32768).reshape(256,256)
  np.save(d/'device.npy',got);np.save(d/'reference.npy',gold)
  err=np.abs(got-gold);s=dict(layer=i,elements=65536,exact=bool(np.array_equal(got,gold)),mismatches=int(np.count_nonzero(err)),max_code_error=int(err.max()),vs_float64_max_codes=int(np.abs(got-truth).max()),params=params)
  save(d/'summary.json',s);print('COMPONENT',s,flush=True);assert s['exact'],s
  results.append(s)
 save(R/'component-summary.json',results)
def reference():
 preflight();assert all(x['exact'] for x in read(R/'component-summary.json'))
 f=read(FIXTURE);ids=f['prompt_ids'][:64]+f['fixed'][:1]
 emb=np.memmap(P/'generation_embedding_weight_f16.bin','<f2','r',shape=(128256,3072));x=np.array(emb[ids],dtype='f4')
 co=np.concatenate([np.fromfile(P/'rope_cos_f16.bin','<f2').reshape(64,128),np.fromfile(P/'generation_decode_rope_cos_00_f16.bin','<f2').reshape(64,128)[:1]])
 si=np.concatenate([np.fromfile(P/'rope_sin_f16.bin','<f2').reshape(64,128),np.fromfile(P/'generation_decode_rope_sin_00_f16.bin','<f2').reshape(64,128)[:1]])
 out=R/'reference';out.mkdir(exist_ok=True);hashes=[];heads={}
 ref.exact_attention_dynamic=lambda q,k,v,past,cfg,cv,divide:floating_attention(q,k,v,past,cfg)
 cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so')
 for i in range(28):
  lp=P/f'layer{i}';q=load_qparams_bin(lp/'qparams_u8.bin');table,_=fp_table(q);ref.table=lambda _,table=table:table
  f=out/f'l{i:02d}.npy'
  if f.exists():x=np.load(f)
  else:
   x,kv,diag=ref.layer(x,lp,q,co,si,None);np.save(f,x)
   if i==0:
    for k,v in diag.items():np.save(out/(k+'.npy'),v)
   for k,v in zip(['k','v'],kv):np.save(out/f'l{i:02d}-{k}.npy',v)
  hashes.append([fnv(x[:64]),fnv(x[64:])]);_cached_w4_projection.cache_clear()
  if i+1 in [1,3,28]:
   hfile=out/f'heads{i+1}.json'
   if hfile.exists():heads[str(i+1)]=read(hfile)
   else:
    gq=load_qparams_bin(P/'generation_qparams_u8.bin');gamma=np.fromfile(P/'generation_final_norm_weight_f16.bin','<f2')
    a=ref.norm(x[63:],gamma,gq['generation_final_norm_output']);logits=project_w4u8(a,P,'generation_lm_head',128256,3072,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)
    ids=logits.argmax(1);heads[str(i+1)]=[dict(token=int(t),code=int(logits[j,t])) for j,t in enumerate(ids)];save(hfile,heads[str(i+1)]);_cached_w4_projection.cache_clear()
  print('REFERENCE',i,flush=True)
 save(out/'summary.json',dict(hashes=hashes,heads=heads))
def execute(tag,repeat=1,audit=False,nl=28,serial=True):
 d=R/tag;d.mkdir(exist_ok=True)
 if (d/'validated.json').exists():return read(d/'validated.json')
 base=read(BASE);prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
 env=dict(t.split('=',1) for t in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args)
 suffix='-phases' if serial else '';binary=REMOTE+f'/binaries-{nl}'+suffix;argv[0]='/data/local/tmp/llama32-htp/l32-0064/int16'
 for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT']:env.pop(k,None)
 env.update(LD_LIBRARY_PATH=binary,DSP_LIBRARY_PATH=binary,ADSP_LIBRARY_PATH=binary,QBH_GENERATION_STEPS='2',QBH_GENERATION_EXPECTED_TOKENS='64',QBH_GENERATION_SEQUENCE='9',QBH_LLAMA_SP2='8',QBH_SP2_DOWN_HVX='0',QBH_WIDE_SCORE='7',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='3' if serial else '0',QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_W4U8_DECODE_AV_REQUANT_ROWS='4')
 f=read(FIXTURE);words=[0x51424556,2,repeat,69]
 for j in range(repeat):words += [j,3,2]+f['prompt_ids'][:64]+f['fixed'][:2]
 ef=d/'trajectory.bin';ef.write_bytes(struct.pack('<'+'I'*len(words),*words));er=binary+'/'+tag+'-trajectory.bin';env['QBH_EVAL_FILE']=er
 if audit:env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=REMOTE+'/'+tag)
 cmd='cd '+binary+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+' '.join(shlex.quote(v) for v in argv)
 if not (d/'protocol.json').exists():save(d/'protocol.json',dict(command=cmd,layers=nl,repeat=repeat,audit=audit,serial=serial,seal_sha256=sha256(R/(f'binaries-{nl}'+suffix)/'seal.json'),package_manifest_sha256=sha256(P/'manifest.json')))
 if not (d/'exit.json').exists():
  adb('push',windows(ef),er)
  if audit:adb('shell','mkdir -p '+env['QBH_GENERATION_AUDIT_DIR'])
  z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);save(d/'exit.json',dict(returncode=z.returncode))
  if audit:adb('pull',env['QBH_GENERATION_AUDIT_DIR']+'/.',windows(d/'audit'))
 assert read(d/'exit.json')['returncode']==0,(d,(d/'stderr.txt').read_text()[-1000:])
 rr=records((d/'stdout.txt').read_text());pp=[x for x in rr if x.get('record')=='generation_profile'];ss=[x for x in rr if 'selected_logit_half_bits' in x and 'generation_step' in x]
 assert len(pp)==len(ss)==2*repeat,(tag,len(pp),len(ss))
 gold=read(R/'reference/summary.json');checks=[]
 for j,(x,t) in enumerate(zip(pp,ss)):
  step=j%2;g=gold['heads'][str(nl)][step]
  assert x['dsp_status']==3 and x['numerical_status']==1
  assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608 and x['vtcm_peak_plan_bytes']<=8388608
  assert x['intermediate_spill_fill_count']==x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==0
  total=sum(sum(x[k] for k in f) for _,f in MODULES)-x['generation_final_norm_ticks'];assert total==x['invocation_ticks']
  assert x['host_wall_ns']/1000>=total/19.2
  for i in range(nl):
   layer=x[f'slice_layer_{i}'];assert layer['status']==3 and layer['layer_unattributed_ticks']==0
   if audit:checks.append(dict(step=step,layer=i,exact=layer['output_hash']==gold['hashes'][i][step],got=layer['output_hash'],expected=gold['hashes'][i][step]))
  assert (t['selected_token_id'],t['selected_logit_half_bits'])==(g['token'],g['code']),(tag,j,'head',t['selected_token_id'],g)
 if audit:
  save(d/'hash-checks.json',checks);assert all(x['exact'] for x in checks),[x for x in checks if not x['exact']][:3]
 save(d/'records.json',rr);save(d/'validated.json',dict(pass_checks=True,boundaries=len(pp),exact_layers=len(checks),prefill_us=[x['host_wall_ns']/1000 for x in pp[::2]],profiles=pp));print('PASS',tag,flush=True)
 return read(d/'validated.json')
def run():
 preflight();execute('audit16',audit=True)
 execute('warmup')
 for kind,n in [('short',5),('formal',10)]:
  for i in range(n):execute(f'{kind}-{i:02d}',repeat=10)
def phases():
 preflight();execute('phases-audit28',audit=True,serial=True)
 execute('phases-warmup',serial=True)
 for kind,n in [('short',5),('formal',10)]:
  for i in range(n):execute(f'phases-{kind}-{i:02d}',repeat=10,serial=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action');a.add_argument('--layers',type=int,default=28);a.add_argument('--tag',default='audit');args=a.parse_args()
 if args.action=='audit':preflight();execute(args.tag,nl=args.layers,audit=True)
 else:globals()[args.action]()
