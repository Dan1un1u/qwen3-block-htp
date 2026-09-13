#!/usr/bin/env python3
"""Sealed Qwen SP2 deployments, exact captures and fixed paired layer gate."""
import argparse,json,subprocess,shutil,shlex,tarfile,struct,statistics
from pathlib import Path
import numpy as np
from common_exp0268 import S,R,O,P,sha,write,preflight
import exp0240_device as old
from measure_exp0218 import LEDGER
from summarize_exp0217 import normalized
REMOTE='/data/local/tmp/qwen3-block-htp/exp0268'
MODELS='/data/local/tmp/qwen3-block-htp/exp0267-models'
BASE='/data/local/tmp/qwen3-block-htp/exp0257-package-v2'
SEED='/data/local/tmp/qwen3-block-htp/exp0257-prefix/prefix_kv_u8.bin'
def adb(*args,check=True):return subprocess.run([old.ADB,*args],check=check,capture_output=True,text=True,timeout=600)
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def read(p):return json.loads(Path(p).read_text())
def records(p):
 rr=[]
 for l in Path(p).read_text().splitlines():
  try:rr.append(json.loads(l))
  except ValueError:pass
 return rr

def stage(count):
 preflight();seal=read(S/'build/qwen3-sp2-build-seal.json');head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip();assert seal['source_head']==head
 for n,h in seal['files'].items():assert sha(n)==h
 for b in ['android_ReleaseG_aarch64','hexagon_ReleaseG_toolv19_v79']:
  c=(S/b/'CMakeCache.txt').read_text();assert 'QBH_MODEL_LLAMA32:BOOL=OFF' in c and f'QBH_EXP0257_LAYER_COUNT:STRING={count}\n' in c
 a=1
 while (R/'binaries'/f'l{count}-a{a}').exists():a+=1
 d=R/'binaries'/f'l{count}-a{a}';d.mkdir(parents=True,exist_ok=False);remote=REMOTE+f'-l{count}-a{a}';assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);shutil.copy2(p,d/p.name);adb('push',win(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 shutil.copy2(S/'build/qwen3-sp2-build-seal.json',d/'build-seal.json');adb('shell','chmod 755 '+remote+'/qwen3_block_cli '+remote+'/llama_sp2_cli')
 (R/f'runtime-l{count}.json').write_text(json.dumps(dict(remote=remote,seal=seal,archive=str(d)),indent=2)+'\n');print('STAGED',count,flush=True)

def deploy():
 from common_exp0268 import OLD
 preflight();assert sha(OLD/'EVIDENCE_SHA256.json')=='a33cb7b09b798ee88e9edc46e2666dd3479eb81ac25b65b3d356b11ac7528755'
 ledger=read(OLD/'EVIDENCE_SHA256.json')
 for n,v in ledger['files'].items():assert sha(OLD/n)==v['sha256'],n
 packages=[(P,BASE),(O/'sp2',MODELS+'/sp2')]+[(O/f'layer{l}-{a}',MODELS+f'/layer{l}-{a}') for l in [0,14,27] for a in ['u8','sp2']]
 proofs=[]
 for local,remote in packages:
  m=read(local/'manifest.json');names=list(m['files'])
  for n in names:assert sha(local/n)==m['files'][n]['sha256'],n
  for first in range(0,len(names),32):
   batch=names[first:first+32];raw=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in batch)).stdout
   got={l.split(None,1)[1].removeprefix(remote+'/'):l.split()[0] for l in raw.splitlines()};assert all(got[n]==m['files'][n]['sha256'] for n in batch)
  proofs.append(dict(local=str(local),remote=remote,manifest_sha256=sha(local/'manifest.json'),files=len(names)));print('VERIFIED',local.name,flush=True)
 assert adb('shell','sha256sum '+SEED).stdout.split()[0]=='7683237318d42ac5cc80052fb53619205a3d82a0d1377bcbbaed78d7c7683b91'
 write(R/'deployment.json',dict(pass_all=True,packages=proofs,parent_evidence_sha256=sha(OLD/'EVIDENCE_SHA256.json'),parent_files=len(ledger['files'])))

def physical(ps,count):
 for q in ps:
  assert q['vtcm_acquired_bytes']==8388608 and q['vtcm_peak_plan_bytes']<=8388608
  for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks']:assert q[k]==0,(k,q[k])
  assert q['block_invocation_count']==count and q['dense_r3_mode']==q['dense_r4_mode']==0 and q['wide_score_mode']==4
  z=normalized([q]);assert sum(z[k] for _,k in LEDGER)==q['invocation_ticks']
  assert q['backend']=='standalone_fastrpc_dsp' and q['qnn']=='none'
  for k,v in q.items():
   if k.startswith('slice_layer_') and isinstance(v,dict):
    assert v['status']==3 and v['layer_unattributed_ticks']==0
    assert v['hidden_ddr_read_bytes']==(131072 if count<28 and v['layer_index']==0 else 0)
    assert v['hidden_ddr_write_bytes']==(131072 if count<28 and v['layer_index']==count-1 else 0)

def run(mode,repeat,tag,count=1,layer=0,dump=False):
 preflight();state=read(R/f'runtime-l{count}.json');root=state['remote'];p=R/tag;p.mkdir(parents=True,exist_ok=False)
 e=dict(old.ENV,LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_SP2=str(8 if mode==8 else 0),QBH_U8_PREFILL_OPT=str(mode if mode!=8 else 0),QBH_WIDE_SCORE='4',QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_KV_CACHE_CAPACITY='128')
 if count==1:pkg=MODELS+f'/layer{layer}-'+('sp2' if mode==8 else 'u8')
 else:pkg=MODELS+'/sp2' if mode==8 else BASE
 if count==28:
  assert read(R/'slice_gate.json')['pass_all'];e.update(QBH_GENERATION_SEQUENCE='9',QBH_GENERATION_STEPS='16',QBH_PREFIX_KV='1',QBH_PREFIX_FILE=SEED)
  ids=np.fromfile(P/'generation_prompt_token_ids_u32.bin','<u4').tolist();assert len(ids)==64
  row=[0,2,16]+ids+[0]*16;f=p/'eval.bin';f.write_bytes(struct.pack('<4I',0x51424556,1,repeat,83)+b''.join(struct.pack('<83I',i,*row[1:]) for i in range(repeat)));adb('push',win(f),root+'/'+tag.replace('/','_')+'.bin');e['QBH_EVAL_FILE']=root+'/'+tag.replace('/','_')+'.bin'
 if dump:
  e.update(QBH_REPLAY_DUMP_DIR=root+'/'+tag.replace('/','_'),QBH_DENSE_R3_AUDIT='1');adb('shell','mkdir '+e['QBH_REPLAY_DUMP_DIR'])
 command='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in e.items())+f' ./qwen3_block_cli {pkg} W4U8 {repeat if count<28 else 1} {old.ARGS}'
 (p/'command.txt').write_text(command+'\n');r=adb('shell',command,check=False);(p/'stdout.jsonl').write_text(r.stdout);(p/'stderr.txt').write_text(r.stderr)
 if dump:adb('pull',e['QBH_REPLAY_DUMP_DIR']+'/.',win(p),check=False)
 if r.returncode:raise RuntimeError((tag,r.returncode,r.stderr[-1300:],r.stdout[-1600:]))
 rs=records(p/'stdout.jsonl');ps=[q for q in rs if q.get('record')==('generation_profile' if count==28 else 'exp0240_profile')];assert len(ps)==repeat*(16 if count==28 else 9),(tag,len(ps));physical(ps,count)
 z=dict(pass_all=True,mode=mode,repeat=repeat,count=count,layer=layer,profiles=len(ps),prefill_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill'),decode_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode'),output_hashes=[q['output_hash'] for q in ps] if count<28 else None)
 if count<28:assert all(h==z['output_hashes'][i%9] for i,h in enumerate(z['output_hashes']))
 else:
  fs=[q for q in rs if q.get('generation_sequence_complete')];assert len(fs)==repeat and all(f['all_steps_pass'] for f in fs)
  for i,f in enumerate(fs):assert f['total_host_wall_ns']==sum(q['host_wall_ns'] for q in ps[i*16:i*16+16])
  zz=[(q['selected_token_id'],q['selected_logit_half_bits']) for q in rs if 'selected_logit_half_bits' in q];assert len(zz)==16*repeat and all(v==zz[i%16] for i,v in enumerate(zz));z['selected_codes']=[list(v) for v in zz[:16]]
 write(p/'validated.json',z);print('RUN_PASS',tag,round(z['prefill_ns']/1000,2),round(z['decode_ns']/1000,2),flush=True);return z

def layer_profile(phase):
 preflight();assert read(R/'single_gate.json')['pass_all']
 if phase=='formal':assert read(R/'layer-short.json')['pass_all']
 n=5 if phase=='short' else 10;rows=[]
 for i in range(n):
  for repeat in [1,10]:
   for mode in ([0,3] if i%2==0 else [3,0]):
    tag=f'layer-{phase}/round{i:02d}-r{repeat}-m{mode}';saved=R/tag/'validated.json';z=read(saved) if saved.exists() else run(mode,repeat,tag);gold=read(R/f'audit-layer0-m{mode}/validated.json');assert z['output_hashes']==gold['output_hashes']*repeat;rows.append(dict(round=i,**z))
 rng=np.random.default_rng(268);idx=rng.integers(0,n,(20000,n));perf={}
 for repeat in [1,10]:
  for key in ['prefill_ns','decode_ns']:
   pair=np.array([[next(z[key] for z in rows if z['round']==i and z['repeat']==repeat and z['mode']==m) for m in [0,3]] for i in range(n)]);ci=np.quantile(pair[idx,1].mean(1)/pair[idx,0].mean(1),[.025,.975]);perf[f'r{repeat}_{key}']=dict(u8_mean_ns=float(pair[:,0].mean()),optimized_u8_mean_ns=float(pair[:,1].mean()),ratio=float(pair[:,1].mean()/pair[:,0].mean()),ci95=ci.tolist(),gate=bool(ci[1]<=1.1),auxiliary_only=repeat==1)
 write(R/('layer-'+phase+'.json'),dict(pass_all=True,rows=rows,performance=perf,speed_pass=all(v['gate'] for k,v in perf.items() if k.startswith('r10_'))));print('PROFILE_DONE',phase,json.dumps(perf),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action');a.add_argument('--count',type=int,default=1);a.add_argument('--layer',type=int,default=0);a.add_argument('--mode',type=int,default=8);a.add_argument('--repeat',type=int,default=1);a.add_argument('--tag',default='audit');a.add_argument('--dump',action='store_true');v=a.parse_args()
 if v.action=='stage':stage(v.count)
 elif v.action=='deploy':deploy()
 elif v.action in ['short','formal']:layer_profile(v.action)
 else:run(v.mode,v.repeat,v.tag,v.count,v.layer,v.dump)
