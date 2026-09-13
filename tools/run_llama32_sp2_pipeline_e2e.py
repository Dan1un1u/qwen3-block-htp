#!/usr/bin/env python3
"""L32-0011 same-build U8/SP2 all-layer M64+15 fixed ten AB/BA pairs."""
import argparse,json,shlex,subprocess,shutil
from pathlib import Path
import numpy as np
from run_llama32_layer import ROOT,adb,windows
from run_llama32_frontend import records
from llama_reference import sha256
from prototype_llama32_sp2 import preflight
OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0011/e2e-a01')
MODELS=Path('/mnt/d/llm_exp/models/llama32-htp')
REMOTE='/data/local/tmp/llama32-htp/l32-0011/e2e-a01'
def save(p,v):
 assert not p.exists(),p
 p.write_text(json.dumps(v,indent=2)+'\n')
def remote_verify(root,m):
 names=list(m['files']);checked={}
 for first in range(0,len(names),32):
  s=adb('shell','sha256sum '+' '.join(shlex.quote(root+'/'+n) for n in names[first:first+32])).stdout
  for line in s.splitlines():
   h,n=line.split(maxsplit=1);checked[n.removeprefix(root+'/')]=h
 assert all(checked.get(n)==m['files'][n]['sha256'] for n in names)
def deploy():
 OUT.mkdir(parents=True,exist_ok=False)
 seal=json.loads((ROOT/'build/llama-build-seal.json').read_text());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert seal['source_head']==head
 for path,h in seal['files'].items():assert sha256(Path(path))==h
 for b in ['android_ReleaseG_aarch64','hexagon_ReleaseG_toolv19_v79']:
  c=(ROOT/b/'CMakeCache.txt').read_text();assert 'QBH_LLAMA_LAYER_COUNT:STRING=16' in c and 'QBH_MODEL_LLAMA32:BOOL=ON' in c
 assert adb('shell','test ! -e '+REMOTE,check=False).returncode==0;adb('shell','mkdir -p '+REMOTE)
 (OUT/'binaries').mkdir();shutil.copy2(ROOT/'build/llama-build-seal.json',OUT/'binaries/build-seal.json')
 for name,b in [('qwen3_block_cli','android_ReleaseG_aarch64'),('libqwen3_probe.so','android_ReleaseG_aarch64'),('libqwen3_probe_skel.so','hexagon_ReleaseG_toolv19_v79')]:
  src=ROOT/b/'ship'/name;shutil.copy2(src,OUT/'binaries'/name);adb('push',windows(src),REMOTE+'/'+name);assert adb('shell','sha256sum '+REMOTE+'/'+name).stdout.split()[0]==sha256(src)
 adb('shell','chmod 755 '+REMOTE+'/qwen3_block_cli')
 oldpath=OUT.parent.parent/'l32-0003/device-frontend-a02/protocol.json';old=json.loads(oldpath.read_text());oldremote=old['command'].split(' && ')[0].removeprefix('cd ');base=MODELS/'l32-0003/frontend-a01';bm=json.loads((base/'manifest.json').read_text());assert sha256(base/'manifest.json')==old['package_manifest_sha256']
 assert adb('shell','sha256sum '+oldremote+'/package/manifest.json').stdout.split()[0]==sha256(base/'manifest.json');remote_verify(oldremote+'/package',bm)
 cfg={}
 for arm,local in [('u8',base),('sp2',MODELS/'l32-0010/frontend-a01')]:
  m=json.loads((local/'manifest.json').read_text())
  for n,h in m['files'].items():assert sha256(local/n)==h['sha256'],n
  dest=REMOTE+'/'+arm;dirs={str(Path(n).parent) for n in m['files']};assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in m['files'])
  adb('shell','mkdir -p '+' '.join(shlex.quote(dest+'/'+n) for n in sorted(dirs)))
  links=[n for n,h in m['files'].items() if bm['files'].get(n,{}).get('sha256')==h['sha256']]
  for first in range(0,len(links),32):adb('shell',' && '.join('ln -s '+shlex.quote(oldremote+'/package/'+n)+' '+shlex.quote(dest+'/'+n) for n in links[first:first+32]))
  for n in sorted(set(m['files'])-set(links))+['manifest.json']:adb('push',windows(local/n),dest+'/'+n)
  remote_verify(dest,m)
  cmd=old['command'].replace(oldremote,REMOTE).replace(REMOTE+'/package',dest)
  options={'QBH_W4U8_DECODE_COMMON_OP_ROWS':4,'QBH_W4U8_DECODE_SWIGLU_ROWS':4,'QBH_W4U8_DECODE_SOFTMAX':'hvx_tile4','QBH_LLAMA_SP2':5 if arm=='sp2' else 0,'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES':32,'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_CONTINUOUS':1,'QBH_W4U8_DECODE_DIRECT_N_O_GATE_PREFETCH':1,'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_SWIGLU_STREAM':1,'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES':16,'QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES':8,'QBH_W4U8_DECODE_DIRECT_N_DOWN_SINGLE_DMA':1,'QBH_W4U8_DECODE_O_BATCH_N_TILES':16,'QBH_W4U8_DECODE_DIRECT_N_O_SINGLE_DMA':1}
  # Parse original environment and overwrite each option once, avoiding shell duplicate assignments.
  prefix,args=cmd.split(' ./qwen3_block_cli ',1);wd,envtext=prefix.split(' && ',1);env=dict(t.split('=',1) for t in shlex.split(envtext));env.update({k:str(v) for k,v in options.items()});assert env['QBH_GENERATION_STEPS']=='16' and env['QBH_GENERATION_SEQUENCE']=='9'
  args=args.replace(' on off fused serial ',' on off hvx_fused_post_norm_pool4 serial ').replace(' serial scalar control ',' serial hvx_tree control ')
  cmd=wd+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+args
  teacher=OUT.parent.parent/'l32-0010/sp2-oracle.json' if arm=='sp2' else OUT.parent.parent/'l32-0003/frontend-reference-a01/teacher.json';assert sha256(teacher)==m['frontend_teacher_sha256']
  cfg[arm]=dict(command=cmd,manifest_sha256=sha256(local/'manifest.json'),oracle=str(teacher),oracle_sha256=sha256(teacher))
 assert (base/'generation_prompt_token_ids_u32.bin').read_bytes()==(MODELS/'l32-0010/frontend-a01/generation_prompt_token_ids_u32.bin').read_bytes()
 save(OUT/'protocol.json',dict(experiment='L32-0011',source_head=head,build_seal=seal,arms=cfg,pairs=10,order='AB/BA',prompt_tokens=64,decode_tokens=15,timing_scope='warm full-model Host wall: embedding,16 layers,final norm,LM head,greedy,FastRPC; excludes tokenizer/loading/session preparation',gate='paired bootstrap95 upper latency ratio <=1.10 in both modes',repeat1='auxiliary only'))
 print('DEPLOY_PASS',flush=True)
def execute(arm,name):
 cfg=json.loads((OUT/'protocol.json').read_text())['arms'][arm];d=OUT/name;d.mkdir(exist_ok=False);assert sha256(Path(cfg['oracle']))==cfg['oracle_sha256'];oracle=json.loads(Path(cfg['oracle']).read_text());save(d/'command.json',dict(command=cfg['command']))
 run=adb('shell',cfg['command'],check=False);(d/'stdout.txt').write_text(run.stdout);(d/'stderr.txt').write_text(run.stderr);rr=records(run.stdout);ss=[x for x in rr if isinstance(x,dict) and 'generation_step' in x and 'selected_token_id' in x];pp=[x for x in rr if isinstance(x,dict) and x.get('record')=='generation_profile']
 ok=run.returncode==0 and len(ss)==len(pp)==16 and [x['selected_token_id'] for x in ss]==oracle['u8_generated_ids'] and [x['selected_logit_half_bits'] for x in ss]==oracle['u8_selected_codes']
 physical=len(pp)==16 and all(q['dsp_status']==3 and q['block_invocation_count']==16 and q['vtcm_acquired_bytes']==8388608 and q['vtcm_peak_plan_bytes']<=8388608 and all(q[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','hmx_fp16_tile_pair_count','generation_lm_head_expand_ticks','w4u8_qkvo_weight_expand_ticks']) for q in pp)
 passed=ok and physical and all(s['pass'] and s['selected_logit_encoding']=='u8_code' for s in ss)
 result=dict(pass_exact=bool(passed),process_exit=run.returncode,arithmetic_match=bool(ok),physical_pass=bool(physical),steps=ss,profiles=pp,prefill_ns=ss[0]['host_wall_ns'] if ss else None,decode_ns=sum(s['host_wall_ns'] for s in ss[1:]))
 save(d/'result.json',result);print(name,passed,flush=True);assert passed,(name,run.returncode,run.stderr[-1500:],run.stdout[-1500:]);return result

def formal():
 for arm in ['u8','sp2']:assert json.loads((OUT/('gate-'+arm)/'result.json').read_text())['pass_exact']
 pairs=[]
 for i in range(10):
  pair={}
  for arm in (['u8','sp2'] if i%2==0 else ['sp2','u8']):pair[arm]=execute(arm,f'formal-{i:02d}-{arm}')
  pairs.append(pair);print('PAIR_COMPLETE',i+1,flush=True)
 rng=np.random.default_rng(11011);idx=rng.integers(0,10,size=(20000,10));report={}
 for mode,tokens in [('prefill',64),('decode',15)]:
  a=np.array([p['u8'][mode+'_ns'] for p in pairs],float);b=np.array([p['sp2'][mode+'_ns'] for p in pairs],float);ci=np.quantile(b[idx].mean(1)/a[idx].mean(1),[.025,.975]);ratio=float(b.mean()/a.mean())
  report[mode]=dict(u8_mean_ns=float(a.mean()),sp2_mean_ns=float(b.mean()),u8_tokens_per_second=tokens*1e9/float(a.mean()),sp2_tokens_per_second=tokens*1e9/float(b.mean()),latency_ratio=ratio,latency_increase_pct=100*(ratio-1),throughput_decrease_pct=100*(1-1/ratio),ci95=ci.tolist(),gate_pass=bool(ci[1]<=1.10))
 save(OUT/'summary.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['deploy','gate','formal']);v=a.parse_args();preflight()
 if v.stage=='gate':
  for arm in ['u8','sp2']:execute(arm,'gate-'+arm)
 else:globals()[v.stage]()
