#!/usr/bin/env python3
"""L32-0071: same binary, fixed trace, full-model HVX R4 versus HVX R3+R4; same frozen package."""
import argparse,copy,json,os,shlex,shutil,struct,subprocess
from pathlib import Path
import numpy as np
from run_llama32_layer import ROOT,adb,windows
from run_llama32_frontend import records
from llama_reference import sha256,EXPECTED
from llama_u8_reference import load_qparams_bin,QPARAM_RECORD,HmxU8Converter,project_w4u8,_cached_w4_projection
import llama32_a8_fp32_reference as ref
from probe_llama32_av_o_fold import configs
from report_llama32_pipeline_profile import MODULES
from ablate_integer_fusion import head as get_head
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0071')
M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0070')
PARENT=M.parent/'l32-0068/A8'
ORDINARY=M.parent/'l32-0054/1B/a8/native64-d42-a2'
BASE=R.parent/'l32-0060/1B/sp2/base.json'
FIXTURE=R.parent/'l32-0057/1B/sp2/fixture.json'
REMOTE='/data/local/tmp/llama32-htp/l32-0071'
PACKAGE_REMOTE='/data/local/tmp/llama32-htp/l32-0070/models'
MODES=['R4','R3R4']
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p;p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def preflight():
 z=subprocess.check_output(['python3',str(ROOT)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],text=True);assert 'ACTIVE_EXPERIMENT=L32-0071' in z;print(z,flush=True)
def fnv(x):
 h=1469598103934665603
 for b in memoryview(np.ascontiguousarray(x,dtype='<f4')).cast('B'):h=((h^b)*1099511628211)&0xffffffffffffffff
 return f'{h:016x}'
def reference():
 preflight();old=R.parent/'l32-0070/oracle-R4/summary.json';d=R/'oracle-R4';d.mkdir(exist_ok=True)
 if not (d/'summary.json').exists():shutil.copy2(old,d/'summary.json');save(d/'reuse.json',dict(source=str(old),sha256=sha256(old),unchanged_manifest_files=True))
 mode='R3R4';p=M/'R4';d=R/'oracle-R3R4';d.mkdir(exist_ok=True);fixture=read(FIXTURE);ids=fixture['prompt_ids'][:64]+fixture['fixed'][:42]
 emb=np.memmap(p/'generation_embedding_weight_f16.bin','<f2','r',shape=(128256,2048));x=np.asarray(emb[ids],dtype='f4')
 co=np.concatenate([np.fromfile(p/'rope_cos_f16.bin','<f2').reshape(64,64)]+[np.fromfile(p/f'generation_decode_rope_cos_{j:02d}_f16.bin','<f2').reshape(64,64)[:1] for j in range(42)])
 si=np.concatenate([np.fromfile(p/'rope_sin_f16.bin','<f2').reshape(64,64)]+[np.fromfile(p/f'generation_decode_rope_sin_{j:02d}_f16.bin','<f2').reshape(64,64)[:1] for j in range(42)])
 hashes=[];heads={}
 for i in range(16):
  lp=p/f'layer{i}';q=load_qparams_bin(lp/'qparams_u8.bin');cfg=configs(lp);ref.configs=lambda _,cfg=cfg:cfg
  f=d/f'l{i:02d}-output.npy'
  if f.exists():x=np.load(f)
  else:
   x,kv,diag=ref.layer(x,lp,q,co,si,None,sp2=False,r4=True,r3=True);assert np.isfinite(x).all();np.save(f,x)
   if i==0:
    for k in ['gate','up','middle','down']:np.save(d/(k+'.npy'),diag[k])
  hashes.append([fnv(x[:64])]+[fnv(x[j:j+1]) for j in range(64,106)]);_cached_w4_projection.cache_clear()
  if i+1 in [1,3,16]:
   hf=d/f'head-{i+1}.json'
   if not hf.exists():save(hf,get_head(x,p))
   heads[str(i+1)]=read(hf)
  print('REFERENCE',mode,i,flush=True)
 save(d/'summary.json',dict(layer_hashes=hashes,heads=heads,trajectory_sha256=sha256(FIXTURE),own_contract=True,quality_claim=False))
def deploy():
 preflight();p=M/'R4';mf=read(p/'manifest.json')['files'];names=list(mf)
 for start in range(0,len(names),24):
  batch=names[start:start+24]
  for n in batch:assert sha256(p/n)==mf[n]['sha256']
  lines=adb('shell','sha256sum '+' '.join(PACKAGE_REMOTE+'/R4/'+n for n in batch)).stdout.splitlines();assert len(lines)==len(batch)
  for l in lines:h,n=l.split(None,1);assert h==mf[n.removeprefix(PACKAGE_REMOTE+'/R4/')]['sha256']
 save(R/'deployment.json',dict(all_payload_hashes_verified=True,reused_package=str(p),manifest_sha256=sha256(p/'manifest.json')))
def stage():
 preflight();seal=read(ROOT/'build/llama-build-seal.json');assert not seal['fp_islands'] and seal['model_size']=='1B';assert seal['source_head']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();nl=int(seal['layer_count']);d=R/f'binaries-I-{nl}-a02';d.mkdir(exist_ok=False);remote=REMOTE+'/'+d.name;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha256(p)==h;shutil.copy2(p,d/p.name);adb('push',windows(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli');save(d/'seal.json',seal)
def execute(mode,tag,nl=16,repeat=1,audit=False,poison=False):
    arm=mode
    d=R/tag
    if (d/'validated.json').exists():return read(d/'validated.json')
    d.mkdir(exist_ok=True);binary_mode='I';root=REMOTE+f'/binaries-{binary_mode}-{nl}-a02';base=read(BASE)
    prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
    env=dict(t.split('=',1) for t in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=PACKAGE_REMOTE+'/R4'
    for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT']:env.pop(k,None)
    env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_GENERATION_STEPS='43',QBH_GENERATION_EXPECTED_TOKENS='64',QBH_GENERATION_SEQUENCE='9',QBH_LLAMA_SP2='0',QBH_SP2_DOWN_HVX='0',QBH_WIDE_SCORE='8',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_DENSE_R3='2' if mode=='R3R4' else '0',QBH_R3_OPT='3' if mode=='R3R4' else '0',QBH_DENSE_R4='4',QBH_R4_OPT='6',QBH_W4U8_DECODE_AV_REQUANT_ROWS='4')
    if poison:env['QBH_W4U8_DECODE_AV_PADDING_POISON']='1'
    fixture=read(FIXTURE);words=[0x51424556,2,repeat,110]
    for j in range(repeat):words += [j,3,43]+fixture['prompt_ids'][:64]+fixture['fixed'][:43]
    ef=d/'fixed-trajectory.bin'
    if not ef.exists():ef.write_bytes(struct.pack('<'+'I'*len(words),*words))
    er=root+'/'+tag+'-trajectory.bin';env['QBH_EVAL_FILE']=er
    if audit:env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=REMOTE+'/'+tag)
    cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+' '.join(shlex.quote(v) for v in argv)
    if not (d/'protocol.json').exists():save(d/'protocol.json',dict(command=cmd,arm=arm,layers=nl,repeat=repeat,audit=audit,seal_sha256=sha256(R/f'binaries-{binary_mode}-{nl}-a02/seal.json'),package_manifest_sha256=sha256(M/'R4'/'manifest.json')))
    if not (d/'exit.json').exists():
        adb('push',windows(ef),er)
        if audit:adb('shell','mkdir -p '+env['QBH_GENERATION_AUDIT_DIR'])
        z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);save(d/'exit.json',dict(returncode=z.returncode))
        if audit:adb('pull',env['QBH_GENERATION_AUDIT_DIR']+'/.',windows(d/'audit'))
    assert read(d/'exit.json')['returncode']==0,(tag,(d/'stderr.txt').read_text()[-1000:])
    rr=records((d/'stdout.txt').read_text())
    if not (d/'records.json').exists():save(d/'records.json',rr)
    pp=[x for x in rr if x.get('record')=='generation_profile'];ss=[x for x in rr if 'selected_logit_half_bits' in x and 'generation_step' in x]
    assert len(pp)==len(ss)==43*repeat,(tag,len(pp),len(ss))
    gold=read(R/('oracle-'+mode)/'summary.json');peak=exact=0;times={k:[] for k in ['prefill','decode']};fields={k:{f:[] for f in ['invocation_ticks','u8_attention_av_requant_ticks','u8_attention_av_hmx_ticks','o_projection_ticks','w4u8_gate_up_swiglu_worker_ticks','u8_attention_softmax_ticks','down_ticks','gate_up_ticks','activation_ticks','dense_r4_prepare_ticks','dense_r4_layout_ticks','dense_r4_finish_ticks','dense_r4_hmx_calls','dense_r3_total_calls','dense_r3_total_hmx_calls','qkv_projection_ticks','qk_norm_rope_ticks']} for k in times}
    for rep in range(repeat):
        for phase,lo,hi in [('prefill',0,1),('decode',1,43)]:
            wall=0;totals={k:0 for k in fields[phase]}
            for j in range(lo,hi):
                x,st=pp[rep*43+j],ss[rep*43+j];g=gold['heads'][str(nl)][j]
                assert (st['selected_token_id'],st['selected_logit_half_bits'])==(g['token'],g['code']),(tag,j,'head',st['selected_token_id'],g)
                assert x['dsp_status']==3 and x['numerical_status']==1
                assert x['vtcm_requested_bytes']==x['vtcm_acquired_bytes']==8388608 and x['vtcm_peak_plan_bytes']<=8388608
                assert x['intermediate_spill_fill_count']==x['intermediate_ddr_read_bytes']==x['intermediate_ddr_write_bytes']==0
                for i in range(nl):
                    l=x[f'slice_layer_{i}'];assert l['status']==3
                    assert l['hidden_ddr_read_bytes']==l['hidden_ddr_write_bytes']==l['layer_unattributed_ticks']==0
                    if audit:
                        assert l['output_hash']==gold['layer_hashes'][i][j],(tag,j,i,l['output_hash'],gold['layer_hashes'][i][j])
                        exact+=1
                total=sum(sum(x[k] for k in f) for _,f in MODULES)-x['generation_final_norm_ticks'];assert total==x['invocation_ticks']
                assert x['host_wall_ns']==st['host_wall_ns'] and x['host_wall_ns']/1000>=total/19.2
                wall+=x['host_wall_ns'];peak=max(peak,x['vtcm_peak_plan_bytes'])
                for k in totals:totals[k]+=x[k]
            times[phase].append(wall)
            for k,v in totals.items():fields[phase][k].append(v)
    result=dict(mode=mode,arm=arm,layers=nl,repeat=repeat,exact_layer_outputs=exact,boundaries=len(pp),peak=peak,times=times,fields=fields)
    save(d/'validated.json',result);print('PASS',tag,{k:float(np.mean(v)) for k,v in times.items()},flush=True);return result

def run():
 preflight()
 for mode in MODES:execute(mode,'warmup-'+mode)
 for stage_name,n in [('short',5),('formal',10)]:
  pairs=[]
  for i in range(n):pairs.append({a:execute(a,f'{stage_name}-{i:02d}-{a}',repeat=10) for a in (MODES if i%2==0 else MODES[::-1])})
  rng=np.random.default_rng(710071);ix=rng.integers(0,n,(20000,n));s={}
  for phase,tokens in [('prefill',64),('decode',42)]:
   a={m:np.array([np.mean(p[m]['times'][phase])/1e6 for p in pairs]) for m in MODES};x,y=a['R4'],a['R3R4']
   s[phase]={m:dict(wall_ms=float(v.mean()),tps=float(tokens*1000/v.mean())) for m,v in a.items()};s[phase]['r3r4_over_r4']=dict(wall_ratio=float(y.mean()/x.mean()),ci95=np.quantile(y[ix].mean(1)/x[ix].mean(1),[.025,.975]).tolist())
  save(R/(stage_name+'-summary.json'),s);print(stage_name,json.dumps(s),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--layers',type=int,default=16);ap.add_argument('--mode',default='A8');a=ap.parse_args()
 if a.action=='audit':preflight();execute(a.mode,f'audit{a.layers}-{a.mode}-a02',nl=a.layers,audit=True)
 else:globals()[a.action]()
