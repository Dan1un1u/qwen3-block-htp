#!/usr/bin/env python3
"""L32-0065: production-scheduled F/I/M full-model paired ablations.
F/I: nonlinear arithmetic plus its necessary QDQ bundle.
I/M: only standalone AV RQ versus scale folding into O. Not all-QDQ attribution.
"""
import argparse, json, os, shlex, shutil, struct, subprocess, time
from pathlib import Path
import numpy as np
from run_llama32_layer import ROOT, adb, windows
from prototype_llama32_sp2 import preflight
from llama_reference import sha256
from llama_u8_reference import load_qparams_bin, project_w4u8, HmxU8Converter, _cached_w4_projection
from run_llama32_frontend import records
from report_llama32_pipeline_profile import MODULES
from probe_llama32_av_o_fullmodel import M, BASE, FIXTURE, fnv
import llama32_a8_fp32_reference as ref
import profile_fp_islands as fp
from llama32_fp_softmax_reference import floating_attention
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0065')
OLD=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0062')
REMOTE='/data/local/tmp/llama32-htp/l32-0065'
PACKAGE_REMOTE='/data/local/tmp/llama32-htp/l32-0062'
MODES=['F','I','M']
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
 p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify():
 preflight();R.mkdir(parents=True,exist_ok=True);checks={}
 for arm in ['control','folded']:
  p=M/arm;mf=read(p/'manifest.json')['files']
  for n,v in mf.items():assert sha256(p/n)==v['sha256'],n
  names=list(mf)
  for i in range(0,len(names),32):
   out=adb('shell','sha256sum '+' '.join(shlex.quote(PACKAGE_REMOTE+'/'+arm+'/'+n) for n in names[i:i+32])).stdout
   lines=out.splitlines();assert len(lines)==len(names[i:i+32])
   for line in lines:
    h,n=line.split(None,1);assert h==mf[n.removeprefix(PACKAGE_REMOTE+'/'+arm+'/')]['sha256'],n
  checks[arm]=dict(files=len(mf),manifest_sha256=sha256(p/'manifest.json'))
 save(R/'verified-packages.json',checks);print('VERIFIED packages',checks,flush=True)
def stage():
 preflight();seal=read(ROOT/'build/llama-build-seal.json');mode='F' if seal['fp_islands'] else 'I';nl=seal['layer_count']
 assert seal['source_head']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 d=R/f'binaries-{mode}-{nl}';d.mkdir(exist_ok=False);remote=REMOTE+'/'+d.name
 adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha256(p)==h;shutil.copy2(p,d/p.name);adb('push',windows(p),remote+'/'+p.name)
  assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli '+remote+'/llama_sp2_cli');save(d/'seal.json',seal)
def head(x,p):
 cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');q=load_qparams_bin(p/'generation_qparams_u8.bin')
 g=np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2');a=ref.norm(x[63:],g,q['generation_final_norm_output'])
 y=project_w4u8(a,p,'generation_lm_head',128256,2048,q['generation_final_norm_output'],q['generation_lm_head_output'],cv)
 ids=y.argmax(1);v=[dict(token=int(t),code=int(y[j,t])) for j,t in enumerate(ids)];_cached_w4_projection.cache_clear();return v
def references():
 preflight()
 for mode,arm in [('I','control'),('M','folded')]:
  dest=R/('oracle-'+mode)/'summary.json'
  if dest.exists():continue
  old=OLD/('oracle-'+arm);gold=read(old/'summary.json');gold['heads']['1']=head(np.load(old/'l00-output.npy'),M/arm)
  gold['reused_reference_sha256']=sha256(old/'summary.json');save(dest,gold)
 p=M/'control';out=R/'oracle-F';out.mkdir(exist_ok=True)
 if (out/'summary.json').exists():return
 fixture=read(FIXTURE);ids=fixture['prompt_ids'][:64]+fixture['fixed'][:42]
 emb=np.memmap(p/'generation_embedding_weight_f16.bin','<f2','r',shape=(128256,2048));x=np.asarray(emb[ids],dtype='f4')
 cs=[np.fromfile(p/'rope_cos_f16.bin','<f2').reshape(64,64)];ss=[np.fromfile(p/'rope_sin_f16.bin','<f2').reshape(64,64)]
 for j in range(42):
  cs.append(np.fromfile(p/f'generation_decode_rope_cos_{j:02d}_f16.bin','<f2').reshape(64,64)[:1]);ss.append(np.fromfile(p/f'generation_decode_rope_sin_{j:02d}_f16.bin','<f2').reshape(64,64)[:1])
 co=np.concatenate(cs);si=np.concatenate(ss);hashes=[];heads={}
 ref.exact_attention_dynamic=lambda q,k,v,past,cfg,cv,divide:floating_attention(q,k,v,past,cfg)
 for i in range(16):
  lp=p/f'layer{i}';q=load_qparams_bin(lp/'qparams_u8.bin');table,_=fp.fp_table(q);ref.table=lambda _,table=table:table
  f=out/f'l{i:02d}-output.npy'
  if f.exists():x=np.load(f)
  else:
   x,kv,diag=ref.layer(x,lp,q,co,si,None,sp2=True);np.save(f,x)
   for k,v in zip(['k','v'],kv):np.save(out/f'l{i:02d}-{k}.npy',v)
  hashes.append([fnv(x[:64])]+[fnv(x[j:j+1]) for j in range(64,106)]);_cached_w4_projection.cache_clear()
  if i+1 in [1,3,16]:
   hf=out/f'head-{i+1}.json'
   if not hf.exists():save(hf,head(x,p))
   heads[str(i+1)]=read(hf)
  print('REFERENCE F',i,flush=True)
 save(out/'summary.json',dict(layer_hashes=hashes,heads=heads,trajectory_sha256=sha256(FIXTURE),own_contract=True,quality_claim=False))
def component():
 # The FP kernel is unchanged; recheck exhaustive Gate/Up pairs on current binary.
 fp.R=R/'component-F';fp.P=M/'control';fp.REMOTE=REMOTE
 code=fp.component.__code__
 import types
 constants=tuple(c.replace('/binaries-16','/binaries-F-16') if isinstance(c,str) else c for c in code.co_consts)
 types.FunctionType(code.replace(co_consts=constants),fp.__dict__)()

def execute(mode,tag,nl=16,repeat=1,audit=False,poison=False):
    arm="folded" if mode=="M" else "control"
    d=R/tag
    if (d/'validated.json').exists():return read(d/'validated.json')
    d.mkdir(exist_ok=True);binary_mode='F' if mode=='F' else 'I';root=REMOTE+f'/binaries-{binary_mode}-{nl}';base=read(BASE)
    prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
    env=dict(t.split('=',1) for t in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=PACKAGE_REMOTE+'/'+arm
    for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT']:env.pop(k,None)
    env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_GENERATION_STEPS='43',QBH_GENERATION_EXPECTED_TOKENS='64',QBH_GENERATION_SEQUENCE='9',QBH_LLAMA_SP2='8',QBH_SP2_DOWN_HVX='0',QBH_WIDE_SCORE='7' if mode=='F' else '8',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_W4U8_DECODE_AV_REQUANT_ROWS='4')
    if poison:env['QBH_W4U8_DECODE_AV_PADDING_POISON']='1'
    fixture=read(FIXTURE);words=[0x51424556,2,repeat,110]
    for j in range(repeat):words += [j,3,43]+fixture['prompt_ids'][:64]+fixture['fixed'][:43]
    ef=d/'fixed-trajectory.bin'
    if not ef.exists():ef.write_bytes(struct.pack('<'+'I'*len(words),*words))
    er=root+'/'+tag+'-trajectory.bin';env['QBH_EVAL_FILE']=er
    if audit:env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=REMOTE+'/'+tag)
    cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+' '.join(shlex.quote(v) for v in argv)
    if not (d/'protocol.json').exists():save(d/'protocol.json',dict(command=cmd,arm=arm,layers=nl,repeat=repeat,audit=audit,seal_sha256=sha256(R/f'binaries-{binary_mode}-{nl}/seal.json'),package_manifest_sha256=sha256(M/arm/'manifest.json')))
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
    gold=read(R/('oracle-'+mode)/'summary.json');peak=exact=0;times={k:[] for k in ['prefill','decode']};fields={k:{f:[] for f in ['invocation_ticks','u8_attention_av_requant_ticks','u8_attention_av_hmx_ticks','o_projection_ticks','gate_up_swiglu_ticks','u8_attention_softmax_ticks']} for k in times}
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
                for k in totals:totals[k]+=x.get(k,0)
            times[phase].append(wall)
            for k,v in totals.items():fields[phase][k].append(v)
    result=dict(mode=mode,arm=arm,layers=nl,repeat=repeat,exact_layer_outputs=exact,boundaries=len(pp),peak=peak,times=times,fields=fields)
    save(d/'validated.json',result);print('PASS',tag,{k:float(np.mean(v)) for k,v in times.items()},flush=True);return result

def run():
 preflight()
 for mode in MODES:execute(mode,'warmup-'+mode)
 for stage_name,n in [('short',5),('formal',10)]:
  rounds=[]
  for i in range(n):
   order=MODES[i%3:]+MODES[:i%3]
   if i%2:order=order[::-1]
   rounds.append({m:execute(m,f'{stage_name}-{i:02d}-{m}',repeat=10) for m in order})
  rng=np.random.default_rng(650065);ix=rng.integers(0,n,(20000,n));s={}
  for phase,tokens in [('prefill',64),('decode',42)]:
   a={m:np.array([np.mean(r[m]['times'][phase])/1e6 for r in rounds]) for m in MODES}
   s[phase]={m:dict(wall_ms=float(x.mean()),tps=float(tokens*1000/x.mean())) for m,x in a.items()}
   for ctl,opt in [('F','I'),('I','M'),('F','M')]:
    x,y=a[ctl],a[opt];s[phase][ctl+'->'+opt]=dict(wall_ratio=float(y.mean()/x.mean()),ci95=np.quantile(y[ix].mean(1)/x[ix].mean(1),[.025,.975]).tolist(),wall_reduction_pct=float((1-y.mean()/x.mean())*100))
  f=R/(stage_name+'-summary.json')
  if not f.exists():save(f,s)
  print(stage_name,json.dumps(s),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--layers',type=int,default=16);ap.add_argument('--mode',default='I');ap.add_argument('--tag');a=ap.parse_args()
 if a.action=='audit':
  preflight();execute(a.mode,a.tag or f'audit{a.layers}-{a.mode}',nl=a.layers,audit=True)
 elif a.action=='poison':
  preflight();execute(a.mode,a.tag or f'poison{a.layers}-{a.mode}',nl=a.layers,audit=True,poison=True)
 else:globals()[a.action]()
