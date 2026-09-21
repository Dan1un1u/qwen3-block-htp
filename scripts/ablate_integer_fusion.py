#!/usr/bin/env python3
"""EXP0307 matched production pipeline: FP nonlinear / integer / AV folded.
No global QDQ-only claim. Original SP2 evidence is not relabeled as INT16.
"""
import argparse, json, os, shlex, shutil, struct, subprocess, copy
from pathlib import Path
import numpy as np
from profile_fp_islands import S as ROOT, adb, win as windows, sha as sha256, fp_table
from reference_w4u8_hmx import load_qparams_bin, QPARAM_RECORD, unpack_w4_codes
from device_exp0284 import records
from summarize_exp0217 import normalized
from measure_exp0218 import LEDGER
from verify_exp0167_generation import load_generation_qparams
import reference_fp_islands as ref
import reference_exp0284 as intref
import profile_fp_islands as oldfp
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0307')
M=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0307')
PARENT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0305/recovered-int16')
BASE=R.parent/'exp0284/fixed-aux-INT16-r1/protocol.json'
FIXTURE=R/'fixture.json'
REMOTE='/data/local/tmp/qwen3-block-htp/exp0307'
PACKAGE_REMOTE=REMOTE+'/models'
MODES=['F','I'] # M rejected: original AV saturation is active under this package.
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
 p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def preflight():
 z=subprocess.check_output(['python3',str(ROOT)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],text=True)
 assert 'EXPERIMENT=EXP-0307' in z;print(z,flush=True)
def fnv(x):
 h=1469598103934665603
 for b in memoryview(np.ascontiguousarray(x,dtype='<f4')).cast('B'):h=((h^b)*1099511628211)&0xffffffffffffffff
 return f'{h:016x}'
def prepare():
 preflight();R.mkdir(parents=True,exist_ok=True);M.mkdir(parents=True,exist_ok=True)
 old=read(PARENT/'manifest.json')
 retired=set(read(R.parent/'exp0305/postflight-verification.json')['retired_old_chain_arrays_not_used'])
 missing={n for n in old['files'] if not (PARENT/n).exists()};assert missing==retired
 old['files']={n:v for n,v in old['files'].items() if n not in retired};assert len(old['files'])==909
 for n,v in old['files'].items():assert sha256(PARENT/n)==v['sha256'],n
 save(R/'recovery-scope.json',dict(retired_not_used=sorted(retired),verified_payload_count=len(old['files']),original_manifest_sha256=sha256(PARENT/'manifest.json')))
 for arm in ['control','folded']:
  p=M/arm;p.mkdir(exist_ok=False)
  for n in old['files']:
   d=p/n;d.parent.mkdir(parents=True,exist_ok=True)
   if arm=='folded' and n.endswith(('qparams_u8.bin','attention_config_all_groups.bin')):shutil.copy2(PARENT/n,d)
   else:os.link(PARENT/n,d)
 folds=[]
 for i in range(28):
  p=M/'control'/f'layer{i}';d=M/'folded'/f'layer{i}';cfg=np.fromfile(p/'attention_config_all_groups.bin','<i4').reshape(8,15)
  assert len(set((int(c[-1]),int(c[8]),int(c[-2])) for c in cfg))==1
  m=int(cfg[0,-1]);zp=int(cfg[0,8]);q=load_qparams_bin(p/'qparams_u8.bin');assert m>=1 and q['attention_concat']['zero_point']==zp
  if m!=1:
   raw=(d/'qparams_u8.bin').read_bytes();rows=[]
   for off in range(0,len(raw),QPARAM_RECORD.size):
    rec=list(QPARAM_RECORD.unpack_from(raw,off))
    if rec[0].split(b'\0')[0]==b'attention_concat':rec[1]=float(np.float32(rec[1]*m));rec[2]=128;rec[3]=-128*rec[1];rec[4]=127*rec[1]
    rows.append(QPARAM_RECORD.pack(*rec))
   (d/'qparams_u8.bin').write_bytes(b''.join(rows));cfg[:,-1]=1;cfg[:,8]=128;cfg.astype('<i4').tofile(d/'attention_config_all_groups.bin')
  w=unpack_w4_codes(p,'o',2048,2048).astype('i8');b24=int(max(255*np.maximum(w,0).sum(1).max(),-255*np.minimum(w,0).sum(1).min()));b32=int(max(zp,255-zp,128)*np.abs(w).sum(1).max());assert b24<2**23 and b32<2**31
  folds.append(dict(layer=i,multiplier=m,zero_point=zp,bound24=b24,bound32=b32,changed=m!=1))
 for arm in ['control','folded']:
  p=M/arm;save(p/'manifest.json',dict(experiment='EXP-0307',arm=arm,parent_manifest_sha256=sha256(PARENT/'manifest.json'),files={str(f.relative_to(p)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in p.rglob('*') if f.is_file() and f.name!='manifest.json'}))
 save(R/'folding-contracts.json',folds)
 fixture()
 print('PREPARED',len(old['files']),'payloads',flush=True)
def fixture():
 preflight();ids=read(R.parent/'exp0284/fixed_tokens.json')['ids'];n=len(ids);assert n>0
 ids=ids+[ids[-1]]*max(0,64-n)
 save(FIXTURE,dict(prompt_ids=np.fromfile(PARENT/'generation_prompt_token_ids_u32.bin','<u4').tolist(),fixed=ids,original_token_count=n,extension='repeat last frozen token to64; fixed before any timing; not generation quality'))
def deploy():
 preflight();parent=read(PARENT/'manifest.json')['files'];parentremote='/data/local/tmp/qwen3-block-htp/exp0284-models/full-int16'
 for arm in ['control','folded']:
  p=M/arm;mf=read(p/'manifest.json')['files'];names=list(mf);remote=PACKAGE_REMOTE+'/'+arm
  adb('shell','mkdir -p '+remote)
  for start in range(0,len(names),24):
   batch=names[start:start+24];cmd=[];copies=[]
   for n in batch:
    assert sha256(p/n)==mf[n]['sha256'];cmd.append('mkdir -p '+shlex.quote(str(Path(remote+'/'+n).parent)))
    if mf[n]['sha256']==parent[n]['sha256']:cmd.append('ln -s '+shlex.quote(parentremote+'/'+n)+' '+shlex.quote(remote+'/'+n))
    else:copies.append(n)
   adb('shell',' && '.join(cmd))
   for n in copies:adb('push',windows(p/n),remote+'/'+n)
   out=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in batch)).stdout.splitlines();assert len(out)==len(batch)
   for line in out:
    h,n=line.split(None,1);assert h==mf[n.removeprefix(remote+'/')]['sha256'],n
  print('DEPLOYED',arm,flush=True)
 save(R/'verified-packages.json',dict(manifests={a:sha256(M/a/'manifest.json') for a in ['control','folded']},local_and_device=True))
def stage(nl):
 preflight();seal=read(ROOT/'build/qwen3-sp2-build-seal.json');mode='F' if seal['fp_islands'] else 'I'
 assert seal['source_head']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 cache=(ROOT/'hexagon_ReleaseG_toolv19_v79/CMakeCache.txt').read_text();assert f'QBH_EXP0257_LAYER_COUNT:STRING={nl}\n' in cache and 'QBH_QWEN_MODEL_SIZE:STRING=1.7B' in cache
 d=R/f'binaries-{mode}-{nl}';d.mkdir(exist_ok=False);remote=REMOTE+'/'+d.name;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha256(p)==h;shutil.copy2(p,d/p.name);adb('push',windows(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli '+remote+'/llama_sp2_cli');save(d/'seal.json',seal)
def component():
 oldfp.R=R/'component-F';oldfp.P=M/'control';oldfp.REMOTE=REMOTE;oldfp.preflight=preflight
 import types
 code=oldfp.component.__code__;const=tuple(c.replace('/binaries-28','/binaries-F-28') if isinstance(c,str) else c for c in code.co_consts)
 types.FunctionType(code.replace(co_consts=const),oldfp.__dict__)()
def references():
 preflight();fixture=read(FIXTURE);ids=fixture['prompt_ids'][:64]+fixture['fixed'][:42]
 seedfile=R.parent/'exp0305/prefix_kv_u8.bin';assert sha256(seedfile)=='7683237318d42ac5cc80052fb53619205a3d82a0d1377bcbbaed78d7c7683b91';seeds=np.fromfile(seedfile,'u1').reshape(28,2,8,128)
 float_oracle=ref.numpy_oracle
 for mode in MODES:
  out=R/('oracle-'+mode);out.mkdir(exist_ok=True)
  if (out/'summary.json').exists():continue
  p=M/('folded' if mode=='M' else 'control');emb=np.memmap(p/'generation_embedding_weight_f16.bin','<f2','r',shape=(151936,2048));x=np.asarray(emb[ids],dtype='f4')
  co=np.concatenate([np.fromfile(p/'rope_cos_f16.bin','<f2').reshape(64,128)]+[np.fromfile(p/f'generation_decode_rope_cos_{j:02d}_f16.bin','<f2').reshape(64,128)[:1] for j in range(42)])
  si=np.concatenate([np.fromfile(p/'rope_sin_f16.bin','<f2').reshape(64,128)]+[np.fromfile(p/f'generation_decode_rope_sin_{j:02d}_f16.bin','<f2').reshape(64,128)[:1] for j in range(42)])
  ref.numpy_oracle=float_oracle if mode=='F' else intref.numpy_oracle;hashes=[];heads={};clipping=[]
  for i in range(28):
   lp=p/f'layer{i}';q=load_qparams_bin(lp/'qparams_u8.bin');tab=fp_table(q)[0] if mode=='F' else np.fromfile(lp/'silu_up_lut_u16.bin','<u2').reshape(256,256).astype('i4')-32768;ref.fp_table=lambda _,tab=tab:(tab,None)
   f=out/f'l{i:02d}-output.npy';cf=out/f'l{i:02d}-clipping.json'
   if f.exists():x=np.load(f)
   else:
    x,kv,diag=ref.layer(x,lp,co,si,seed=seeds[i]);np.save(f,x)
    for k,v in zip(['k','v'],kv):np.save(out/f'l{i:02d}-{k}.npy',v)
    if mode=='M':
     cfg=np.fromfile(M/'control'/f'layer{i}/attention_config_all_groups.bin','<i4').reshape(8,15);m=int(cfg[0,-1]);zp=int(cfg[0,8]);av=diag['attention'].astype('i4');lift=av if m==1 else (av-128)*m+zp
     save(cf,dict(layer=i,elements=lift.size,saturated=int(np.count_nonzero((lift<0)|(lift>255)))))
   hashes.append([fnv(x[:64])]+[fnv(x[j:j+1]) for j in range(64,106)]);ref.weight.cache_clear()
   if cf.exists():clipping.append(read(cf))
   if i+1 in [1,3,28]:
    hf=out/f'head-{i+1}.json'
    if not hf.exists():
     qg=load_generation_qparams(p/'generation_qparams_u8.bin');a=ref.norm(x[63:],np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2'),qg['generation_final_norm_output']);y=ref.project(a,p,'generation_lm_head',151936,qg['generation_final_norm_output'],qg['generation_lm_head_output']);tid=y.argmax(1);save(hf,[dict(token=int(t),code=int(y[j,t])) for j,t in enumerate(tid)]);ref.weight.cache_clear()
    heads[str(i+1)]=read(hf)
   print('REFERENCE',mode,i,flush=True)
  save(out/'summary.json',dict(layer_hashes=hashes,heads=heads,clipping=clipping,trajectory_sha256=sha256(FIXTURE),own_contract=True,quality_claim=False))
  if mode=='M':assert all(c['saturated']==0 for c in clipping),clipping

def execute(mode,tag,nl=28,repeat=1,audit=False,poison=False):
    assert mode in MODES, "Unconditional AV folding rejected by clipping contract"
    arm="control"
    d=R/tag
    if (d/'validated.json').exists():return read(d/'validated.json')
    d.mkdir(exist_ok=True);binary_mode='F' if mode=='F' else 'I';root=REMOTE+f'/binaries-{binary_mode}-{nl}';base=read(BASE)
    prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
    env=dict(t.split('=',1) for t in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=PACKAGE_REMOTE+'/'+arm
    for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT']:env.pop(k,None)
    env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_GENERATION_STEPS='43',QBH_GENERATION_EXPECTED_TOKENS='64',QBH_GENERATION_SEQUENCE='9',QBH_SP2='8',QBH_SP2_DOWN_HVX='0',QBH_WIDE_SCORE='7' if mode=='F' else '8',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_W4U8_DECODE_AV_REQUANT_ROWS='4')
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
    rr=records(d/'stdout.txt')
    if not (d/'records.json').exists():save(d/'records.json',rr)
    pp=[x for x in rr if x.get('record')=='generation_profile'];ss=[x for x in rr if 'selected_logit_half_bits' in x and 'generation_step' in x]
    assert len(pp)==len(ss)==43*repeat,(tag,len(pp),len(ss))
    gold=read(R/('oracle-'+mode)/'summary.json');peak=exact=0;times={k:[] for k in ['prefill','decode']};fields={k:{f:[] for f in ['invocation_ticks','u8_attention_av_requant_ticks','u8_attention_av_hmx_ticks','o_projection_ticks','w4u8_gate_up_swiglu_worker_ticks','u8_attention_softmax_ticks']} for k in times}
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
                z=normalized([x]);total=sum(z[k] for _,k in LEDGER);assert total==x['invocation_ticks']
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
  rounds=[]
  for i in range(n):
   order=MODES[i%len(MODES):]+MODES[:i%len(MODES)]
   if i%2:order=order[::-1]
   rounds.append({m:execute(m,f'{stage_name}-{i:02d}-{m}',repeat=10) for m in order})
  rng=np.random.default_rng(650065);ix=rng.integers(0,n,(20000,n));s={}
  for phase,tokens in [('prefill',64),('decode',42)]:
   a={m:np.array([np.mean(r[m]['times'][phase])/1e6 for r in rounds]) for m in MODES}
   s[phase]={m:dict(wall_ms=float(x.mean()),tps=float(tokens*1000/x.mean())) for m,x in a.items()}
   for ctl,opt in [('F','I')]:
    x,y=a[ctl],a[opt];s[phase][ctl+'->'+opt]=dict(wall_ratio=float(y.mean()/x.mean()),ci95=np.quantile(y[ix].mean(1)/x[ix].mean(1),[.025,.975]).tolist(),wall_reduction_pct=float((1-y.mean()/x.mean())*100))
  f=R/(stage_name+'-summary.json')
  if not f.exists():save(f,s)
  print(stage_name,json.dumps(s),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--layers',type=int,default=28);ap.add_argument('--mode',default='I');ap.add_argument('--tag');a=ap.parse_args()
 if a.action=='audit':
  preflight();execute(a.mode,a.tag or f'audit{a.layers}-{a.mode}',nl=a.layers,audit=True)
 elif a.action=='poison':
  preflight();execute(a.mode,a.tag or f'poison{a.layers}-{a.mode}',nl=a.layers,audit=True,poison=True)
 elif a.action=='stage':stage(a.layers)
 else:globals()[a.action]()
