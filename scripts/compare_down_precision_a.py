#!/usr/bin/env python3
"""EXP-0308 same-binary, same-trace A8 versus uniform INT16 Down."""
import argparse,copy,json,os,shlex,shutil,struct,subprocess,sys
from pathlib import Path
import numpy as np
from profile_fp_islands import S as ROOT, adb, win as windows, sha as sha256
from reference_w4u8_hmx import load_qparams_bin,QPARAM_RECORD
from device_exp0284 import records
from summarize_exp0217 import normalized
from measure_exp0218 import LEDGER
import long_reference_exp0295 as ref
from verify_exp0167_generation import load_generation_qparams
SIZE=os.environ.get('QBH_QWEN_MODEL_SIZE','0.6B');assert SIZE in ['0.6B','1.7B']
H=1024 if SIZE=='0.6B' else 2048
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0308')/SIZE
M=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0308')/SIZE
PARENT=M.parent.parent/('exp-0304/int16' if SIZE=='0.6B' else 'exp0307/control-v2')
ORDINARY=M.parent.parent/'exp0298'/SIZE/'a8/native64-d42'
HIST=R.parent.parent/('exp0301' if SIZE=='0.6B' else 'exp0302')/SIZE/'a8'
BASE=HIST/'base.json';FIXTURE=HIST/'fixture.json'
REMOTE='/data/local/tmp/qwen3-block-htp/exp0308/'+SIZE
PACKAGE_REMOTE=REMOTE+'/models'
PARENT_REMOTE='/data/local/tmp/qwen3-block-htp/exp-0304/models/int16' if SIZE=='0.6B' else '/data/local/tmp/qwen3-block-htp/exp0307/models/control-v2'
MODES=['A8','INT16']
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p;p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def preflight():
 z=subprocess.check_output(['python3',str(ROOT)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],text=True);assert 'EXPERIMENT=EXP-0308' in z;print(z,flush=True)
def fnv(x):
 h=1469598103934665603
 for b in memoryview(np.ascontiguousarray(x,dtype='<f4')).cast('B'):h=((h^b)*1099511628211)&0xffffffffffffffff
 return f'{h:016x}'
def recover():
 preflight();R.mkdir(parents=True,exist_ok=True);M.mkdir(parents=True,exist_ok=True)
 if SIZE=='0.6B':
  sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools');from restore_verified_payload import run
  run(PARENT_REMOTE,PARENT/'manifest.json',M/'recovered',R/'recovery.json')
def prepare():
 preflight();R.mkdir(parents=True,exist_ok=True);M.mkdir(parents=True,exist_ok=True)
 parent=M/'recovered' if SIZE=='0.6B' else PARENT;old=read(parent/'manifest.json');mf=old['files'];om=read(ORDINARY/'manifest.json')['files'];missing=[n for n in mf if not (parent/n).exists()]
 assert all(n.endswith('.npy') or n.startswith('audit_hidden_') or n.startswith('chain_step') for n in missing),missing
 mf={n:v for n,v in mf.items() if n not in missing}
 for n,v in mf.items():assert sha256(parent/n)==v['sha256'],n
 save(R/'recovery-scope.json',dict(missing_not_runtime=missing,verified_files=len(mf),original_manifest_sha256=sha256(parent/'manifest.json')))
 for arm in MODES:
  p=M/arm;p.mkdir(exist_ok=False)
  for n in mf:
   src=parent/n;d=p/n;d.parent.mkdir(parents=True,exist_ok=True)
   if arm=='A8' and n.endswith('qparams_u8.bin') and n.startswith('layer'):
    assert sha256(ORDINARY/n)==om[n]['sha256'];a=load_qparams_bin(src);b=load_qparams_bin(ORDINARY/n)
    for k in a:
     if k!='middle':assert a[k]==b[k],(n,k,a[k],b[k])
    raw=(ORDINARY/n).read_bytes();mid=next(raw[i:i+QPARAM_RECORD.size] for i in range(0,len(raw),QPARAM_RECORD.size) if QPARAM_RECORD.unpack_from(raw,i)[0].split(b'\0')[0]==b'middle')
    raw=src.read_bytes();rows=[mid if QPARAM_RECORD.unpack_from(raw,i)[0].split(b'\0')[0]==b'middle' else raw[i:i+QPARAM_RECORD.size] for i in range(0,len(raw),QPARAM_RECORD.size)];d.write_bytes(b''.join(rows))
   elif arm=='A8' and n.endswith('silu_up_lut_u16.bin'):
    assert sha256(ORDINARY/n)==om[n]['sha256'];os.link(ORDINARY/n,d)
   else:os.link(src,d)
  save(p/'manifest.json',dict(experiment='EXP-0308',arm=arm,size=SIZE,parent_manifest_sha256=sha256(parent/'manifest.json'),ordinary_manifest_sha256=sha256(ORDINARY/'manifest.json'),files={n:dict(bytes=(p/n).stat().st_size,sha256=sha256(p/n)) for n in mf}))
 dif=[n for n in mf if sha256(M/'A8'/n)!=sha256(M/'INT16'/n)];assert len(dif)==56 and all(n.endswith(('qparams_u8.bin','silu_up_lut_u16.bin')) for n in dif)
 save(R/'payload-fairness.json',dict(different_files=dif,common_weights_and_other_metadata=True,only_middle_qparam_fields_changed=True,fixture_sha256=sha256(FIXTURE)))
def reference():
 preflight();fixture=read(FIXTURE);ids=fixture['prompt_ids'][:64]+fixture['fixed'][:42];assert len(ids)==106
 seeds=None
 if SIZE=='1.7B':
  f=R.parent.parent/'exp0305/prefix_kv_u8.bin';assert sha256(f)=='7683237318d42ac5cc80052fb53619205a3d82a0d1377bcbbaed78d7c7683b91';seeds=np.fromfile(f,'u1').reshape(28,2,8,128)
 for mode in MODES:
  p=M/mode;d=R/('oracle-'+mode);d.mkdir(exist_ok=True)
  if (d/'summary.json').exists():continue
  os.environ['QBH_SDK_REFERENCE_DIR']=str(d/'sdk-rsqrt');emb=np.memmap(p/'generation_embedding_weight_f16.bin','<f2','r',shape=(151936,H));x=np.asarray(emb[ids],dtype='f4')
  co=np.concatenate([np.fromfile(p/'rope_cos_f16.bin','<f2').reshape(64,128)]+[np.fromfile(p/f'generation_decode_rope_cos_{j:02d}_f16.bin','<f2').reshape(64,128)[:1] for j in range(42)])
  si=np.concatenate([np.fromfile(p/'rope_sin_f16.bin','<f2').reshape(64,128)]+[np.fromfile(p/f'generation_decode_rope_sin_{j:02d}_f16.bin','<f2').reshape(64,128)[:1] for j in range(42)])
  hashes=[];heads={}
  for i in range(28):
   f=d/f'l{i:02d}-output.npy'
   if f.exists():x=np.load(f)
   else:
    x,kv,diag=ref.layer(x,p/f'layer{i}',co,si,sp2=(mode=='INT16'),seed=None if seeds is None else seeds[i]);np.save(f,x)
   hashes.append([fnv(x[:64])]+[fnv(x[j:j+1]) for j in range(64,106)]);ref.weight.cache_clear()
   if i+1 in [1,3,28]:
    hf=d/f'head-{i+1}.json'
    if not hf.exists():
     q=load_generation_qparams(p/'generation_qparams_u8.bin');a=ref.norm(x[63:],np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2'),q['generation_final_norm_output']);y=ref.project(a,p,'generation_lm_head',151936,q['generation_final_norm_output'],q['generation_lm_head_output']);t=y.argmax(1);save(hf,[dict(token=int(k),code=int(y[j,k])) for j,k in enumerate(t)]);ref.weight.cache_clear()
    heads[str(i+1)]=read(hf)
   print('REFERENCE',SIZE,mode,i,flush=True)
  save(d/'summary.json',dict(layer_hashes=hashes,heads=heads,trajectory_sha256=sha256(FIXTURE),own_contract=True,quality_claim=False))
def deploy():
 preflight();parent=read(PARENT/'manifest.json')['files']
 for arm in MODES:
  p=M/arm;mf=read(p/'manifest.json')['files'];remote=PACKAGE_REMOTE+'/'+arm;names=list(mf);adb('shell','mkdir -p '+remote)
  for start in range(0,len(names),24):
   batch=names[start:start+24];cmd=[];copies=[]
   for n in batch:
    assert sha256(p/n)==mf[n]['sha256'];cmd.append('mkdir -p '+shlex.quote(str(Path(remote+'/'+n).parent)))
    if mf[n]['sha256']==parent[n]['sha256']:cmd.append('ln -s '+PARENT_REMOTE+'/'+n+' '+remote+'/'+n)
    else:copies.append(n)
   adb('shell',' && '.join(cmd))
   for n in copies:adb('push',windows(p/n),remote+'/'+n)
   lines=adb('shell','sha256sum '+' '.join(remote+'/'+n for n in batch)).stdout.splitlines();assert len(lines)==len(batch)
   for l in lines:h,n=l.split(None,1);assert h==mf[n.removeprefix(remote+'/')]['sha256']
  print('DEPLOYED',SIZE,arm,flush=True)
 save(R/'deployment.json',dict(all_payload_hashes_verified=True))
def stage():
 preflight();seal=read(ROOT/'build/qwen3-sp2-build-seal.json');assert not seal['fp_islands'] and seal['model_size']==SIZE;assert seal['source_head']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();cache=(ROOT/'hexagon_ReleaseG_toolv19_v79/CMakeCache.txt').read_text();nl=int(next(l.split('=',1)[1] for l in cache.splitlines() if l.startswith('QBH_EXP0257_LAYER_COUNT:STRING=')));d=R/f'binaries-I-{nl}';d.mkdir(exist_ok=False);remote=REMOTE+'/'+d.name;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha256(p)==h;shutil.copy2(p,d/p.name);adb('push',windows(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli');save(d/'seal.json',seal)
def execute(mode,tag,nl=28,repeat=1,audit=False,poison=False):
    assert mode in MODES, "Unconditional AV folding rejected by clipping contract"
    arm=mode
    d=R/tag
    if (d/'validated.json').exists():return read(d/'validated.json')
    d.mkdir(exist_ok=True);binary_mode='I';root=REMOTE+f'/binaries-{binary_mode}-{nl}';base=read(BASE)
    prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
    env=dict(t.split('=',1) for t in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=PACKAGE_REMOTE+'/'+arm
    for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT']:env.pop(k,None)
    env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_GENERATION_STEPS='43',QBH_GENERATION_EXPECTED_TOKENS='64',QBH_GENERATION_SEQUENCE='9',QBH_SP2='0' if mode=='A8' else '8',QBH_U8_PREFILL_OPT='3' if mode=='A8' else '0',QBH_PREFIX_KV='0' if SIZE=='0.6B' else '1',QBH_SP2_DOWN_HVX='0',QBH_WIDE_SCORE='7' if mode=='F' else '4',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_W4U8_DECODE_AV_REQUANT_ROWS='4')
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
    gold=read(R/('oracle-'+mode)/'summary.json');peak=exact=0;times={k:[] for k in ['prefill','decode']};fields={k:{f:[] for f in ['invocation_ticks','u8_attention_av_requant_ticks','u8_attention_av_hmx_ticks','o_projection_ticks','w4u8_gate_up_swiglu_worker_ticks','u8_attention_softmax_ticks','down_ticks','gate_up_ticks']} for k in times}
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
  pairs=[]
  for i in range(n):pairs.append({a:execute(a,f'{stage_name}-{i:02d}-{a}',repeat=10) for a in (MODES if i%2==0 else MODES[::-1])})
  rng=np.random.default_rng(680068);ix=rng.integers(0,n,(20000,n));s={}
  for phase,tokens in [('prefill',64),('decode',42)]:
   a={m:np.array([np.mean(p[m]['times'][phase])/1e6 for p in pairs]) for m in MODES};x,y=a['A8'],a['INT16']
   s[phase]={m:dict(wall_ms=float(v.mean()),tps=float(tokens*1000/v.mean())) for m,v in a.items()};s[phase]['int16_over_a8']=dict(wall_ratio=float(y.mean()/x.mean()),ci95=np.quantile(y[ix].mean(1)/x[ix].mean(1),[.025,.975]).tolist())
  save(R/(stage_name+'-summary.json'),s);print(stage_name,json.dumps(s),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--layers',type=int,default=28);ap.add_argument('--mode',default='A8');a=ap.parse_args()
 if a.action=='audit':preflight();execute(a.mode,f'audit{a.layers}-{a.mode}',nl=a.layers,audit=True)
 else:globals()[a.action]()
