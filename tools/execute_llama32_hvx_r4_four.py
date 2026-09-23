#!/usr/bin/env python3
"""L32-0072: same binary, fixed trace, full-model ordinary A8 versus online HVX R4."""
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
SIZE=os.environ.get('QBH_LLAMA_MODEL_SIZE','1B')
assert SIZE in ['1B','3B']
H,D,NL=(2048,64,16) if SIZE=='1B' else (3072,128,28)
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0072')/SIZE
M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0072')/SIZE
PARENT=M.parents[1]/('l32-0070/A8' if SIZE=='1B' else 'l32-0066/folded')
ORDINARY=M.parents[1]/'l32-0054/3B/a8/native64-d42'
BASE=R.parents[1]/('l32-0060/1B/sp2/base.json' if SIZE=='1B' else 'l32-0064/base.json')
FIXTURE=R.parents[1]/('l32-0057/1B/sp2/fixture.json' if SIZE=='1B' else 'l32-0066/fixture.json')
REMOTE='/data/local/tmp/llama32-htp/l32-0072/'+SIZE
PACKAGE_REMOTE=REMOTE+'/models'
PARENT_REMOTE='/data/local/tmp/llama32-htp/'+('l32-0070/models/A8' if SIZE=='1B' else 'l32-0066/folded')
if SIZE=='3B':
 import llama32_3b_a8_reference as ref
 def get_head(x,p):
  cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');q=load_qparams_bin(p/'generation_qparams_u8.bin')
  gamma=np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2');act=ref.norm(x[63:],gamma,q['generation_final_norm_output'])
  logits=project_w4u8(act,p,'generation_lm_head',128256,H,q['generation_final_norm_output'],q['generation_lm_head_output'],cv)
  tokens=logits.argmax(1);v=[dict(token=int(t),code=int(logits[j,t])) for j,t in enumerate(tokens)];_cached_w4_projection.cache_clear();return v
MODES=['A8','R4']
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p;p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def preflight():
 z=subprocess.check_output(['python3',str(ROOT)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],text=True);assert 'ACTIVE_EXPERIMENT=L32-0072' in z;print(z,flush=True)
def fnv(x):
 h=1469598103934665603
 for b in memoryview(np.ascontiguousarray(x,dtype='<f4')).cast('B'):h=((h^b)*1099511628211)&0xffffffffffffffff
 return f'{h:016x}'
def prepare():
 preflight();import torch
 from safetensors import safe_open
 from quantize_llama32 import pack
 from export_llama32_u8 import write_qparams
 from llama_u8_reference import unpack_w4_codes
 torch.set_num_threads(4);R.mkdir(exist_ok=True,parents=True);M.mkdir(exist_ok=True,parents=True)
 mf=read(PARENT/'manifest.json')['files']
 for n,v in mf.items():assert sha256(PARENT/n)==v['sha256'],n
 changed=set()
 for i in range(NL):
  for name in ['down_weight_w4_hmx.bin','down_weight_w4_scale_f32.bin','silu_up_lut_u16.bin','qparams_u8.bin']:changed.add(f'layer{i}/'+name)
 for arm in MODES:
  dst=M/arm;dst.mkdir(exist_ok=False)
  for n in mf:
   if arm=='R4' and n in changed:continue
   d=dst/n;d.parent.mkdir(parents=True,exist_ok=True);os.link(PARENT/n,d)
 original=Path('/mnt/d/llm_exp/models/llama3.2-'+SIZE+'-Instruct-origin');folds=[]
 if SIZE=='1B':
  assert sha256(original/'model.safetensors')==EXPECTED['model.safetensors']
  original_sha=EXPECTED['model.safetensors']
 else:
  from prepare_llama32_3b import original as verify_original
  verify_original();original_sha=sha256(original/'model.safetensors.index.json')
 def tensor(name):
  file='model.safetensors' if SIZE=='1B' else read(original/'model.safetensors.index.json')['weight_map'][name]
  with safe_open(original/file,framework='pt',device='cpu') as f:return f.get_tensor(name)
 # A8 control reuses folded AV and latest common optimizations; restore only the ordinary middle contract.
 if SIZE=='3B':
  om=read(ORDINARY/'manifest.json')['files']
  for i in range(NL):
   for name in ['qparams_u8.bin','silu_up_lut_u16.bin']:
    n=f'layer{i}/'+name;assert sha256(ORDINARY/n)==om[n]['sha256']
   q=load_qparams_bin(M/'A8'/f'layer{i}/qparams_u8.bin');oq=load_qparams_bin(ORDINARY/f'layer{i}/qparams_u8.bin');q['middle']=oq['middle']
   dst=M/'A8'/f'layer{i}';(dst/'qparams_u8.bin').unlink();write_qparams(dst/'qparams_u8.bin',q)
   (dst/'silu_up_lut_u16.bin').unlink();os.link(ORDINARY/f'layer{i}/silu_up_lut_u16.bin',dst/'silu_up_lut_u16.bin')
 def had(x):
  x=x.clone()
  for h in [1<<j for j in range(13)]:
   v=x.reshape(x.shape[0],-1,2*h);lo=v[:,:,:h].clone();hi=v[:,:,h:].clone();v[:,:,:h]=lo+hi;v[:,:,h:]=lo-hi
  return x/np.sqrt(8192)
 if True:
  for i in range(NL):
   dst=M/'R4'/f'layer{i}';dst.mkdir(exist_ok=True);w=tensor(f'model.layers.{i}.mlp.down_proj.weight').double();wr=had(w)
   torch.manual_seed(700070+i);x=torch.randn(3,8192,dtype=torch.float64);lhs=x@w.T;rhs=had(x)@wr.T;err=float(torch.linalg.vector_norm(lhs-rhs)/torch.linalg.vector_norm(lhs));assert err<2e-12
   wf=wr.float();scale=wf.abs().amax(1).clamp_min(1e-8)/7;code=(wf/scale[:,None]).round().clamp(-7,7).to(torch.int8)
   pack(code).tofile(dst/'down_weight_w4_hmx.bin');scale.numpy().astype('<f4').tofile(dst/'down_weight_w4_scale_f32.bin');assert np.array_equal(code.numpy(),unpack_w4_codes(dst,'down',H,8192))
   codes=code.numpy().astype('i8');lo=int(255*np.minimum(codes,0).sum(1).min());hi=int(255*np.maximum(codes,0).sum(1).max());assert lo>=-8388608 and hi<=8388607
   q=load_qparams_bin(M/'A8'/f'layer{i}/qparams_u8.bin');alpha=max(abs(q['middle']['minimum']),abs(q['middle']['maximum']));sc=float(np.float32(alpha/127));q['middle'].update(scale=sc,zero_point=128,minimum=-128*sc,maximum=127*sc);write_qparams(dst/'qparams_u8.bin',q)
   g=(np.arange(256,dtype='f8')-q['gate']['zero_point'])*q['gate']['scale'];u=(np.arange(256,dtype='f8')-q['up']['zero_point'])*q['up']['scale'];lut=((g/(1+np.exp(-np.clip(g,-700,700))))[:,None]*u[None,:]).astype('<f2');assert np.isfinite(lut).all();lut.tofile(dst/'silu_up_lut_u16.bin')
   folds.append(dict(layer=i,ideal_invariance_relative_l2=err,partial24_bounds=[lo,hi],middle=q['middle'],method='original BF16 -> FP64 normalized Sylvester H8192 -> FP32 per-output RTN W4[-7,7]',fp16_lut_finite=True));print('FOLD',i,err,flush=True)
   del w,wr,wf,scale,code,codes
 for arm in MODES:
  p=M/arm;save(p/'manifest.json',dict(experiment='L32-0072',arm=arm,parent_manifest_sha256=sha256(PARENT/'manifest.json'),files={n:dict(bytes=(p/n).stat().st_size,sha256=sha256(p/n)) for n in mf}))
 save(R/'FOLD_VALIDATION.json',dict(original=str(original),original_sha256=original_sha,layers=folds,changed_files=sorted(changed),range_selection='same original absolute maximum, symmetric U8; not fitted on eval or calibrated for quality',other_weights_unchanged=True))
def reference():
 preflight();mode=os.environ.get('R4_ARM','R4');p=M/mode;d=R/('oracle-'+mode);d.mkdir(exist_ok=True);fixture=read(FIXTURE);ids=fixture['prompt_ids'][:64]+fixture['fixed'][:42]
 emb=np.memmap(p/'generation_embedding_weight_f16.bin','<f2','r',shape=(128256,H));x=np.asarray(emb[ids],dtype='f4')
 co=np.concatenate([np.fromfile(p/'rope_cos_f16.bin','<f2').reshape(64,D)]+[np.fromfile(p/f'generation_decode_rope_cos_{j:02d}_f16.bin','<f2').reshape(64,D)[:1] for j in range(42)])
 si=np.concatenate([np.fromfile(p/'rope_sin_f16.bin','<f2').reshape(64,D)]+[np.fromfile(p/f'generation_decode_rope_sin_{j:02d}_f16.bin','<f2').reshape(64,D)[:1] for j in range(42)])
 hashes=[];heads={}
 for i in range(NL):
  lp=p/f'layer{i}';q=load_qparams_bin(lp/'qparams_u8.bin');cfg=configs(lp);ref.configs=lambda _,cfg=cfg:cfg
  f=d/f'l{i:02d}-output.npy'
  if f.exists():x=np.load(f)
  else:
   x,kv,diag=ref.layer(x,lp,q,co,si,None,**({'sp2':False} if SIZE=='1B' else {}),r4=mode=='R4');assert np.isfinite(x).all();np.save(f,x)
   if i==0:
    for k in ['gate','up','middle','down']:np.save(d/(k+'.npy'),diag[k])
  hashes.append([fnv(x[:64])]+[fnv(x[j:j+1]) for j in range(64,106)]);_cached_w4_projection.cache_clear()
  if i+1 in [1,3,NL]:
   hf=d/f'head-{i+1}.json'
   if not hf.exists():save(hf,get_head(x,p))
   heads[str(i+1)]=read(hf)
  print('REFERENCE',mode,i,flush=True)
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
  print('DEPLOYED',arm,flush=True)
 save(R/'deployment.json',dict(all_payload_hashes_verified=True))
def stage():
 preflight();seal=read(ROOT/'build/llama-build-seal.json');assert not seal['fp_islands'] and seal['model_size']==SIZE;assert seal['source_head']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();nl=int(seal['layer_count']);d=R/f'binaries-I-{nl}';d.mkdir(exist_ok=False);remote=REMOTE+'/'+d.name;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha256(p)==h;shutil.copy2(p,d/p.name);adb('push',windows(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli');save(d/'seal.json',seal)
def execute(mode,tag,nl=NL,repeat=1,audit=False,poison=False):
    arm=mode
    d=R/tag
    if (d/'validated.json').exists():return read(d/'validated.json')
    d.mkdir(exist_ok=True);binary_mode='I';root=REMOTE+f'/binaries-{binary_mode}-{nl}';base=read(BASE)
    prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
    env=dict(t.split('=',1) for t in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=PACKAGE_REMOTE+'/'+arm
    for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT']:env.pop(k,None)
    env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_GENERATION_STEPS='43',QBH_GENERATION_EXPECTED_TOKENS='64',QBH_GENERATION_SEQUENCE='9',QBH_LLAMA_SP2='0',QBH_SP2_DOWN_HVX='0',QBH_WIDE_SCORE='8',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_DENSE_R3='0',QBH_DENSE_R4='4' if mode=='R4' else '0',QBH_R4_OPT='6',QBH_W4U8_DECODE_AV_REQUANT_ROWS='4')
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
    gold=read(R/('oracle-'+mode)/'summary.json');peak=exact=0;times={k:[] for k in ['prefill','decode']};fields={k:{f:[] for f in ['invocation_ticks','u8_attention_av_requant_ticks','u8_attention_av_hmx_ticks','o_projection_ticks','w4u8_gate_up_swiglu_worker_ticks','u8_attention_softmax_ticks','down_ticks','gate_up_ticks','activation_ticks','dense_r4_prepare_ticks','dense_r4_layout_ticks','dense_r4_finish_ticks','dense_r4_hmx_calls']} for k in times}
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
  rng=np.random.default_rng(680068);ix=rng.integers(0,n,(20000,n));s={}
  for phase,tokens in [('prefill',64),('decode',42)]:
   a={m:np.array([np.mean(p[m]['times'][phase])/1e6 for p in pairs]) for m in MODES};x,y=a['A8'],a['R4']
   s[phase]={m:dict(wall_ms=float(v.mean()),tps=float(tokens*1000/v.mean())) for m,v in a.items()};s[phase]['r4_over_a8']=dict(wall_ratio=float(y.mean()/x.mean()),ci95=np.quantile(y[ix].mean(1)/x[ix].mean(1),[.025,.975]).tolist())
  save(R/(stage_name+'-summary.json'),s);print(stage_name,json.dumps(s),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--layers',type=int,default=NL);ap.add_argument('--mode',default='A8');a=ap.parse_args()
 if a.action=='audit':preflight();execute(a.mode,f'audit{a.layers}-{a.mode}',nl=a.layers,audit=True)
 else:globals()[a.action]()
