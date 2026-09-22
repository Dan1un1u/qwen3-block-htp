#!/usr/bin/env python3
"""L32-0066: paired full-model AV scale folding, independent actual arithmetic."""
import argparse, copy, json, os, shlex, shutil, struct, subprocess
from pathlib import Path
import numpy as np
import torch
import llama32_3b_reference as ref
from llama_u8_reference import (load_qparams_bin, QPARAM_RECORD, unpack_w4_codes,
    HmxU8Converter, project_w4u8, _cached_w4_projection, exact_attention_dynamic)
from llama_reference import sha256, rope
from prototype_llama32_sp2 import preflight, oracle
from run_llama32_layer import adb, windows, ROOT
from run_llama32_frontend import records
from uniform_int16_down import make as make_int16
from probe_llama32_av_o_fold import configs, FMT
from report_llama32_pipeline_profile import MODULES

M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0066')
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0066')
REMOTE='/data/local/tmp/llama32-htp/l32-0066'
BASE=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0064/base.json')
FIXTURE=R/'fixture.json'
PARENT=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0064/int16')

def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def fnv(x):
    h=1469598103934665603
    for b in memoryview(np.ascontiguousarray(x,dtype='<f4')).cast('B'):h=((h^b)*1099511628211)&0xffffffffffffffff
    return f'{h:016x}'
def manifest(p,**extra):
    save(p/'manifest.json',dict(experiment='L32-0066',**extra,files={str(f.relative_to(p)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in p.rglob('*') if f.is_file() and f.name!='manifest.json'}))

def prepare():
    preflight();R.mkdir(parents=True,exist_ok=True);M.mkdir(parents=True,exist_ok=True);parent=PARENT;old=read(parent/'manifest.json')
    prior=read('/mnt/d/llm_exp/results/llama32-htp/l32-0058/fixture.json')
    assert len(prior['fixed'])>=43
    save(FIXTURE,dict(prompt_ids=prior['prompt_ids'][:64],fixed=prior['fixed'][:43],source='Frozen L32-0058 continuation; M64+42'))
    for n,v in old['files'].items():assert sha256(parent/n)==v['sha256'],n
    ctl=M/'control';ctl.mkdir(exist_ok=False)
    for n in old['files']:
        dest=ctl/n;dest.parent.mkdir(parents=True,exist_ok=True)
        # Independent small metadata; immutable large payloads can be linked.
        if n.endswith(('qparams_u8.bin','silu_up_lut_u16.bin','attention_config_all_groups.bin')):shutil.copy2(parent/n,dest)
        else:os.link(parent/n,dest)
    assert np.fromfile(ctl/'generation_prompt_token_ids_u32.bin','<u4').tolist()==read(FIXTURE)['prompt_ids']
    dst=M/'folded';dst.mkdir(exist_ok=False);folds=[]
    for f in ctl.rglob('*'):
        if f.is_file():
            d=dst/f.relative_to(ctl);d.parent.mkdir(parents=True,exist_ok=True)
            if f.name in ['qparams_u8.bin','attention_config_all_groups.bin']:shutil.copy2(f,d)
            else:os.link(f,d)
    for i in range(28):
        p=ctl/f'layer{i}';d=dst/f'layer{i}';cs=configs(p);q=load_qparams_bin(p/'qparams_u8.bin')
        assert len(set((c[-1],c[8],c[-2]) for c in cs))==1
        m=cs[0][-1];zp=cs[0][8];assert m>=1 and q['attention_concat']['zero_point']==zp
        if m!=1:
            raw=(d/'qparams_u8.bin').read_bytes();out=[]
            for off in range(0,len(raw),QPARAM_RECORD.size):
                rec=list(QPARAM_RECORD.unpack_from(raw,off))
                if rec[0].split(b'\0')[0]==b'attention_concat':
                    rec[1]=float(np.float32(rec[1]*m));rec[2]=128;rec[3]=-128*rec[1];rec[4]=127*rec[1]
                out.append(QPARAM_RECORD.pack(*rec))
            (d/'qparams_u8.bin').write_bytes(b''.join(out))
            for c in cs:c[-1]=1;c[8]=128
            (d/'attention_config_all_groups.bin').write_bytes(b''.join(FMT.pack(*c) for c in cs))
        w=unpack_w4_codes(p,'o',3072,3072).astype('i8')
        b24=int(max(255*np.maximum(w,0).sum(1).max(),-255*np.minimum(w,0).sum(1).min()))
        b32=int(max(zp,255-zp,128)*np.abs(w).sum(1).max());assert b24<2**23 and b32<2**31
        folds.append(dict(layer=i,multiplier=m,old_zero=zp,new_zero=cs[0][8],old_scale=q['attention_concat']['scale'],new_scale=load_qparams_bin(d/'qparams_u8.bin')['attention_concat']['scale'],signed24_bound=b24,signed32_bound=b32,changed=m!=1))
    save(R/'folding-contracts.json',folds)
    for arm in ['control','folded']:manifest(M/arm,arm=arm,parent_manifest_sha256=sha256(parent/'manifest.json'),recipe='W4A8-INT16-Down',rotation=False,fp32_residual=True)

def reference(arm):
    preflight();p=M/arm;d=R/('oracle-'+arm);d.mkdir(exist_ok=True)
    fixture=read(FIXTURE);ids=fixture['prompt_ids'][:64]+fixture['fixed'][:42]
    emb=np.memmap(p/'generation_embedding_weight_f16.bin','<f2','r',shape=(128256,3072));x=np.asarray(emb[ids],dtype='f4')
    # Use deployed RoPE tables, avoiding any alternative library convention.
    cs=[];ss=[]
    cs.append(np.fromfile(p/'rope_cos_f16.bin','<f2').reshape(64,128));ss.append(np.fromfile(p/'rope_sin_f16.bin','<f2').reshape(64,128))
    for j in range(42):
        cs.append(np.fromfile(p/f'generation_decode_rope_cos_{j:02d}_f16.bin','<f2').reshape(64,128)[:1])
        ss.append(np.fromfile(p/f'generation_decode_rope_sin_{j:02d}_f16.bin','<f2').reshape(64,128)[:1])
    co=np.concatenate(cs);si=np.concatenate(ss);hashes=[];saturation=[];heads={}
    cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so')
    original_attention=ref.exact_attention_dynamic
    for i in range(28):
        lp=p/f'layer{i}';q=load_qparams_bin(lp/'qparams_u8.bin');cfg=configs(lp)
        ref.configs=lambda q,cfg=cfg:cfg
        observed={};original_cfg=configs(M/'control'/f'layer{i}')
        def attention(qr,k,v,count,cfg,converter,divide):
            av,sc,pr=original_attention(qr,k,v,count,cfg,converter,divide)
            rawcfg=copy.deepcopy(original_cfg)
            for c in rawcfg:
                if c[-1]!=1:c[-1]=1;c[8]=128
            carrier=av if arm=='folded' else original_attention(qr,k,v,count,rawcfg,converter,divide)[0]
            m=original_cfg[0][-1];zero=original_cfg[0][8]
            lifted=carrier.astype('i4') if m==1 else (carrier.astype('i4')-128)*m+zero
            clipped=np.clip(lifted,0,255)
            if arm=='control':assert np.array_equal(clipped,av)
            for phase,sl in [('prefill',slice(0,64)),('decode',slice(64,None))]:
                t=lifted[sl];delta=t-np.clip(t,0,255)
                observed[phase]=dict(elements=t.size,saturated=int(np.count_nonzero(delta)),max_excess_codes=int(np.abs(delta).max()),integer_clipping_identity=True)
            np.save(d/f'l{i:02d}-av.npy',av)
            return av,sc,pr
        ref.exact_attention_dynamic=attention
        # Resumable only from immutable per-layer output; complete reference later.
        out=d/f'l{i:02d}-output.npy';info=d/f'l{i:02d}-info.json'
        if info.exists():
            x=np.load(out);a=read(info)
        else:
            x,kv,diag=ref.layer(x,lp,q,co,si,None)
            np.save(out,x)
            for name,v in zip(['k','v'],kv):np.save(d/f'l{i:02d}-{name}.npy',v)
            hs=[fnv(x[:64])]+[fnv(x[j:j+1]) for j in range(64,106)]
            a=dict(layer=i,hashes=hs,saturation=observed);save(info,a)
        hashes.append(a['hashes']);saturation.append(a['saturation']);_cached_w4_projection.cache_clear()
        if i+1 in [1,3,28]:
            hf=d/f'head-{i+1}.json'
            if hf.exists():heads[str(i+1)]=read(hf)
            else:
                gq=load_qparams_bin(p/'generation_qparams_u8.bin');gamma=np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2')
                act=ref.norm(x[63:],gamma,gq['generation_final_norm_output'])
                logits=project_w4u8(act,p,'generation_lm_head',128256,3072,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)
                tokens=logits.argmax(1);v=[dict(token=int(t),code=int(logits[j,t])) for j,t in enumerate(tokens)]
                save(hf,v);heads[str(i+1)]=v;_cached_w4_projection.cache_clear()
        print('REFERENCE',arm,i,observed or a['saturation'],flush=True)
    ref.exact_attention_dynamic=original_attention
    save(d/'summary.json',dict(arm=arm,layer_hashes=hashes,saturation=saturation,heads=heads,quality_claim=False,trajectory_sha256=sha256(FIXTURE)))

def deploy():
    preflight();base=read(BASE);parent=read(PARENT/'manifest.json')
    for arm in ['control','folded']:
        remote=REMOTE+'/'+arm;p=M/arm;mf=read(p/'manifest.json')['files'];names=list(mf)
        adb('shell','mkdir -p '+remote)
        for start in range(0,len(names),24):
            batch=names[start:start+24];commands=[]
            for n in batch:
                assert sha256(p/n)==mf[n]['sha256']
                commands.append('mkdir -p '+shlex.quote(str(Path(remote+'/'+n).parent)))
                if mf[n]['sha256']==parent['files'][n]['sha256']:commands.append('ln -s '+shlex.quote('/data/local/tmp/llama32-htp/l32-0064/int16'+'/'+n)+' '+shlex.quote(remote+'/'+n))
            adb('shell',' && '.join(commands))
            for n in batch:
                if mf[n]['sha256']!=parent['files'][n]['sha256']:adb('push',windows(p/n),remote+'/'+n)
            out=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in batch)).stdout
            for line in out.splitlines():h,n=line.split(None,1);assert h==mf[n.removeprefix(remote+'/')]['sha256']
        print('DEPLOYED',arm,flush=True)

def stage():
    preflight();seal=read(ROOT/'build/llama-build-seal.json');assert seal['source_head']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert seal['model_size']=='3B' and not seal['fp_islands']
    nl=int(seal['layer_count']);dst=R/f'binaries-{nl}';dst.mkdir(exist_ok=False);remote=REMOTE+f'/binaries-{nl}'
    adb('shell','mkdir -p '+remote)
    for n,h in seal['files'].items():
        p=Path(n);assert sha256(p)==h;shutil.copy2(p,dst/p.name);adb('push',windows(p),remote+'/'+p.name)
        assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
    adb('shell','chmod 755 '+remote+'/qwen3_block_cli');save(dst/'seal.json',seal)

def execute(arm,tag,nl=28,repeat=1,audit=False):
    d=R/tag
    if (d/'validated.json').exists():return read(d/'validated.json')
    d.mkdir(exist_ok=True);root=REMOTE+f'/binaries-{nl}';base=read(BASE)
    prefix,args=base['command'].split(' ./qwen3_block_cli ',1)
    env=dict(t.split('=',1) for t in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=REMOTE+'/'+arm
    for k in ['QBH_EVAL_FILE','QBH_GENERATION_AUDIT_DIR','QBH_GENERATION_BOUNDARY_AUDIT']:env.pop(k,None)
    env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_GENERATION_STEPS='43',QBH_GENERATION_EXPECTED_TOKENS='64',QBH_GENERATION_SEQUENCE='9',QBH_LLAMA_SP2='8',QBH_SP2_DOWN_HVX='0',QBH_WIDE_SCORE='8',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_W4U8_DECODE_AV_REQUANT_ROWS='4')
    fixture=read(FIXTURE);words=[0x51424556,2,repeat,110]
    for j in range(repeat):words += [j,3,43]+fixture['prompt_ids'][:64]+fixture['fixed'][:43]
    ef=d/'fixed-trajectory.bin'
    if not ef.exists():ef.write_bytes(struct.pack('<'+'I'*len(words),*words))
    er=root+'/'+tag+'-trajectory.bin';env['QBH_EVAL_FILE']=er
    if audit:env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=REMOTE+'/'+tag)
    cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+' '.join(shlex.quote(v) for v in argv)
    if not (d/'protocol.json').exists():save(d/'protocol.json',dict(command=cmd,arm=arm,layers=nl,repeat=repeat,audit=audit,seal_sha256=sha256(R/f'binaries-{nl}/seal.json'),package_manifest_sha256=sha256(M/arm/'manifest.json')))
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
    gold=read(R/('oracle-'+arm)/'summary.json');peak=exact=0;times={k:[] for k in ['prefill','decode']};fields={k:{f:[] for f in ['invocation_ticks','u8_attention_av_requant_ticks','u8_attention_av_hmx_ticks','o_projection_ticks']} for k in times}
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
    result=dict(arm=arm,layers=nl,repeat=repeat,exact_layer_outputs=exact,boundaries=len(pp),peak=peak,times=times,fields=fields)
    save(d/'validated.json',result);print('PASS',tag,{k:float(np.mean(v)) for k,v in times.items()},flush=True);return result

def run():
    preflight();assert read(R/'eligibility.json')['eligible']
    for arm in ['control','folded']:execute(arm,'audit28-'+arm,audit=True)
    for arm in ['control','folded']:execute(arm,'warmup-'+arm)
    for stage_name,n in [('short',5),('formal',10)]:
        pairs=[]
        for i in range(n):
            pairs.append({a:execute(a,f'{stage_name}-{i:02d}-{a}',repeat=10) for a in (['control','folded'] if i%2==0 else ['folded','control'])})
        rng=np.random.default_rng(660066);idx=rng.integers(0,n,(20000,n));summary={}
        for phase,tokens in [('prefill',64),('decode',42)]:
            def stat(x,y):
                x=np.asarray(x);y=np.asarray(y)
                return dict(control=float(x.mean()),folded=float(y.mean()),ratio=float(y.mean()/x.mean()) if x.mean() else None,ci95=np.quantile(y[idx].mean(1)/x[idx].mean(1),[.025,.975]).tolist() if np.all(x) else None)
            x=[np.mean(p['control']['times'][phase])/1e6 for p in pairs];y=[np.mean(p['folded']['times'][phase])/1e6 for p in pairs]
            s=stat(x,y);s.update(control_tps=tokens*1000/s['control'],folded_tps=tokens*1000/s['folded'],units='ms total phase wall')
            s['components_us']={k:stat([np.mean(p['control']['fields'][phase][k])/19.2 for p in pairs],[np.mean(p['folded']['fields'][phase][k])/19.2 for p in pairs]) for k in pairs[0]['control']['fields'][phase]};summary[phase]=s
        out=R/(stage_name+'-summary.json')
        if not out.exists():save(out,summary)
        print(stage_name,json.dumps(summary),flush=True)


def eligibility():
    preflight();summaries={a:read(R/('oracle-'+a)/'summary.json') for a in ['control','folded']}
    sat={a:sum(v['saturated'] for layer in s['saturation'] for v in layer.values()) for a,s in summaries.items()}
    count={a:sum(v['elements'] for layer in s['saturation'] for v in layer.values()) for a,s in summaries.items()}
    changes=[]
    for i in range(28):
        x=np.load(R/'oracle-control'/f'l{i:02d}-output.npy');y=np.load(R/'oracle-folded'/f'l{i:02d}-output.npy');d=x.astype('f8')-y.astype('f8')
        changes.append(dict(layer=i,changed=int(np.count_nonzero(x.view('u4')!=y.view('u4'))),max_abs=float(np.abs(d).max()),relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(x),1e-30))))
    result=dict(eligible=not any(sat.values()),saturated=sat,elements=count,layer_differences=changes,quality_claim=False,bitwise_equivalence=False)
    save(R/'eligibility.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action');ap.add_argument('--arm',default='control');a=ap.parse_args()
    if a.action=='reference':reference(a.arm)
    elif a.action=='audit1':
        preflight()
        for arm in ['control','folded']:execute(arm,'audit1-'+arm,nl=1,audit=True)
    elif a.action=='audit3':
        preflight()
        for arm in ['control','folded']:execute(arm,'audit3-'+arm,nl=3,audit=True)
    else:globals()[a.action]()
