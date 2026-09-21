#!/usr/bin/env python3
"""L32-0061: bounded AV multiplier folding, immutable paired packages/results."""
import argparse, copy, json, os, re, shlex, shutil, struct, subprocess
from pathlib import Path
import numpy as np
import llama32_a8_fp32_reference as ref
from llama_u8_reference import load_qparams_bin, QPARAM_RECORD, unpack_w4_codes
from llama_reference import sha256
from prototype_llama32_sp2 import preflight, oracle
from run_llama32_layer import adb, windows, ROOT
from uniform_int16_down import make as make_int16

M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0061')
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0061')
FMT=struct.Struct('<IIIIiiiiiIIIIII')
REMOTE='/data/local/tmp/llama32-htp/l32-0061'

def save(p,v):
    assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def configs(p):
    raw=(p/'attention_config_all_groups.bin').read_bytes()
    return [list(FMT.unpack_from(raw,i)) for i in range(0,len(raw),FMT.size)]

def pad(p,x):
    a=np.zeros((64,2048),dtype='<f4');a[:len(x)]=x;a.tofile(p)

def prepare():
    preflight(); R.mkdir(parents=True,exist_ok=True)
    parent=M/'recovered-parent'
    historical=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0016/layer0-a02/manifest.json')
    assert (parent/'manifest.json').read_bytes()==historical.read_bytes()
    old=json.loads(historical.read_text())
    for n,v in old['files'].items(): assert sha256(parent/n)==v['sha256'],n
    save(R/'recovery.json',dict(source_device='/data/local/tmp/llama32-htp/l32-0018/layer0-a01/package',historical_manifest_sha256=sha256(historical),verified_payloads=len(old['files']),recovered_bytes=sum(v['bytes'] for v in old['files'].values())))
    control=M/'control';control.mkdir(exist_ok=False)
    for n in old['files']:
        if n.endswith('.npy'): continue
        out=control/n;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/n,out)
    save(R/'int16-audit.json',make_int16(parent/'layer0',control/'layer0'))
    candidate=M/'folded';shutil.copytree(control,candidate)
    cfg=configs(control/'layer0');assert len(set((c[-1],c[8]) for c in cfg))==1
    multiplier=cfg[0][-1];oldzero=cfg[0][8];assert multiplier>1
    q=load_qparams_bin(control/'layer0/qparams_u8.bin');oldscale=q['attention_concat']['scale']
    raw=(candidate/'layer0/qparams_u8.bin').read_bytes();records=[]
    for off in range(0,len(raw),QPARAM_RECORD.size):
        rec=list(QPARAM_RECORD.unpack_from(raw,off))
        if rec[0].split(b'\0')[0]==b'attention_concat':
            rec[1]=float(np.float32(oldscale*multiplier));rec[2]=128
            rec[3]=-128*rec[1];rec[4]=127*rec[1]
        records.append(QPARAM_RECORD.pack(*rec))
    (candidate/'layer0/qparams_u8.bin').write_bytes(b''.join(records))
    folded=copy.deepcopy(cfg)
    for c in folded:c[-1]=1;c[8]=128
    (candidate/'layer0/attention_config_all_groups.bin').write_bytes(b''.join(FMT.pack(*c) for c in folded))
    # AV HMX scale and center are IDENTICAL: control m!=1 uses center128;
    # candidate m=1 uses output_zero128. Only post-HMX RQ is removed.
    for a,b in zip(cfg,folded):
        assert a[-2]==b[-2] and all(a[i]==b[i] for i in range(len(a)) if i not in [8,14])
    reports={};diags={}
    for arm in ['control','folded']:
        p=M/arm;qs=load_qparams_bin(p/'layer0/qparams_u8.bin');cs=configs(p/'layer0')
        ref.configs=lambda q,cs=cs:cs
        past=None;diags[arm]={}
        for step,phase in enumerate(['prefill','decode']):
            rows=64 if step==0 else 1
            x=np.fromfile(p/('reference_w4u8_block_input_f32.bin' if not step else 'replay_decode_input_00_f32.bin'),'<f4').reshape(64,2048)[:rows]
            prefix='' if not step else 'replay_decode_';suffix='' if not step else '_00'
            co=np.fromfile(p/(prefix+'rope_cos'+suffix+'_f16.bin'),'<f2').reshape(64,64)
            si=np.fromfile(p/(prefix+'rope_sin'+suffix+'_f16.bin'),'<f2').reshape(64,64)
            y,cache,d=ref.layer(x,p/'layer0',qs,co,si,past,sp2=True)
            if step==0:past=cache
            pad(p/('reference_w4u8_block_output_f32.bin' if not step else 'replay_decode_reference_00_f32.bin'),y)
            diags[arm][phase]=d
            for n,v in d.items():np.save(p/f'{phase}_{n}.npy',v)
            for j,n in enumerate(['k','v']):
                z=qs['k_rope' if n=='k' else 'v']['zero_point'];v=np.full((8,80,64),z,dtype='u1');v[:,:cache[j].shape[1]]=cache[j];v.tofile(p/f'layer0/reference_kv_cache_{n}_u8.bin')
            print('REFERENCE',arm,phase,flush=True)
        save(p/'manifest.json',dict(experiment='L32-0061',arm=arm,source_layer=0,recipe='W4A8-INT16-Down',fp32_residual=True,rotation='OFF',parent_manifest_sha256=sha256(historical),files={str(f.relative_to(p)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in p.rglob('*') if f.is_file()}))
    w=unpack_w4_codes(control/'layer0','o',2048,2048).astype('i8')
    pos=np.maximum(w,0).sum(1);neg=np.minimum(w,0).sum(1)
    bound24=int(max(255*pos.max(),-255*neg.min()));bound32=int((128*np.abs(w).sum(1)).max())
    assert bound24<2**23 and bound32<2**31
    for phase in ['prefill','decode']:
        a=diags['control'][phase];b=diags['folded'][phase]
        for n in ['input_norm','q','k','score','probability']:assert np.array_equal(a[n],b[n]),n
        carrier=b['attention'].astype('i8');unclipped=(carrier-128)*multiplier+oldzero
        clipped=np.clip(unclipped,0,255);assert np.array_equal(clipped,a['attention'])
        saturated=(unclipped<0)|(unclipped>255);delta=unclipped-clipped
        ai=oracle(clipped-oldzero,w);bi=oracle(carrier-128,w)
        # Exact identity including the removed saturation contribution.
        assert np.array_equal(multiplier*bi-ai,oracle(delta,w))
        unsaturated_rows=~saturated.any(1);assert np.array_equal(ai[unsaturated_rows],multiplier*bi[unsaturated_rows])
        od=a['o'].astype('f8')-b['o'].astype('f8')
        reports[phase]=dict(elements=carrier.size,saturated=int(saturated.sum()),saturation_fraction=float(saturated.mean()),max_excess_codes=int(np.abs(delta).max()),unsaturated_rows=int(unsaturated_rows.sum()),integer_identity_exact=True,carrier_min=int(carrier.min()),carrier_max=int(carrier.max()),old_o_acc_abs=int(np.abs(ai).max()),new_o_acc_abs=int(np.abs(bi).max()),o_max_abs_difference=float(np.abs(od).max()),o_relative_l2_difference=float(np.linalg.norm(od)/max(np.linalg.norm(a['o']),1e-30)),o_changed=int(np.count_nonzero(od)))
    save(R/'offline-audit.json',dict(multiplier=multiplier,old_zero=oldzero,new_zero=128,old_scale=oldscale,new_scale=load_qparams_bin(candidate/'layer0/qparams_u8.bin')['attention_concat']['scale'],raw_signed24_bound=bound24,centered_signed32_bound=bound32,phases=reports,new_contract_not_quality_claim=True))
    print(json.dumps(reports),flush=True)

def deploy():
    preflight();seal=json.loads((ROOT/'build/llama-build-seal.json').read_text());head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();assert seal['source_head']==head and seal['layer_count']=='1'
    assert adb('shell','test ! -e '+REMOTE+'/binaries',check=False).returncode==0
    adb('shell','mkdir -p '+REMOTE+'/binaries')
    for name,build in [('qwen3_block_cli','android_ReleaseG_aarch64'),('libqwen3_probe.so','android_ReleaseG_aarch64'),('libqwen3_probe_skel.so','hexagon_ReleaseG_toolv19_v79')]:
        f=ROOT/build/'ship'/name;assert sha256(f)==seal['files'][str(f)]
        adb('push',windows(f),REMOTE+'/binaries/'+name)
        assert adb('shell','sha256sum '+REMOTE+'/binaries/'+name).stdout.split()[0]==sha256(f)
    adb('shell','chmod 755 '+REMOTE+'/binaries/qwen3_block_cli')
    for arm in ['control','folded']:
        p=M/arm;m=json.loads((p/'manifest.json').read_text())
        for n,v in m['files'].items():assert sha256(p/n)==v['sha256']
        adb('push',windows(p),REMOTE+'/'+arm)
        names=list(m['files'])
        for start in range(0,len(names),24):
            out=adb('shell','sha256sum '+' '.join(shlex.quote(REMOTE+'/'+arm+'/'+n) for n in names[start:start+24])).stdout
            for line in out.splitlines():h,n=line.split(maxsplit=1);assert h==m['files'][n.removeprefix(REMOTE+'/'+arm+'/')]['sha256']
    save(R/'build-seal.json',seal)

def execute(arm,tag,repeats=1,audit=False):
    dest=R/tag
    if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
    dest.mkdir(exist_ok=True)
    old=json.loads(Path('/mnt/d/llm_exp/results/llama32-htp/l32-0018/layer0-a01/protocol.json').read_text())['command']
    toks=shlex.split(old.split(' && ',1)[1]);index=toks.index('./qwen3_block_cli');env=dict(t.split('=',1) for t in toks[:index]);args=toks[index:]
    for k in ['LD_LIBRARY_PATH','DSP_LIBRARY_PATH','ADSP_LIBRARY_PATH']:env[k]=REMOTE+'/binaries'
    env.pop('QBH_REPLAY_DUMP_DIR',None)
    env.update(QBH_WIDE_SCORE='8',QBH_W4U8_DECODE_AV_REQUANT_ROWS='4')
    if repeats>1:env['QBH_LLAMA_REPLAY_REPEATS']=str(repeats+1)
    if audit:
        adb('shell','mkdir -p '+REMOTE+'/'+tag);env['QBH_REPLAY_DUMP_DIR']=REMOTE+'/'+tag
        env['QBH_DENSE_R3_AUDIT']='1'
    # Replay owns independent output/KV checks; generic numerical-audit mode
    # is forbidden with vertical slice. OFF-rotation chain capture is separate.
    args[1]=REMOTE+'/'+arm;args[3]='1';args[8]='off'
    command='cd '+REMOTE+'/binaries && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' '+' '.join(shlex.quote(v) for v in args)
    protocol=dict(command=command,arm=arm,repeats=repeats,audit=audit,package_manifest_sha256=sha256(M/arm/'manifest.json'),build_seal_sha256=sha256(R/'build-seal.json'))
    if (dest/'protocol.json').exists():assert json.loads((dest/'protocol.json').read_text())==protocol
    else:save(dest/'protocol.json',protocol)
    if (dest/'stdout.txt').exists():
        if (dest/'process.json').exists():assert json.loads((dest/'process.json').read_text())['returncode']==0
        run=subprocess.CompletedProcess(command,0,(dest/'stdout.txt').read_text(),(dest/'stderr.txt').read_text())
    else:
        run=adb('shell',command,check=False);(dest/'stdout.txt').write_text(run.stdout);(dest/'stderr.txt').write_text(run.stderr)
        save(dest/'process.json',dict(returncode=run.returncode))
    records=[]
    for line in run.stdout.splitlines():
        try: records.append(json.loads(re.sub(r':-?(?:nan|inf)([,}])',r':null\1',line)))
        except ValueError:pass
    if not (dest/'records.json').exists():save(dest/'records.json',records)
    if run.returncode:print(run.stdout[-2500:]+run.stderr[-1500:],flush=True)
    assert run.returncode==0,(tag,run.returncode)
    profiles=[p for p in records if p.get('record')=='replay_profile']
    assert len(profiles)==2*(repeats+1 if repeats>1 else 1),(tag,len(profiles))
    for p in profiles:
        for k in ['output_mismatches','cache_mismatches','cache_prefix_mismatches','cache_structure_mismatches','intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count']:
            assert p[k]==0,(tag,k,p[k])
        assert p['vtcm_requested_bytes']==p['vtcm_acquired_bytes']==8*1024*1024
        assert p['vtcm_peak_plan_bytes']<=8*1024*1024
    if audit:
        for step,phase in enumerate(['prefill','decode']):
            n=f'actual_replay_output_{step:02d}_f32.bin';adb('pull',REMOTE+'/'+tag+'/'+n,windows(dest/n))
            golden='reference_w4u8_block_output_f32.bin' if step==0 else 'replay_decode_reference_00_f32.bin'
            actual=np.fromfile(dest/n,'<u4').reshape(64,2048)[:64 if step==0 else 1];expected=np.fromfile(M/arm/golden,'<u4').reshape(64,2048)[:len(actual)]
            assert np.array_equal(actual,expected),(tag,phase,'not bit exact',int(np.count_nonzero(actual!=expected)))
            n=f'actual_replay_chain_{step:02d}.bin';adb('pull',REMOTE+'/'+tag+'/'+n,windows(dest/n))
    if repeats>1:profiles=profiles[2:] # per-process priming replay excluded
    result=dict(arm=arm,profiles=profiles)
    save(dest/'result.json',result);print('PASS',tag,flush=True);return result

def run():
    preflight()
    for arm in ['control','folded']:execute(arm,'audit-a02-'+arm,audit=True)
    for arm in ['control','folded']:execute(arm,'warmup-'+arm)
    for stage,rounds in [('short',5),('formal',10)]:
        pairs=[]
        for i in range(rounds):
            pairs.append({a:execute(a,f'{stage}-{i:02d}-{a}',10) for a in (['control','folded'] if i%2==0 else ['folded','control'])})
        rng=np.random.default_rng(610061);idx=rng.integers(0,rounds,(20000,rounds));summary={}
        for step,phase in enumerate(['prefill','decode']):
            summary[phase]={}
            fields=['host_wall_ns','invocation_ticks','u8_attention_av_requant_ticks','u8_attention_av_hmx_ticks','o_projection_ticks']
            for field in fields:
                if field not in pairs[0]['control']['profiles'][0]:continue
                values={arm:np.array([np.mean([p[field] for p in pair[arm]['profiles'] if p['replay_step']==step]) for pair in pairs]) for arm in ['control','folded']}
                x=values['control'];y=values['folded'];scale=.001 if field=='host_wall_ns' else 1/19.2
                summary[phase][field]=dict(control_us=float(x.mean()*scale),folded_us=float(y.mean()*scale),wall_ratio=float(y.mean()/x.mean()) if x.mean() else None,ci95=np.quantile(y[idx].mean(1)/x[idx].mean(1),[.025,.975]).tolist() if np.all(x) else None)
        save(R/(stage+'-summary.json'),summary)
        print(stage,json.dumps(summary),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','deploy','run']);a=ap.parse_args();globals()[a.action]()
