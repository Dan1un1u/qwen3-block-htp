#!/usr/bin/env python3
"""L32-0049 generic segmented FP16 hardware verification and timing."""
import sys,os,json,shlex,shutil,subprocess,struct,statistics
from pathlib import Path
import numpy as np
from prepare_llama32_3b import ROOT as S,OUT as M,RES as BASE,preflight,save
from run_llama32_layer import adb,windows
from llama_reference import sha256 as sha
from run_llama32_frontend import records
from report_llama32_pipeline_profile import MODULES
R=BASE.parent/'l32-0049'
def read(p):return json.loads(Path(p).read_text())
def stage(tag):
    preflight();seal=read(S/'build/llama-build-seal.json');assert not seal['paper_trace'];assert seal['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
    d=R/tag;d.mkdir(exist_ok=False);remote='/data/local/tmp/llama32-htp/l32-0049/'+tag
    assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
    for n,h in seal['files'].items():
        p=Path(n);assert sha(p)==h;shutil.copyfile(p,d/p.name);adb('push',windows(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
    adb('shell','chmod 755 '+remote+'/qwen3_block_cli');save(d/'runtime.json',dict(seal=seal,remote=remote));print('STAGED',tag,flush=True)

def deploy(name):
    preflight();p=M.parent/'l32-0049'/name;mf=read(p/'manifest.json')
    remote='/data/local/tmp/llama32-htp/l32-0049/models/'+name
    assert adb('shell','test ! -e '+remote,check=False).returncode==0
    adb('shell','mkdir -p '+remote)
    parents=[]
    if name!='weights':
        parents.append(read(R/'package-weights.json'))
        old=R.parent/'l32-0047'/('package-frontend64-fixed.json' if name.startswith('frontend') else 'package-'+name+'.json')
        if old.exists():parents.append(read(old))
    known={}
    for parent in parents:
        pm=read(Path(parent['package'])/'manifest.json')
        for n,h in pm['files'].items():known[h['sha256']]=parent['remote']+'/'+n
    links=[]
    for n,h in mf['files'].items():
        assert sha(p/n)==h['sha256'],n
        q=remote+'/'+n;directory=q.rsplit('/',1)[0]
        if h['sha256'] in known:
            links.append('mkdir -p '+shlex.quote(directory)+' && ln -s '+shlex.quote(known[h['sha256']])+' '+shlex.quote(q))
        else:
            adb('shell','mkdir -p '+shlex.quote(directory));adb('push',windows(p/n),q)
    for i in range(0,len(links),30):adb('shell',' && '.join(links[i:i+30]))
    names=list(mf['files'])
    for i in range(0,len(names),32):
        lines=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in names[i:i+32])).stdout.splitlines()
        assert len(lines)==len(names[i:i+32])
        for line in lines:
            h,n=line.split(None,1);assert h==mf['files'][n.removeprefix(remote+'/')]['sha256'],n
    save(R/('package-'+name+'.json'),dict(package=str(p),remote=remote,layers=mf['layers'],manifest_sha256=sha(p/'manifest.json')))
    print('DEPLOYED',name,flush=True)
def execute(runtime,name,tag,repeat=1,full=False,audit=False,segmented=True):
    preflight();rt=read(R/runtime/'runtime.json');cfg=read(R/('package-'+name+'.json'));assert int(rt['seal']['layer_count'])==cfg['layers'];assert rt['seal']['model_size']=='3B'
    root=rt['remote'];p=Path(cfg['package']);d=R/tag;d.mkdir(parents=True,exist_ok=False)
    env=dict(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,
        QBH_VERTICAL_SLICE='1',QBH_REPLAY_SEQUENCE='1',QBH_REPLAY_DECODE_STEPS='1',
        QBH_SCAN_MODE='prefill',QBH_LOGICAL_M='64',QBH_KV_CACHE_LENGTH='0',
        QBH_KV_CACHE_CAPACITY='128' if full else '80',
        QBH_W4F16_DECODE_OPT='3',QBH_F16_SEGMENTED='1' if segmented else '0')
    argv=['./qwen3_block_cli',cfg['remote'],'F16F16','1','2','32','hvx','on','off','fused',
        'gate8_interleaved','control','hvx','crouton_native_batch8','4','64',
        'parallel_qk_norm_rope','4','norms','serial','scalar','input_norm_pool_post_norm_pool','4','3','1','0']
    if full:
        teacher=read(R/'frontend64-fixed-teacher.json');steps=43;words=67+steps
        row=[0,3,steps]+teacher['prompt_ids']+teacher['input_generated_ids']
        f=d/'eval.bin';f.write_bytes(struct.pack('<4I',0x51424556,2,repeat,words)+b''.join(struct.pack(f'<{words}I',i,*row[1:]) for i in range(repeat)))
        env.update(QBH_GENERATION_SEQUENCE='10',QBH_GENERATION_STEPS='43',QBH_GENERATION_EXPECTED_TOKENS='64',QBH_KV_CACHE_LAYOUT='hmx_native_f16',QBH_EVAL_FILE=root+'/'+tag.replace('/','_')+'.bin')
        adb('push',windows(f),env['QBH_EVAL_FILE'])
        if audit:
            assert repeat==1
            env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=root+'/'+tag.replace('/','_')+'-audit')
            adb('shell','mkdir '+env['QBH_GENERATION_AUDIT_DIR'])
    else:
        env['QBH_REPLAY_DUMP_DIR']=root+'/'+tag.replace('/','_');adb('shell','mkdir '+env['QBH_REPLAY_DUMP_DIR'])
    command='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' '+shlex.join(argv)
    save(d/'protocol.json',dict(command=command,runtime=rt,package=cfg,repeat=repeat,full=full,audit=audit,segmented=segmented))
    z=adb('shell',command,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);save(d/'exit.json',dict(returncode=z.returncode))
    rs=records(z.stdout);save(d/'records.json',rs)
    if not full:adb('pull',env['QBH_REPLAY_DUMP_DIR']+'/.',windows(d),check=False)
    if audit:
        (d/'audit').mkdir();adb('pull',env['QBH_GENERATION_AUDIT_DIR']+'/.',windows(d/'audit'),check=False)
    assert z.returncode==0,(tag,z.returncode,z.stderr[-1500:],z.stdout[-2500:])
    ps=[v for v in rs if isinstance(v,dict) and v.get('record')==('generation_profile' if full else 'replay_profile')]
    assert len(ps)==repeat*(43 if full else 2),(tag,len(ps))
    for v in ps:
        assert v['block_invocation_count']==cfg['layers'] and v['llama_fp32_residual']==0
        expected_segments=(cfg['layers']+3)//4 if segmented else 0
        assert v['weight_segment_count']==v['weight_segment_map_count']==v['weight_segment_unmap_count']==expected_segments
        assert v['weight_segment_error']==0
        assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
        assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode','paper_format_disable','paper_pipeline_disable'])
        ticks=sum(sum(v[k] for k in fs) for _,fs in MODULES)-v['generation_final_norm_ticks'];assert ticks==v['invocation_ticks']
    if not full:
        gates=[v for v in rs if isinstance(v,dict) and 'output_nrmse' in v and 'pass' in v]
        assert len(gates)==2 and all(v['pass'] for v in gates),gates
    else:
        selected=[v for v in rs if isinstance(v,dict) and 'selected_logit_half_bits' in v]
        assert len(selected)==repeat*43
        trajectory=[(v['selected_token_id'],v['selected_logit_half_bits']) for v in selected]
        frozen=R/('hardware-trajectory-'+runtime+'.json')
        if frozen.exists():assert trajectory==[tuple(x) for x in read(frozen)['tokens']]*repeat
        elif audit:save(frozen,dict(tokens=trajectory,origin='hardware repeatability only; independent mathematical oracle retained separately'))
        else:raise AssertionError('Freeze audited trajectory before timing')
    out=dict(pass_all=True,layers=cfg['layers'],repeat=repeat,full=full,profiles=len(ps),prefill_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='prefill'),decode_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='decode'),peak=max(v['vtcm_peak_plan_bytes'] for v in ps))
    save(d/'validated.json',out);print('PASS',tag,64e9/out['prefill_ns'],1e9/out['decode_ns'],flush=True);return out
if __name__=='__main__':
    a=sys.argv
    if a[1]=='stage':stage(a[2])
    elif a[1]=='deploy':deploy(a[2])
    elif a[1]=='run':execute(a[2],a[3],a[4],int(a[5]) if len(a)>5 else 1,len(a)>6 and a[6]=='full',len(a)>7 and a[7]=='audit')
    elif a[1]=='mono':execute(a[2],a[3],a[4],1,False,False,False)
    elif a[1]=='timing':
        execute(a[2],a[3],'aux-repeat1',1,True)
        for phase,n in [('short',5),('formal',10)]:
            runs=[execute(a[2],a[3],f'{phase}/{i:02d}',10,True) for i in range(n)]
            save(R/(phase+'.json'),dict(runs=runs))
    else:raise ValueError(a[1])
