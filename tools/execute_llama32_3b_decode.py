#!/usr/bin/env python3
"""L32-0044 exact 3B optimization against frozen L32-0041 payloads."""
import sys,os,json,shlex,shutil,subprocess,struct,statistics
from pathlib import Path
import numpy as np
from prepare_llama32_3b import ROOT as S,OUT as M,RES as BASE,preflight,save
R=BASE.parent/"l32-0044"
LONG_M=M.parent/"l32-0044"
from run_llama32_layer import adb,windows
from llama_reference import sha256 as sha
from run_llama32_frontend import records
from report_llama32_pipeline_profile import MODULES

def read(p):return json.loads(Path(p).read_text())
def stage(tag):
    preflight();seal=read(S/'build/llama-build-seal.json');assert not seal['paper_trace'];assert seal['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
    d=R/tag;d.mkdir(exist_ok=False);remote='/data/local/tmp/llama32-htp/l32-0044/'+tag
    assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
    for n,h in seal['files'].items():
        p=Path(n);assert sha(p)==h;shutil.copyfile(p,d/p.name);adb('push',windows(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
    adb('shell','chmod 755 '+remote+'/qwen3_block_cli');save(d/'runtime.json',dict(seal=seal,remote=remote));print('STAGED',tag,flush=True)
def deploy(name):
    preflight();p=(LONG_M if name.startswith('frontend64') else M)/name;mf=read(p/'manifest.json');names=[n for n in mf['files'] if not n.endswith('.npy')]
    remote='/data/local/tmp/llama32-htp/l32-0044/models/'+name
    assert adb('shell','test ! -e '+remote,check=False).returncode==0
    staging=R/('staging-'+name);staging.mkdir(exist_ok=False)
    for n in names:
        assert sha(p/n)==mf['files'][n]['sha256'],n
        d=staging/n;d.parent.mkdir(exist_ok=True,parents=True);os.link(p/n,d)
    adb('shell','mkdir -p '+remote);adb('push',windows(staging)+'/.',remote)
    for i in range(0,len(names),32):
        lines=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(lines)==len(names[i:i+32])
        for line in lines:
            h,n=line.split(None,1);assert h==mf['files'][n.removeprefix(remote+'/')]['sha256'],n
    save(R/('package-'+name+'.json'),dict(package=str(p),remote=remote,manifest_sha256=sha(p/'manifest.json'),layers=mf['layers']));print('DEPLOYED',name,flush=True)
def execute(runtime,name,tag,repeat=1,full=False):
    preflight();rt=read(R/runtime/'runtime.json');cfg=read((R if name=='layer0-a01' or name.startswith('frontend64') else BASE)/('package-'+name+'.json'));assert int(rt['seal']['layer_count'])==cfg['layers'];assert rt['seal']['model_size']=='3B';root=rt['remote'];p=Path(cfg['package']);d=R/tag;d.mkdir(exist_ok=False,parents=True)
    old=read(R.parent/'l32-0040'/('package-full.json' if full else 'package-l7.json'))['command'];prefix,args=old.split(' ./qwen3_block_cli ',1)
    env=dict(v.split('=',1) for v in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=cfg['remote'];argv[2]='1'
    env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_LLAMA_SP2='8',QBH_LLAMA_FP32_RESIDUAL='1',QBH_WIDE_SCORE='8',QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0',QBH_DENSE_R3='0',QBH_DENSE_R4='0');env.pop('QBH_REPLAY_DUMP_DIR',None)
    env['QBH_W4U8_DECODE_AV_REQUANT_ROWS']='4'
    if full:
        teacher=read((R if name.startswith('frontend64') else BASE)/(name+'-teacher.json'));ids=teacher['prompt_ids'];steps=int(os.environ.get('QBH_3B_DECODE_COUNT','15'))+1;tokens=teacher['u8_generated_ids'][:steps];assert len(tokens)==steps
        if name.startswith('frontend64'):env.update(QBH_KV_CACHE_CAPACITY='128',QBH_GENERATION_EXPECTED_TOKENS='64')
        row=[0,2 if os.environ.get('QBH_3B_GREEDY')=='1' else 3,steps]+ids+tokens;words=67+steps
        f=d/'eval.bin';f.write_bytes(struct.pack('<4I',0x51424556,2,repeat,words)+b''.join(struct.pack(f'<{words}I',i,*row[1:]) for i in range(repeat)))
        env['QBH_EVAL_FILE']=root+'/'+tag.replace('/','_')+'.bin';adb('push',windows(f),env['QBH_EVAL_FILE'])
    else:
        assert repeat==1
        env['QBH_REPLAY_DUMP_DIR']=root+'/'+tag.replace('/','_');adb('shell','mkdir '+env['QBH_REPLAY_DUMP_DIR'])
    command='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+shlex.join(argv)
    save(d/'protocol.json',dict(command=command,runtime=rt,package=cfg,repeat=repeat,full=full));z=adb('shell',command,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);save(d/'exit.json',dict(returncode=z.returncode));rs=records(z.stdout);save(d/'records.json',rs)
    if not full:adb('pull',env['QBH_REPLAY_DUMP_DIR']+'/.',windows(d),check=False)
    assert z.returncode==0,(tag,z.returncode,z.stderr[-2000:])
    ps=[v for v in rs if isinstance(v,dict) and v.get('record')==('generation_profile' if full else 'replay_profile')];assert len(ps)==repeat*(steps if full else 2),(tag,len(ps))
    for v in ps:
        assert v['block_invocation_count']==cfg['layers'] and v['llama_fp32_residual']==1
        assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
        assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode','paper_format_disable','paper_pipeline_disable'])
        ticks=sum(sum(v[k] for k in fs) for _,fs in MODULES)-v['generation_final_norm_ticks'];assert ticks==v['invocation_ticks'],(ticks,v['invocation_ticks'])
    if full:
        selected=[v for v in rs if isinstance(v,dict) and 'selected_logit_half_bits' in v];assert [v['selected_token_id'] for v in selected]==tokens*repeat;assert [v['selected_logit_half_bits'] for v in selected]==teacher['u8_selected_codes'][:steps]*repeat
    else:
        for v in ps:assert all(v[k]==0 for k in ['output_mismatches','output_nonfinite_count','cache_mismatches','cache_structure_mismatches','cache_nonfinite_count','scan_cache_append_mismatch_count'])
        for i in range(2):
            a=np.fromfile(d/f'actual_replay_output_{i:02d}_f32.bin','<f4').reshape(64,3072)[:64 if i==0 else 1]
            b=np.fromfile(p/('reference_w4u8_block_output_f32.bin' if i==0 else 'replay_decode_reference_00_f32.bin'),'<f4').reshape(64,3072)[:len(a)];assert np.array_equal(a,b) and np.isfinite(a).all()
    out=dict(pass_all=True,layers=cfg['layers'],repeat=repeat,full=full,profiles=len(ps),prefill_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='prefill'),decode_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='decode'),peak=max(v['vtcm_peak_plan_bytes'] for v in ps));save(d/'validated.json',out);print('PASS',tag,64e9/out['prefill_ns'],1e9/out['decode_ns'],flush=True);return out
if __name__=='__main__':
    a=sys.argv
    if a[1]=='stage':stage(a[2])
    elif a[1]=='deploy':deploy(a[2])
    elif a[1]=='run':execute(a[2],a[3],a[4],int(a[5]) if len(a)>5 else 1,len(a)>6 and a[6]=='full')
    elif a[1]=='timing':
        execute(a[2],a[3],'full-warmup',1,True)
        for phase,n in [('short',5),('formal',10)]:
            results=[execute(a[2],a[3],f'{phase}/{i:02d}',10,True) for i in range(n)];save(R/(phase+'.json'),dict(pass_all=True,runs=results))
    else:raise ValueError(a[1])