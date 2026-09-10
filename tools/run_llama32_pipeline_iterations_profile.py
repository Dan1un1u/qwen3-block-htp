#!/usr/bin/env python3
"""L32-0006 matched three-arm A8 relative speed experiment."""
import argparse, hashlib, json, shlex, subprocess
from pathlib import Path
import numpy as np
from run_llama32_layer import ROOT,adb,windows
from run_llama32_frontend import records

OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0006')
OLD=OUT.parent
REMOTE='/data/local/tmp/llama32-htp/l32-0006/profile-a01'
RECIPES={'w4a16':('l32-0002','device-frontend-a03',8),'w4a8':('l32-0003','device-frontend-a02',8)}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    assert not p.exists(),p
    p.write_text(json.dumps(d,indent=2)+'\n')
def preflight():
    subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
def deploy():
    preflight(); dest=OUT/'profile-a01';dest.mkdir(exist_ok=False)
    assert adb('shell','test ! -e '+REMOTE,check=False).returncode==0
    protocol={'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pairs':10,'order':'ABC/BCA/CAB, A=old A8 B=new A8 C=W4A16 OPT2','recipes':{},'builds':{},'timing_scope':'complete warm model Host wall; excludes tokenizer/loading/session preparation','gate':'each mode paired document-free run bootstrap95 upper ratio<=1.10; repeat1 auxiliary'}
    for variant in ['baseline','candidate']:
        remote=REMOTE+'/'+variant;adb('shell','mkdir -p '+remote); hashes={}
        for name,build in [('qwen3_block_cli','android_ReleaseG_aarch64'),('libqwen3_probe.so','android_ReleaseG_aarch64'),('libqwen3_probe_skel.so','hexagon_ReleaseG_toolv19_v79')]:
            src=OUT/'baseline-build16'/name if variant=='baseline' else ROOT/build/'ship'/name
            if variant=='candidate':
                cache=(ROOT/build/'CMakeCache.txt').read_text();assert 'QBH_LLAMA_LAYER_COUNT:STRING=16' in cache and 'QBH_MODEL_LLAMA32:BOOL=ON' in cache
            hashes[name]=digest(src);adb('push',windows(src),remote+'/'+name)
            assert adb('shell','sha256sum '+remote+'/'+name).stdout.split()[0]==hashes[name]
        adb('shell','chmod 755 '+remote+'/qwen3_block_cli');protocol['builds'][variant]=hashes
    for recipe,(exp,attempt,steps) in RECIPES.items():
        old=json.loads((OLD/exp/attempt/'protocol.json').read_text()); command=old['command']; oldremote=command.split(' && ')[0].removeprefix('cd ')
        package=oldremote+'/package';local=Path('/mnt/d/llm_exp/models/llama32-htp')/exp/'frontend-a01';manifest=json.loads((local/'manifest.json').read_text())
        assert digest(local/'manifest.json')==old['package_manifest_sha256']
        assert adb('shell','sha256sum '+package+'/manifest.json').stdout.split()[0]==old['package_manifest_sha256']
        names=list(manifest['files']);checked={}
        for first in range(0,len(names),32):
            text=adb('shell','sha256sum '+' '.join(shlex.quote(package+'/'+n) for n in names[first:first+32])).stdout
            for line in text.splitlines():
                h,n=line.split(maxsplit=1);checked[n.removeprefix(package+'/')]=h
        assert all(checked[n]==manifest['files'][n]['sha256'] for n in names)
        commands={}
        for variant in ['baseline','candidate']:
            c=command.replace(oldremote,REMOTE+'/'+variant).replace(REMOTE+'/'+variant+'/package',package)
            c=c.replace('QBH_GENERATION_STEPS=16','QBH_GENERATION_STEPS=8')
            if recipe=='w4a16':c=c.replace(' && ',' && QBH_W4F16_DECODE_OPT=2 ',1)
            if recipe=='w4a8':
                c=c.replace(' && ',' && QBH_W4U8_DECODE_COMMON_OP_ROWS=4 QBH_W4U8_DECODE_SWIGLU_ROWS=4 QBH_W4U8_DECODE_SOFTMAX=hvx_tile4 ',1)
                c=c.replace(' on off fused serial ',' on off hvx_fused_post_norm_pool4 serial ').replace(' serial scalar control ',' serial hvx_tree control ')

            if recipe=='w4a8' and variant=='candidate':
                options={'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES':32,'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_CONTINUOUS':1,'QBH_W4U8_DECODE_DIRECT_N_O_GATE_PREFETCH':1,'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_SWIGLU_STREAM':1,'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES':16,'QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES':8,'QBH_W4U8_DECODE_DIRECT_N_DOWN_SINGLE_DMA':1,'QBH_W4U8_DECODE_O_BATCH_N_TILES':16,'QBH_W4U8_DECODE_DIRECT_N_O_SINGLE_DMA':1}
                c=c.replace(' && ',' && '+' '.join(k+'='+str(v) for k,v in options.items())+' ',1)
            commands[variant]=c
        protocol['recipes'][recipe]={'commands':commands,'steps':steps,'package_manifest_sha256':old['package_manifest_sha256'],'original_protocol':str(OLD/exp/attempt/'protocol.json'),'original_result':str(OLD/exp/attempt/'result.json')}
    roots=Path('/mnt/d/llm_exp/models/llama32-htp')
    a=(roots/'l32-0002/frontend-a01/generation_prompt_token_ids_u32.bin').read_bytes()
    b=(roots/'l32-0003/frontend-a01/generation_prompt_token_ids_u32.bin').read_bytes()
    assert a==b and len(a)==64*4
    protocol['shared_prompt_sha256']=hashlib.sha256(a).hexdigest()
    save(dest/'protocol.json',protocol);print('DEPLOYED',flush=True)
def execute(recipe,variant,name,audit=False,long=False):
    p=OUT/'profile-a01';protocol=json.loads((p/'protocol.json').read_text());cfg=protocol['recipes'][recipe];d=p/name;d.mkdir(exist_ok=False)
    cmd=cfg['commands'][variant]
    steps=16 if long else cfg['steps']
    if long:
        assert recipe=='w4a8'
        cmd=cmd.replace('QBH_GENERATION_STEPS=8','QBH_GENERATION_STEPS=16')
    if audit and recipe=='w4a16':cmd=cmd.replace(' && ',' && QBH_W4F16_DECODE_AUDIT=1 ',1)
    save(d/'command.json',{'command':cmd,'variant':variant,'recipe':recipe})
    run=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(run.stdout);(d/'stderr.txt').write_text(run.stderr)
    rr=records(run.stdout);ss=[s for s in rr if isinstance(s,dict) and 'selected_token_id' in s];pp=[s for s in rr if isinstance(s,dict) and s.get('record')=='generation_profile']
    assert run.returncode==0 and len(ss)==len(pp)==steps,(name,run.returncode,len(ss),len(pp))
    old=json.loads(Path(cfg['original_result']).read_text())['generation_steps'][:steps]
    assert [s['selected_token_id'] for s in ss]==[s['selected_token_id'] for s in old]
    assert [s['selected_logit_half_bits'] for s in ss]==[s['selected_logit_half_bits'] for s in old]
    for s,q in zip(ss,pp):
        assert s['pass'] and q['dsp_status']==3 and q['block_invocation_count']==16
        assert q['vtcm_acquired_bytes']==8388608 and q['vtcm_peak_plan_bytes']<=8388608
        assert all(q[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks'])
        if recipe=='w4a8':assert q['hmx_fp16_tile_pair_count']==q['generation_lm_head_expand_ticks']==q['w4u8_qkvo_weight_expand_ticks']==0
        if audit:assert q['w4f16_decode_conversion_audit_mismatches']==0
    summary={'pass':True,'prefill_ns':ss[0]['host_wall_ns'],'decode_ns':sum(s['host_wall_ns'] for s in ss[1:]),'decode_tokens':len(ss)-1,'steps':ss,'profiles':pp}
    save(d/'result.json',summary);return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['deploy','gate','formal','long','formal-long']);args=ap.parse_args()
    if args.stage=='deploy':return deploy()
    preflight()
    if args.stage=='long':
        for v in ['baseline','candidate']:print(v,execute('w4a8',v,'long-w4a8-'+v,long=True)['pass'],flush=True)
        return
    if args.stage=='formal-long':
        p=OUT/'profile-a01'
        assert all(json.loads((p/('long-w4a8-'+v)/'result.json').read_text())['pass'] for v in ['baseline','candidate'])
        pairs=[]
        for i in range(10):
            pair={}
            for v in (['baseline','candidate'] if i%2==0 else ['candidate','baseline']):
                pair[v]=execute('w4a8',v,f'formal-long-{i:02d}-{v}',long=True)
            pairs.append(pair);print('LONG_PAIR_COMPLETE',i+1,flush=True)
        rng=np.random.default_rng(6006);idx=rng.integers(0,10,size=(20000,10));report={}
        for mode in ['prefill','decode']:
            a=np.array([x['baseline'][mode+'_ns'] for x in pairs],float);b=np.array([x['candidate'][mode+'_ns'] for x in pairs],float)
            ci=np.quantile(b[idx].mean(1)/a[idx].mean(1),[.025,.975]);tokens=64 if mode=='prefill' else 15
            report[mode]={'baseline_mean_ns':float(a.mean()),'candidate_mean_ns':float(b.mean()),'baseline_tokens_per_second':tokens*1e9/float(a.mean()),'candidate_tokens_per_second':tokens*1e9/float(b.mean()),'candidate_host_ratio':float(b.mean()/a.mean()),'ci95':ci.tolist(),'slowdown_gate_pass':bool(ci[1]<=1.10),'tokens_per_run':tokens}
        save(p/'summary-long.json',report);print(json.dumps(report),flush=True);return
    if args.stage=='gate':
        for recipe in RECIPES:
            for variant in ['baseline','candidate']:
                s=execute(recipe,variant,'gate-'+recipe+'-'+variant,audit=variant=='candidate' and recipe=='w4a16')
                print(recipe,variant,'GENERATION_PASS',s['prefill_ns'],s['decode_ns'],flush=True)
        return
    p=OUT/'profile-a01'
    for recipe in RECIPES:
        assert all(json.loads((p/('gate-'+recipe+'-'+v)/'result.json').read_text())['pass'] for v in ['baseline','candidate'])
    arms=[('w4a8','baseline'),('w4a8','candidate'),('w4a16','baseline')]
    cycles=[]
    for i in range(10):
        cycle={}
        for recipe,v in arms[i%3:]+arms[:i%3]:
            cycle[recipe+'_'+v]=execute(recipe,v,f'formal-{recipe}-{i:02d}-{v}')
        cycles.append(cycle);print('CYCLE_COMPLETE',i+1,flush=True)
    rng=np.random.default_rng(6006);idx=rng.integers(0,10,size=(20000,10));report={}
    for mode in ['prefill','decode']:
        tokens=64 if mode=='prefill' else 7
        vals={a:np.array([c[a][mode+'_ns'] for c in cycles],dtype=float) for a in cycles[0]}
        entry={'tokens_per_run':tokens,'arms':{a:{'mean_ns':float(v.mean()),'tokens_per_second':tokens*1e9/float(v.mean())} for a,v in vals.items()},'comparisons':{}}
        for reference in ['w4a8_baseline','w4a16_baseline']:
            a=vals[reference];b=vals['w4a8_candidate'];ci=np.quantile(b[idx].mean(1)/a[idx].mean(1),[.025,.975])
            entry['comparisons'][reference]={'candidate_host_ratio':float(b.mean()/a.mean()),'ci95':ci.tolist(),'slowdown_gate_pass':bool(ci[1]<=1.10),'speed_advantage_supported':bool(ci[1]<1)}
        report[mode]=entry
    save(p/'summary.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
