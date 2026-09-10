#!/usr/bin/env python3
"""L32-0004 immutable paired profiling of sealed quantized Llama packages."""
import argparse, hashlib, json, shlex, subprocess
from pathlib import Path
import numpy as np
from run_llama32_layer import ROOT,adb,windows
from run_llama32_frontend import records

OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0004')
OLD=OUT.parent
REMOTE='/data/local/tmp/llama32-htp/l32-0004/profile-a01'
RECIPES={'w4a16':('l32-0002','device-frontend-a03',8),'w4a8':('l32-0003','device-frontend-a02',16)}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    assert not p.exists(),p
    p.write_text(json.dumps(d,indent=2)+'\n')
def preflight():
    subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
def deploy():
    preflight(); dest=OUT/'profile-a01';dest.mkdir(exist_ok=False)
    assert adb('shell','test ! -e '+REMOTE,check=False).returncode==0
    protocol={'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pairs':10,'order':'even AB, odd BA','recipes':{},'builds':{},'timing_scope':'complete warm model Host wall; excludes tokenizer/loading/session preparation','gate':'each mode paired document-free run bootstrap95 upper ratio<=1.10; repeat1 auxiliary'}
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
            if variant=='candidate' and recipe=='w4a16':c=c.replace(' && ',' && QBH_W4F16_DECODE_OPT=2 ',1)
            commands[variant]=c
        protocol['recipes'][recipe]={'commands':commands,'steps':steps,'package_manifest_sha256':old['package_manifest_sha256'],'original_protocol':str(OLD/exp/attempt/'protocol.json'),'original_result':str(OLD/exp/attempt/'result.json')}
    save(dest/'protocol.json',protocol);print('DEPLOYED',flush=True)
def execute(recipe,variant,name,audit=False):
    p=OUT/'profile-a01';protocol=json.loads((p/'protocol.json').read_text());cfg=protocol['recipes'][recipe];d=p/name;d.mkdir(exist_ok=False)
    cmd=cfg['commands'][variant]
    if audit and recipe=='w4a16':cmd=cmd.replace(' && ',' && QBH_W4F16_DECODE_AUDIT=1 ',1)
    save(d/'command.json',{'command':cmd,'variant':variant,'recipe':recipe})
    run=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(run.stdout);(d/'stderr.txt').write_text(run.stderr)
    rr=records(run.stdout);ss=[s for s in rr if isinstance(s,dict) and 'selected_token_id' in s];pp=[s for s in rr if isinstance(s,dict) and s.get('record')=='generation_profile']
    assert run.returncode==0 and len(ss)==len(pp)==cfg['steps'],(name,run.returncode,len(ss),len(pp))
    old=json.loads(Path(cfg['original_result']).read_text())['generation_steps']
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
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['deploy','gate','formal']);args=ap.parse_args()
    if args.stage=='deploy':return deploy()
    preflight()
    if args.stage=='gate':
        for recipe in RECIPES:
            for variant in ['baseline','candidate']:
                s=execute(recipe,variant,'gate-'+recipe+'-'+variant,audit=variant=='candidate' and recipe=='w4a16')
                print(recipe,variant,'GENERATION_PASS',s['prefill_ns'],s['decode_ns'],flush=True)
        return
    p=OUT/'profile-a01'
    for recipe in RECIPES:
        assert all(json.loads((p/('gate-'+recipe+'-'+v)/'result.json').read_text())['pass'] for v in ['baseline','candidate'])
    results={}
    for recipe in RECIPES:
        pairs=[]
        for i in range(10):
            pair={}
            for v in (['baseline','candidate'] if i%2==0 else ['candidate','baseline']):
                pair[v]=execute(recipe,v,f'formal-{recipe}-{i:02d}-{v}')
            pairs.append(pair);print('PAIR_COMPLETE',recipe,i+1,flush=True)
        rng=np.random.default_rng(4004);idx=rng.integers(0,10,size=(20000,10));report={}
        for mode in ['prefill','decode']:
            a=np.array([x['baseline'][mode+'_ns'] for x in pairs],dtype=float);b=np.array([x['candidate'][mode+'_ns'] for x in pairs],dtype=float)
            ratios=b[idx].mean(1)/a[idx].mean(1);ci=np.quantile(ratios,[.025,.975]);tokens=64 if mode=='prefill' else RECIPES[recipe][2]-1
            report[mode]={'baseline_mean_ns':float(a.mean()),'candidate_mean_ns':float(b.mean()),'ratio':float(b.mean()/a.mean()),'ci95':ci.tolist(),'slowdown_gate_pass':bool(ci[1]<=1.10),'baseline_tokens_per_second':tokens*1e9/float(a.mean()),'candidate_tokens_per_second':tokens*1e9/float(b.mean()),'tokens_per_run':tokens}
        results[recipe]=report
    save(p/'summary.json',results);print(json.dumps(results),flush=True)
if __name__=='__main__':main()
