#!/usr/bin/env python3
"""Frozen-runtime, resumable paired PPL measurement. Never changes model bytes."""
import argparse,json,math,os,struct,subprocess,tarfile,time
from pathlib import Path
from data_exp0229 import RESULT,BASE,SOURCE,MODEL,sha,write,preflight,verified
from measure_exp0219 import adb,windows

VARIANTS=['F','A0','A']
PACKAGES={
 'F':('exp0217/f16f16_greedy16','exp0218-f16f16','0f8a359b559f252cb13329f57d4cabd56f17ca9bfb64bc5643f4d6014795070f'),
 'A0':('exp0224/A','exp0224-A','a5de4e6c4e02ac913e69fbddb0d4b0b9e12b5cfe88ff606cf1ea18842dc0c179'),
 'A':('exp0227/A','exp0227-A','00cb9e3a02b4b4b851bdcfb01befa9f81007ff47ef7b687c3ad54d203c0cf5a4')}
REMOTE='/data/local/tmp/qwen3-block-htp/exp0229-'
MODEL_BASE=Path('/mnt/d/llm_exp/models/qwen3-block-htp')
FIELDS=['token_id','target_token','target_code','nll','rank','target_ties','max_ties','saturated']
FREEZE_SHA='932b47a1c787b31dd6b510e037fe5c8633c52972fe56e369ce030cf6e45c4a89'

def frozen():
    assert sha(RESULT/'dataset_freeze.json')==FREEZE_SHA
    for n,h in json.loads((RESULT/'dataset_freeze.json').read_text())['files'].items():
        assert sha(RESULT/n)==h,n

def deploy():
    preflight();frozen()
    old=json.loads(verified('exp0218','closure.json').read_text())
    expected=old['speed_runtime']['binaries']
    results={}
    for v in VARIANTS:
        local,previous,h=PACKAGES[v];package=MODEL_BASE/local;assert sha(package/'manifest.json')==h
        oldfiles=json.loads((package/'manifest.json').read_text())['files']
        files={n.replace(chr(92),'/'):oldfiles.get(n.replace(chr(92),'/'),record) for n,record in oldfiles.items()}
        for n,record in files.items():assert sha(package/n)==record['sha256'],(v,n)
        root=REMOTE+v;prior='/data/local/tmp/qwen3-block-htp/'+previous
        assert adb('shell',f'test ! -e {root} && echo absent').strip()=='absent'
        adb('shell',f'mkdir {root}')
        links={'block_package_layer14_m64':prior+'/block_package_layer14_m64',
               **{n:prior+'/'+n for n in expected}}
        for n,target in links.items():adb('shell',f'ln -s {target} {root}/{n}')
        checksum=RESULT/(v+'_files.sha256')
        checksum.write_text(''.join(record['sha256']+'  '+n+'\n' for n,record in files.items()))
        archive=RESULT/(v+'_inputs.tar')
        with tarfile.open(archive,'x') as tar:
            tar.add(checksum,arcname='files.sha256')
            for p in sorted((RESULT/'inputs').glob('*.bin')):tar.add(p,arcname=p.name)
        adb('push',windows(archive),root+'/inputs.tar')
        adb('shell',f'cd {root} && tar -xf inputs.tar')
        check=adb('shell',f'cd {root}/block_package_layer14_m64 && sha256sum -c ../files.sha256')
        assert check.count(': OK')==len(files) and 'FAILED' not in check
        binary=adb('shell',f'cd {root} && sha256sum '+' '.join(expected))
        assert all(h in binary for h in expected.values())
        for p in sorted((RESULT/'inputs').glob('*.bin')):
            assert adb('shell',f'sha256sum {root}/{p.name}').split()[0]==sha(p)
        results[v]=dict(package=str(package),manifest_sha256=h,remote=root,previous=prior,
            files=len(files),runtime=expected,input_archive_sha256=sha(archive))
        print('DEVICE_VERIFIED',v,len(files),flush=True)
    write('device_verified.json',dict(variants=results,boot_id=adb('shell','cat /proc/sys/kernel/random/boot_id').strip()))

def parse(path,suite):
    raw=(RESULT/'inputs'/(suite+'.bin')).read_bytes()
    magic,version,count,width=struct.unpack_from('<4I',raw)
    assert (magic,version,width)==(0x51424556,1,83)
    rows=[struct.unpack_from('<83I',raw,16+i*332) for i in range(count)]
    records=[json.loads(l) for l in path.read_text().splitlines() if l.startswith('{')]
    starts=[r for r in records if r.get('record')=='eval_sample_begin']
    steps=[r for r in records if r.get('record')=='eval_step']
    ends=[r for r in records if r.get('record')=='eval_suite_complete']
    assert len(starts)==count and len(steps)==count*16 and len(ends)==1 and ends[0]['samples']==count
    samples=[]
    for i,row in enumerate(rows):
        values=steps[i*16:(i+1)*16]
        for j,r in enumerate(values):
            assert r['sample_id']==row[0] and r['step']==j and r['target_token']==row[67+j]
            assert r['teacher_forcing'] and r['pass'] and r['vtcm_bytes']==8388608
            assert r['intermediate_read']==r['intermediate_write']==r['spill']==r['nonfinite']==0
            assert r['cache_valid']==64+j and r['vocab_count']==151936
            assert math.isfinite(r['nll']) and r['nll']>=0
            assert 1<=r['rank']<=151936 and r['max_ties']>=1 and r['target_ties']>=1
            if r['token_id']==r['target_token']:assert r['rank']==1
        samples.append(dict(id=row[0],steps=[{k:r[k] for k in FIELDS} for r in values]))
    return dict(raw_sha256=sha(path),input_sha256=sha(RESULT/'inputs'/(suite+'.bin')),suite=suite,
                samples=samples,physical_gate='pass_8MiB_no_intermediate_DDR_or_spill',suite_wall_ns=ends[0]['suite_wall_ns'])

def one(v,suite,phase):
    root=RESULT/'device'/phase;root.mkdir(parents=True,exist_ok=True)
    path=root/(suite+'_'+v+'.jsonl');validated=path.with_suffix('.validated.json')
    if validated.exists():
        old=json.loads(validated.read_text());assert old==parse(path,suite)
        assert json.loads(path.with_suffix('.execution.json').read_text())['returncode']==0
        print('RESUME_VERIFIED',v,suite,flush=True);return old
    assert not path.exists(),('preserve incomplete attempt and repair explicitly',path)
    env=dict(os.environ,EXP0218_REMOTE_ROOT=REMOTE+v)
    command=['bash',str(SOURCE/'scripts/run_exp0218.sh'),'f16f16' if v=='F' else 'w4f16',suite]
    started=time.monotonic()
    with path.open('x') as f:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
        for line in process.stdout:f.write(line);f.flush()
        code=process.wait()
    meta=dict(command=command,remote=REMOTE+v,returncode=code,elapsed_s=time.monotonic()-started,
              source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip())
    path.with_suffix('.execution.json').write_text(json.dumps(meta,indent=2)+'\n')
    assert code==0,(path,code)
    data=parse(path,suite);validated.write_text(json.dumps(data,indent=2)+'\n')
    print('SUITE_COMPLETE',v,phase,suite,round(meta['elapsed_s'],2),flush=True)
    return data

def run(phase):
    preflight();frozen()
    device=json.loads((RESULT/'device_verified.json').read_text())
    assert adb('shell','cat /proc/sys/kernel/random/boot_id').strip()==device['boot_id']
    if phase=='before':
        old218=json.loads(verified('exp0218','quality_summary.json').read_text())
        old227=json.loads(verified('exp0227','quality_summary.json').read_text())
        for v in VARIANTS:
            data=one(v,'regression','before')
            expected=old218['samples']['f16f16'] if v=='F' else old227['samples']['control_A' if v=='A0' else 'selected_A']
            for row in data['samples']:
                old=next(r for r in expected if r['id']==row['id'])
                assert [s['nll'] for s in row['steps']]==old['nll'],(v,'qbh nll')
                assert [s['token_id'] for s in row['steps']]==old['top1'],(v,'qbh top1')
            data=one(v,'sentinel','before')
            assert data['samples'][:8]==data['samples'][8:]
        write('before_gate.json',dict(pass_all=True,regression='all_32_old_targets_per_variant_exact',sentinel='128targets_per_variant_exact_twice'))
    elif phase=='after':
        for v in VARIANTS:
            now=one(v,'sentinel','after')
            before=json.loads((RESULT/'device/before'/('sentinel_'+v+'.validated.json')).read_text())
            primary=json.loads((RESULT/'device/primary'/('primary_00_'+v+'.validated.json')).read_text())
            assert now['samples']==before['samples'] and now['samples'][:8]==primary['samples'][:8]
        write('after_gate.json',dict(pass_all=True,before_full_after_equal=True))
    else:
        assert phase in ['primary','reserve']
        assert json.loads((RESULT/'before_gate.json').read_text())['pass_all']
        if phase=='reserve':assert json.loads((RESULT/'primary_summary.json').read_text())['reserve_required']
        for i in range(16):
            for v in VARIANTS[i%3:]+VARIANTS[:i%3]:one(v,f'{phase}_{i:02d}',phase)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['deploy','before','primary','reserve','after']);a=p.parse_args()
    deploy() if a.phase=='deploy' else run(a.phase)
