#!/usr/bin/env python3
"""Actual-DSP paired evaluation with unchanged ABI108 binaries."""
import argparse,json,tarfile
import numpy as np
import measure_exp0229 as reuse
from data_exp0232 import RESULT,OUTPUT,SOURCE,write,sha,preflight,verified
from awq_exp0232 import frozen,DATA_HASH,C64,C64_HASH
from data_exp0232 import BASE
from measure_exp0219 import adb,windows

REMOTE='/data/local/tmp/qwen3-block-htp/exp0232-'
reuse.RESULT=RESULT;reuse.REMOTE=REMOTE
reuse.FREEZE_SHA=DATA_HASH
one=reuse.one

def package(v):
    return OUTPUT.parent/'exp0217/f16f16_greedy16' if v=='F' else C64 if v=='C64' else OUTPUT/'E64'

def qualified():
    frozen();primary=json.loads((RESULT/'summary_primary.json').read_text())
    name='summary_combined.json' if primary['reserve_trigger'] else 'summary_primary.json'
    final=json.loads((RESULT/name).read_text());assert final['status']=='pass' and not final['reserve_trigger']
    for n,h in final['inputs'].items():assert sha(RESULT/n)==h
    expected=json.loads((RESULT/'E64/package.json').read_text())['manifest_sha256']
    assert sha(package('E64')/'manifest.json')==expected and sha(C64/'manifest.json')==C64_HASH
    return dict(summary=name,summary_sha256=sha(RESULT/name),E64_manifest_sha256=expected)

def variants():
    qualified();return ['F','C64','E64']

def deployed(v):
    d=json.loads((RESULT/f'deploy_{v}.json').read_text())
    assert sha(package(v)/'manifest.json')==d['manifest_sha256']
    assert adb('shell','cat /proc/sys/kernel/random/boot_id').strip()==d['boot_id']
    return d

def deploy(v):
    preflight();proof=qualified();assert v in variants()
    if not (RESULT/'software_qualification.json').exists():write('software_qualification.json',proof)
    legacy=verified('exp0230','inputs/sentinel.bin');target=RESULT/'inputs/legacy230_sentinel.bin'
    if target.exists():assert sha(target)==sha(legacy)
    else:target.write_bytes(legacy.read_bytes())
    root=REMOTE+v;pkg=package(v);manifest=json.loads((pkg/'manifest.json').read_text())
    prior='/data/local/tmp/qwen3-block-htp/exp0230-'+('F' if v=='F' else 'C64')
    old=manifest['files'];files={n.replace(chr(92),'/'):old.get(n.replace(chr(92),'/'),r) for n,r in old.items()}
    for n,r in files.items():assert sha(pkg/n)==r['sha256'],(v,n)
    runtime=json.loads(verified('exp0218','closure.json').read_text())['speed_runtime']['binaries']
    assert adb('shell',f'test ! -e {root} && echo absent').strip()=='absent'
    archive=OUTPUT/(v+'_deploy.tar');checks=RESULT/(v+'_files.sha256')
    checks.write_text(''.join(r['sha256']+'  '+n+'\n' for n,r in files.items()))
    with tarfile.open(archive,'x') as tar:
        if v in ['F','C64']:
            item=tarfile.TarInfo('block_package_layer14_m64');item.type=tarfile.SYMTYPE
            item.linkname=prior+'/block_package_layer14_m64';item.mode=0o777;tar.addfile(item)
        else:
            alias=tarfile.TarInfo('b');alias.type=tarfile.SYMTYPE;alias.linkname=prior+'/block_package_layer14_m64';alias.mode=0o777;tar.addfile(alias)
            for directory in ['block_package_layer14_m64']+[f'block_package_layer14_m64/layer{i}' for i in range(28)]:
                item=tarfile.TarInfo(directory);item.type=tarfile.DIRTYPE;item.mode=0o755;tar.addfile(item)
            changed=set(manifest['changed_files'])
            for n in files:
                target='block_package_layer14_m64/'+n
                if n in changed:tar.add(pkg/n,arcname=target)
                else:
                    item=tarfile.TarInfo(target);item.type=tarfile.SYMTYPE
                    item.linkname='../'*(1+n.count('/'))+'b/'+n;item.mode=0o777
                    assert len(item.linkname.encode())<100;tar.addfile(item)
            tar.add(pkg/'manifest.json',arcname='block_package_layer14_m64/manifest.json')
        for n in runtime:
            item=tarfile.TarInfo(n);item.type=tarfile.SYMTYPE;item.linkname=prior+'/'+n;item.mode=0o777;tar.addfile(item)
        tar.add(checks,arcname='files.sha256')
        for p in sorted((RESULT/'inputs').glob('*.bin')):
            if 'calibration' not in p.name:tar.add(p,arcname=p.name)
    adb('shell',f'mkdir {root}');adb('push',windows(archive),root+'/inputs.tar')
    adb('shell',f'cd {root} && tar -xf inputs.tar')
    status=adb('shell',f'cd {root}/block_package_layer14_m64 && sha256sum -c {root}/files.sha256')
    assert status.count(': OK')==len(files) and 'FAILED' not in status
    for n,h in runtime.items():assert adb('shell',f'sha256sum {root}/{n}').split()[0]==h
    for p in sorted((RESULT/'inputs').glob('*.bin')):
        if 'calibration' not in p.name:assert adb('shell',f'sha256sum {root}/{p.name}').split()[0]==sha(p)
    write(f'deploy_{v}.json',dict(remote=root,manifest_sha256=sha(pkg/'manifest.json'),files=len(files),
        runtime=runtime,boot_id=adb('shell','cat /proc/sys/kernel/random/boot_id').strip(),
        archive=str(archive),archive_sha256=sha(archive)))
    print('DEVICE_DEPLOYED_VERIFIED',v,len(files),flush=True)

def development_check():
    preflight();qualified();details={}
    for v in variants():
        deployed(v)
        if v in ['F','C64']:
            regression=one(v,'legacy230_sentinel','reference_check')
            expected=json.loads(verified('exp0230',f'device/before/sentinel_{v}.validated.json').read_text())
            assert regression['samples']==expected['samples'],('frozen legacy control',v)
        if v=='F':
            regression=one(v,'regression','reference_check')
            expected=json.loads(verified('exp0229','device/before/regression_F.validated.json').read_text())
            assert regression['samples']==expected['samples']
        current=one(v,'sentinel','reference_check');assert current['samples'][:8]==current['samples'][8:]
        actual=np.array([[s['nll'] for s in row['steps']] for row in current['samples'][:8]])
        sw=json.loads((RESULT/f'software/primary_{v}.json').read_text());by_id={r['id']:r['nll'] for r in sw['samples']}
        reference=np.array([by_id[r['id']] for r in current['samples'][:8]])
        delta=actual-reference
        result=dict(mean_nll_delta=float(delta.mean()),mean_absolute_error=float(np.abs(delta).mean()),max_absolute_error=float(np.abs(delta).max()))
        assert abs(result['mean_nll_delta'])<=.02 and result['mean_absolute_error']<=.05,(v,result)
        details[v]=result
    write('development_device_check.json',dict(pass_all=True,controls_exact=True,repeat_exact=True,comparisons=details,
        reference_scope='8fixed_primary_documents_GPU_software_already_scored_no_selection_or_tuning',
        thresholds='unchanged_EXP230_abs_mean_delta_le.02_MAE_le.05'))
    print('DEVICE_REFERENCE_CHECK_PASS',flush=True)

def run(phase):
    preflight();qualified();vs=variants()
    for v in vs:deployed(v)
    assert json.loads((RESULT/'development_device_check.json').read_text())['pass_all']
    if phase=='before':
        for v in vs:
            d=one(v,'sentinel','before');assert d['samples'][:8]==d['samples'][8:]
            reference=json.loads((RESULT/f'device/reference_check/sentinel_{v}.validated.json').read_text());assert d['samples']==reference['samples']
        write('before_gate.json',dict(pass_all=True,repeat_exact=True))
    elif phase=='after':
        for v in vs:
            now=one(v,'sentinel','after')
            before=json.loads((RESULT/f'device/before/sentinel_{v}.validated.json').read_text())
            primary=json.loads((RESULT/f'device/primary/primary_00_{v}.validated.json').read_text())
            assert now['samples']==before['samples'] and now['samples'][:8]==primary['samples'][:8]
        write('after_gate.json',dict(pass_all=True,before_full_after_equal=True))
    else:
        assert phase in ['primary','reserve'];assert json.loads((RESULT/'before_gate.json').read_text())['pass_all']
        if phase=='reserve':assert json.loads((RESULT/'device_primary_summary.json').read_text())['reserve_required']
        for i in range(16):
            for v in vs[i%3:]+vs[:i%3]:one(v,f'{phase}_{i:02d}',phase)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['deploy','development-check','before','primary','reserve','after'])
    p.add_argument('variant',nargs='?',choices=['F','C64','E64']);a=p.parse_args()
    if a.phase=='deploy':deploy(a.variant)
    elif a.phase=='development-check':development_check()
    else:run(a.phase)
