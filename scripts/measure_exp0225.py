#!/usr/bin/env python3
"""Isolated device ablations with frozen runtime and qbh-lite-v1."""
import argparse
import json
import os
import subprocess
import tarfile
import numpy as np
import eval_exp0218 as ev
import measure_exp0218 as measure
from learned_rotation_exp0225 import RESULT,OUTPUT
SELECTED=json.loads((RESULT/'selection.json').read_text())['selected']
VARIANTS=[SELECTED]
from rotation_exp0219 import write_json
from prepare_exp0164_generation_package import sha256_file
from measure_exp0219 import adb,windows

REMOTE='/data/local/tmp/qwen3-block-htp/exp0225-'
CONTROLS=['A0']
ALL=['A0',SELECTED]

def remote(v):
    return '/data/local/tmp/qwen3-block-htp/exp0224-A' if v=='A0' else REMOTE+v

def package_path(v):return OUTPUT.parent/'exp0224/A' if v=='A0' else OUTPUT/v

def deploy(variant):
    ev.dataset();root=remote(variant)
    assert variant in VARIANTS
    FROZEN_REMOTE=remote('A0')
    assert adb('shell',f'test ! -e {root} && echo absent').strip()=='absent'
    package=package_path(variant)
    manifest=json.loads((package/'manifest.json').read_text());old=manifest['files']
    files={n.replace(chr(92),'/'):old.get(n.replace(chr(92),'/'),v) for n,v in old.items()}
    changed=set(manifest['changed_files'])
    assert all(sha256_file(package/n)==files[n]['sha256'] for n in changed)
    OUTPUT.mkdir(parents=True,exist_ok=True);archive=OUTPUT/(variant+'_deploy.tar')
    with tarfile.open(archive,'x') as tar:
        # Android tar truncates PAX long link targets to the 100-byte ustar
        # linkname field. One short directory alias bounds every file target.
        info=tarfile.TarInfo('b');info.type=tarfile.SYMTYPE
        info.linkname=FROZEN_REMOTE+'/block_package_layer14_m64';info.mode=0o777;tar.addfile(info)
        for directory in ['block_package_layer14_m64']+[f'block_package_layer14_m64/layer{i}' for i in range(28)]:
            info=tarfile.TarInfo(directory);info.type=tarfile.DIRTYPE;info.mode=0o755;tar.addfile(info)
        for n in files:
            target='block_package_layer14_m64/'+n
            if n in changed:tar.add(package/n,arcname=target)
            else:
                info=tarfile.TarInfo(target);info.type=tarfile.SYMTYPE
                info.linkname='../'*(1+n.count('/'))+'b/'+n
                assert len(info.linkname.encode())<100
                info.mode=0o777;tar.addfile(info)
        tar.add(package/'manifest.json',arcname='block_package_layer14_m64/manifest.json')
        for n in ['qwen3_block_cli','libqwen3_probe.so','libqwen3_probe_skel.so','quick.bin','full.bin','repeat.bin']:
            info=tarfile.TarInfo(n);info.type=tarfile.SYMTYPE;info.linkname=FROZEN_REMOTE+'/'+n;info.mode=0o777;tar.addfile(info)
    adb('shell',f'mkdir {root}');print(adb('push',windows(archive),root+'/package.tar'),flush=True)
    adb('shell',f'cd {root} && tar -xf package.tar')
    checksum=OUTPUT/(variant+'_files.sha256');checksum.write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in files.items()))
    adb('push',windows(checksum),root+'/files.sha256')
    status=adb('shell',f'cd {root}/block_package_layer14_m64 && sha256sum -c ../files.sha256')
    assert status.count(': OK')==len(files)
    binaries=adb('shell',f'cd {root} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
    expected=json.loads((ev.ROOT/'closure.json').read_text())['speed_runtime']['binaries']
    assert all(h in binaries for h in expected.values())
    write_json(RESULT/f'deploy_{variant}.json',dict(remote=root,frozen_dependency=FROZEN_REMOTE,verified_files=len(files),
        changed_files=sorted(changed),manifest_sha256=sha256_file(package/'manifest.json'),runtime_hashes=expected,
        runtime_source='d981072513d06ed61731c14743c76ac6bc81617f',runtime_abi=108,
        device_boot_id=adb('shell','cat /proc/sys/kernel/random/boot_id').strip(),archive_bytes=archive.stat().st_size))

def run(variant,phase,index=None):
    os.environ['EXP0218_REMOTE_ROOT']=remote(variant);measure.ROOT=ev.ROOT
    suite=phase if phase in ['quick','full','repeat'] else None
    path=RESULT/phase/(f'round_{index:02d}_{variant}.jsonl' if index else variant+'.jsonl')
    meta=measure.run('w4f16',suite,path)
    if suite:
        d=ev.parse_device(path,suite)
        if suite=='repeat':
            fields=['token_id','target_code','nll','rank','target_ties','max_ties','saturated'];s=d['samples']
            d['repeat_equal']=all(all(all(a[k]==b[k] for k in fields) for a,b in zip(s[i]['steps'],s[i+2]['steps'])) for i in [0,1])
            assert d['repeat_equal'],path
        write_json(path.with_suffix('.validated.json'),d)
    else:
        measure.ROOT=RESULT/variant;measure.speed_validate(path,'w4f16')
    write_json(path.with_suffix('.provenance.json'),dict(remote_root=remote(variant),variant=variant,experiment='EXP-0225'))
    print(json.dumps(dict(variant=variant,phase=phase,round=index,returncode=meta['returncode'],elapsed_s=meta['elapsed_s'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['deploy','quick','full','repeat','warmup','short','formal']);p.add_argument('--variant',choices=ALL);a=p.parse_args()
    if a.phase=='deploy':assert a.variant;deploy(a.variant);return
    variants=[a.variant] if a.variant else (VARIANTS if a.phase in ['quick','full','repeat'] else ALL)
    for variant in variants:
        d=RESULT/variant;d.mkdir(parents=True,exist_ok=True);expected=d/'regression_expected.json'
        root=package_path(variant)
        if not expected.exists():write_json(expected,dict(w4f16=np.fromfile(root/'generation_expected_token_ids_u32.bin',dtype='<u4').tolist()))
    if a.phase in ['quick','full','repeat']:
        for v in variants:run(v,a.phase)
    else:
        for i in range(dict(warmup=1,short=5,formal=10)[a.phase]):
            for v in variants[i%len(variants):]+variants[:i%len(variants)]:run(v,a.phase,i+1)

if __name__=='__main__':main()
