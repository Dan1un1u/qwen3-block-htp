#!/usr/bin/env python3
"""W4F16-only A/B/C quality and alternating A/C speed using frozen ABI108."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import eval_exp0218 as evaluation
import measure_exp0218 as measure
from rotation_exp0219 import RESULT, OUTPUT, BASE, write_json
from prepare_exp0164_generation_package import sha256_file, file_record

ADB='/mnt/c/adb/adb.exe'
REMOTE='/data/local/tmp/qwen3-block-htp/exp0219-'
RUNTIME=evaluation.ROOT/'speed_runtime'

def adb(*args):
    return subprocess.run([ADB,*args],check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout

def windows(path):
    return subprocess.check_output(['wslpath','-w',str(path)],text=True).strip()

def deploy(variant):
    evaluation.dataset()
    root=REMOTE+variant
    assert adb('shell',f'test ! -e {root} && echo absent').strip()=='absent'
    package=BASE if variant=='A' else OUTPUT/variant
    manifest=json.loads((package/'manifest.json').read_text())
    for name,record in manifest['files'].items():
        path=package/name.replace(chr(92),'/')
        assert file_record(path)==record, path
    archive=OUTPUT/f'deploy_{variant}.tar'
    with tarfile.open(archive,'x') as tar:
        tar.add(package,arcname='block_package_layer14_m64')
        for name in ['qwen3_block_cli','libqwen3_probe.so','libqwen3_probe_skel.so']:
            tar.add(RUNTIME/name,arcname=name)
        for suite in ['quick','full','repeat']:
            tar.add(evaluation.ROOT/(suite+'.bin'),arcname=suite+'.bin')
    adb('shell',f'mkdir {root}')
    print(adb('push',windows(archive),root+'/package.tar'),flush=True)
    print(adb('shell',f'cd {root} && tar -xf package.tar && chmod 755 qwen3_block_cli'),flush=True)
    # Verify every deployed inference byte against the local manifest.
    verification=''.join(record['sha256']+'  '+name.replace(chr(92),'/')+'\n' for name,record in manifest['files'].items())
    check=OUTPUT/f'deploy_{variant}.sha256';check.write_text(verification)
    adb('push',windows(check),root+'/files.sha256')
    status=adb('shell',f'cd {root}/block_package_layer14_m64 && sha256sum -c ../files.sha256')
    assert status.count(': OK')==len(manifest['files'])
    binaries=adb('shell',f'cd {root} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
    for name in ['qwen3_block_cli','libqwen3_probe.so','libqwen3_probe_skel.so']:
        assert sha256_file(RUNTIME/name) in binaries
    write_json(RESULT/f'deploy_{variant}.json',dict(remote=root,package=str(package),
        package_manifest_sha256=sha256_file(package/'manifest.json'),verified_files=len(manifest['files']),
        runtime_source_commit='d981072513d06ed61731c14743c76ac6bc81617f',
        runtime_abi=108,runtime_experiment_label=218,experiment='EXP-0219',binary_hashes=binaries,
        device=adb('shell','cat /proc/sys/kernel/random/boot_id; getprop ro.product.model')))

def run(variant,phase,round_index=None):
    os.environ['EXP0218_REMOTE_ROOT']=REMOTE+variant
    measure.ROOT=evaluation.ROOT  # Frozen tokenizer/prompt reference location.
    suite=phase if phase in ['quick','full','repeat'] else None
    path=RESULT/phase/(f'round_{round_index:02d}_{variant}.jsonl' if round_index else variant+'.jsonl')
    meta=measure.run('w4f16',suite,path)
    if suite:
        validated=evaluation.parse_device(path,suite)
        if suite=='repeat':
            fields=['token_id','target_code','nll','rank','target_ties','max_ties','saturated']
            samples=validated['samples']
            validated['repeat_equal']=all(all(all(a[k]==b[k] for k in fields) for a,b in zip(samples[i]['steps'],samples[i+2]['steps'])) for i in [0,1])
            assert validated['repeat_equal'],path
        write_json(path.with_suffix('.validated.json'),validated)
    else:
        measure.ROOT=RESULT/variant
        measure.speed_validate(path,'w4f16')
    print(json.dumps(dict(variant=variant,phase=phase,round=round_index,**meta)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['deploy','quick','full','repeat','warmup','short','formal']);p.add_argument('--variant',choices=['A','B','C']);a=p.parse_args()
    if a.phase=='deploy':
        assert a.variant;deploy(a.variant);return
    variants=[a.variant] if a.variant else (['A','B','C'] if a.phase in ['quick','full','repeat'] else ['A','C'])
    for variant in variants:
        directory=RESULT/variant;directory.mkdir(exist_ok=True)
        expected=BASE/'generation_expected_token_ids_u32.bin' if variant=='A' else OUTPUT/variant/'generation_expected_token_ids_u32.bin'
        import numpy as np
        target=directory/'regression_expected.json'
        if not target.exists():write_json(target,dict(w4f16=np.fromfile(expected,dtype='<u4').tolist()))
    if a.phase in ['quick','full','repeat']:
        for variant in variants:run(variant,a.phase)
    else:
        for i in range(dict(warmup=1,short=5,formal=10)[a.phase]):
            for variant in variants[i%len(variants):]+variants[:i%len(variants)]:run(variant,a.phase,i+1)

if __name__=='__main__':main()
