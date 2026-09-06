#!/usr/bin/env python3
"""Verify both retained device controls against authority-backed manifests."""
import json,hashlib
from learned_rotation_exp0226 import RESULT,OUTPUT
from measure_exp0219 import adb,windows
from prepare_exp0164_generation_package import sha256_file as sha
from rotation_exp0219 import write_json

def main():
    closure=RESULT.parent/'exp0224/closure.json'
    assert sha(closure)=='949ab21493b4471968af9e18f68e3b0d01a950826f286dc7db0f7b557a28c801'
    old=json.loads(closure.read_text());records={}
    for v,path,remote,expected in [('A0','exp0224/A','exp0224-A',old['packages']['A']['manifest_sha256']),('old100','exp0225/step100','exp0225-step100','44ac941297f1aab82ec4daff2ca8191379def73f6796a1c9c5d7d700820c5673')]:
        root=OUTPUT.parent/path;manifest=root/'manifest.json';assert sha(manifest)==expected
        record=json.loads(manifest.read_text());files=record['files'];checksum=OUTPUT/(v+'_control_verify.sha256')
        checksum.write_text(''.join(r['sha256']+'  '+n.replace(chr(92),'/')+'\n' for n,r in files.items()))
        remote='/data/local/tmp/qwen3-block-htp/'+remote;check='/data/local/tmp/qwen3-block-htp/exp0226_'+v+'_verify.sha256'
        adb('push',windows(checksum),check)
        output=adb('shell',f'cd {remote}/block_package_layer14_m64 && sha256sum -c {check}')
        assert output.count(': OK')==len(files) and 'FAILED' not in output
        binaries=adb('shell',f'cd {remote} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
        assert all(value in binaries for value in old['runtime_binaries'].values())
        records[v]=dict(remote=remote,manifest_sha256=expected,verified_files=len(files),runtime_binaries=old['runtime_binaries'],checksum_file_sha256=sha(checksum),verification_output_sha256=hashlib.sha256(output.encode()).hexdigest())
        print('CONTROL_DEVICE_VERIFIED',v,len(files),flush=True)
    write_json(RESULT/'device_controls_verified.json',dict(controls=records,device_boot_id=adb('shell','cat /proc/sys/kernel/random/boot_id').strip()))
if __name__=='__main__':main()
