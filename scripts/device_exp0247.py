#!/usr/bin/env python3
"""Bounded real layer0 R3 runner: audit first, then paired whole-layer gates."""
from pathlib import Path
import json,subprocess,argparse,hashlib,os
import exp0240_device as old
S=old.SOURCE;R=old.RESULT.parent/'exp0247';MODELS=old.MODELS/'exp0247'
REMOTE='/data/local/tmp/qwen3-block-htp/exp0247-layer0';ADB=old.ADB
ENV=dict(old.ENV,LD_LIBRARY_PATH=REMOTE,DSP_LIBRARY_PATH=REMOTE,ADSP_LIBRARY_PATH=REMOTE)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def preflight():old.preflight()
def adb(*args):return subprocess.run([ADB,*args],check=True,capture_output=True,text=True,timeout=180).stdout
def binaries():return [S/'android_ReleaseG_aarch64/ship/qwen3_block_cli',S/'android_ReleaseG_aarch64/ship/libqwen3_probe.so',S/'hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so']
def deploy():
    preflight();assert json.loads((R/'export_audit.json').read_text())['pass_all']
    for cell in ['control','r3']:
        p=MODELS/cell;d=json.loads((p/'manifest.json').read_text())
        for n,v in d['files'].items():assert sha(p/n)==v['sha256'],n
    adb('shell',f'mkdir -p {REMOTE}')
    for p in binaries():adb('push',old.win(p),REMOTE+'/'+p.name)
    for cell in ['control','r3']:adb('push',old.win(MODELS/cell),REMOTE+'/'+cell)
    adb('shell',f'chmod 755 {REMOTE}/qwen3_block_cli')
    remote=adb('shell',f'cd {REMOTE} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
    assert all(sha(p) in remote for p in binaries())
    x=dict(binaries={p.name:sha(p) for p in binaries()},source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),
        remote_sha256=remote,boot=adb('shell','cat /proc/sys/kernel/random/boot_id'),device=adb('shell','getprop ro.product.model'),actual_layer=0)
    with (R/'device_binary_manifest.json').open('x') as f:json.dump(x,f,indent=2)
    print('DEPLOYED',flush=True)
def run(cell,repeat,tag,dump=False,steps=8):
    assert cell in ['control','r3','scalar','identity'] and repeat in [1,10] and 1<=steps<=8
    env=dict(ENV);env['QBH_DENSE_R3']=str(dict(control=0,r3=1,scalar=2,identity=3)[cell]);env['QBH_REPLAY_DECODE_STEPS']=str(steps)
    package='control' if cell=='control' else 'r3';path=R/tag;path.mkdir(exist_ok=False)
    if dump:
        env.update(QBH_REPLAY_DUMP_DIR=REMOTE+'/'+tag,QBH_DENSE_R3_AUDIT='1')
        adb('shell',f'mkdir -p {REMOTE}/{tag}')
    cmd=f'cd {REMOTE} && '+' '.join(k+'='+v for k,v in env.items())+f' ./qwen3_block_cli {REMOTE}/{package} W4U8 {repeat} {old.ARGS}'
    (path/'command.txt').write_text(cmd+'\n')
    p=subprocess.run([ADB,'shell',cmd],capture_output=True,text=True,timeout=300)
    (path/'stdout.jsonl').write_text(p.stdout);(path/'stderr.txt').write_text(p.stderr)
    if dump:adb('pull',REMOTE+'/'+tag+'/.',old.win(path))
    records=[];extra=[]
    for line in p.stdout.splitlines():
        try:d=json.loads(line)
        except ValueError:continue
        if d.get('record')=='exp0240_profile':records.append(d)
        if d.get('record')=='dense_r3':extra.append(d)
    print(tag,'exit',p.returncode,'records',len(records),'stderr',p.stderr[-600:],flush=True)
    assert p.returncode==0 and len(records)==repeat*(steps+1) and len(extra)==len(records),tag
    assert all(d['mode']==int(env['QBH_DENSE_R3']) for d in extra)
    if cell=='r3':assert all(d['hmx_calls']==1 and d['rows']==(1536 if d['step']==0 else 24) for d in extra)
    print('host_us',[(d['replay_step'],round(d['host_wall_ns']/1000,1)) for d in records[:steps+1]],flush=True)
    return records
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--cell',default='control');p.add_argument('--tag',default='smoke_control');p.add_argument('--repeat',type=int,default=1);p.add_argument('--dump',action='store_true');p.add_argument('--steps',type=int,default=8);a=p.parse_args()
    if a.action=='deploy':deploy()
    else:preflight();run(a.cell,a.repeat,a.tag,a.dump,a.steps)
