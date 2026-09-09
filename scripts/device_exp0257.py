#!/usr/bin/env python3
"""Own immutable deployments and real-device text/timing captures for EXP0257."""
from pathlib import Path
import subprocess,json,shutil,tarfile,shlex,os,argparse
from export_exp0257 import S,M,R,O,C,sha,write,preflight
import exp0240_device as old
ADB=old.ADB;REMOTE='/data/local/tmp/qwen3-block-htp/exp0257'
PARENT='/data/local/tmp/qwen3-block-htp/exp0230-C64/block_package_layer14_m64'
def adb(*args,timeout=300):return subprocess.run([ADB,*args],check=True,capture_output=True,text=True,timeout=timeout).stdout
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def runtime_root(count):
 p=R/f'runtime_l{count}.json'
 return json.loads(p.read_text())['remote'] if p.exists() else REMOTE+f'-l{count}'
def stage(count):
 preflight();attempt=1
 while (R/'binaries'/f'l{count}_attempt{attempt}').exists():attempt+=1
 root=REMOTE+f'-l{count}_attempt{attempt}';d=R/'binaries'/f'l{count}_attempt{attempt}';d.mkdir(parents=True,exist_ok=False)
 bins=[S/'android_ReleaseG_aarch64/ship/qwen3_block_cli',S/'android_ReleaseG_aarch64/ship/libqwen3_probe.so',S/'hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so']
 adb('shell',f'mkdir {root}')
 for p in bins:shutil.copy2(p,d/p.name);adb('push',win(p),root+'/'+p.name)
 hashes=adb('shell',f'cd {root} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
 assert all(sha(p) in hashes for p in bins)
 adb('shell',f'chmod 755 {root}/qwen3_block_cli')
 if count==1:adb('shell',f'ln -s /data/local/tmp/qwen3-block-htp/exp0252-layer0/control {root}/control')
 write(d/'manifest.json',dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),layer_count=count,binaries={p.name:sha(p) for p in bins},remote=root,boot=adb('shell','cat /proc/sys/kernel/random/boot_id').strip()))
 (R/f'runtime_l{count}.json').write_text(json.dumps(dict(remote=root,manifest=str(d/'manifest.json'))))
 print('STAGED',count,flush=True)
def deploy():
 preflight();pkg=O/'package';manifest=json.loads((pkg/'manifest.json').read_text());cm=json.loads((C/'manifest.json').read_text());files=manifest['files']
 audit=json.loads((R/'export_audit.json').read_text());assert audit['pass_all'] and sha(pkg/'manifest.json')==audit['manifest_sha256']
 archive=O/'device_payload_v2.tar';checks=R/'device_files.sha256';checks.write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in files.items()))
 links=[]
 with tarfile.open(archive,'x',dereference=True) as tar:
  for n,v in files.items():
   assert sha(pkg/n)==v['sha256'],n
   if n in cm['files'] and cm['files'][n]['sha256']==v['sha256'] and 'weight' in n:
    links.append('mkdir -p '+shlex.quote(str(Path(n).parent))+' && ln -s '+shlex.quote(PARENT+'/'+n)+' '+shlex.quote(n))
   else:tar.add(pkg/n,arcname=n)
  tar.add(pkg/'manifest.json',arcname='manifest.json');tar.add(checks,arcname='files.sha256')
 linkscript=R/'device_links_v2.sh';linkscript.write_text('set -e\n'+'\n'.join(links)+'\n')
 root=REMOTE+'-package-v2';adb('shell',f'mkdir {root}');adb('push',win(archive),root+'/payload.tar',timeout=600)
 adb('shell',f'cd {root} && tar -xf payload.tar',timeout=600)
 adb('push',win(linkscript),root+'/links.sh');adb('shell',f'cd {root} && sh links.sh')
 result=adb('shell',f'cd {root} && sha256sum -c files.sha256',timeout=600)
 assert result.count(': OK')==len(files) and 'FAILED' not in result
 write(R/'device_package.json',dict(remote=root,manifest_sha256=sha(pkg/'manifest.json'),verified_files=len(files),archive_sha256=sha(archive)))
 print('DEPLOYED',len(files),flush=True)
def env(count,wide,capacity=128):
 root=runtime_root(count);e=dict(old.ENV);e.update(QBH_WIDE_SCORE=str(wide),QBH_DENSE_R3='0',QBH_KV_CACHE_CAPACITY=str(capacity),LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root)
 seed=R/'prefix_device.json'
 if seed.exists():e.update(QBH_PREFIX_KV='1',QBH_PREFIX_FILE=json.loads(seed.read_text())['remote'])
 return root,e
def execute(count,e,pkg,repeat,tag):
 preflight();root=runtime_root(count);p=R/tag;p.mkdir(parents=True,exist_ok=False)
 command=f'cd {root} && '+' '.join(k+'='+shlex.quote(v) for k,v in e.items())+f' ./qwen3_block_cli {pkg} W4U8 {repeat} {old.ARGS}'
 (p/'command.txt').write_text(command+'\n');r=subprocess.run([ADB,'shell',command],capture_output=True,text=True,timeout=600)
 (p/'stdout.jsonl').write_text(r.stdout);(p/'stderr.txt').write_text(r.stderr)
 print('RUN',tag,'rc',r.returncode,'stderr',r.stderr[-800:],flush=True)
 if r.returncode:raise RuntimeError(tag)
 return p

def replay(count,wide,tag,pkg=None,dump=True):
 root,e=env(count,wide,72 if pkg is None else 128)
 if dump:
  e['QBH_REPLAY_DUMP_DIR']=root+'/'+tag;e['QBH_DENSE_R3_AUDIT']='1';adb('shell',f'mkdir -p {e["QBH_REPLAY_DUMP_DIR"]}')
 p=execute(count,e,pkg or root+'/control',1,tag)
 if dump:adb('pull',e['QBH_REPLAY_DUMP_DIR']+'/.',win(p))
 records=[json.loads(l) for l in (p/'stdout.jsonl').read_text().splitlines() if l.startswith('{')]
 prof=[r for r in records if r.get('record')=='exp0240_profile'];assert len(prof)==9
 return p

def generate(wide,tag,sample=0,steps=16,audit=False):
 root,e=env(28,wide);e.update(QBH_GENERATION_SEQUENCE='9',QBH_GENERATION_STEPS=str(steps))
 prompts=json.loads((R/'prompts.json').read_text())['samples'];ids=prompts[sample]['token_ids']
 # Separate immutable prompt overlay per run; all model files are verified shared links.
 overlay=root+'/'+tag.replace('/','_');adb('shell',f'mkdir {overlay} && ln -s {REMOTE}-package-v2/* {overlay}/')
 local=R/(tag.replace('/','_')+'_tokens.bin');local.write_bytes(__import__('struct').pack('<64I',*ids))
 adb('shell',f'rm {overlay}/generation_prompt_token_ids_u32.bin');adb('push',win(local),overlay+'/generation_prompt_token_ids_u32.bin')
 if audit:
  e['QBH_GENERATION_BOUNDARY_AUDIT']='1';e['QBH_GENERATION_AUDIT_DIR']=root+'/'+tag.replace('/','_')+'_audit';adb('shell',f'mkdir {e["QBH_GENERATION_AUDIT_DIR"]}')
 p=execute(28,e,overlay,1,tag)
 if audit:adb('pull',e['QBH_GENERATION_AUDIT_DIR']+'/.',win(p))
 records=[json.loads(l) for l in (p/'stdout.jsonl').read_text().splitlines() if l.startswith('{')];final=[r for r in records if r.get('generation_sequence_complete')];profiles=[r for r in records if r.get('record')=='generation_profile']
 assert len(final)==1 and final[0]['all_steps_pass'] and len(profiles)==steps
 for i,r in enumerate(profiles):
  assert r['block_invocation_count']==28 and r['vtcm_acquired_bytes']==8388608 and r['wide_score_mode']==wide
  assert not r['intermediate_ddr_read_bytes'] and not r['intermediate_ddr_write_bytes'] and not r['intermediate_spill_fill_count']
 write(p/'validated.json',dict(sample=prompts[sample]['id'],wide=wide,steps=steps,token_ids=final[0]['token_ids'],prefill_host_ns=profiles[0]['host_wall_ns'],decode_host_ns=sum(r['host_wall_ns'] for r in profiles[1:]),pass_all=True))
 return p
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--count',type=int,default=1);p.add_argument('--wide',type=int,default=4);p.add_argument('--tag',default='audit_nr64');p.add_argument('--sample',type=int,default=0);p.add_argument('--steps',type=int,default=16);a=p.parse_args()
 if a.action=='stage':stage(a.count)
 elif a.action=='deploy':deploy()
 elif a.action=='replay':replay(a.count,a.wide,a.tag)
 elif a.action=='generate':generate(a.wide,a.tag,a.sample,a.steps)
