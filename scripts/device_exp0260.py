#!/usr/bin/env python3
"""Frozen C64 W4F16 device experiments, exact checks and raw evidence."""
from pathlib import Path
import subprocess,json,hashlib,shutil,shlex,statistics,argparse
S=Path('/home/daniuniu/work/qwen3-block-htp');M=S.parent/'qwen3-block-htp-project-memory'
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0260');C=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0230/C64')
PARENT='/data/local/tmp/qwen3-block-htp/exp0230-C64';PKG=PARENT+'/block_package_layer14_m64'
ADB='/mnt/c/adb/adb.exe';REMOTE='/data/local/tmp/qwen3-block-htp/exp0260'
ARGS='4 32 hvx on off fused serial adaptive_down96_gate4_dma8_cross hvx crouton_native_batch8 4 64 gqa_qkv_overlap 4 qkv_norms serial scalar input_norm_pool_post_norm_pool 4 3 1 0'
ENV=dict(QBH_KV_CACHE_LAYOUT='hmx_native_f16',QBH_QKV_SCHEDULE='control',QBH_W4F16_GROUP_FENCE='join_only_down',QBH_W4F16_EXPAND_CLAIM_REGIONS='1',QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER='1',QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER='1',QBH_W4F16_GATE_UP_STREAM_GROUP_TILES='4',QBH_VERTICAL_SLICE='1',QBH_REPLAY_SEQUENCE='1',QBH_SCAN_MODE='prefill',QBH_LOGICAL_M='64',QBH_KV_CACHE_LENGTH='0',QBH_KV_CACHE_CAPACITY='80')
def preflight():subprocess.run(['python3',str(M/'scripts/project_memory.py'),'preflight','--source-worktree',str(S)],check=True)
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def adb(*args,timeout=600):return subprocess.run([ADB,*args],capture_output=True,text=True,check=True,timeout=timeout).stdout
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def records(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.startswith('{')]
def verify():
 preflight();R.mkdir(exist_ok=True)
 assert sha(C/'manifest.json')=='7de4f0758d83f2ba3b58696c695bfbfed72a25dd3bf308475abee0a0f0575a89'
 m=read(C/'manifest.json')
 for n,v in m['files'].items():assert sha(C/n)==v['sha256'],n
 p=R.parent/'exp0230';assert sha(p/'evidence_sha256.json')=='bb90bd21fb0d9ba979fcad9f6224b8dd739d9939f493636f5d85b4d8b07160f2'
 raw=adb('shell',f'cd {PKG} && sha256sum -c {PARENT}/files.sha256');assert 'FAILED' not in raw
 write(R/'package_verified.json',dict(local_files=len(m['files']),manifest_sha256=sha(C/'manifest.json'),remote_check=raw,remote=PKG))
def stage(count):
 preflight();attempt=1
 while (R/f'binaries/l{count}_attempt{attempt}').exists():attempt+=1
 d=R/f'binaries/l{count}_attempt{attempt}';d.mkdir(parents=True);root=REMOTE+f'-l{count}_attempt{attempt}';adb('shell','mkdir '+root)
 bins=[S/'android_ReleaseG_aarch64/ship/qwen3_block_cli',S/'android_ReleaseG_aarch64/ship/libqwen3_probe.so',S/'hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so']
 for p in bins:shutil.copyfile(p,d/p.name);adb('push',win(p),root+'/'+p.name)
 hs=adb('shell',f'cd {root} && chmod 755 qwen3_block_cli && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so');assert all(sha(p) in hs for p in bins)
 write(d/'manifest.json',dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),layer_count=count,binaries={p.name:sha(p) for p in bins},remote=root,boot=adb('shell','cat /proc/sys/kernel/random/boot_id').strip()))
 (R/f'runtime_l{count}.json').write_text(json.dumps(dict(remote=root,manifest=str(d/'manifest.json'))));print('STAGED',count,flush=True)
def run(count,arm,rep,tag,dump=False,audit=False):
 preflight();root=read(R/f'runtime_l{count}.json')['remote'];e=dict(ENV);e.update(QBH_W4F16_DECODE_OPT=str(arm),LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root)
 if count==28:e.update(QBH_GENERATION_SEQUENCE='7',QBH_GENERATION_STEPS='16')
 if dump:e['QBH_REPLAY_DUMP_DIR']=root+'/'+tag.replace('/','_');adb('shell','mkdir '+e['QBH_REPLAY_DUMP_DIR'])
 args=ARGS
 if audit:e['QBH_W4F16_DECODE_AUDIT']='1'
 cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in e.items())+f' ./qwen3_block_cli {PKG} W4F16 {rep} {args}'
 p=R/tag;p.mkdir(parents=True,exist_ok=False);(p/'command.txt').write_text(cmd+'\n');z=subprocess.run([ADB,'shell',cmd],capture_output=True,text=True,timeout=600);(p/'stdout.jsonl').write_text(z.stdout);(p/'stderr.txt').write_text(z.stderr);assert z.returncode==0,(tag,z.stderr[-1800:])
 if dump:adb('pull',e['QBH_REPLAY_DUMP_DIR']+'/.',win(p))
 ps=[q for q in records(p/'stdout.jsonl') if q.get('record') in ['exp0240_profile','generation_profile']];assert len(ps)==rep*(16 if count==28 else 9),(tag,len(ps))
 for q in ps:
  assert q['variant']=='W4F16' and q['block_invocation_count']==count and q['vtcm_acquired_bytes']==q['vtcm_requested_bytes']==8388608
  assert q['intermediate_ddr_read_bytes']==q['intermediate_ddr_write_bytes']==q['intermediate_spill_fill_count']==0
  assert q['w4f16_decode_conversion_audit_mismatches']==0
  assert q['w4f16_decode_opt_calls']==(count*8 if arm and q['mode']=='decode' else 0)
 write(p/'validated.json',dict(tag=tag,arm=arm,repeat=rep,count=count,output_hashes=[q['output_hash'] for q in ps],prefill_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='prefill'),decode_ns=statistics.mean(q['host_wall_ns'] for q in ps if q['mode']=='decode')));print('PASS',tag,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--count',type=int,default=1);p.add_argument('--arm',type=int,default=0);p.add_argument('--repeat',type=int,default=1);p.add_argument('--tag',default='smoke');p.add_argument('--dump',action='store_true');p.add_argument('--audit',action='store_true');a=p.parse_args()
 if a.action=='verify':verify()
 elif a.action=='stage':stage(a.count)
 else:run(a.count,a.arm,a.repeat,a.tag,a.dump,a.audit)
