#!/usr/bin/env python3
"""EXP0240 bounded layer replay runner; no full-model continuation here."""
import os, argparse, hashlib, json, shutil, subprocess, time
from pathlib import Path
SOURCE=Path('/home/daniuniu/work/qwen3-block-htp')
MODELS=Path('/mnt/d/llm_exp/models/qwen3-block-htp')
RESULT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0241')
ADB='/mnt/c/adb/adb.exe'
REMOTE='/data/local/tmp/qwen3-block-htp/exp0241-layer14'
ARGS='2 32 rms_rope_softmax on off fused_pool6_shuffle4 serial control hvx w4u8_streaming_persistent_mlp_hvx 3 64 u8_log2_gqa_qkv_overlap_vgather_vdeal_fused_qk_requant_hmx_batch_lut_templates_gqa_batch_dependency_stream_softmax_shuffle4 6 w4u8_mlp_io_qkv_o qkvo_batch4_qk_head_pairs hvx_tree_qk_batched_rsqrt_shared_rope_parallel_input control 4 4 4 2'
ENV={
'QBH_W4U8_STREAM_FENCE':'single_fence','QBH_W4U8_GATE_UP_RING_SLOTS':'16','QBH_W4U8_QKV_RING_EXPAND_WORKERS':'3',
'QBH_KV_CACHE_LAYOUT':'hmx_native_u8_segmented_vtcm_k7_session_v9','QBH_W4U8_PREFILL_CACHE_MODE':'reuse','QBH_W4U8_DELTA_RECONSTRUCTION':'serial',
'QBH_W4U8_DECODE_SOFTMAX':'hvx_tile4','QBH_W4U8_DECODE_LM_HEAD_GROUP_TILES':'32','QBH_W4U8_DECODE_O_BATCH_N_TILES':'16',
'QBH_W4U8_DECODE_AV_REQUANT_ROWS':'4','QBH_W4U8_DECODE_COMMON_OP_ROWS':'4','QBH_W4U8_DECODE_QK_NORM_ROPE_ROWS':'4',
'QBH_W4U8_DECODE_PROJECTION_MODE':'direct_n','QBH_W4U8_DECODE_DIRECT_N_MASK':'63','QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES':'32',
'QBH_W4U8_DECODE_DIRECT_N_GATE_UP_CONTINUOUS':'1','QBH_W4U8_DECODE_DIRECT_N_O_GATE_PREFETCH':'1','QBH_W4U8_DECODE_DIRECT_N_GATE_UP_SWIGLU_STREAM':'1',
'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES':'16','QBH_W4U8_DECODE_DIRECT_N_Q_BATCH_N_TILES':'32','QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES':'8',
'QBH_W4U8_DECODE_DIRECT_N_DOWN_SINGLE_DMA':'1','QBH_W4U8_DECODE_DIRECT_N_O_SINGLE_DMA':'1','QBH_W4U8_DECODE_SWIGLU_ROWS':'4',
'QBH_VERTICAL_SLICE':'1','QBH_REPLAY_SEQUENCE':'1','QBH_SCAN_MODE':'prefill','QBH_LOGICAL_M':'64','QBH_KV_CACHE_LENGTH':'0','QBH_KV_CACHE_CAPACITY':'72',
'LD_LIBRARY_PATH':REMOTE,'DSP_LIBRARY_PATH':REMOTE,'ADSP_LIBRARY_PATH':REMOTE}
def preflight():subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(SOURCE)],check=True)
def adb(*args):return subprocess.run([ADB,*args],check=True,text=True,capture_output=True).stdout
def win(p):return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def deploy():
 preflight();stage=MODELS/'exp0241/device';stage.mkdir(parents=True,exist_ok=True)
 for variant,src in [('control',MODELS/'exp0148/w4u8'),('lpbq32',MODELS/'exp0240/lpbq32'),('unit_multiplier',MODELS/'exp0240/unit_multiplier')]:
  manifest=json.loads((src/'manifest.json').read_text())
  for name,info in manifest['files'].items():assert sha(src/name)==info['sha256'],name
  d=stage/variant;d.mkdir(exist_ok=True);(d/'layer14').mkdir(exist_ok=True)
  for f in src.iterdir():
   if f.is_file():
    shutil.copy2(f,d/f.name);shutil.copy2(f,d/'layer14'/f.name)
  for kind,size in [('k',102400),('v',106496)]:
   for prefix,suffix in [('', ''),('reference_','_step00')]:
    (d/'layer14'/f'{prefix}kv_cache_{kind}_hmx_u8_segmented{suffix}.bin').write_bytes(bytes(size))
  (d/'device_manifest.json').write_text(json.dumps({'scope':'layer14','empty_cache_length':0,'unused_reference_slots':'zero allocation placeholders; never compared by EXP0240 runner','files':{str(f.relative_to(d)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in d.rglob('*') if f.is_file() and f.name!='device_manifest.json'}},indent=2))
 adb('shell',f'mkdir -p {REMOTE}')
 for f in [SOURCE/'android_ReleaseG_aarch64/ship/qwen3_block_cli',SOURCE/'android_ReleaseG_aarch64/ship/libqwen3_probe.so',SOURCE/'hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so']:
  adb('push',win(f),REMOTE+'/'+f.name)
 for name in ['control','lpbq32','unit_multiplier']:adb('push',win(stage/name),REMOTE+'/'+name)
 adb('shell',f'chmod 755 {REMOTE}/qwen3_block_cli')
 RESULT.mkdir(exist_ok=True)
 (RESULT/'device_binary_manifest.json').write_text(json.dumps({'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),'single_layer_build':True,'binaries':{f.name:sha(f) for f in [SOURCE/'android_ReleaseG_aarch64/ship/qwen3_block_cli',SOURCE/'android_ReleaseG_aarch64/ship/libqwen3_probe.so',SOURCE/'hexagon_ReleaseG_toolv19_v79/ship/libqwen3_probe_skel.so']},'device':adb('shell','getprop ro.product.model; cat /proc/sys/kernel/random/boot_id')},indent=2))
 print('DEPLOYED',flush=True)
def run(cell,repeat,tag,dump=False):
 env=dict(ENV);package='control'
 if os.getenv('QBH_LPBQ32_AUDIT'):env['QBH_LPBQ32_AUDIT']='1'
 if cell in ['lpbq32','scalar','unit_multiplier','unit_scalar','direct','unit_direct']:
  package='unit_multiplier' if cell.startswith('unit') else 'lpbq32'
  env['QBH_LPBQ32']='2' if cell in ['scalar','unit_scalar','direct','unit_direct'] else '1'
 if cell=='matched':
  env.update({'QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES':'4','QBH_W4U8_DECODE_DIRECT_N_Q_BATCH_N_TILES':'0','QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES':'8','QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES':'2','QBH_W4U8_DECODE_O_BATCH_N_TILES':'8','QBH_W4U8_DECODE_DIRECT_N_DOWN_SINGLE_DMA':'0','QBH_W4U8_DECODE_DIRECT_N_O_SINGLE_DMA':'0'})
 if cell!='control':
  for key in ['QBH_W4U8_DECODE_DIRECT_N_GATE_UP_CONTINUOUS','QBH_W4U8_DECODE_DIRECT_N_O_GATE_PREFETCH','QBH_W4U8_DECODE_DIRECT_N_GATE_UP_SWIGLU_STREAM']:env[key]='0'
 path=RESULT/tag;path.mkdir(exist_ok=False)
 if dump:
  env['QBH_REPLAY_DUMP_DIR']=REMOTE+'/'+tag
  adb('shell',f'mkdir -p {REMOTE}/{tag}')
 command=f'cd {REMOTE} && '+ ' '.join(k+'='+v for k,v in env.items())+f' ./qwen3_block_cli {REMOTE}/{package} W4U8 {repeat} {ARGS}'
 (path/'command.txt').write_text(command+'\n')
 r=subprocess.run([ADB,'shell',command],capture_output=True,text=True,timeout=180)
 (path/'stdout.jsonl').write_text(r.stdout);(path/'stderr.txt').write_text(r.stderr)
 if dump:adb('pull',REMOTE+'/'+tag+'/.',win(path))
 records=[]
 for line in r.stdout.splitlines():
  try:x=json.loads(line)
  except ValueError:continue
  if x.get('record')=='exp0240_profile':records.append(x)
 print(tag,'exit',r.returncode,'profiles',len(records),r.stderr[-700:],flush=True)
 if r.returncode!=0 or len(records)!=repeat*9:raise RuntimeError(str(path))
 print('host_us',[(x['replay_step'],round(x['host_wall_ns']/1000,2)) for x in records[:9]],flush=True)
 return records
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action');a.add_argument('--cell',default='control');a.add_argument('--repeat',type=int,default=1);a.add_argument('--tag',default='smoke_control');a.add_argument('--dump',action='store_true');z=a.parse_args()
 if z.action=='deploy':deploy()
 else:preflight();run(z.cell,z.repeat,z.tag,z.dump)
