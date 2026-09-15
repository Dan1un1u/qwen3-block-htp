"""Paired complete-model longer KV ablation; single serialized device owner."""
import os,sys,json,shlex,subprocess,shutil,struct,statistics
from pathlib import Path
import numpy as np
from run_llama32_layer import adb,windows
from run_llama32_frontend import records
from llama_reference import sha256 as sha
from report_llama32_pipeline_profile import MODULES
S=Path('/home/daniuniu/work/llama32-htp');R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0037');STEPS=64;CASE='A64';CASES={'A16':('A',16),'A64':('A',64),'B64':('B',64)};ARMS={'LATEST':(0,8),'ORIGINAL':(8,0)}
def read(p):return json.loads(p.read_text())
def write(p,z):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,indent=2);f.write('\n')
def preflight():
 t=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'ACTIVE_EXPERIMENT=L32-0037' in t

def prepare():
 preflight();cfg=read(R/'package.json');p=Path(cfg['package']);assert sha(p/'manifest.json')==cfg['manifest_sha256'];m=read(p/'manifest.json');pm=read(Path(cfg['parent_package'])/'manifest.json');remote='/data/local/tmp/llama32-htp/l32-0037-model-full-a01'
 assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 names=[n for n in m['files'] if not n.endswith('.npy')];dirs=sorted({str(Path(n).parent) for n in names if str(Path(n).parent)!='.'});adb('shell','mkdir -p '+' '.join(shlex.quote(remote+'/'+n) for n in dirs));links=[];copies=[]
 for n,v in m['files'].items():
  assert sha(p/n)==v['sha256'],n
  if n.endswith('.npy'):continue
  if n in pm['files'] and v['sha256']==pm['files'][n]['sha256']:links.append((cfg['parent_remote']+'/'+n,remote+'/'+n))
  else:copies.append(n)
 for i in range(0,len(links),24):adb('shell',' && '.join('ln -s '+shlex.quote(a)+' '+shlex.quote(b) for a,b in links[i:i+24]))
 for n in copies:adb('push',windows(p/n),remote+'/'+n)
 for i in range(0,len(names),32):
  ls=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(ls)==len(names[i:i+32])
  for l in ls:
   h,n=l.split(None,1);assert h==m['files'][n.removeprefix(remote+'/')]['sha256'],n
 write(R/'deployment.json',dict(cfg,remote=remote,immutable_links=len(links),fresh_files=copies))

def stage():
 preflight();seal=read(S/'build/llama-build-seal.json');assert not seal['paper_trace'];assert seal['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 for b in ['android_ReleaseG_aarch64','hexagon_ReleaseG_toolv19_v79']:
  t=(S/b/'CMakeCache.txt').read_text();assert 'QBH_PAPER_TRACE:BOOL=OFF' in t and 'QBH_LLAMA_LAYER_COUNT:STRING=16\n' in t
 d=R/'binaries-l16-a01';d.mkdir();remote='/data/local/tmp/llama32-htp/l32-0037-l16-a01';assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha(p)==h;shutil.copy2(p,d/p.name);adb('push',windows(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli');write(d/'seal.json',seal);write(R/'runtime.json',dict(remote=remote,seal=seal,archive=str(d)));print('STAGED',flush=True)

def one(arm,tag,repeat,audit=False):
 preflight();cfg=read(R/'deployment.json');runtime=read(R/'runtime.json');root=runtime['remote'];p=Path(cfg['package']);d=R/tag;d.mkdir(parents=True,exist_ok=False);teacher_path=R/f'teacher-{CASES[CASE][0]}.json';teacher=read(teacher_path);assert sha(teacher_path)==cfg['teacher_hashes'][CASES[CASE][0]]
 prefix,args=cfg['original_command'].split(' ./qwen3_block_cli ',1);env=dict(v.split('=',1) for v in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=cfg['remote'];argv[2]='1'
 env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_LLAMA_SP2='8',QBH_LLAMA_FP32_RESIDUAL='1',QBH_WIDE_SCORE=str(ARMS[arm][1]),QBH_DENSE_R3='0',QBH_DENSE_R4='0',QBH_PAPER_FORMAT_DISABLE=str(ARMS[arm][0]),QBH_PAPER_PIPELINE_DISABLE='0',QBH_KV_CACHE_CAPACITY='128',QBH_GENERATION_STEPS=str(STEPS));env.pop('QBH_REPLAY_DUMP_DIR',None)
 ids=read(R/'prompts.json')[CASES[CASE][0]]['ids'];target=np.fromfile(p/'generation_expected_token_ids_u32.bin','<u4').tolist();assert len(ids)==64 and len(target)==16
 row=[0,2,STEPS]+ids+target;f=d/'eval.bin';f.write_bytes(struct.pack('<4I',0x51424556,1,repeat,83)+b''.join(struct.pack('<83I',i,*row[1:]) for i in range(repeat)));env['QBH_EVAL_FILE']=root+'/'+tag.replace('/','_')+'.bin';adb('push',windows(f),env['QBH_EVAL_FILE'])
 if audit:
  env.update(QBH_GENERATION_BOUNDARY_AUDIT='1',QBH_GENERATION_AUDIT_DIR=root+'/'+tag.replace('/','_')+'-audit');adb('shell','mkdir '+env['QBH_GENERATION_AUDIT_DIR'])
 cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+' '.join(shlex.quote(v) for v in argv)
 write(d/'protocol.json',dict(command=cmd,runtime=runtime,package=cfg,arm=arm,repeat=repeat,steps=STEPS));z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);write(d/'exit.json',dict(returncode=z.returncode));
 if audit:adb('pull',env['QBH_GENERATION_AUDIT_DIR']+'/.',windows(d),check=False)
 assert z.returncode==0,(tag,z.returncode,z.stdout[-3000:],z.stderr[-1000:])
 rs=records(z.stdout);ps=[v for v in rs if v.get('record')=='generation_profile'];assert len(ps)==repeat*STEPS,(tag,len(ps));ts=[v for v in rs if 'selected_logit_half_bits' in v];assert [v['selected_token_id'] for v in ts]==teacher['ids'][:STEPS]*repeat;assert [v['selected_logit_half_bits'] for v in ts]==teacher['codes'][:STEPS]*repeat
 for i,v in enumerate(ps):
  step=i%STEPS;assert v['block_invocation_count']==16 and v['llama_fp32_residual']==1
  assert v['generation_step']==step and v['valid_length']==64+step and v['logical_m']==(64 if step==0 else 1)
  assert v['wide_score_mode']==ARMS[arm][1] and v['scan_cache_append_mismatch_count']==0
  assert v['paper_format_disable']==ARMS[arm][0] and v['paper_pipeline_disable']==0
  assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
  assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode'])
  ticks=sum(sum(v[k] for k in fields) for _,fields in MODULES)-v['generation_final_norm_ticks'];assert ticks==v['invocation_ticks']
 if audit:
  from llama32_fp32_residual import norm
  from llama_u8_reference import load_qparams_bin
  q=load_qparams_bin(p/'generation_qparams_u8.bin')['generation_final_norm_output'];gamma=np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2')
  for step in range(STEPS):
   x=np.fromfile(d/f'generation_hidden_step{step:02d}_f32.bin','<f4');expected=np.load(R/'oracle'/CASES[CASE][0]/f'hidden-{step:02d}.npy')[-1];assert np.isfinite(x).all() and np.array_equal(x,expected),(tag,step,'independent full16 hidden')
   n=np.fromfile(d/f'generation_norm_step{step:02d}_u8_native.bin','u1').reshape(64,64,32).transpose(1,0,2).reshape(64,2048)[:1];assert np.array_equal(n,norm(x[None],gamma,q)),(tag,step,'norm')
 out=dict(pass_all=True,case=CASE,steps=STEPS,audit=audit,arm=arm,repeat=repeat,profiles=len(ps),prefill_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='prefill'),decode_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='decode'),peak=max(v['vtcm_peak_plan_bytes'] for v in ps))
 write(d/'validated.json',out);print('PASS',tag,64e9/out['prefill_ns'],1e9/out['decode_ns'],flush=True);return out

def setcase(c):
 global CASE,STEPS
 CASE=c;STEPS=CASES[c][1]
def gates():
 out=[]
 for c in CASES:
  setcase(c)
  for a in ARMS:out.append(one(a,f'gate/{c}-{a}',1,audit=True))
  for step in range(STEPS):
   for n in [f'generation_hidden_step{step:02d}_f32.bin',f'generation_norm_step{step:02d}_u8_native.bin']:assert (R/f'gate/{c}-LATEST'/n).read_bytes()==(R/f'gate/{c}-ORIGINAL'/n).read_bytes(),(c,step,n)
 write(R/'full_gate.json',dict(pass_all=True,runs=out,independent_full16_hidden_and_codes_exact=True))
def timing():
 assert read(R/'full_gate.json')['pass_all']
 for case in CASES:
  setcase(case)
  for phase,n in [('short',5),('formal',10)]:
   out=[];keys=list(ARMS)
   for c in range(n):
    for a in (keys if c%2==0 else keys[::-1]):out.append(dict(cycle=c,**one(a,f'{case}/{phase}/{c:02d}-{a}',5)))
   write(R/f'{case}-{phase}.json',dict(pass_all=True,runs=out))
if __name__=='__main__':{'prepare':prepare,'stage':stage,'gate':gates,'timing':timing}[sys.argv[1]]()
