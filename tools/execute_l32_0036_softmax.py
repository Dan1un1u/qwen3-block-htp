"""L32-0036 fixed full-model factorial, independent frozen references."""
import sys,os,json,shlex,subprocess,shutil,struct,statistics
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/llama32-htp');sys.path.insert(0,str(S/'tools'))
from run_llama32_layer import adb,windows
from run_llama32_frontend import records
from llama_reference import sha256 as sha
from report_llama32_pipeline_profile import MODULES
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0036/opt-a2');OLD=R.parents[1]/'l32-0018';M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0016')
ARMS={'LOG2':(0,0),'FAST':(0,0),'FP':(0,0)}
PACKAGES={'l0':('layer0-a03','layer0-a02',1),'l7':('layer7-a04','layer7-a02',1),'l15':('layer15-a03','layer15-a02',1),'chain3':('chain3-a03','chain3-a01',3),'chain16':('chain16-a03','chain16-a01',16),'full':('gate-fp32-final','frontend-a02',16)}
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,indent=2);f.write('\n')
def preflight():
 t=subprocess.check_output(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True)
 assert 'ACTIVE_EXPERIMENT=L32-0036' in t

def prepare():
 preflight();R.mkdir(exist_ok=False)
 prior=R.parents[1]/'l32-0033'
 for key in PACKAGES:
  cfg=read(prior/f'package-{key}.json');assert sha(Path(cfg['package'])/'manifest.json')==cfg['manifest_sha256'];write(R/f'package-{key}.json',cfg)
 for key in ['l7','chain3','chain16','full']:
  cfg=read(prior/f'package-{key}-fp.json');assert sha(Path(cfg['package'])/'manifest.json')==cfg['manifest_sha256'];write(R/f'package-{key}-fp.json',cfg)
 for mode in ['greedy','fixed']:shutil.copy2(prior/(mode+'-fp-teacher-a02.json'),R/(mode+'-fp-teacher-a02.json'))
 write(R/'oracle_freeze.json',{mode:sha(R/(mode+'-fp-teacher-a02.json')) for mode in ['greedy','fixed']})
 for key in ['full','full-fp']:
  cfg=read(R/f'package-{key}.json');pkg=Path(cfg['package']);mf=read(pkg/'manifest.json');names=[n for n in mf['files'] if not n.endswith('.npy')]
  for n in names:assert sha(pkg/n)==mf['files'][n]['sha256'],n
  for i in range(0,len(names),32):
   ls=adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(ls)==len(names[i:i+32])
   for line in ls:
    h,n=line.split(None,1);assert h==mf['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
 write(R/'model_gate.json',dict(pass_all=True,scope='full original and FP local/remote hashes verified; prior selected and slice immutable packages retained'))
 print('PREPARED original manifests and frozen oracles',flush=True)
def stage(count,trace=False):
 preflight();seal=read(S/'build/llama-build-seal.json');assert seal['paper_trace']==trace;assert seal['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 for b in ['android_ReleaseG_aarch64','hexagon_ReleaseG_toolv19_v79']:
  t=(S/b/'CMakeCache.txt').read_text();assert ('QBH_PAPER_TRACE:BOOL='+('ON' if trace else 'OFF')) in t;assert 'QBH_MODEL_LLAMA32:BOOL=ON' in t and f'QBH_LLAMA_LAYER_COUNT:STRING={count}\n' in t
 attempt=1
 while (R/f'binaries-l{count}-a{attempt}').exists():attempt+=1
 d=R/f'binaries-l{count}-a{attempt}';d.mkdir();remote=f'/data/local/tmp/llama32-htp/l32-0036-opt-a2-l{count}-a{attempt}'
 assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha(p)==h;shutil.copy2(p,d/p.name);adb('push',windows(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli');write(d/'seal.json',seal)
 (R/f'runtime-l{count}.json').write_text(json.dumps(dict(remote=remote,seal=seal,archive=str(d)),indent=2))
 print('STAGED',count,flush=True)
def execute(key,arm,tag,repeat=1,audit=False):
 preflight();cfg=read(R/f"package-{key}{'-fp' if arm=='FP' else ''}.json");state=read(R/f"runtime-l{cfg['layers']}.json");root=state['remote'];assert audit or state['seal']['paper_trace'] is False;d=R/tag;d.mkdir(parents=True,exist_ok=False)
 prefix,args=cfg['command'].split(' ./qwen3_block_cli ',1);oldroot=prefix.split(' && ')[0].removeprefix('cd ');env=dict(v.split('=',1) for v in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=cfg['remote'];argv[2]=str(repeat if key!='full' else 1)
 env.update(QBH_LLAMA_SP2='8',QBH_WIDE_SCORE='7' if arm=='FP' else '8' if arm=='FAST' else '0',LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_PAPER_FORMAT_DISABLE=str(ARMS[arm][0]),QBH_PAPER_PIPELINE_DISABLE=str(ARMS[arm][1]),QBH_DENSE_R3='0',QBH_DENSE_R4='0');env.pop('QBH_REPLAY_DUMP_DIR',None)
 if key!='full' and repeat>1:env['QBH_LLAMA_REPLAY_REPEATS']=str(repeat)
 if key=='full':
  ids=np.fromfile(M/'frontend-a02/generation_prompt_token_ids_u32.bin','<u4').tolist();assert len(ids)==64
  forced=os.environ.get('QBH_PAPER_FIXED_TOKENS')=='1';fixed=read(R.parents[1]/'l32-0016/frontend-a02-reference/teacher.json')['u8_generated_ids'];row=[0,3 if forced else 2,16]+ids+fixed;f=d/'eval.bin';f.write_bytes(struct.pack('<4I',0x51424556,1,repeat,83)+b''.join(struct.pack('<83I',i,*row[1:]) for i in range(repeat)))
  env['QBH_EVAL_FILE']=root+'/'+tag.replace('/','_')+'.bin';adb('push',windows(f),env['QBH_EVAL_FILE'])
 if audit:
  env['QBH_REPLAY_DUMP_DIR']=root+'/'+tag.replace('/','_');adb('shell','mkdir '+env['QBH_REPLAY_DUMP_DIR'])
 cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+' '.join(shlex.quote(v) for v in argv)
 write(d/'protocol.json',dict(command=cmd,runtime=state,package=cfg,arm=arm,repeat=repeat,audit=audit));z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);write(d/'exit.json',dict(returncode=z.returncode))
 if audit:adb('pull',env['QBH_REPLAY_DUMP_DIR']+'/.',windows(d),check=False)
 assert z.returncode==0,(tag,z.returncode,z.stdout[-2000:],z.stderr[-1000:])
 rs=records(z.stdout);ps=[v for v in rs if isinstance(v,dict) and v.get('record')==('generation_profile' if key=='full' else 'replay_profile')];assert len(ps)==repeat*(16 if key=='full' else 2),(tag,len(ps))
 for v in ps:
  assert v['block_invocation_count']==cfg['layers'] and v['llama_fp32_residual']==1
  assert v['paper_format_disable']==ARMS[arm][0] and v['paper_pipeline_disable']==ARMS[arm][1]
  assert v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
  assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode'])
  ticks=sum(sum(v[k] for k in fields) for _,fields in MODULES)-v['generation_final_norm_ticks'];assert ticks==v['invocation_ticks'],(tag,ticks,v['invocation_ticks'])
 if key=='full':
  oracle=(read(R.parents[1]/'l32-0016/frontend-a02-reference/teacher.json') if arm!='FP' else dict(zip(['u8_generated_ids','u8_selected_codes'],(lambda z:(z['ids'],z['codes']))(read(R/(('fixed' if os.environ.get('QBH_PAPER_FIXED_TOKENS')=='1' else 'greedy')+'-fp-teacher-a02.json'))))));steps=[v for v in rs if isinstance(v,dict) and 'selected_logit_half_bits' in v]
  assert [v['selected_token_id'] for v in steps]==oracle['u8_generated_ids']*repeat
  assert [v['selected_logit_half_bits'] for v in steps]==oracle['u8_selected_codes']*repeat
 else:
  for v in ps:
   assert all(v[k]==0 for k in ['output_mismatches','output_nonfinite_count','cache_mismatches','cache_structure_mismatches','cache_nonfinite_count','scan_cache_append_mismatch_count'])
  if audit:
   for step in range(2):
    name=f'actual_replay_output_{step:02d}_f32.bin';rows=64 if step==0 else 1
    x=np.fromfile(d/name,'<f4').reshape(64,2048)[:rows];ref='reference_w4u8_block_output_f32.bin' if step==0 else 'replay_decode_reference_00_f32.bin';y=np.fromfile(Path(cfg['package'])/ref,'<f4').reshape(64,2048)[:rows]
    assert np.isfinite(x).all() and np.array_equal(x,y),(tag,step,int(np.count_nonzero(x!=y)))
    if arm!='FP':assert (d/name).read_bytes()==(Path(cfg['old'])/name).read_bytes(),(tag,step,'sealed output')
 out=dict(pass_all=True,arm=arm,repeat=repeat,key=key,profiles=len(ps),prefill_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='prefill'),decode_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='decode'),peak=max(v['vtcm_peak_plan_bytes'] for v in ps),audit=audit)
 write(d/'validated.json',out);print('PASS',tag,64e9/out['prefill_ns'],1e9/out['decode_ns'],flush=True);return out

def gates(keys,filename):
 out=[]
 for key in keys:
  for arm in ARMS:out.append(execute(key,arm,f'{key}-{arm}-audit',audit=key!='full'))
 write(R/filename,dict(pass_all=True,runs=out))
def fullgates():
 assert read(R/'slice_gate.json')['pass_all'];out=[]
 for a in ARMS:out.append(execute('chain16',a,f'chain16-{a}-audit',audit=True))
 for mode in ['greedy','fixed']:
  os.environ['QBH_PAPER_FIXED_TOKENS']='1' if mode=='fixed' else '0'
  for a in ARMS:out.append(execute('full',a,f'full-{a}-{mode}-audit'))
 write(R/'full_gate.json',dict(pass_all=True,runs=out))
def timing():
 assert read(R/'full_gate.json')['pass_all']
 for mode in ['fixed']:
  os.environ['QBH_PAPER_FIXED_TOKENS']='1' if mode=='fixed' else '0'
  aux=[execute('full',a,f'{mode}-aux-{a}-r1') for a in ARMS];write(R/f'{mode}_auxiliary.json',aux)
  for phase,cycles in [('short',5),('formal',10)]:
   out=[];keys=list(ARMS)
   for c in range(cycles):
    for a in keys[c%len(keys):]+keys[:c%len(keys)]:out.append(dict(cycle=c,**execute('full',a,f'{mode}-{phase}/{c:02d}-{a}',repeat=10)))
   write(R/f'{mode}_{phase}.json',dict(pass_all=True,runs=out))
if __name__=='__main__':
 action=sys.argv[1]
 if action=='prepare':prepare()
 elif action=='stage':stage(int(sys.argv[2]),trace='trace' in sys.argv[3:])
 elif action=='selected':gates(['l7'],'single_gate.json')
 elif action=='slice':
  assert read(R/'single_gate.json')['pass_all'];gates(['chain3'],'slice_gate.json')
 elif action=='fullgate':fullgates()
 elif action=='timing':timing()
