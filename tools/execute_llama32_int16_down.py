"""Latest optimized Llama paper ablations; immutable independent teachers."""
import sys,os,json,shlex,subprocess,shutil,struct,statistics
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/llama32-htp');sys.path.insert(0,str(S/'tools'))
from run_llama32_layer import adb,windows
from run_llama32_frontend import records
from llama_reference import sha256 as sha
from report_llama32_pipeline_profile import MODULES
ROOT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0040');R=ROOT
M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0016')
PHASES={'A1':{'B11':(0,0),'B01':(3,0),'B10':(0,31),'B00':(3,31)},'A2':{'ALL':(0,0),'QKV':(0,1),'FFN':(0,2),'EPILOGUE':(0,12),'LOOKAHEAD':(0,16)},'A3':{'ALL':(0,0),'SPLIT':(4,0)},'A5':{'SP2':(0,0),'A8':(0,0)},'A9':{'LOG2':(0,0),'FP':(0,0)}}
ARMS={'A8':(0,0),'SP2':(0,0),'INT16':(0,0)}
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,indent=2);f.write('\n')
def preflight():
 t=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'ACTIVE_EXPERIMENT=L32-0040' in t

def prepare():
 preflight();old=ROOT.parent/'l32-0038';ledger=read(old/'EVIDENCE_SHA256.json');assert sha(old/'EVIDENCE_SHA256.json')=='a7ba00b82f2b76983f33320c658eaab86754ed366f0c103e06aba7db5e93f93f'
 for key in ['l7','chain3','chain16','full']:
  for suffix in ['', '-a8']:
   f='package-'+key+suffix+'.json';src=old/'A5'/f;assert sha(src)==ledger['files']['A5/'+f]['sha256'];shutil.copyfile(src,R/f)
   cfg=read(src);pkg=Path(cfg['package']);assert sha(pkg/'manifest.json')==cfg['manifest_sha256'];mf=read(pkg/'manifest.json')
   for n,v in mf['files'].items():assert sha(pkg/n)==v['sha256'],n
   names=[n for n in mf['files'] if not n.endswith('.npy')]
   for i in range(0,len(names),32):
    z=adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(z)==len(names[i:i+32])
    for line in z:
     h,n=line.split(None,1);assert h==mf['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
   print('CONTROL_VERIFIED',key,suffix,flush=True)
  parent=read(R/f'package-{key}.json');name={'l7':'layer7-int16','chain3':'chain3-int16','chain16':'chain16-int16','full':'full-int16'}[key];pkg=M.parent/'l32-0040'/name;ref=read(R/(name+'-package.json'));assert sha(pkg/'manifest.json')==ref['manifest_sha256'];mf=read(pkg/'manifest.json');pm=read(Path(parent['package'])/'manifest.json');remote='/data/local/tmp/llama32-htp/l32-0040-models/'+name
  assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
  names=[n for n in mf['files'] if not n.endswith('.npy')];dirs=sorted({str(Path(n).parent) for n in names if str(Path(n).parent)!='.'});adb('shell','mkdir -p '+' '.join(shlex.quote(remote+'/'+n) for n in dirs));links=[];copies=[]
  for n in names:
   assert sha(pkg/n)==mf['files'][n]['sha256'],n
   if n in pm['files'] and mf['files'][n]['sha256']==pm['files'][n]['sha256']:links.append((parent['remote']+'/'+n,remote+'/'+n))
   else:copies.append(n)
  for i in range(0,len(links),24):adb('shell',' && '.join('ln -s '+shlex.quote(a)+' '+shlex.quote(b) for a,b in links[i:i+24]))
  for n in copies:adb('push',windows(pkg/n),remote+'/'+n)
  for i in range(0,len(names),32):
   z=adb('shell','sha256sum '+' '.join(shlex.quote(remote+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(z)==len(names[i:i+32])
   for line in z:
    h,n=line.split(None,1);assert h==mf['files'][n.removeprefix(remote+'/')]['sha256'],n
  write(R/f'package-{key}-int16.json',dict(parent,package=str(pkg),manifest_sha256=ref['manifest_sha256'],remote=remote,fresh_files=copies,immutable_links=len(links)))
  print('INT16_DEPLOYED',key,flush=True)
 for mode in ['greedy','fixed']:
  f=mode+'-a8-teacher.json';src=old/'A5'/f;assert sha(src)==ledger['files']['A5/'+f]['sha256'];shutil.copyfile(src,R/f)
 write(R/'provenance.json',dict(prior_ledger=sha(old/'EVIDENCE_SHA256.json'),int16_oracles={m:sha(R/(m+'-int16-teacher.json')) for m in ['greedy','fixed']},same_native_path=True,other_weights_unchanged=True))
def stage(count,trace=False):
 preflight();seal=read(S/'build/llama-build-seal.json');assert seal['paper_trace']==trace;assert seal['source_head']==subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 for b in ['android_ReleaseG_aarch64','hexagon_ReleaseG_toolv19_v79']:
  t=(S/b/'CMakeCache.txt').read_text();assert ('QBH_PAPER_TRACE:BOOL='+('ON' if trace else 'OFF')) in t;assert 'QBH_MODEL_LLAMA32:BOOL=ON' in t and f'QBH_LLAMA_LAYER_COUNT:STRING={count}\n' in t
 attempt=1
 while (R/f'binaries-l{count}-a{attempt}').exists():attempt+=1
 d=R/f'binaries-l{count}-a{attempt}';d.mkdir();remote=f'/data/local/tmp/llama32-htp/l32-0040-{R.name}-l{count}-a{attempt}'
 assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir -p '+remote)
 for n,h in seal['files'].items():
  p=Path(n);assert sha(p)==h;shutil.copy2(p,d/p.name);adb('push',windows(p),remote+'/'+p.name);assert adb('shell','sha256sum '+remote+'/'+p.name).stdout.split()[0]==h
 adb('shell','chmod 755 '+remote+'/qwen3_block_cli '+remote+'/llama_sp2_cli');write(d/'seal.json',seal)
 (R/f'runtime-l{count}.json').write_text(json.dumps(dict(remote=remote,seal=seal,archive=str(d)),indent=2))
 print('STAGED',count,flush=True)
def execute(key,arm,tag,repeat=1,audit=False):
 preflight();cfg=read(R/f"package-{key}{'-int16' if arm=='INT16' else '-a8' if arm=='A8' else ''}.json");state=read(R/f"runtime-l{cfg['layers']}.json");root=state['remote'];assert audit or state['seal']['paper_trace'] is False;d=R/tag;d.mkdir(parents=True,exist_ok=False)
 prefix,args=cfg['command'].split(' ./qwen3_block_cli ',1);oldroot=prefix.split(' && ')[0].removeprefix('cd ');env=dict(v.split('=',1) for v in shlex.split(prefix.split(' && ')[1]));argv=shlex.split(args);argv[0]=cfg['remote'];argv[2]=str(repeat if key!='full' else 1)
 env.update(QBH_LLAMA_SP2='0' if arm=='A8' else '8',QBH_WIDE_SCORE='7' if arm=='FP' else '8',LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_PAPER_FORMAT_DISABLE=str(ARMS[arm][0]),QBH_PAPER_PIPELINE_DISABLE=str(ARMS[arm][1]),QBH_DENSE_R3='0',QBH_DENSE_R4='0');env.pop('QBH_REPLAY_DUMP_DIR',None)
 if key!='full' and repeat>1:env['QBH_LLAMA_REPLAY_REPEATS']=str(repeat)
 if key=='full':
  ids=np.fromfile(M/'frontend-a02/generation_prompt_token_ids_u32.bin','<u4').tolist();assert len(ids)==64
  forced=os.environ.get('QBH_PAPER_FIXED_TOKENS')=='1';fixed=read(ROOT.parent/'l32-0016/frontend-a02-reference/teacher.json')['u8_generated_ids'];row=[0,3 if forced else 2,16]+ids+fixed;f=d/'eval.bin';f.write_bytes(struct.pack('<4I',0x51424556,1,repeat,83)+b''.join(struct.pack('<83I',i,*row[1:]) for i in range(repeat)))
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
  oracle=(read(ROOT.parent/'l32-0016/frontend-a02-reference/teacher.json') if arm=='SP2' else dict(zip(['u8_generated_ids','u8_selected_codes'],(lambda z:(z['ids'],z['codes']))(read(R/(('fixed' if os.environ.get('QBH_PAPER_FIXED_TOKENS')=='1' else 'greedy')+('-int16-teacher.json' if arm=='INT16' else '-a8-teacher.json')))))));steps=[v for v in rs if isinstance(v,dict) and 'selected_logit_half_bits' in v]
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
    if arm=='SP2':assert (d/name).read_bytes()==(Path(cfg['old'])/name).read_bytes(),(tag,step,'sealed output')
 out=dict(pass_all=True,arm=arm,repeat=repeat,key=key,profiles=len(ps),prefill_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='prefill'),decode_ns=statistics.mean(v['host_wall_ns'] for v in ps if v['mode']=='decode'),peak=max(v['vtcm_peak_plan_bytes'] for v in ps),audit=audit)
 write(d/'validated.json',out);print('PASS',tag,64e9/out['prefill_ns'],1e9/out['decode_ns'],flush=True);return out

def gates(keys,filename):
 out=[]
 for key in keys:
  for a in ARMS:out.append(execute(key,a,f'{key}-{a}-audit',audit=True))
 write(R/filename,dict(pass_all=True,runs=out))
def fullgates():
 assert read(R/'slice_gate.json')['pass_all'];out=[]
 for a in ARMS:out.append(execute('chain16',a,f'chain16-{a}-audit',audit=True))
 for mode in (['greedy','fixed'] if True else ['greedy']):
  os.environ['QBH_PAPER_FIXED_TOKENS']='1' if mode=='fixed' else '0'
  for a in ARMS:out.append(execute('full',a,f'full-{a}-{mode}-audit'))
 write(R/'full_gate.json',dict(pass_all=True,runs=out))
def timing():
 assert read(R/'full_gate.json')['pass_all']
 for mode in ['fixed','greedy']:
  os.environ['QBH_PAPER_FIXED_TOKENS']='1' if mode=='fixed' else '0'
  for a in ARMS:execute('full',a,f'{mode}-warmup-{a}',repeat=1)
  write(R/(mode+'_auxiliary.json'),[execute('full',a,f'{mode}-aux-{a}-r1') for a in ARMS])
  for phase,n in [('short',5),('formal',10)]:
   out=[];keys=list(ARMS)
   for c in range(n):
    j=c%3;order=keys[j:]+keys[:j]
    if c%2:order=order[::-1]
    for a in order:out.append(dict(cycle=c,**execute('full',a,f'{mode}-{phase}/{c:02d}-{a}',repeat=10)))
   write(R/(mode+'_'+phase+'.json'),dict(pass_all=True,runs=out))
if __name__=='__main__':
 action=sys.argv[1]
 if action=='prepare':prepare()
 elif action=='stage':stage(int(sys.argv[2]))
 elif action=='selected':gates(['l7'],'single_gate.json')
 elif action=='slice':
  assert read(R/'single_gate.json')['pass_all'];gates(['chain3'],'slice_gate.json')
 elif action=='fullgate':fullgates()
 elif action=='timing':timing()
 else:raise ValueError(action)
