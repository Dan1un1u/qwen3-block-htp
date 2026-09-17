import sys,json,shlex,subprocess
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/llama32-htp');sys.path.insert(0,str(S/'tools'))
from execute_llama32_3b_opt import preflight,stage,read,save,R,adb,windows,records
preflight();stage('binaries-1b-l1-regression-a01');rt=read(R/'binaries-1b-l1-regression-a01/runtime.json');assert rt['seal']['model_size']=='1B'
cfg=read(R.parent/'l32-0040/package-l7.json');root=rt['remote'];d=R/'1b-regression-a01';d.mkdir(exist_ok=False)
prefix,args=cfg['command'].split(' ./qwen3_block_cli ',1);env=dict(v.split('=',1) for v in shlex.split(prefix.split(' && ')[1]));env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_WIDE_SCORE='8',QBH_REPLAY_DUMP_DIR=root,QBH_PAPER_FORMAT_DISABLE='0',QBH_PAPER_PIPELINE_DISABLE='0')
cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+args;save(d/'protocol.json',dict(command=cmd,runtime=rt,package=cfg));z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);assert z.returncode==0,z.stderr
rs=records(z.stdout);save(d/'records.json',rs);ps=[v for v in rs if isinstance(v,dict) and v.get('record')=='replay_profile'];assert len(ps)==2
for v in ps:assert all(v[k]==0 for k in ['output_mismatches','output_nonfinite_count','cache_mismatches','cache_structure_mismatches','intermediate_spill_fill_count','ledger_unattributed_ticks'])
for step in range(2):
 n=f'actual_replay_output_{step:02d}_f32.bin';adb('pull',root+'/'+n,windows(d/n));a=np.fromfile(d/n,'<f4').reshape(64,2048)[:64 if step==0 else 1];f='reference_w4u8_block_output_f32.bin' if step==0 else 'replay_decode_reference_00_f32.bin';b=np.fromfile(Path(cfg['package'])/f,'<f4').reshape(64,2048)[:len(a)];assert np.array_equal(a,b)
save(d/'validated.json',dict(pass_all=True,numerical_exact=True,cache_exact=True,model='1B',scope='frozen layer7 prefill/decode bounded shared-native regression'))
print('1B_REGRESSION_PASS')