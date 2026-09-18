import sys,shlex
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
from execute_llama32_3b_f16 import R,read,save,preflight,adb,windows,records
preflight();rt=read(R/'f16-l28d/runtime.json');root=rt['remote']
for arm in ['A8','SP2']:
 parent=R.parent/'l32-0046'/('audit-'+arm)
 proto=read(parent/'protocol.json');prefix,tail=proto['command'].split(' ./qwen3_block_cli ',1)
 env=dict(v.split('=',1) for v in shlex.split(prefix.split(' && ')[1]))
 d=R/('regression-3B-'+arm);d.mkdir(exist_ok=False)
 env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root,QBH_GENERATION_AUDIT_DIR=root+'/regression-'+arm)
 adb('shell','mkdir '+env['QBH_GENERATION_AUDIT_DIR'])
 cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+tail
 save(d/'protocol.json',dict(command=cmd,runtime=rt,parent=str(parent/'protocol.json')))
 z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);save(d/'exit.json',dict(returncode=z.returncode));rs=records(z.stdout);save(d/'records.json',rs);assert z.returncode==0,z.stderr[-1000:]
 (d/'audit').mkdir();adb('pull',env['QBH_GENERATION_AUDIT_DIR']+'/.',windows(d/'audit'))
 selected=lambda rows:[(v['selected_token_id'],v['selected_logit_half_bits']) for v in rows if 'selected_logit_half_bits' in v]
 assert selected(rs)==selected(read(parent/'records.json'))
 checks=[]
 for p in (parent/'audit').glob('*.bin'):
  a=d/'audit'/p.name;assert a.read_bytes()==p.read_bytes(),p.name;checks.append(p.name)
 save(d/'validated.json',dict(pass_all=True,exact_audit_files=len(checks),exact_token_steps=43,files=checks))
 print('REGRESSION_PASS',arm,len(checks),flush=True)
