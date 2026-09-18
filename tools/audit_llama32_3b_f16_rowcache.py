import json,sys,shlex
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
from execute_llama32_3b_f16 import preflight,adb,windows,R,read,save,records
preflight();proto=read(R/'full-audit-c/protocol.json');old=proto['command'];prefix,tail=old.split(' ./qwen3_block_cli ',1)
env=dict(x.split('=',1) for x in shlex.split(prefix.split(' && ')[1]));root=proto['runtime']['remote']
root=read(R/'f16-l28d/runtime.json')['remote'];env.update(LD_LIBRARY_PATH=root,DSP_LIBRARY_PATH=root,ADSP_LIBRARY_PATH=root)
tag='rowcache-audit-b';d=R/tag;d.mkdir(exist_ok=False);env['QBH_KV_CACHE_LAYOUT']='row_major';env['QBH_GENERATION_AUDIT_DIR']=root+'/'+tag
adb('shell','mkdir '+env['QBH_GENERATION_AUDIT_DIR'])
cmd='cd '+root+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' ./qwen3_block_cli '+tail
save(d/'protocol.json',dict(parent=proto,command=cmd,scope='same model/inputs/math, row-major KV reconstruction control'))
z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);save(d/'exit.json',dict(returncode=z.returncode));rs=records(z.stdout);save(d/'records.json',rs)
(d/'audit').mkdir();adb('pull',env['QBH_GENERATION_AUDIT_DIR']+'/.',windows(d/'audit'))
assert z.returncode==0
oldrs=read(R/'full-audit-c/records.json')
tok=lambda xs:[(x['selected_token_id'],x['selected_logit_half_bits']) for x in xs if 'selected_logit_half_bits' in x]
assert tok(rs)==tok(oldrs)
checks=[]
for i in range(43):
 a=np.fromfile(d/f'audit/generation_hidden_norm_step{i:02d}_f16.bin','<u2')
 b=np.fromfile(R/f'full-audit-c/audit/generation_hidden_norm_step{i:02d}_f16.bin','<u2')
 m=int(np.count_nonzero(a!=b));checks.append(dict(step=i,mismatches=m,elements=len(a)));assert m==0,(i,m)
save(d/'validated.json',dict(pass_all=True,exact_hidden_norm_elements=sum(v['elements'] for v in checks),exact_token_steps=43,checks=checks))
print('ROW/NATIVE EXACT',sum(v['elements'] for v in checks),flush=True)
