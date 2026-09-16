"""Untimed full INT16 hidden-state audit against independent complete CPU trajectories."""
import sys,json,shlex,numpy as np
from pathlib import Path
S=Path('/home/daniuniu/work/llama32-htp');sys.path.insert(0,str(S/'tools'))
from execute_llama32_int16_down import R,read,write,preflight,adb,windows,records
P=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0040/full-int16')
preflight();out=[]
for mode in ['fixed','greedy']:
 old=R/f'full-INT16-{mode}-audit';cfg=read(old/'protocol.json');remote=cfg['runtime']['remote']+'/hidden-'+mode+'-a01';d=R/('full-hidden-'+mode+'-a01');d.mkdir(exist_ok=False);assert adb('shell','test ! -e '+remote,check=False).returncode==0;adb('shell','mkdir '+remote)
 cmd=cfg['command'].replace(' ./qwen3_block_cli ',' QBH_GENERATION_BOUNDARY_AUDIT=1 QBH_GENERATION_AUDIT_DIR='+shlex.quote(remote)+' ./qwen3_block_cli ');write(d/'protocol.json',dict(command=cmd,parent=str(old),runtime=cfg['runtime']))
 z=adb('shell',cmd,check=False);(d/'stdout.txt').write_text(z.stdout);(d/'stderr.txt').write_text(z.stderr);write(d/'exit.json',dict(returncode=z.returncode));adb('pull',remote+'/.',windows(d));assert z.returncode==0
 rs=records(z.stdout);steps=[v for v in rs if isinstance(v,dict) and 'selected_logit_half_bits' in v];oracle=read(R/(mode+'-int16-teacher.json'));assert [v['selected_token_id'] for v in steps]==oracle['ids'];assert [v['selected_logit_half_bits'] for v in steps]==oracle['codes']
 for step in range(16):
  actual=np.fromfile(d/f'generation_hidden_step{step:02d}_f32.bin','<f4');ref=np.load(P/f'{mode}_step{step:02d}_layer15_hidden.npy')[-1:].reshape(-1);assert np.array_equal(actual,ref),(mode,step,int(np.count_nonzero(actual!=ref)),float(np.max(np.abs(actual-ref))))
 row=dict(pass_all=True,mode=mode,independent_full_model_hidden_exact_values=32768,token_ids_and_logits_exact=True);write(d/'validated.json',row);out.append(row);print('HIDDEN_EXACT',mode,flush=True)
write(R/'full_hidden_gate.json',dict(pass_all=True,runs=out))
