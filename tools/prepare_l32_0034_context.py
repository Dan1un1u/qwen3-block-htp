"""Frozen linked longer-context overlay; no calibration, weights or DSP changes."""
import os,sys,json,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
from llama_reference import rope,sha256 as sha
from llama32_fp32_residual import layer,norm
from llama_u8_reference import load_qparams_bin,project_w4u8,HmxU8Converter
S=Path('/home/daniuniu/work/llama32-htp');O=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0016/frontend-a02');P=O.parent.parent/'l32-0034/full-a01';R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0034')
def read(p):return json.loads(p.read_text())
def write(p,z):
 with p.open('x') as f:json.dump(z,f,indent=2);f.write('\n')
def main():
 t=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'ACTIVE_EXPERIMENT=L32-0034' in t
 prior=read(R.parent/'l32-0033/package-full.json');assert sha(O/'manifest.json')==prior['manifest_sha256'];pm=read(O/'manifest.json')
 P.mkdir(parents=True,exist_ok=False);R.mkdir(parents=True,exist_ok=True)
 for n,v in pm['files'].items():
  assert sha(O/n)==v['sha256'],n
  d=P/n;d.parent.mkdir(parents=True,exist_ok=True)
  if 'cache_' in Path(n).name:
   x=np.fromfile(O/n,'u1').reshape(8,80,64);q=load_qparams_bin(O/Path(n).parent/'qparams_u8.bin');key='k_rope' if '_k_' in n else 'v';y=np.full((8,128,64),q[key]['zero_point'],'u1');y[:,:80]=x;y.tofile(d)
  else:os.link(O/n,d)
 cfg=read(Path('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin/config.json'));torch.set_num_threads(8);assert torch.cuda.is_available()
 for step in range(1,34):
  cos,sin=rope(cfg,torch.arange(63+step,127+step,device='cuda')[None],torch.float16)
  for name,x in [('cos',cos),('sin',sin)]:
   b=x.cpu().numpy().astype('<f2').tobytes();n=f'generation_decode_rope_{name}_{step-1:02d}_f16.bin'
   if (P/n).exists():assert b==(P/n).read_bytes(),n
   else:(P/n).write_bytes(b)
 write(R/'overlay_freeze.json',dict(parent_manifest_sha256=sha(O/'manifest.json'),first15_rope_exact=True,new_decode_rope=18,cache_capacity=128,generation_steps=34,weights_changed=False))
 embed=np.memmap(P/'generation_embedding_weight_f16.bin','<f2','r',shape=(128256,2048));prompt=np.fromfile(P/'generation_prompt_token_ids_u32.bin','<u4').tolist();qs=[load_qparams_bin(P/f'layer{i}/qparams_u8.bin') for i in range(16)];gq=load_qparams_bin(P/'generation_qparams_u8.bin');gamma=np.fromfile(P/'generation_final_norm_weight_f16.bin','<f2');cv=HmxU8Converter(S/'build/l32-0003/qbh_hmx_u8_reference.so');caches=[None]*16;tokens=[];codes=[]
 old=read(R.parent/'l32-0016/frontend-a02-reference/teacher.json')
 for step in range(34):
  x=np.array(embed[prompt if not step else [tokens[-1]]],dtype='f4');n='rope_cos_f16.bin' if not step else f'generation_decode_rope_cos_{step-1:02d}_f16.bin';c=np.fromfile(P/n,'<f2').reshape(64,64);s=np.fromfile(P/n.replace('cos','sin'),'<f2').reshape(64,64)
  for i in range(16):x,caches[i],_=layer(x,P/f'layer{i}',qs[i],c,s,caches[i])
  np.save(P/f'oracle_step{step:02d}_hidden.npy',x)
  a=norm(x[-1:],gamma,gq['generation_final_norm_output']);logits=project_w4u8(a,P,'generation_lm_head',128256,2048,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)[0];tok=int(logits.argmax());tokens.append(tok);codes.append(int(logits[tok]))
  if step<16:assert tok==old['u8_generated_ids'][step] and codes[-1]==old['u8_selected_codes'][step],(step,tok,codes[-1])
  print('LONG_CONTEXT_ORACLE',step,tok,codes[-1],flush=True)
 write(R/'teacher.json',dict(ids=tokens,codes=codes,steps=34,first16_original_exact=True,contract='original log2 wide0 nativeW4 SP2 FP32residual1; independent all16 layer HMX converter/reference; no PPL'))
 files={str(p.relative_to(P)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(P.rglob('*')) if p.is_file()}
 for n,v in files.items():
  if 'weight' in Path(n).name:assert v['sha256']==pm['files'][n]['sha256']
 write(P/'manifest.json',dict(experiment='L32-0034',layers=16,cache_capacity=128,generation_steps=34,parent=str(O),parent_manifest_sha256=sha(O/'manifest.json'),frontend_teacher_sha256=sha(R/'teacher.json'),original16_ids_file_retained=True,files=files))
 write(R/'package.json',dict(package=str(P),manifest_sha256=sha(P/'manifest.json'),original_command=prior['command'],parent_remote=prior['remote'],parent_package=str(O),teacher_sha256=sha(R/'teacher.json')))
 print('LOCAL_PACKAGE_READY',sha(P/'manifest.json'),flush=True)
if __name__=='__main__':main()
