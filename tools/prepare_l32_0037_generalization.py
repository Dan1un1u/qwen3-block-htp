"""Frozen64-token prompts,128-slot cache,63 decode positions and CPU HMX oracle."""
import os,sys,json,subprocess
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer
from llama_reference import rope,sha256 as sha
from llama32_fp32_residual import layer,norm
from llama_u8_reference import load_qparams_bin,project_w4u8,HmxU8Converter
S=Path('/home/daniuniu/work/llama32-htp');O=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0034/full-a01');P=O.parent.parent/'l32-0037/full-a01';R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0037')
def read(p):return json.loads(p.read_text())
def write(p,z):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 z=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'ACTIVE_EXPERIMENT=L32-0037' in z
 prior=read(R.parent/'l32-0034/deployment.json');assert sha(O/'manifest.json')==prior['manifest_sha256'];pm=read(O/'manifest.json');P.mkdir(parents=True,exist_ok=False);R.mkdir(parents=True,exist_ok=False)
 for n,v in pm['files'].items():
  if n.endswith('.npy'):continue
  assert sha(O/n)==v['sha256'],n;d=P/n;d.parent.mkdir(parents=True,exist_ok=True);os.link(O/n,d)
 cfg=read(Path('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin/config.json'));torch.set_num_threads(8);assert torch.cuda.is_available()
 for step in range(1,64):
  c,s=rope(cfg,torch.arange(63+step,127+step,device='cuda')[None],torch.float16)
  for name,x in [('cos',c),('sin',s)]:
   b=x.cpu().numpy().astype('<f2').tobytes();n=f'generation_decode_rope_{name}_{step-1:02d}_f16.bin'
   if (P/n).exists():assert (P/n).read_bytes()==b,n
   else:(P/n).write_bytes(b)
 tok=AutoTokenizer.from_pretrained('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin',local_files_only=True);texts=read(Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0281/prompts.json'));prompts={}
 for k,v in texts.items():
  ids=tok.encode(v['text'],add_special_tokens=False);assert len(ids)>=64;ids=ids[:64];prompts[k]=dict(text=v['text'],ids=ids,decoded=tok.decode(ids),token_sha256=__import__('hashlib').sha256(__import__('struct').pack('<64I',*ids)).hexdigest())
 write(R/'prompts.json',prompts);write(R/'protocol_freeze.json',dict(cases={'A16':['A',16],'A64':['A',64],'B64':['B',64]},arms={'LATEST':dict(format=0,wide=8),'ORIGINAL':dict(format=8,wide=0)},repeat=5,short=5,formal=10,capacity=128,max_kv=127,no_ppl=True,weights_changed=False))
 embed=np.memmap(P/'generation_embedding_weight_f16.bin','<f2','r',shape=(128256,2048));qs=[load_qparams_bin(P/f'layer{i}/qparams_u8.bin') for i in range(16)];gq=load_qparams_bin(P/'generation_qparams_u8.bin');gamma=np.fromfile(P/'generation_final_norm_weight_f16.bin','<f2');cv=HmxU8Converter(S/'build/l32-0003/qbh_hmx_u8_reference.so')
 for label,prompt in prompts.items():
  caches=[None]*16;tokens=[];codes=[]
  for step in range(64):
   x=np.array(embed[prompt['ids'] if not step else [tokens[-1]]],dtype='f4');n='rope_cos_f16.bin' if not step else f'generation_decode_rope_cos_{step-1:02d}_f16.bin';c=np.fromfile(P/n,'<f2').reshape(64,64);sn=np.fromfile(P/n.replace('cos','sin'),'<f2').reshape(64,64)
   for i in range(16):x,caches[i],_=layer(x,P/f'layer{i}',qs[i],c,sn,caches[i])
   d=R/'oracle'/label;d.mkdir(parents=True,exist_ok=True);np.save(d/f'hidden-{step:02d}.npy',x)
   a=norm(x[-1:],gamma,gq['generation_final_norm_output']);logits=project_w4u8(a,P,'generation_lm_head',128256,2048,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)[0];tokid=int(logits.argmax());tokens.append(tokid);codes.append(int(logits[tokid]));print('ORACLE',label,step,tokid,codes[-1],flush=True)
  write(R/f'teacher-{label}.json',dict(ids=tokens,codes=codes,steps=64,contract='independent original exact log2 full16 CPU HMX converter oracle; latest preserves probabilities, no PPL'))
 files={str(p.relative_to(P)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(P.rglob('*')) if p.is_file()}
 for n,v in files.items():
  if 'weight' in Path(n).name:assert v['sha256']==pm['files'][n]['sha256']
 write(P/'manifest.json',dict(experiment='L32-0037',layers=16,cache_capacity=128,generation_steps=64,parent=str(O),parent_manifest_sha256=sha(O/'manifest.json'),teacher_hashes={k:sha(R/f'teacher-{k}.json') for k in prompts},files=files))
 write(R/'package.json',dict(package=str(P),manifest_sha256=sha(P/'manifest.json'),original_command=prior['original_command'],parent_remote=prior['remote'],parent_package=str(O),teacher_hashes={k:sha(R/f'teacher-{k}.json') for k in prompts}))
 print('PREPARED',flush=True)
if __name__=='__main__':main()
