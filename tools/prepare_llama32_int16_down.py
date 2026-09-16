import sys,os,json,subprocess
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/llama32-htp');sys.path.insert(0,str(S/'tools'))
from llama32_a8_fp32_reference import layer,norm
from llama_u8_reference import load_qparams_bin,HmxU8Converter,project_w4u8
from llama_reference import sha256 as sha
M=Path('/mnt/d/llm_exp/models/llama32-htp');O=M/'l32-0040';R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0040');PR=R.parent/'l32-0038/A5'
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,indent=2)
def verify(p,h):
 assert sha(p/'manifest.json')==h
 z=read(p/'manifest.json')
 for n,v in z['files'].items():assert sha(p/n)==v['sha256'],n
 return z
def manifest(dst,parent,pm,changes,base):
 files={str(p.relative_to(dst)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(dst.rglob('*')) if p.is_file()}
 for n,v in files.items():
  if 'weight' in Path(n).name:assert v['sha256']==pm['files'][n]['sha256'],n
 write(dst/'manifest.json',dict(experiment='L32-0040',recipe='uniform INT16 Down nativeW4 FP32residual1',parent=str(parent),parent_sha256=sha(parent/'manifest.json'),ordinary_metadata_parent=str(base),ordinary_manifest_sha256=sha(base/'manifest.json'),changed_metadata=changes,weight_codes_scales_unchanged=True,files=files))
 write(R/(dst.name+'-package.json'),dict(path=str(dst),manifest_sha256=sha(dst/'manifest.json'),files=len(files)))
def clone(key,basekey,origlayer,base):
 cfg=read(PR/f'package-{basekey}.json');parent=Path(cfg['package']);pm=verify(parent,cfg['manifest_sha256']);dst=O/key;dst.mkdir();changes=[]
 for n,v in pm['files'].items():
  name=Path(n).name
  if name.endswith('.npy') or name.startswith('reference_kv_cache_') or name in ['reference_w4u8_block_output_f32.bin','replay_decode_reference_00_f32.bin','generation_expected_token_ids_u32.bin']:continue
  src=parent/n
  if name in ['qparams_u8.bin','silu_up_lut_u16.bin']:
   changes.append(n);continue
  target=dst/n;target.parent.mkdir(parents=True,exist_ok=True);os.link(src,target)
 from uniform_int16_down import make
 audits={}
 for root in sorted(parent.glob('layer*')):
  if root.is_dir() and (root/'qparams_u8.bin').exists():audits[root.name]=make(root,dst/root.name)
 write(R/(key+'-lut-audit.json'),audits)
 return dst,parent,pm,changes

def replay(key,basekey,layers,origlayer,base):
 dst,parent,pm,changes=clone(key,basekey,origlayer,base);past=[None]*layers
 for step in range(2):
  rows=64 if step==0 else 1;inp='reference_w4u8_block_input_f32.bin' if not step else 'replay_decode_input_00_f32.bin';x=np.fromfile(dst/inp,'<f4').reshape(64,2048)[:rows]
  cos=np.fromfile(dst/('rope_cos_f16.bin' if not step else 'replay_decode_rope_cos_00_f16.bin'),'<f2').reshape(64,64);sin=np.fromfile(dst/('rope_sin_f16.bin' if not step else 'replay_decode_rope_sin_00_f16.bin'),'<f2').reshape(64,64)
  for i in range(layers):
   root=dst/f'layer{i}';q=load_qparams_bin(root/'qparams_u8.bin');x,past[i],diag=layer(x,root,q,cos,sin,past[i],sp2=True);np.save(dst/f'a8_step{step}_hidden{i}.npy',x)
   for j,n in enumerate(['k','v']):
    ref=np.full((8,80,64),q['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');ref[:,:past[i][j].shape[1]]=past[i][j];ref.tofile(root/f'reference_kv_cache_{n}_u8.bin')
   print('REPLAY_REFERENCE',key,step,i,flush=True)
  out=np.zeros((64,2048),'<f4');out[:rows]=x;out.tofile(dst/('reference_w4u8_block_output_f32.bin' if not step else 'replay_decode_reference_00_f32.bin'))
  if layers==1:
   for n,v in diag.items():np.save(dst/f'a8_step{step}_{n}.npy',v)
 manifest(dst,parent,pm,changes,base)

def frontend(base):
 dst,parent,pm,changes=clone('full-int16','full',None,base)
 embed=np.memmap(dst/'generation_embedding_weight_f16.bin',dtype='<f2',mode='r',shape=(128256,2048));prompt=np.fromfile(dst/'generation_prompt_token_ids_u32.bin','<u4').tolist();fixed=read(R.parent/'l32-0016/frontend-a02-reference/teacher.json')['u8_generated_ids'];qs=[load_qparams_bin(dst/f'layer{i}/qparams_u8.bin') for i in range(16)];gq=load_qparams_bin(dst/'generation_qparams_u8.bin');gamma=np.fromfile(dst/'generation_final_norm_weight_f16.bin','<f2');cv=HmxU8Converter(S/'build/l32-0003/qbh_hmx_u8_reference.so')
 for mode in ['greedy','fixed']:
  caches=[None]*16;tokens=[];codes=[]
  for step in range(16):
   inp=prompt if not step else [tokens[-1] if mode=='greedy' else fixed[step-1]];x=np.array(embed[inp],dtype='f4')
   name='rope_cos_f16.bin' if not step else f'generation_decode_rope_cos_{step-1:02d}_f16.bin';cos=np.fromfile(dst/name,'<f2').reshape(64,64);sin=np.fromfile(dst/name.replace('cos','sin'),'<f2').reshape(64,64)
   for i in range(16):
    x,caches[i],_=layer(x,dst/f'layer{i}',qs[i],cos,sin,caches[i],sp2=True)
    np.save(dst/f'{mode}_step{step:02d}_layer{i:02d}_hidden.npy',x)
    if mode=='greedy' and step==0:
     for j,n in enumerate(['k','v']):
      ref=np.full((8,80,64),qs[i]['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');ref[:,:64]=caches[i][j];ref.tofile(dst/f'layer{i}/reference_kv_cache_{n}_u8.bin')
   if step==0 and mode=='greedy':x.tofile(dst/'reference_w4u8_block_output_f32.bin')
   act=norm(x[-1:],gamma,gq['generation_final_norm_output']);logits=project_w4u8(act,dst,'generation_lm_head',128256,2048,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)[0];token=int(logits.argmax());tokens.append(token);codes.append(int(logits[token]));print('FRONTEND_REFERENCE',mode,step,token,codes[-1],flush=True)
  write(R/(mode+'-int16-teacher.json'),dict(ids=tokens,codes=codes,fixed_inputs=fixed if mode=='fixed' else None,reference='independent FP32 ordered residual, exact HMX converter and integer attention, all16 layers'))
  if mode=='greedy':np.array(tokens,'<u4').tofile(dst/'generation_expected_token_ids_u32.bin')
 manifest(dst,parent,pm,changes,base)

def main():
 z=subprocess.check_output(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'ACTIVE_EXPERIMENT=L32-0040' in z
 O.mkdir(exist_ok=False);R.mkdir(exist_ok=True)
 base=M/'l32-0016/frontend-a02'
 replay('layer7-int16','l7',1,7,base);replay('chain3-int16','chain3',3,None,base);replay('chain16-int16','chain16',16,None,base);frontend(base)
if __name__=='__main__':main()
