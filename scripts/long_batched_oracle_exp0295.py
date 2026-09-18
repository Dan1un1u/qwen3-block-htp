"""Independent teacher-forced causal oracle, batched across known input tokens.
Integer pointwise/projection and causal attention are row-independent; verifies
all chunk and decode boundaries without reloading weights for every token.
This path never consumes device activations or output codes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys,argparse,numpy as np,torch
from pathlib import Path
S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S/'tools'));sys.path.insert(0,str(S/'scripts'))
from long_prefill_exp0295 import R,M,read,put,preflight,SIZE
from long_oracle_exp0295 import fnv
from long_reference_exp0295 import layer as qlayer,norm,H
def layer(x,p,q,c,s,seed=None):return qlayer(x,p,c,s,seed=seed)
from llama_u8_reference import load_qparams_bin,project_w4u8,HmxU8Converter
VOCAB=151936

def main(length,decode,tag):
 preflight();torch.set_num_threads(4);p=M/str(length);d=R/tag;d.mkdir(exist_ok=False)
 os.environ['QBH_SDK_REFERENCE_DIR']=str(d/'sdk-rsqrt')
 seeds=np.fromfile('/mnt/d/llm_exp/models/qwen3-block-htp/exp0257/prefix/prefix_kv_u8.bin','u1').reshape(28,2,8,128) if SIZE=='1.7B' else None
 ids=np.fromfile(p/'long_prompt_u32.bin','<u4');fixed=np.fromfile(p/'long_fixed_u32.bin','<u4')[:decode];ids=np.concatenate([ids,fixed])
 cos=np.fromfile(p/'long_rope_cos_f16.bin','<f2').reshape(-1,128)[:len(ids)];sin=np.fromfile(p/'long_rope_sin_f16.bin','<f2').reshape(-1,128)[:len(ids)]
 embedding=np.memmap(p/'generation_embedding_weight_f16.bin','<f2','r',shape=(VOCAB,H));x=embedding[ids].astype('f4')
 starts=list(range(0,length,64))+list(range(length,length+decode));ends=[min(i+64,length) for i in range(0,length,64)]+list(range(length+1,length+decode+1))
 first=[[] for _ in starts];out=[dict(step=j,position=a,rows=b-a,head=b>=length,token=None,code=None,layer_hashes=[]) for j,(a,b) in enumerate(zip(starts,ends))]
 for i in range(28):
  lp=p/f'layer{i}';q=load_qparams_bin(lp/'qparams_u8.bin');x,kv,_=layer(x,lp,q,cos,sin,seed=None if seeds is None else seeds[i])
  for j,(a,b) in enumerate(zip(starts,ends)):
   out[j]['layer_hashes'].append(fnv(x[a:b]));first[j].append(x[a].copy())
  for n,v in zip(['k','v'],kv):np.save(d/f'final_l{i:02d}_{n}.npy',v)
  if i in [0,13,27]:np.save(d/f'layer{i:02d}_all.npy',x)
  print('BATCHED_LAYER',i,flush=True)
 for j,v in enumerate(first):np.save(d/f's{j:02d}_layer_first_rows.npy',np.stack(v))
 gq=load_qparams_bin(p/'generation_qparams_u8.bin');gamma=np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2');indices=[ends[j]-1 for j,v in enumerate(out) if v['head']]
 act=norm(x[indices],gamma,gq['generation_final_norm_output']);cv=HmxU8Converter(S/'build/reference/qbh_hmx_u8_reference.so')
 logits=project_w4u8(act,p,'generation_lm_head',VOCAB,H,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)
 k=0
 for j,v in enumerate(out):
  if v['head']:
   v['token']=int(logits[k].argmax());v['code']=int(logits[k,v['token']]);k+=1
  put(d/f'step-{j:02d}.json',v)
 put(d/'summary.json',dict(length=length,decode=decode,steps=out,scope='independent full causal teacher-forced actual arithmetic; row batching, not floating/PPL teacher'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('length',type=int);a.add_argument('--decode',type=int,default=3);a.add_argument('--tag',required=True);v=a.parse_args();main(v.length,v.decode,v.tag)
