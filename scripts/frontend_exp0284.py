"""Independent full28 INT16 trajectory, preserving original fixed token0 KV seed."""
from pathlib import Path
import os,json,hashlib,sys,numpy as np
from reference_exp0284 import layer,norm
from reference_w4u8_hmx import unpack_w4_codes,HmxU8Converter,projection_bias_words
from verify_exp0167_generation import load_generation_qparams
from common_exp0277 import sha,read,write,S
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0284');P=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0284/full-int16')
def main():
 import subprocess
 z=subprocess.check_output(['python3',str(S)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],text=True);assert 'EXPERIMENT=EXP-0284' in z
 seedpath=P.parents[1]/'exp0257/prefix/prefix_kv_u8.bin';assert sha(seedpath)=='7683237318d42ac5cc80052fb53619205a3d82a0d1377bcbbaed78d7c7683b91';seeds=np.fromfile(seedpath,'u1').reshape(28,2,8,128)
 out=R/'frontend-reference';out.mkdir(exist_ok=False);prompt=np.fromfile(P/'generation_prompt_token_ids_u32.bin','<u4').tolist();fixed=read(R.parent/'exp0282/A5-fair-a02/fixed_tokens.json')['ids'];embed=np.memmap(P/'generation_embedding_weight_f16.bin','<f2',mode='r',shape=(151936,2048));gamma=np.fromfile(P/'generation_final_norm_weight_f16.bin','<f2');gq=load_generation_qparams(P/'generation_qparams_u8.bin');iq=gq['generation_final_norm_output'];oq=gq['generation_lm_head_output'];cv=HmxU8Converter(S/'build/reference/qbh_hmx_u8_reference.so');w=unpack_w4_codes(P,'generation_lm_head',151936,2048);ws=np.fromfile(P/'generation_lm_head_weight_w4_scale_f32.bin','<f4')
 for mode in ['fixed','greedy']:
  caches=[None]*28;codes=[]
  for step in range(16):
   ids=prompt if not step else [fixed[step-1] if mode=='fixed' else codes[-1][0]];x=np.array(embed[ids],dtype='f4');name='rope_cos_f16.bin' if not step else f'generation_decode_rope_cos_{step-1:02d}_f16.bin';cos=np.fromfile(P/name,'<f2').reshape(64,128);sin=np.fromfile(P/name.replace('cos','sin'),'<f2').reshape(64,128)
   for i in range(28):
    x,caches[i],_=layer(x,P/f'layer{i}',cos,sin,caches[i],seed=seeds[i] if not step else None);np.save(out/f'{mode}_step{step:02d}_layer{i:02d}_hidden.npy',x)
   x[-1:].tofile(out/f'{mode}_step{step:02d}_hidden_f32.bin');act=norm(x[-1:],gamma,iq);logits=[]
   for start in range(0,len(w),1024):
    ww=w[start:start+1024];lo,hi=projection_bias_words(ww,ws[start:start+1024],iq,oq);acc=act.astype('f8')@ww.astype('f8').T;logits.append(cv.convert(acc.astype('i8'),lo,hi))
   logits=np.concatenate(logits,1)[0];idx=int(logits.argmax());codes.append([idx,int(logits[idx])]);print('Q_REFERENCE',mode,step,codes[-1],flush=True)
  write(R/(mode+'-int16-teacher.json'),dict(selected_codes=codes,reference='independent full28 FP32 residual, uniform INT16 Down, original prefixKV/attention/head; no device inputs',seed_sha256=sha(seedpath)))
if __name__=='__main__':main()
