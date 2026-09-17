"""Independent full28 FP16 residual contract, fixed trajectory; no PPL."""
from common_exp0285 import *
from reference_exp0285 import layer,norm
from reference_w4u8_hmx import unpack_w4_codes,HmxU8Converter,projection_bias_words
from verify_exp0167_generation import load_generation_qparams
import numpy as np
def fnv(b):
 v=14695981039346656037
 for a in b:v=((v^a)*1099511628211)&0xffffffffffffffff
 return v
def main():
 preflight();p=O/'sp2-fp32';out=R/'frontend-reference-a01';out.mkdir(exist_ok=False)
 seeds=np.fromfile(p.parents[1]/'exp0257/prefix/prefix_kv_u8.bin','u1').reshape(28,2,8,128)
 ids=np.fromfile(p/'generation_prompt_token_ids_u32.bin','<u4').tolist();fixed=read(R/'fixed_tokens.json')['ids']
 embed=np.memmap(p/'generation_embedding_weight_f16.bin','<f2',mode='r',shape=(151936,2048))
 g=np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2');q=load_generation_qparams(p/'generation_qparams_u8.bin')
 caches=[None]*28;hashes=[];acts=[]
 for step in range(16):
  x=np.array(embed[ids if not step else [fixed[step-1]]],dtype='f4')
  name='rope_cos_f16.bin' if not step else f'generation_decode_rope_cos_{step-1:02d}_f16.bin'
  cos=np.fromfile(p/name,'<f2').reshape(64,128);sin=np.fromfile(p/name.replace('cos','sin'),'<f2').reshape(64,128)
  row=[]
  for i in range(28):
   x,caches[i],_=layer(x,p/f'layer{i}',cos,sin,caches[i],seed=seeds[i] if not step else None)
   # Hexagon vcvt canonicalizes negative zero; no observed nonfinite values allowed.
   assert np.isfinite(x).all();x[x==0]=0
   row.append(fnv(x.astype('<f2').tobytes()))
   if step==0:print('CPU_LAYER',i,flush=True)
  hashes.append(row);x[-1:].astype('<f2').tofile(out/f'step{step:02d}_hidden_f16.bin')
  a=norm(x[-1:],g,q['generation_final_norm_output']);acts.append(a[0]);a.tofile(out/f'step{step:02d}_norm_u8.bin')
  print('CPU_STEP',step,flush=True)
 w=unpack_w4_codes(p,'generation_lm_head',151936,2048);ws=np.fromfile(p/'generation_lm_head_weight_w4_scale_f32.bin','<f4');cv=HmxU8Converter(S/'build/reference/qbh_hmx_u8_reference.so')
 acts=np.asarray(acts,'f8');logits=[]
 for st in range(0,len(w),1024):
  ww=w[st:st+1024];lo,hi=projection_bias_words(ww,ws[st:st+1024],q['generation_final_norm_output'],q['generation_lm_head_output'])
  logits.append(cv.convert((acts@ww.astype('f8').T).astype('i8'),lo,hi))
 logits=np.concatenate(logits,1);inds=logits.argmax(1)
 write(out/'reference.json',dict(layer_hashes=hashes,selected_codes=[[int(t),int(logits[i,t])] for i,t in enumerate(inds)],quality_claim=False))
 print('CPU_FULL_PASS',flush=True)
if __name__=='__main__':main()
