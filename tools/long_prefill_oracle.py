"""Independent actual-arithmetic oracle for L32-0050 (not a floating/PPL teacher)."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,argparse,ctypes
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S/'tools'))
from llama32_a8_fp32_reference import layer,norm
from llama_u8_reference import load_qparams_bin,project_w4u8,HmxU8Converter
from long_prefill import R,M,read,put,preflight
def fnv(x):
 h=1469598103934665603
 for b in memoryview(np.ascontiguousarray(x,dtype='<f4')).cast('B'):
  h=((h^b)*1099511628211)&0xffffffffffffffff
 return f'{h:016x}'
def main(length,decode):
 preflight();p=M/str(length);d=R/f'oracle-{length}';d.mkdir(exist_ok=False)
 ids=np.fromfile(p/'long_prompt_u32.bin','<u4');fixed=np.fromfile(p/'long_fixed_u32.bin','<u4')
 cos=np.fromfile(p/'long_rope_cos_f16.bin','<f2').reshape(-1,64);sin=np.fromfile(p/'long_rope_sin_f16.bin','<f2').reshape(-1,64)
 embed=np.memmap(p/'generation_embedding_weight_f16.bin','<f2','r',shape=(128256,2048))
 qs=[load_qparams_bin(p/f'layer{i}/qparams_u8.bin') for i in range(16)]
 caches=[None]*16;out=[];pos=0;chunks=(length+63)//64
 cv=HmxU8Converter(S/'build/l32-0003/qbh_hmx_u8_reference.so')
 for step in range(chunks+decode):
  inp=ids[pos:pos+64] if step<chunks else fixed[step-chunks:step-chunks+1]
  rows=len(inp);x=embed[inp].astype('f4');hashes=[]
  for i in range(16):
   x,caches[i],diag=layer(x,p/f'layer{i}',qs[i],cos[pos:pos+rows],sin[pos:pos+rows],caches[i],sp2=True)
   hashes.append(fnv(x))
   if i in [0,7,15]:
    np.save(d/f's{step:02d}_l{i:02d}_hidden.npy',x)
   print('ORACLE',length,step,i,hashes[-1],flush=True)
  head=step>=chunks-1;token=code=None
  if head:
   gq=load_qparams_bin(p/'generation_qparams_u8.bin');gamma=np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2')
   act=norm(x[-1:],gamma,gq['generation_final_norm_output'])
   logits=project_w4u8(act,p,'generation_lm_head',128256,2048,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)[0]
   token=int(logits.argmax());code=int(logits[token])
  out.append(dict(step=step,position=pos,rows=rows,layer_hashes=hashes,head=head,token=token,code=code))
  put(d/f'step-{step:02d}.json',out[-1]);pos+=rows
 for i,kv in enumerate(caches):
  for n,x in zip(['k','v'],kv):np.save(d/f'final_l{i:02d}_{n}.npy',x)
 put(d/'summary.json',dict(length=length,decode=decode,steps=out,scope='independent integer HMX arithmetic and ordered FP32 boundaries, not PPL'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('length',type=int);a.add_argument('--decode',type=int,default=3);v=a.parse_args();main(v.length,v.decode)
