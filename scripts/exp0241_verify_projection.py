#!/usr/bin/env python3
import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from reference_w4u8_hmx import HmxU8Converter, load_qparams_bin, projection_bias_words, raw_u8s8_accumulator, unpack_u8_hmx_activation
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0241')
REF=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0240')
P=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0240/lpbq32')
A=R/(sys.argv[1] if len(sys.argv)>1 else 'projection_audit_01')
C=Path('/home/daniuniu/work/qwen3-block-htp/build/reference/qbh_hmx_u8_reference.so')
def main():
 torch.set_num_threads(4)
 conv=HmxU8Converter(C);qp=load_qparams_bin(P/'qparams_u8.bin');results=[]
 for name,inp,out in [('q','input_norm','q_projection'),('k','input_norm','k_projection'),('v','input_norm','v'),('o','attention_concat','attention_projection'),('gate','post_attention_norm','gate'),('up','post_attention_norm','up'),('down','middle','down')]:
  w=np.load(REF/f'{name}_reference_s8.npy');n,k=w.shape
  scale=np.fromfile(P/f'{name}_weight_w4_scale_f32.bin',dtype='<f4')
  lo,hi=projection_bias_words(w,scale,qp[inp],qp[out]);actualbias=np.fromfile(A/f'{name}_bias.bin',dtype='<u4').reshape(n//32,64)
  assert np.array_equal(lo,actualbias[:,:32].reshape(n)) and np.array_equal(hi.view('<u4'),actualbias[:,32:].reshape(n)),name+' bias'
  for step in range(9):
   raw=np.fromfile(A/f'step{step:02d}_{name}_projection.bin',dtype=np.uint8)
   x=unpack_u8_hmx_activation(raw[:64*k],k);actual=unpack_u8_hmx_activation(raw[64*k:],n)
   m=64 if step==0 else 1;x=x[:m];actual=actual[:m]
   accumulator=raw_u8s8_accumulator(x,w)
   # Independent int64 row reduction confirms the optimized integer matmul.
   assert np.array_equal(accumulator[0],w.astype(np.int64)@x[0].astype(np.int64))
   expected=conv.convert(accumulator,lo,hi)
   delta=actual.astype(np.int16)-expected.astype(np.int16)
   row={'projection':name,'step':step,'elements':int(delta.size),'mismatches':int(np.count_nonzero(delta)),'max_lsb':int(np.abs(delta).max()),'max_abs_accumulator':int(np.abs(accumulator).max())}
   results.append(row);print(row,flush=True)
 result={'pass':all(x['mismatches']==0 for x in results),'method':'independent S8 codes; U8xS8 integer matmul checked by int64 reduction; independently reconstructed affine bias; SDK native HMX output conversion','converter_sha256':hashlib.sha256(C.read_bytes()).hexdigest(),'all_seven_biases_exact':True,'projections':results}
 (R/(A.name+'_integer_audit.json')).write_text(json.dumps(result,indent=2)+'\n');assert result['pass']
if __name__=='__main__':main()
