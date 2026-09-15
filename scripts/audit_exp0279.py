"""Independent chain outputs/native boundaries and full frontend/head checks."""
from common_exp0279 import *
from device_exp0279 import records
import numpy as np
from reference_w4u8_hmx import unpack_u8_hmx_activation,unpack_w4_codes,HmxU8Converter,projection_bias_words
from reference_exp0269 import norm
from verify_exp0167_generation import load_generation_qparams,audit_bias
def head(tag):
 preflight();d=R/tag;p=O/'sp2-fp32';v=read(d/'validated.json');assert v['audit'] and v['fp32']==2
 q=load_generation_qparams(p/'generation_qparams_u8.bin');iq=q['generation_final_norm_output'];oq=q['generation_lm_head_output'];g=np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2');xs=[]
 for step in range(34):
  x=np.fromfile(d/f'generation_hidden_step{step:02d}_f32.bin','<f4').reshape(1,2048);assert np.isfinite(x).all();expected=norm(x,g,iq)
  actual=unpack_u8_hmx_activation(np.fromfile(d/f'generation_norm_step{step:02d}_u8_native.bin','u1'),2048)[:1];assert np.array_equal(actual,expected),(step,'finalnorm');xs.append(actual[0])
 assert audit_bias(p,iq,oq)==0
 w=unpack_w4_codes(p,'generation_lm_head',151936,2048);ws=np.fromfile(p/'generation_lm_head_weight_w4_scale_f32.bin','<f4');cv=HmxU8Converter(S/'build/reference/qbh_hmx_u8_reference.so');x=np.asarray(xs,dtype='f8');logits=[]
 for start in range(0,len(w),1024):
  ww=w[start:start+1024];lo,hi=projection_bias_words(ww,ws[start:start+1024],iq,oq);acc=x@ww.astype('f8').T;logits.append(cv.convert(acc.astype('i8'),lo,hi))
 logits=np.concatenate(logits,1);ids=logits.argmax(1);codes=[[int(i),int(logits[j,i])] for j,i in enumerate(ids)];assert codes==v['selected_codes'],(codes,v['selected_codes'])
 write(d/'independent_gate.json',dict(pass_all=True,finalnorm_exact_values=34*2048,head_full_vocab_steps=34,head_bias_exact=True,selected_token_and_code_exact=True,embedding='original manifest-verified C64 FP16 embedding, expanded exactly to FP32 by unchanged scalar cast',scope='chain3 exact plus full physical/deterministic/frontend boundary checks, not full28 CPU transformer equivalence or quality/PPL'))
 print('HEAD_PASS',tag,flush=True)
