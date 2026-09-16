"""Independent chain outputs/native boundaries and full frontend/head checks."""
from common_exp0284 import *
from device_exp0284 import records
import numpy as np
from reference_w4u8_hmx import unpack_u8_hmx_activation,unpack_w4_codes,HmxU8Converter,projection_bias_words
from reference_exp0269 import norm
from verify_exp0167_generation import load_generation_qparams,audit_bias
BASE=983040;CAP=393216

def chain(tag, repeat_tag):
 preflight();p=package_path('full-int16' if 'INT16' in tag else 'full-a8' if 'A8' in tag else 'sp2-fp32');d=R/tag;a=read(d/'validated.json');r=read(R/repeat_tag/'validated.json');assert a['numerical_pass'] and r['output_hashes']==a['output_hashes']*10
 for step in range(2):
  rows=64 if step==0 else 1;b=np.fromfile(d/f'step{step:02d}_r3.bin','u1');assert len(b)==BASE+11*CAP
  for slot,name,k in [(2,'attention',2048),(5,'post',2048),(6,'gate',6144),(7,'up',6144)]:
   actual=unpack_u8_hmx_activation(b[BASE+slot*CAP:BASE+slot*CAP+64*k],k)[:rows];expect=np.load(p/f'chain_step{step}_last_{name}.npy');assert np.array_equal(actual,expect),(step,name)
  codes=b[2*CAP:BASE].reshape(24,4,64,32).transpose(0,2,1,3).reshape(24,64,128)[:,:rows]
  for sl,name in [(slice(0,16),'q'),(slice(16,24),'k')]:assert np.array_equal(codes[sl],np.load(p/f'chain_step{step}_last_{name}.npy').transpose(1,0,2)),(step,name)
  mid=np.load(p/f'chain_step{step}_last_middle.npy');actual=unpack_u8_hmx_activation(b[BASE+8*CAP:BASE+9*CAP],6144)
  sp2='QBH_SP2=8 ' in read(d/'protocol.json')['command']
  assert np.array_equal(actual[:rows],((mid+32768)&255).astype('u1') if sp2 else mid.astype('u1'))
  if step and sp2:assert np.array_equal(actual[4:5],((mid+32768)>>8).astype('u1'))
 write(d/'chain_gate.json',dict(pass_all=True,layers=3,fp32_output_exact=True,last_layer_QK_AV_post_gate_up_SP2_exact=True,repeat10_exact=True,independent_fp32_values=65*2048,physical_pass=True))
 print('CHAIN_PASS',flush=True)

def head(tag):
 preflight();d=R/tag;p=O/'sp2-fp32';v=read(d/'validated.json');assert v['audit'] and v['fp32']==2
 q=load_generation_qparams(p/'generation_qparams_u8.bin');iq=q['generation_final_norm_output'];oq=q['generation_lm_head_output'];g=np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2');xs=[]
 for step in range(16):
  x=np.fromfile(d/f'generation_hidden_step{step:02d}_f32.bin','<f4').reshape(1,2048);assert np.isfinite(x).all();expected=norm(x,g,iq)
  actual=unpack_u8_hmx_activation(np.fromfile(d/f'generation_norm_step{step:02d}_u8_native.bin','u1'),2048)[:1];assert np.array_equal(actual,expected),(step,'finalnorm');xs.append(actual[0])
 assert audit_bias(p,iq,oq)==0
 w=unpack_w4_codes(p,'generation_lm_head',151936,2048);ws=np.fromfile(p/'generation_lm_head_weight_w4_scale_f32.bin','<f4');cv=HmxU8Converter(S/'build/reference/qbh_hmx_u8_reference.so');x=np.asarray(xs,dtype='f8');logits=[]
 for start in range(0,len(w),1024):
  ww=w[start:start+1024];lo,hi=projection_bias_words(ww,ws[start:start+1024],iq,oq);acc=x@ww.astype('f8').T;logits.append(cv.convert(acc.astype('i8'),lo,hi))
 logits=np.concatenate(logits,1);ids=logits.argmax(1);codes=[[int(i),int(logits[j,i])] for j,i in enumerate(ids)];assert codes==v['selected_codes'],(codes,v['selected_codes'])
 if 'INT16' in tag:
  mode='fixed' if '-fixed-' in tag else 'greedy';teacher=read(R/(mode+'-int16-teacher-a03.json'));assert codes==teacher['selected_codes'],(codes,teacher['selected_codes'])
  for step in range(16):
   actual=np.fromfile(d/f'generation_hidden_step{step:02d}_f32.bin','<f4');ref=np.fromfile(R/'frontend-reference-a03'/f'{mode}_step{step:02d}_hidden_f32.bin','<f4');assert np.array_equal(actual,ref),(tag,step,int(np.count_nonzero(actual!=ref)))
  write(d/'full28_cpu_reference_gate.json',dict(pass_all=True,steps=16,hidden_values=16*2048,selected_codes_exact=True,reference='independent complete28-layer transformer including frozen prefixKV'))
 write(d/'independent_gate.json',dict(pass_all=True,finalnorm_exact_values=16*2048,head_full_vocab_steps=16,head_bias_exact=True,selected_token_and_code_exact=True,embedding='original manifest-verified C64 FP16 embedding, expanded exactly to FP32 by unchanged scalar cast',scope='chain3 exact plus full physical/deterministic/frontend boundary checks, not full28 CPU transformer equivalence or quality/PPL'))
 print('HEAD_PASS',tag,flush=True)
if __name__=='__main__':
 import sys
 if sys.argv[1]=='chain':chain(sys.argv[2],sys.argv[3])
 else:head(sys.argv[2])
