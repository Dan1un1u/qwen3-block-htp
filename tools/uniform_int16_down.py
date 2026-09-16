"""Uniform INT16 Down input: offline LUT and independent exhaustive audits."""
import math,struct
import numpy as np
from llama_u8_reference import load_qparams_bin,QPARAM_RECORD,unpack_w4_codes

def make(parent,dest,width=8192):
 q=load_qparams_bin(parent/'qparams_u8.bin');alpha=max(abs(q['middle']['minimum']),abs(q['middle']['maximum']))
 delta=float(np.float32(alpha/32767));assert delta>0 and q['middle']['zero_point']==0
 g=(np.arange(256,dtype='f8')-q['gate']['zero_point'])*q['gate']['scale'];u=(np.arange(256,dtype='f8')-q['up']['zero_point'])*q['up']['scale']
 z=(g/(1+np.exp(-np.clip(g,-700,700))))[:,None]*u[None,:]
 v=np.clip(np.rint(z/delta),-32767,32767).astype('i4')
 independent=np.array([max(-32767,min(32767,round((float(gg)/(1+math.exp(-max(-700,min(700,float(gg)))))*float(uu))/delta))) for gg in g for uu in u],dtype='i4').reshape(256,256)
 assert np.array_equal(v,independent),'scalar/vector LUT disagreement'
 carrier=(v+32768).astype('<u2');assert np.array_equal((carrier.astype('i4')&255)+256*(carrier.astype('i4')>>8)-32768,v)
 carrier.tofile(dest/'silu_up_lut_u16.bin')
 raw=(parent/'qparams_u8.bin').read_bytes();out=[]
 for off in range(0,len(raw),QPARAM_RECORD.size):
  rec=list(QPARAM_RECORD.unpack_from(raw,off))
  if rec[0].split(b'\0')[0]==b'middle':rec[1]=delta;rec[2]=0
  out.append(QPARAM_RECORD.pack(*rec))
 (dest/'qparams_u8.bin').write_bytes(b''.join(out))
 nq=load_qparams_bin(dest/'qparams_u8.bin')
 for name in q:
  if name!='middle':assert nq[name]==q[name]
 w=unpack_w4_codes(parent,'down',2048,width).astype('i8');positive=np.maximum(w,0).sum(1);negative=np.minimum(w,0).sum(1);l1=np.abs(w).sum(1)
 # Production read_i32 uses signed24 storage. Bounds apply to every byte input,
 # including both components; checking only final signed32 would be insufficient.
 bound24=int(max(255*positive.max(),-255*negative.min()));bound32=int(32768*l1.max());assert bound24<2**23 and bound32<2**31,(bound24,bound32)
 full=np.arange(-32768,32768,dtype='i8');code=full+32768;assert np.array_equal((code&255)+256*(code>>8)-32768,full)
 # Adversarial extrema maximize positive/negative sums per output channel.
 for row in w:
  for sign in [-1,1]:
   x=np.where(row>=0,32767,-32768)*sign;x=np.clip(x,-32768,32767);c=x+32768
   low=int((c&255)@row);high=int((c>>8)@row);ws=int(row.sum());exact=int(x@row)
   merged=(low+256*high-32768*ws)&0xffffffff;merged=merged-(1<<32) if merged>=1<<31 else merged
   assert exact==merged and -2**23<=low<2**23 and -2**23<=high<2**23 and abs(exact)<2**31
 return dict(alpha=alpha,delta_f32=delta,sp2_delta=q['middle']['scale'],lut_entries=65536,unique_levels=int(np.unique(v).size),min_code=int(v.min()),max_code=int(v.max()),signed24_abs_bound=bound24,signed32_abs_bound=bound32,all_signed16_values=65536,adversarial_rows=len(w)*2,scalar_reference_exact=True,weights_unchanged=True,rounding='ties-to-even; symmetric [-32767,32767]',physical_carrier='v+32768 low/high bytes; unchanged native mode8')
