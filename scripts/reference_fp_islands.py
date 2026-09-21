"""Independent Qwen no-rotation FP32 residual + native SP2 contract."""
import argparse,os,struct,functools
import numpy as np
from common_exp0269 import *
from reference_w4u8_hmx import load_qparams_bin,unpack_w4_codes,HmxU8Converter,projection_bias_words,exact_qk_norm_rope_u8
from integer_attention_exp0252 import numpy_oracle as legacy_numpy_oracle,config
from profile_fp_islands import fp_table
def numpy_oracle(q,k,v,valid,c,mode):
 # AV multiplier==1 is converted directly with output_zero_point by the
 # frozen DSP pack_v contract. The historical oracle incorrectly saturated
 # at center128 first, then translated; clipping and translation do not commute.
 res=legacy_numpy_oracle(q,k,v,valid,c,'wide_float')
 if c['avm']==1:
  vv=np.asarray(v,np.int64)-c['vz']
  vv=np.clip(np.sign(vv)*((np.abs(vv)*c['vn']+c['vd']//2)//c['vd']),-128,127)
  div=1<<c['avs'];acc=res['probability']@vv
  res['av']=np.clip((acc+c['oz']*div+(div//2 if c['avs'] else 0))//div,0,255)
 return res

CV=HmxU8Converter(S/'build/reference/qbh_hmx_u8_reference.so')
def verify(p):
 m=read(p/'manifest.json')
 for n,v in m['files'].items():assert sha(p/n)==v['sha256'],str(p/n)
 return m
def norm(x,g,q):
 x=np.asarray(x,dtype='f4');s=np.cumsum(x*x,axis=1,dtype='f4')[:,-1:]
 inv=np.float32(1)/np.sqrt(s/np.float32(2048)+np.float32(1e-6))
 val=(x*inv)*g.astype('f4');z=val/np.float32(q['scale'])+np.float32(q['zero_point'])
 return np.clip(np.copysign(np.floor(np.abs(z)+np.float32(.5)),z),0,255).astype('u1')
@functools.lru_cache(maxsize=256)
def weight(p,name,n,k):
 p=Path(p);w=unpack_w4_codes(p,name,n,k).astype('f8');ws=np.fromfile(p/(name+'_weight_w4_scale_f32.bin'),'<f4');return w,ws

def project(a,p,name,n,iq,oq,raw=False):
 w,ws=weight(str(p),name,n,a.shape[1]);acc=(a.astype('f8')-int(iq['zero_point']))@w.T
 assert np.max(np.abs(acc))<2**31 and np.array_equal(acc,np.round(acc))
 if raw:return acc.astype('f4')*(np.float32(iq['scale'])*ws)
 lo,hi=projection_bias_words(w.astype('i1'),ws,iq,oq)
 # converter bias includes -zero*sum, so feed uncentered raw accumulator.
 acc+=int(iq['zero_point'])*w.sum(1)
 return CV.convert(acc.astype('i8'),lo,hi)

def layer(x,p,cos,sin,past=None,sp2=True,seed=None):
 q=load_qparams_bin(p/'qparams_u8.bin');n=norm(x,np.fromfile(p/'input_norm_weight_f16.bin','<f2'),q['input_norm'])
 qr=project(n,p,'q',2048,q['input_norm'],q['q_projection']);kr=project(n,p,'k',1024,q['input_norm'],q['k_projection']);v=project(n,p,'v',1024,q['input_norm'],q['v'])
 qr=exact_qk_norm_rope_u8(qr,16,q['q_projection'],q['q_rope'],np.fromfile(p/'q_norm_weight_f16.bin','<f2'),cos,sin).reshape(-1,16,128)
 kr=exact_qk_norm_rope_u8(kr,8,q['k_projection'],q['k_rope'],np.fromfile(p/'k_norm_weight_f16.bin','<f2'),cos,sin).reshape(-1,8,128)
 k=kr.transpose(1,0,2);v=v.reshape(-1,8,128).transpose(1,0,2);count=0
 if seed is not None:
  assert past is None and len(x)>=64;k[:,0]=seed[0];v[:,0]=seed[1]
 if past is not None:count=past[0].shape[1];k=np.concatenate([past[0],k],1);v=np.concatenate([past[1],v],1)
 cfg=np.fromfile(p/'attention_config_all_groups.bin','<i4').reshape(8,15);av=np.empty((len(x),16,128),'u1')
 valid=np.arange(k.shape[1])[None,:]<=count+np.arange(len(x))[:,None]
 for group in range(8):
  c=config(tuple(int(a) for a in cfg[group]));res=numpy_oracle(qr[:,2*group:2*group+2].transpose(1,0,2),k[group],v[group],valid,c,'wide_nr64')
  av[:,2*group:2*group+2]=res['av'].transpose(1,0,2)
 av=av.reshape(-1,2048);o=project(av,p,'o',2048,q['attention_concat'],None,True);res=x+o
 post=norm(res,np.fromfile(p/'post_norm_weight_f16.bin','<f2'),q['post_attention_norm'])
 g=project(post,p,'gate',6144,q['post_attention_norm'],q['gate']);u=project(post,p,'up',6144,q['post_attention_norm'],q['up'])
 lut,_=fp_table(q);mid=lut[g,u]
 down=project(mid,p,'down',2048,q['middle'],None,True);y=res+down
 return y,(k,v),dict(input_norm=n,q=qr,k=kr,attention=av,o=o,residual=res,post=post,gate=g,up=u,middle=mid,down=down)
