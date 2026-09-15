#!/usr/bin/env python3
"""L32-0016 immutable fixed-weight FP32 residual reference and device replay."""
import argparse,json,os,shlex,shutil,subprocess,re
from pathlib import Path
import numpy as np
from export_llama32_u8 import configs,divide,ROOT
from llama_reference import sha256
from llama_u8_reference import load_qparams_bin,project_w4u8,exact_qk_norm_rope_u8,exact_attention_dynamic,HmxU8Converter,unpack_w4_codes
from llama_sp2_reference import table
from prototype_llama32_sp2 import oracle,preflight
from run_llama32_layer import adb,windows
M=Path('/mnt/d/llm_exp/models/llama32-htp');R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0016')
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify(p):
 m=json.loads((p/'manifest.json').read_text())
 for n,v in m['files'].items():assert sha256(p/n)==v['sha256'],n
 return m

def norm(x,gamma,q):
 x=np.asarray(x,dtype='f4');s=np.cumsum(x*x,axis=1,dtype='f4')[:,-1:]
 inv=np.float32(1)/np.sqrt(s/np.float32(x.shape[1])+np.float32(1e-5))
 v=(x*inv)*gamma.astype('f4')
 code=v/np.float32(q['scale'])+np.float32(q['zero_point'])
 return np.clip(np.copysign(np.floor(np.abs(code)+np.float32(.5)),code),0,255).astype('u1')
def floating_attention(q,k,v,past,cfg):
 # Independent Float64 continuous softmax; raw QK and AV integer conversion
 # stay identical to the retained hardware contract.
 rows=len(q);width=k.shape[1];score=np.empty((32,rows,width),'u1');prob=np.zeros_like(score);av=np.empty((rows,32,64),'u1')
 for g,c in enumerate(cfg):
  qq=q[:,4*g:4*g+4].transpose(1,0,2).astype('i8');kk=k[g].astype('i8');vv=v[g].astype('i8')
  acc=(qq-int(c[4]))@np.clip(kk-int(c[5]),-128,127).T;div=2**int(c[11]);raw=np.clip((acc+div//2)//div+128,0,255);pp=np.zeros_like(raw)
  for h in range(4):
   for row in range(rows):
    n=past+row+1;xs=raw[h,row,:n].astype('f8');xs=(xs-xs.max())*int(c[12])/2**int(c[2]);ex=np.exp2(xs);pp[h,row,:n]=np.clip(np.floor(255*ex/ex.sum()+.5),0,255)
  vc=vv-int(c[6]);vc=np.clip(np.sign(vc)*((np.abs(vc)*int(c[9])+int(c[10])//2)//int(c[10])),-128,127)
  accum=pp@vc;div=2**int(c[13]);temp=np.clip((accum+div//2)//div+128,0,255);value=np.clip((temp-128)*int(c[14])+int(c[8]),0,255)
  av[:,4*g:4*g+4]=value.transpose(1,0,2);score[4*g:4*g+4]=raw;prob[4*g:4*g+4]=pp
 return av,score,prob

def layer(x,p,q,cos,sin,past=None,sp2=True):
 cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so')
 def project(a,name,iq,oq):
  n={'q':2048,'k':512,'v':512,'gate':8192,'up':8192}[name]
  return project_w4u8(a,p,name,n,a.shape[1],q[iq],q[oq],cv)
 def raw(a,name,scale,zero=0):
  w=unpack_w4_codes(p,name,2048,a.shape[1]);ws=np.fromfile(p/(name+'_weight_w4_scale_f32.bin'),dtype='<f4')
  acc=oracle(a.astype('i4')-zero,w)
  assert np.max(np.abs(acc))<2**31
  return acc.astype('f4')*(np.float32(scale)*ws)
 a=norm(x,np.fromfile(p/'input_norm_weight_f16.bin',dtype='<f2'),q['input_norm'])
 qr=project(a,'q','input_norm','q_projection');kr=project(a,'k','input_norm','k_projection');v=project(a,'v','input_norm','v')
 qr=exact_qk_norm_rope_u8(qr,32,q['q_projection'],q['q_rope'],None,cos,sin).reshape(-1,32,64)
 kr=exact_qk_norm_rope_u8(kr,8,q['k_projection'],q['k_rope'],None,cos,sin).reshape(-1,8,64)
 k=kr.transpose(1,0,2);v=v.reshape(-1,8,64).transpose(1,0,2);count=0
 if past is not None:count=past[0].shape[1];k=np.concatenate([past[0],k],1);v=np.concatenate([past[1],v],1)
 av,score,prob=floating_attention(qr,k,v,count,configs(q));av=av.reshape(len(x),2048)
 o=raw(av,'o',q['attention_concat']['scale'],q['attention_concat']['zero_point']);res=x+o
 post=norm(res,np.fromfile(p/'post_norm_weight_f16.bin',dtype='<f2'),q['post_attention_norm'])
 g=project(post,'gate','post_attention_norm','gate');u=project(post,'up','post_attention_norm','up')
 lut=table(str(p/'silu_up_lut_u16.bin')) if sp2 else np.fromfile(p/'silu_up_lut_u16.bin','<u2').reshape(256,256);mid=lut[g,u];down=raw(mid,'down',q['middle']['scale'],0 if sp2 else q['middle']['zero_point']);y=res+down
 return y,(k,v),dict(input_norm=a,q=qr,k=kr,attention=av,o=o,residual=res,post=post,gate=g,up=u,middle=mid,down=down,score=score,probability=prob)
