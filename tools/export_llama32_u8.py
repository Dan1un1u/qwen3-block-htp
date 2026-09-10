#!/usr/bin/env python3
"""Export independent actual-arithmetic Llama W4A8 fixtures, no quality claim."""
import argparse,json,struct,os,math
from pathlib import Path
import numpy as np
import torch
from llama_reference import sha256
from llama_u8_reference import *
ROOT=Path(__file__).resolve().parents[1]
CAL=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0003/calibration-a01/calibration.json')
QUANT=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0002/quant-a01')
LAYERS=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0002/layers-a02')

def quantize(a,q):return np.clip(np.floor(np.asarray(a,dtype=np.float32)/np.float32(q['scale'])+np.float32(q['zero_point'])+.5),0,255).astype('u1')
def carrier(x):
 choices=[(abs(m/(1<<s)-x),s,m) for s in range(16) for m in range(1,19)]
 _,s,m=min(choices);return s,m

def configs(q):
 score=carrier(q['q_rope']['scale']*q['k_rope']['scale']*.125/(math.log(2)/8));den=max(q['v']['zero_point'],255-q['v']['zero_point']);av=carrier(q['attention_probability']['scale']*q['v']['scale']*den/127/q['attention_concat']['scale'])
 return [(1,g,3,1,q['q_rope']['zero_point'],q['k_rope']['zero_point'],q['v']['zero_point'],0,q['attention_concat']['zero_point'],127,den,*score,*av) for g in range(8)]

def lut(q):
 g=(np.arange(256,dtype=np.float32)-q['gate']['zero_point'])*q['gate']['scale'];u=(np.arange(256,dtype=np.float32)-q['up']['zero_point'])*q['up']['scale']
 # Match frozen device table-index semantics; weights of the table are newly calibrated.
 return quantize((g/(1+np.exp(-g)))[:,None]*u[None,:],q['middle']).astype('<u2')

def write_qparams(path,q):path.write_bytes(b''.join(struct.pack('<32sfi2f',n.encode(),v['scale'],v['zero_point'],v['minimum'],v['maximum']) for n,v in sorted(q.items())))
def divide(e,total,mode,count):return 255 if count==1 else min(255,((1<<(15-e))*255+total//2)//total)

def layer(x,package,q,cos,sin,past=None,converter=None):
 if converter is None:converter=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so')
 def project(x,n,inq,outq):
  k=x.shape[1];out={'q':2048,'k':512,'v':512,'o':2048,'gate':8192,'up':8192,'down':2048}[n]
  return project_w4u8(x,package,n,out,k,q[inq],q[outq],converter)
 norm=exact_rms_norm_u8(x,q['block_input'],np.fromfile(package/'input_norm_weight_f16.bin',dtype='<f2'),q['input_norm'])
 qr=project(norm,'q','input_norm','q_projection');kr=project(norm,'k','input_norm','k_projection');v=project(norm,'v','input_norm','v')
 qr=exact_qk_norm_rope_u8(qr,32,q['q_projection'],q['q_rope'],None,cos,sin).reshape(-1,32,64)
 kr=exact_qk_norm_rope_u8(kr,8,q['k_projection'],q['k_rope'],None,cos,sin).reshape(-1,8,64)
 k=kr.transpose(1,0,2);v=v.reshape(-1,8,64).transpose(1,0,2);count=0
 if past is not None:count=past[0].shape[1];k=np.concatenate([past[0],k],1);v=np.concatenate([past[1],v],1)
 av,score,prob=exact_attention_dynamic(qr,k,v,count,configs(q),converter,divide);av=av.reshape(len(x),2048)
 o=project(av,'o','attention_concat','attention_projection');res=exact_residual_add_u8(x,q['block_input'],o,q['attention_projection'],q['post_attention_residual'])
 post=exact_rms_norm_u8(res,q['post_attention_residual'],np.fromfile(package/'post_norm_weight_f16.bin',dtype='<f2'),q['post_attention_norm'])
 g=project(post,'gate','post_attention_norm','gate');u=project(post,'up','post_attention_norm','up');mid=lut(q)[g,u].astype('u1');down=project(mid,'down','middle','down');out=exact_residual_add_u8(res,q['post_attention_residual'],down,q['down'],q['block_output'])
 return out,(k,v),dict(q=qr,k=kr,attention=av,o=o,residual=res,post=post,gate=g,up=u,middle=mid,down=down,score=score,probability=prob)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 c=json.loads(CAL.read_text());assert c['experiment']=='L32-0003';assert sha256(QUANT/'manifest.json')==c['weight_manifest_sha256'];a.output.mkdir(parents=True);torch.set_num_threads(8)
 for index in [0,7,15]:
  q=c['qparams'][index];past=None
  for phase,count,start in [('prefill',64,0),('decode',1,64)]:
   old=LAYERS/f'layer{index}-{phase}';om=json.loads((old/'manifest.json').read_text());out=a.output/f'layer{index}-{phase}';out.mkdir()
   for name,v in om['files'].items():
    if 'weight' in name or name.startswith('rope_'):
     assert sha256(old/name)==v['sha256'];os.link(old/name,out/name)
   x=quantize(np.fromfile(old/'block_input_f16.bin',dtype='<f2').reshape(64,2048)[:count],q['block_input']);padded=np.full((64,2048),q['block_input']['zero_point'],dtype='u1');padded[:count]=x;padded.tofile(out/'block_input_u8.bin')
   write_qparams(out/'qparams_u8.bin',q);lut(q).tofile(out/'silu_up_lut_u16.bin');(out/'attention_config_all_groups.bin').write_bytes(b''.join(struct.pack('<IIIIiiiiiIIIIII',*v) for v in configs(q)))
   cos=np.fromfile(out/'rope_cos_f16.bin',dtype='<f2').reshape(64,64);sin=np.fromfile(out/'rope_sin_f16.bin',dtype='<f2').reshape(64,64)
   y,cache,diag=layer(x,out,q,cos,sin,past if start else None)
   ref=np.full((64,2048),q['block_output']['zero_point'],dtype='u1');ref[:count]=y;ref.tofile(out/'reference_w4u8_integer_attention_block_output_u8.bin')
   for j,n in enumerate(['k','v']):
    initial=np.full((8,80,64),q['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1')
    if start:initial[:,:64]=past[j]
    initial.tofile(out/f'kv_cache_{n}_u8.bin');initial[:,:start+count]=cache[j];initial.tofile(out/f'reference_kv_cache_{n}_u8.bin')
   if not start:past=cache
   for n,t in diag.items():np.save(out/f'reference_{n}.npy',t)
   m=dict(experiment='L32-0003',recipe='W4A8',rotation='OFF',layer=index,phase=phase,logical_rows=count,past_tokens=start,cache_capacity=80,calibration_sha256=sha256(CAL),reference='independent signed-W4 int accumulator plus SDK libnative HMX conversion; Llama RoPE-only; exact division log2 softmax; Q14 residual and freshly calibrated SwiGLU LUT',files={f.name:dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in out.iterdir()});(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('EXPORTED',out,flush=True)
if __name__=='__main__':main()
