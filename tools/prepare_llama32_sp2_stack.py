#!/usr/bin/env python3
"""L32-0009 fixed SP2 fused-LUT, Q31 Down and Q14 residual independent oracle."""
import argparse,json,os,struct
from pathlib import Path
import numpy as np
from llama_reference import sha256
from llama_u8_reference import load_qparams_bin,unpack_w4_codes,exact_residual_add_u8,QPARAM_RECORD
from prototype_llama32_sp2 import oracle,preflight

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 preflight();a.output.mkdir(parents=True,exist_ok=False)
 code=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0008/codebook.json')
 levels=np.array(json.loads(code.read_text())['levels'],dtype='i2');assert len(levels)==241
 base=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0003/layers-a01')
 for layer in [0,7,15]:
  roots=[base/f'layer{layer}-{phase}' for phase in ['prefill','decode']]
  for root in roots:
   for n,v in json.loads((root/'manifest.json').read_text())['files'].items():assert sha256(root/n)==v['sha256'],root/n
  q=load_qparams_bin(roots[0]/'qparams_u8.bin')
  alpha=float(np.float32(max(abs(q['middle']['minimum']),abs(q['middle']['maximum']))/24576))
  g=(np.arange(256,dtype='f8')-q['gate']['zero_point'])*q['gate']['scale']
  u=(np.arange(256,dtype='f8')-q['up']['zero_point'])*q['up']['scale']
  x=(g/(1+np.exp(-np.clip(g,-700,700))))[:,None]*u[None,:]
  grid=levels.astype('f8')*alpha
  ix=np.searchsorted(grid,x).clip(1,len(grid)-1);ix-=np.abs(x-grid[ix-1])<=np.abs(x-grid[ix])
  lut=levels[ix]
  w=unpack_w4_codes(roots[0],'down',2048,8192)
  ws=np.fromfile(roots[0]/'down_weight_w4_scale_f32.bin',dtype='<f4').astype('f8')
  ratio=alpha*ws/q['down']['scale'];mult=np.floor(ratio*2**31+.5).astype('i8')
  assert np.all((mult>0)&(mult<2**31))
  outputs=[];diags=[]
  for root in roots:
   gate=np.load(root/'reference_gate.npy');up=np.load(root/'reference_up.npy')
   v=lut[gate,up]
   acc=oracle(v,w)
   down=np.clip(((acc*mult+2**30)>>31)+q['down']['zero_point'],0,255).astype('u1')
   out=exact_residual_add_u8(np.load(root/'reference_residual.npy'),q['post_attention_residual'],down,q['down'],q['block_output'])
   padded=np.full((64,2048),q['block_output']['zero_point'],dtype='u1');padded[:len(out)]=out;outputs.append(padded)
   ideal=np.clip(np.floor(acc*ratio+.5)+q['down']['zero_point'],0,255).astype('u1')
   diags.append(dict(rows=len(out),q31_vs_exact_scale_mismatches=int(np.count_nonzero(down!=ideal)),q31_vs_exact_scale_max_abs=int(np.max(np.abs(down.astype('i4')-ideal))),acc_max_abs=int(np.max(np.abs(acc)))))
  for sp2 in [False,True]:
   out=a.output/f"layer{layer}-{'sp2' if sp2 else 'u8'}"
   out.mkdir();l=out/'layer0';l.mkdir()
   p,d=roots
   for f in p.iterdir():
    if f.name=='manifest.json':continue
    if sp2 and f.name in ['silu_up_lut_u16.bin','qparams_u8.bin','reference_w4u8_integer_attention_block_output_u8.bin']:continue
    src=d/f.name if f.name.startswith('reference_kv_cache') else f
    os.link(src,l/f.name)
   for n in ['block_input_u8.bin','rope_cos_f16.bin','rope_sin_f16.bin']:
    os.link(p/n,out/('reference_w4u8_block_input_u8.bin' if n=='block_input_u8.bin' else n))
   for dst,src in [('replay_decode_input_00_u8.bin','block_input_u8.bin'),('replay_decode_rope_cos_00_f16.bin','rope_cos_f16.bin'),('replay_decode_rope_sin_00_f16.bin','rope_sin_f16.bin')]:os.link(d/src,out/dst)
   if sp2:
    (lut.astype('i4')+32768).astype('<u2').tofile(l/'silu_up_lut_u16.bin')
    raw=(p/'qparams_u8.bin').read_bytes();records=[]
    for off in range(0,len(raw),QPARAM_RECORD.size):
     rec=list(QPARAM_RECORD.unpack_from(raw,off))
     if rec[0].split(b'\0')[0]==b'middle':rec[1]=alpha;rec[2]=0
     records.append(QPARAM_RECORD.pack(*rec))
    (l/'qparams_u8.bin').write_bytes(b''.join(records))
    outputs[0].tofile(l/'reference_w4u8_integer_attention_block_output_u8.bin')
    outputs[0].tofile(out/'reference_w4u8_integer_attention_block_output_u8.bin')
    outputs[1].tofile(out/'replay_decode_reference_00_u8.bin')
   else:
    os.link(p/'reference_w4u8_integer_attention_block_output_u8.bin',out/'reference_w4u8_integer_attention_block_output_u8.bin')
    os.link(d/'reference_w4u8_integer_attention_block_output_u8.bin',out/'replay_decode_reference_00_u8.bin')
   manifest=dict(experiment='L32-0009',recipe='W4A8',layers=1,source_layer=layer,cache_capacity=80,decode_steps=1,sp2=sp2,
     alpha_f32=alpha,codebook_sha256=sha256(code),q31_multiplier_range=[int(mult.min()),int(mult.max())],diagnostics=diags,
     source_manifests={str(t):sha256(t/'manifest.json') for t in roots},
     files={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in out.rglob('*') if f.is_file()})
   (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
   print(out,diags,flush=True)
if __name__=='__main__':main()
