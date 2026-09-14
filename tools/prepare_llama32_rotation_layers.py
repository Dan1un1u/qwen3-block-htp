#!/usr/bin/env python3
"""Frozen single-layer packages, ideal dense R4 reference and native-SP2 integer tail."""
import json,os
from pathlib import Path
import numpy as np
from llama_reference import sha256
from llama_u8_reference import load_qparams_bin,unpack_w4_codes,exact_residual_add_u8
from prototype_llama32_sp2 import oracle,preflight
from probe_llama32_rotations import r4,save
M=Path('/mnt/d/llm_exp/models/llama32-htp')
NEW=M/'l32-0015/rotated-down-a01';OUT=M/'l32-0015/layers-a01'
def tail(z,root,q,residual):
 lut=np.fromfile(root/'silu_up_lut_u16.bin',dtype='<u2')[65536:];v=lut[z.view('u2')].astype('i4')-32768
 w=unpack_w4_codes(root,'down',2048,8192);ws=np.fromfile(root/'down_weight_w4_scale_f32.bin',dtype='<f4').astype('f8');mult=np.floor(q['middle']['scale']*ws/q['down']['scale']*2**31+.5).astype('i8')
 acc=oracle(v,w);assert np.abs(acc).max()<2**31
 down=np.clip(((acc*mult+2**30)>>31)+q['down']['zero_point'],0,255).astype('u1')
 return exact_residual_add_u8(residual,q['post_attention_residual'],down,q['down'],q['block_output']),down

def main():
 preflight();OUT.mkdir(exist_ok=False)
 for i in [0,7,15]:
  base=M/f'l32-0009/packages-a01/layer{i}-sp2';bm=json.loads((base/'manifest.json').read_text());fresh=NEW/f'layer{i}';fm=json.loads((fresh/'manifest.json').read_text())
  for n,h in fm['files'].items():assert sha256(fresh/n)==h
  dst=OUT/f'layer{i}';dst.mkdir();(dst/'layer0').mkdir()
  replacing={f'layer0/{n}' for n in ['down_weight_w4_hmx.bin','down_weight_w4_scale_f32.bin','silu_up_lut_u16.bin','qparams_u8.bin','reference_w4u8_integer_attention_block_output_u8.bin']}|{'reference_w4u8_integer_attention_block_output_u8.bin','replay_decode_reference_00_u8.bin'}
  for n,h in bm['files'].items():
   assert sha256(base/n)==h['sha256'],base/n
   if n not in replacing:os.link(base/n,dst/n)
  for n in ['down_weight_w4_hmx.bin','down_weight_w4_scale_f32.bin','silu_up_lut_u16.bin','qparams_u8.bin']:os.link(fresh/n,dst/'layer0'/n)
  q=load_qparams_bin(fresh/'qparams_u8.bin');lut=np.fromfile(fresh/'silu_up_lut_u16.bin',dtype='<f2',count=65536).reshape(256,256)
  for step,phase in enumerate(['prefill','decode']):
   ref=M/f'l32-0003/layers-a01/layer{i}-{phase}';g=np.load(ref/'reference_gate.npy');u=np.load(ref/'reference_up.npy');z=lut[g,u];a,b=r4(z)
   y,d=tail(b,fresh,q,np.load(ref/'reference_residual.npy'));pad=np.full((64,2048),q['block_output']['zero_point'],dtype='u1');pad[:len(y)]=y
   if not step:
    pad.tofile(dst/'layer0/reference_w4u8_integer_attention_block_output_u8.bin');pad.tofile(dst/'reference_w4u8_integer_attention_block_output_u8.bin')
   else:pad.tofile(dst/'replay_decode_reference_00_u8.bin')
   np.save(dst/f'ideal_r4_{step:02d}.npy',b);np.save(dst/f'ideal_down_{step:02d}.npy',d)
  save(dst/'manifest.json',dict(experiment='L32-0015',recipe='W4A8',layers=1,source_layer=i,sp2=True,rotation='r4',baseline_manifest_sha256=sha256(base/'manifest.json'),fresh_manifest_sha256=sha256(fresh/'manifest.json'),reference='FP64 dense factors with explicit FP16 stage rounding; independent S32/Q31/Q14 tail',files={str(p.relative_to(dst)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in dst.rglob('*') if p.is_file()}))
  print('LAYER_PACKAGE',i,flush=True)
if __name__=='__main__':main()
