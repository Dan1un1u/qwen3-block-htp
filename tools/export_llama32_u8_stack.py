#!/usr/bin/env python3
"""Consecutive Llama A8 transformer validation with independent exact integer replay."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from export_llama32_u8 import CAL,QUANT,layer,quantize,write_qparams,lut,configs
from llama_reference import sha256
import struct
OLD=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0002/stack16-a01')

def main():
 p=argparse.ArgumentParser();p.add_argument('--layers',type=int,choices=[3,16],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 c=json.loads(CAL.read_text());m=json.loads((OLD/'manifest.json').read_text());assert c['weight_manifest_sha256']==sha256(QUANT/'manifest.json');a.output.mkdir(parents=True);torch.set_num_threads(8)
 def copy(n,dest):assert sha256(OLD/n)==m['files'][n]['sha256'];os.link(OLD/n,dest)
 q=c['qparams'][0];x=quantize(np.fromfile(OLD/'block_input_f16.bin',dtype='<f2').reshape(64,2048),q['block_input']);dx=quantize(np.fromfile(OLD/'replay_decode_input_00_f16.bin',dtype='<f2').reshape(64,2048)[:1],q['block_input'])
 x.tofile(a.output/'reference_w4u8_block_input_u8.bin')
 def padded(x,q):y=np.full((64,2048),q['zero_point'],dtype='u1');y[:len(x)]=x;return y
 padded(dx,q['block_input']).tofile(a.output/'replay_decode_input_00_u8.bin')
 for n in ['rope_cos_f16.bin','rope_sin_f16.bin','replay_decode_rope_cos_00_f16.bin','replay_decode_rope_sin_00_f16.bin']:copy(n,a.output/n)
 cos=np.fromfile(a.output/'rope_cos_f16.bin',dtype='<f2').reshape(64,64);sin=np.fromfile(a.output/'rope_sin_f16.bin',dtype='<f2').reshape(64,64)
 dc=np.fromfile(a.output/'replay_decode_rope_cos_00_f16.bin',dtype='<f2').reshape(64,64);ds=np.fromfile(a.output/'replay_decode_rope_sin_00_f16.bin',dtype='<f2').reshape(64,64)
 for i in range(a.layers):
  out=a.output/f'layer{i}';out.mkdir();q=c['qparams'][i]
  for f in (OLD/f'layer{i}').glob('*weight*.bin'):copy(str(f.relative_to(OLD)),out/f.name)
  write_qparams(out/'qparams_u8.bin',q);lut(q).tofile(out/'silu_up_lut_u16.bin');(out/'attention_config_all_groups.bin').write_bytes(b''.join(struct.pack('<IIIIiiiiiIIIIII',*v) for v in configs(q)))
  x,cache,_=layer(x,out,q,cos,sin);dx,full,_=layer(dx,out,q,dc,ds,cache)
  for j,n in enumerate(['k','v']):
   initial=np.full((8,80,64),q['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');initial.tofile(out/f'kv_cache_{n}_u8.bin');initial[:,:65]=full[j];initial.tofile(out/f'reference_kv_cache_{n}_u8.bin')
  np.save(out/'reference_output_prefill.npy',x);np.save(out/'reference_output_decode.npy',dx);print('EXPORTED_LAYER',i,flush=True)
 x.tofile(a.output/'reference_w4u8_integer_attention_block_output_u8.bin');padded(dx,q['block_output']).tofile(a.output/'replay_decode_reference_00_u8.bin')
 m=dict(experiment='L32-0003',recipe='W4A8',rotation='OFF',layers=a.layers,cache_capacity=80,decode_steps=1,calibration_sha256=sha256(CAL),reference='independent SDK integer HMX consecutive replay; never substitute device outputs for reference',files={str(f.relative_to(a.output)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in a.output.rglob('*') if f.is_file()});(a.output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
