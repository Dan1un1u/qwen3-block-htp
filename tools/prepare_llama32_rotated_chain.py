#!/usr/bin/env python3
"""Independent dense-factor reference for continuous FP32 rotation chains."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from llama32_rotated_fp32 import layer,save,M,ROOT
from llama32_fp32_residual import verify
from llama_u8_reference import load_qparams_bin
from llama_reference import sha256
from prototype_llama32_sp2 import preflight

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--layers',type=int,required=True);ap.add_argument('--attempt',required=True);ap.add_argument('--extension',required=True);a=ap.parse_args();preflight()
 old=M/'l32-0016'/f'chain{a.layers}-a01';om=verify(old);out=M/'l32-0021'/a.attempt;out.mkdir(parents=True,exist_ok=False);shutil.copytree(old,out,dirs_exist_ok=True);(out/'manifest.json').unlink();folds=[]
 for i in range(a.layers):
  fresh=M/'l32-0015/rotated-down-a01'/f'layer{i}' if i in [0,7,15] else M/'l32-0021'/a.extension/f'layer{i}'
  fm=json.loads((fresh/'manifest.json').read_text())
  for n,h in fm['files'].items():assert sha256(fresh/n)==h
  for n in ['down_weight_w4_hmx.bin','down_weight_w4_scale_f32.bin','silu_up_lut_u16.bin','qparams_u8.bin']:shutil.copyfile(fresh/n,out/f'layer{i}'/n)
  folds.append(dict(path=str(fresh/'manifest.json'),sha256=sha256(fresh/'manifest.json')))
 caches=[None]*a.layers
 for step in range(2):
  rows=64 if step==0 else 1;name='reference_w4u8_block_input_f32.bin' if not step else'replay_decode_input_00_f32.bin';x=np.fromfile(out/name,dtype='<f4').reshape(64,2048)[:rows]
  n='rope_cos_f16.bin' if not step else'replay_decode_rope_cos_00_f16.bin';cos=np.fromfile(out/n,dtype='<f2').reshape(64,64);sin=np.fromfile(out/n.replace('cos','sin'),dtype='<f2').reshape(64,64)
  for i in range(a.layers):
   p=out/f'layer{i}';q=load_qparams_bin(p/'qparams_u8.bin');x,cache,diag=layer(x,p,q,cos,sin,caches[i],arm='both')
   if not step:caches[i]=cache
   np.save(out/f'rotation_{step}_hidden{i}.npy',x)
   for j,n in enumerate(['k','v']):
    ref=np.full((8,80,64),q['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');ref[:,:cache[j].shape[1]]=cache[j];ref.tofile(p/f'reference_kv_cache_{n}_u8.bin')
   print('IDEAL_CHAIN',step,i,flush=True)
  pad=np.zeros((64,2048),dtype='<f4');pad[:rows]=x;pad.tofile(out/('reference_w4u8_block_output_f32.bin' if not step else'replay_decode_reference_00_f32.bin'))
 save(out/'manifest.json',dict(experiment='L32-0021',layers=a.layers,arm='both',reference='independent FP64 dense factors roundedFP16, integer/FP32 tail; no hardware captures',baseline_manifest_sha256=sha256(old/'manifest.json'),folds=folds,files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in out.rglob('*') if p.is_file()}));print(out,flush=True)
if __name__=='__main__':main()
