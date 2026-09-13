#!/usr/bin/env python3
"""L32-0010 frozen all-layer SP2 package, consecutive gate and greedy oracle."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from export_llama32_u8 import layer,write_qparams
from llama_u8_reference import load_qparams_bin,unpack_w4_codes,HmxU8Converter,exact_rms_norm_u8,project_w4u8
from llama_reference import sha256
from prototype_llama32_sp2 import preflight
ROOT=Path(__file__).resolve().parents[1]
MODELS=Path('/mnt/d/llm_exp/models/llama32-htp')
OUT=MODELS/'l32-0010'
BASE=MODELS/'l32-0003/frontend-a01'
RESULTS=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0010')
FRONT=OUT/'frontend-a01'
def save(p,v):
 assert not p.exists(),p
 p.write_text(json.dumps(v,indent=2)+'\n')
def manifest(root,**kw):
 save(root/'manifest.json',dict(experiment='L32-0010',recipe='W4A8',rotation='OFF',sp2=True,cache_capacity=80,**kw,files={str(f.relative_to(root)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in root.rglob('*') if f.is_file()}))
def verify(root):
 m=json.loads((root/'manifest.json').read_text())
 for n,h in m['files'].items():assert sha256(root/n)==h['sha256'],root/n
 return m
def prepare():
 m=verify(BASE);FRONT.mkdir(parents=True,exist_ok=False);RESULTS.mkdir(parents=True,exist_ok=True)
 for n in m['files']:
  if Path(n).name.startswith('reference_') or Path(n).name in ['qparams_u8.bin','silu_up_lut_u16.bin','generation_expected_token_ids_u32.bin']:continue
  dst=FRONT/n;dst.parent.mkdir(parents=True,exist_ok=True);os.link(BASE/n,dst)
 os.link(BASE/'reference_w4u8_block_input_u8.bin',FRONT/'reference_w4u8_block_input_u8.bin')
 code=RESULTS.parent/'l32-0008/codebook.json';levels=np.array(json.loads(code.read_text())['levels'],dtype='i2');assert len(levels)==241
 bounds=[]
 for i in range(16):
  old=BASE/f'layer{i}';dst=FRONT/f'layer{i}';q=load_qparams_bin(old/'qparams_u8.bin')
  alpha=float(np.float32(max(abs(q['middle']['minimum']),abs(q['middle']['maximum']))/24576))
  g=(np.arange(256,dtype='f8')-q['gate']['zero_point'])*q['gate']['scale'];u=(np.arange(256,dtype='f8')-q['up']['zero_point'])*q['up']['scale']
  x=(g/(1+np.exp(-np.clip(g,-700,700))))[:,None]*u[None,:];grid=levels.astype('f8')*alpha
  ix=np.searchsorted(grid,x).clip(1,len(grid)-1);ix-=np.abs(x-grid[ix-1])<=np.abs(x-grid[ix]);v=levels[ix].astype('i4')
  encoded=(v+32768).astype('<u2');encoded.tofile(dst/'silu_up_lut_u16.bin')
  assert np.array_equal((encoded&255).astype('i4')+256*(encoded>>8).astype('i4')-32768,v)
  q['middle']['scale']=alpha;q['middle']['zero_point']=0;write_qparams(dst/'qparams_u8.bin',q)
  w=unpack_w4_codes(old,'down',2048,8192).astype('i4');pos=np.maximum(w,0).sum(1);neg=np.minimum(w,0).sum(1)
  lo=int(255*neg.min());hi=int(255*pos.max());assert lo>=-8388608 and hi<=8388607
  merged=int(24576*np.abs(w).sum(1).max());assert merged<2**31
  ws=np.fromfile(old/'down_weight_w4_scale_f32.bin',dtype='<f4').astype('f8');mult=np.floor(alpha*ws/q['down']['scale']*2**31+.5).astype('i8');assert np.all((mult>0)&(mult<2**31))
  bounds.append(dict(layer=i,alpha=alpha,partial_dot_bounds=[lo,hi],merged_abs_bound=merged,q31_range=[int(mult.min()),int(mult.max())],lut_reconstruction_mismatches=0))
 save(FRONT/'preparation.json',dict(parent_package_manifest_sha256=sha256(BASE/'manifest.json'),codebook_sha256=sha256(code),bounds=bounds))
 print('ALL_16_BOUNDS_PASS',flush=True)
def stack():
 old=MODELS/'l32-0003/stack16-a01';m=verify(old);out=OUT/'stack3-a01';out.mkdir(exist_ok=False)
 for n in ['reference_w4u8_block_input_u8.bin','replay_decode_input_00_u8.bin','rope_cos_f16.bin','rope_sin_f16.bin','replay_decode_rope_cos_00_f16.bin','replay_decode_rope_sin_00_f16.bin']:os.link(old/n,out/n)
 def rope(n):return np.fromfile(out/n,dtype='<f2').reshape(64,64)
 c=rope('rope_cos_f16.bin');s=rope('rope_sin_f16.bin');dc=rope('replay_decode_rope_cos_00_f16.bin');ds=rope('replay_decode_rope_sin_00_f16.bin')
 x=np.fromfile(out/'reference_w4u8_block_input_u8.bin',dtype='u1').reshape(64,2048);dx=np.fromfile(out/'replay_decode_input_00_u8.bin',dtype='u1').reshape(64,2048)[:1]
 for i in range(3):
  dst=out/f'layer{i}';dst.mkdir()
  for f in (FRONT/f'layer{i}').iterdir():
   if not f.name.startswith('reference_'):os.link(f,dst/f.name)
  q=load_qparams_bin(dst/'qparams_u8.bin');x,cache,_=layer(x,dst,q,c,s,sp2=True);dx,full,_=layer(dx,dst,q,dc,ds,cache,sp2=True)
  for j,n in enumerate(['k','v']):
   v=np.full((8,80,64),q['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');v[:,:65]=full[j];v.tofile(dst/f'reference_kv_cache_{n}_u8.bin')
  np.save(dst/'reference_output_prefill.npy',x);np.save(dst/'reference_output_decode.npy',dx);print('STACK_LAYER',i,flush=True)
 x.tofile(out/'reference_w4u8_integer_attention_block_output_u8.bin');v=np.full((64,2048),q['block_output']['zero_point'],dtype='u1');v[:1]=dx;v.tofile(out/'replay_decode_reference_00_u8.bin');manifest(out,layers=3,decode_steps=1)
def frontend():
 old=json.loads((BASE/'manifest.json').read_text());qs=[load_qparams_bin(FRONT/f'layer{i}/qparams_u8.bin') for i in range(16)];gq=load_qparams_bin(FRONT/'generation_qparams_u8.bin');embed=np.memmap(FRONT/'generation_embedding_weight_u8.bin',dtype='u1',mode='r',shape=(128256,2048));ids=np.fromfile(FRONT/'generation_prompt_token_ids_u32.bin',dtype='<u4').tolist();assert len(ids)==64
 caches=[None]*16;tokens=[];codes=[];conv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');gamma=np.fromfile(FRONT/'generation_final_norm_weight_f16.bin',dtype='<f2')
 for step in range(16):
  x=np.array(embed[ids if step==0 else [tokens[-1]]]);name='rope_cos_f16.bin' if step==0 else f'generation_decode_rope_cos_{step-1:02d}_f16.bin';c=np.fromfile(FRONT/name,dtype='<f2').reshape(64,64);s=np.fromfile(FRONT/name.replace('cos','sin'),dtype='<f2').reshape(64,64)
  for i in range(16):
   x,caches[i],_=layer(x,FRONT/f'layer{i}',qs[i],c,s,caches[i],conv,sp2=True)
   if step==0:
    for j,n in enumerate(['k','v']):
     ref=np.full((8,80,64),qs[i]['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');ref[:,:64]=caches[i][j];ref.tofile(FRONT/f'layer{i}/reference_kv_cache_{n}_u8.bin')
   print('ORACLE_LAYER',step,i,flush=True)
  if step==0:x.tofile(FRONT/'reference_w4u8_integer_attention_block_output_u8.bin')
  norm=exact_rms_norm_u8(x[-1:],qs[-1]['block_output'],gamma,gq['generation_final_norm_output']);logits=project_w4u8(norm,FRONT,'generation_lm_head',128256,2048,gq['generation_final_norm_output'],gq['generation_lm_head_output'],conv)
  token=int(logits[0].argmax());tokens.append(token);codes.append(int(logits[0,token]));save(RESULTS/f'oracle-step-{step:02d}.json',dict(step=step,token=token,code=codes[-1]));print('ORACLE_STEP',step,token,codes[-1],flush=True)
 np.array(tokens,dtype='<u4').tofile(FRONT/'generation_expected_token_ids_u32.bin');teacher=RESULTS/'sp2-oracle.json';save(teacher,dict(u8_generated_ids=tokens,u8_selected_codes=codes,prompt_ids=ids,scope='independent SP2 Q31 integer reference; no PPL/quality claim'))
 manifest(FRONT,layers=16,generation_tokens=16,prompt_tokens=64,original=old['original'],frontend_teacher_sha256=sha256(teacher),parent_package_manifest_sha256=sha256(BASE/'manifest.json'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','stack','frontend']);v=a.parse_args();preflight();torch.set_num_threads(8);globals()[v.stage]()
