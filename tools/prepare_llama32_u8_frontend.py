#!/usr/bin/env python3
"""Llama W4A8 tokenizer-to-logits package with independent greedy HMX oracle."""
import argparse,json,os,struct
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from export_llama32_u8 import CAL,QUANT,layer,quantize,write_qparams
from llama_u8_reference import exact_rms_norm_u8,project_w4u8,projection_bias_words,_cached_w4_projection,HmxU8Converter
from llama_reference import sha256,rope
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0002/frontend-a01')
REF=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0002/frontend-reference-a01')
STACK=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0003/stack16-a01')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);a=ap.parse_args()
 if a.output.exists() or a.reference.exists():raise FileExistsError('immutable frontend attempts')
 cal=json.loads(CAL.read_text());old=json.loads((OLD/'manifest.json').read_text());stack=json.loads((STACK/'manifest.json').read_text());teacher=json.loads((REF/'teacher.json').read_text())
 assert sha256(REF/'teacher.json')==old['frontend_teacher_sha256'];assert cal['weight_manifest_sha256']==sha256(QUANT/'manifest.json')
 a.output.mkdir(parents=True);a.reference.mkdir();torch.set_num_threads(8)
 for n,h in json.loads((REF/'freeze.json').read_text()).items():assert sha256(REF/n)==h;os.link(REF/n,a.reference/n)
 os.link(REF/'freeze.json',a.reference/'freeze.json')
 def oldlink(n):assert sha256(OLD/n)==old['files'][n]['sha256'];os.link(OLD/n,a.output/n)
 for n in ['generation_prompt_token_ids_u32.bin','generation_final_norm_weight_f16.bin','generation_lm_head_weight_w4_hmx.bin','generation_lm_head_weight_w4_scale_f32.bin']+[f'generation_decode_rope_{k}_{i:02d}_f16.bin' for i in range(15) for k in ['cos','sin']]+['rope_cos_f16.bin','rope_sin_f16.bin']:oldlink(n)
 for i in range(16):
  out=a.output/f'layer{i}';out.mkdir()
  for f in (STACK/f'layer{i}').iterdir():
   assert sha256(f)==stack['files'][str(f.relative_to(STACK))]['sha256']
   if not f.name.startswith('reference_'):os.link(f,out/f.name)
 q0=cal['qparams'][0]['block_input'];source=OLD/'generation_embedding_weight_f16.bin';assert sha256(source)==old['files'][source.name]['sha256'];e=np.memmap(source,dtype='<f2',mode='r',shape=(128256,2048));dest=a.output/'generation_embedding_weight_u8.bin'
 with dest.open('xb') as f:
  for start in range(0,128256,1024):quantize(e[start:start+1024],q0).tofile(f)
 embed=np.memmap(dest,dtype='u1',mode='r',shape=(128256,2048));gq=cal['generation_qparams'];write_qparams(a.output/'generation_qparams_u8.bin',gq)
 weights,scales=_cached_w4_projection(str(a.output.resolve()),'generation_lm_head',128256,2048)
 lo,hi=projection_bias_words(weights,scales,gq['generation_final_norm_output'],gq['generation_lm_head_output']);bias=np.concatenate((lo.reshape(-1,32),hi.view('<u4').reshape(-1,32)),1);bias.astype('<u4').tofile(a.output/'generation_lm_head_bias_u32.bin')
 ids=teacher['prompt_ids'];x=np.array(embed[ids]);x.tofile(a.output/'reference_w4u8_block_input_u8.bin');caches=[None]*16;tokens=[];codes=[];converter=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');gamma=np.fromfile(a.output/'generation_final_norm_weight_f16.bin',dtype='<f2')
 for step in range(16):
  x=np.array(embed[ids if step==0 else [tokens[-1]]]);cosfile='rope_cos_f16.bin' if step==0 else f'generation_decode_rope_cos_{step-1:02d}_f16.bin';sinfile=cosfile.replace('cos','sin');c=np.fromfile(a.output/cosfile,dtype='<f2').reshape(64,64);s=np.fromfile(a.output/sinfile,dtype='<f2').reshape(64,64)
  for i in range(16):
   x,caches[i],_=layer(x,a.output/f'layer{i}',cal['qparams'][i],c,s,caches[i],converter)
   if step==0:
    for j,n in enumerate(['k','v']):
     ref=np.full((8,80,64),cal['qparams'][i]['k_rope' if n=='k' else 'v']['zero_point'],dtype='u1');ref[:,:64]=caches[i][j];ref.tofile(a.output/f'layer{i}/reference_kv_cache_{n}_u8.bin')
  if step==0:x.tofile(a.output/'reference_w4u8_integer_attention_block_output_u8.bin')
  norm=exact_rms_norm_u8(x[-1:],cal['qparams'][-1]['block_output'],gamma,gq['generation_final_norm_output']);logits=project_w4u8(norm,a.output,'generation_lm_head',128256,2048,gq['generation_final_norm_output'],gq['generation_lm_head_output'],converter)
  token=int(logits[0].argmax());tokens.append(token);codes.append(int(logits[0,token]));print('REFERENCE_STEP',step,token,codes[-1],flush=True)
 np.asarray(tokens,dtype='<u4').tofile(a.output/'generation_expected_token_ids_u32.bin')
 tok=AutoTokenizer.from_pretrained(old['original']['original_root'],local_files_only=True);result=dict(nll=teacher['nll'],prompt_ids=ids,prompt_messages=teacher['prompt_messages'],u8_generated_ids=tokens,u8_selected_codes=codes,u8_text=tok.decode(tokens,skip_special_tokens=True),arithmetic='SDK HMX conversion with signed per-channel W4; independent integer transformer and U8 final head; no quality threshold',teacher_source_sha256=sha256(REF/'teacher.json'))
 (a.reference/'teacher.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 m=dict(experiment='L32-0003',recipe='W4A8',rotation='OFF',layers=16,generation_tokens=16,prompt_tokens=64,cache_capacity=80,original=old['original'],frontend_teacher_sha256=sha256(a.reference/'teacher.json'),dataset_freeze_sha256=sha256(a.reference/'freeze.json'),calibration_sha256=sha256(CAL),quality_gate=None,files={str(f.relative_to(a.output)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in a.output.rglob('*') if f.is_file()});(a.output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('FRONTEND_COMPLETE',result['u8_text'],flush=True)
if __name__=='__main__':main()
