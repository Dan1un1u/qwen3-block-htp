#!/usr/bin/env python3
"""Independent arithmetic audit on first scored token of first EN and ZH heldout docs."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from export_llama32_u8 import CAL,layer
from llama_u8_reference import HmxU8Converter,exact_rms_norm_u8,project_w4u8
from llama_reference import sha256
ROOT=Path(__file__).resolve().parents[1]
PKG=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0003/frontend-a01')
REF=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0003/frontend-reference-a01')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 m=json.loads((PKG/'manifest.json').read_text());c=json.loads(CAL.read_text());assert sha256(CAL)==m['calibration_sha256']
 for n,v in m['files'].items():assert sha256(PKG/n)==v['sha256'],n
 ds=json.loads((REF/'dataset.json').read_text());samples=[next(s for s in ds['samples'] if s['language']==lang) for lang in ['en','zh']];torch.set_num_threads(8)
 embed=np.memmap(PKG/'generation_embedding_weight_u8.bin',dtype='u1',mode='r',shape=(128256,2048));cos=np.fromfile(PKG/'rope_cos_f16.bin',dtype='<f2').reshape(64,64);sin=np.fromfile(PKG/'rope_sin_f16.bin',dtype='<f2').reshape(64,64);gamma=np.fromfile(PKG/'generation_final_norm_weight_f16.bin',dtype='<f2');conv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');rows=[]
 for sample in samples:
  x=np.array(embed[sample['prompt_ids']])
  for i in range(16):x,_,_=layer(x,PKG/f'layer{i}',c['qparams'][i],cos,sin,converter=conv)
  q=c['generation_qparams'];norm=exact_rms_norm_u8(x[-1:],c['qparams'][-1]['block_output'],gamma,q['generation_final_norm_output']);codes=project_w4u8(norm,PKG,'generation_lm_head',128256,2048,q['generation_final_norm_output'],q['generation_lm_head_output'],conv)[0]
  target=sample['target_ids'][0];logits=(codes.astype(np.float64)-q['generation_lm_head_output']['zero_point'])*q['generation_lm_head_output']['scale'];mx=float(logits.max());nll=float(mx+np.log(np.exp(logits-mx).sum())-logits[target]);r=dict(sample_id=sample['id'],step=0,language=sample['language'],target_token=target,target_code=int(codes[target]),max_code=int(codes.max()),histogram_total=int(np.bincount(codes,minlength=256).sum()),nll=nll);rows.append(r);print('NLL_ORACLE',json.dumps(r),flush=True)
 result=dict(package_manifest_sha256=sha256(PKG/'manifest.json'),dataset_sha256=sha256(REF/'dataset.json'),selection='first heldout document of each language, first target; numerical validation only, never model selection',rows=rows)
 a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
