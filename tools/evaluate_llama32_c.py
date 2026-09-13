#!/usr/bin/env python3
"""Matched native-context BF16 teacher and exact integer software PPL."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from llama_reference import sha256
from export_llama32_u8 import layer
from llama_u8_reference import load_qparams_bin,HmxU8Converter,exact_rms_norm_u8
from llama32_c_integer_oracle import NativeOracle
ROOT=Path(__file__).resolve().parents[1]
RESULTS=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0013');MODELS=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0013')
DATA=RESULTS/'data';FRONT=MODELS/'frontend-a01'
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def dataset():
    m=json.loads((DATA/'freeze.json').read_text())
    for n,h in m['files'].items():assert sha256(DATA/n)==h
    return json.loads((DATA/'dataset.json').read_text())
@torch.inference_mode()
def teacher():
    from transformers import AutoModelForCausalLM
    ds=dataset();out=RESULTS/'teacher-a01';out.mkdir(exist_ok=False)
    model=AutoModelForCausalLM.from_pretrained('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin',torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).cuda().eval();rows=[]
    for sample in ds['samples']:
        ids=torch.tensor([sample['prompt_ids']+sample['target_ids']],device='cuda')
        logits=model(ids,use_cache=False).logits[0,63:79].float()
        nll=F.cross_entropy(logits,ids[0,64:80],reduction='none').cpu().tolist()
        rows.append(dict(id=sample['id'],nll=nll))
        if len(rows)%16==0:print('BF16_BRIDGE_SAMPLES',len(rows),flush=True)
    vals=[v for r in rows for v in r['nll']];save(out/'bridge.json',dict(scope=ds['name'],dataset_sha256=sha256(DATA/'dataset.json'),rows=rows,nll=sum(vals)/len(vals),ppl=math.exp(sum(vals)/len(vals)),targets=len(vals)))
    ids=np.fromfile(DATA/'validation.bin',dtype='<u4');segments=[]
    for start in range(0,len(ids),2048):
        chunk=ids[start:start+2048]
        if len(chunk)<2:continue
        x=torch.tensor(chunk.astype('i8')[None],device='cuda');logits=model(x,use_cache=False).logits[0,:-1].float()
        loss=F.cross_entropy(logits,x[0,1:],reduction='sum');segments.append(dict(start=start,length=len(chunk),targets=len(chunk)-1,nll_sum=float(loss)))
        if len(segments)%16==0:print('BF16_FULL_VALIDATION_WINDOWS',len(segments),flush=True)
    count=sum(v['targets'] for v in segments);nll=sum(v['nll_sum'] for v in segments)/count;assert count==252728
    save(out/'full-validation.json',dict(scope='full WT2 validation, disjoint2048 windows including948-token tail; no BOS/EOS; BF16 teacher',segments=segments,nll=nll,ppl=math.exp(nll),targets=count,validation_sha256=sha256(DATA/'validation.bin'),historical_teacher_ppl=13.6346577256))
@torch.inference_mode()
def integer():
    ds=dataset();m=json.loads((FRONT/'manifest.json').read_text())
    for n,h in m['files'].items():assert sha256(FRONT/n)==h['sha256']
    out=RESULTS/'integer-ppl-a01';out.mkdir(exist_ok=False);oracle=NativeOracle();oracle.install()
    conv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');qs=[load_qparams_bin(FRONT/f'layer{i}/qparams_u8.bin') for i in range(16)];gq=load_qparams_bin(FRONT/'generation_qparams_u8.bin')
    embed=np.memmap(FRONT/'generation_embedding_weight_u8.bin',dtype='u1',mode='r',shape=(128256,2048));gamma=np.fromfile(FRONT/'generation_final_norm_weight_f16.bin',dtype='<f2');allrows=[]
    ropes=[]
    for step in range(16):
        name='rope_cos_f16.bin' if step==0 else f'generation_decode_rope_cos_{step-1:02d}_f16.bin'
        ropes.append((np.fromfile(FRONT/name,dtype='<f2').reshape(64,64),np.fromfile(FRONT/name.replace('cos','sin'),dtype='<f2').reshape(64,64)))
    for sample in ds['samples']:
        caches=[None]*16;rows=[]
        for step,target in enumerate(sample['target_ids']):
            x=np.array(embed[sample['prompt_ids'] if step==0 else [sample['target_ids'][step-1]]]);c,s=ropes[step]
            for i in range(16):x,caches[i],_=layer(x,FRONT/f'layer{i}',qs[i],c,s,caches[i],conv,sp2=True)
            n=exact_rms_norm_u8(x[-1:],qs[-1]['block_output'],gamma,gq['generation_final_norm_output'])
            codes=oracle.project(n,FRONT,'generation_lm_head',128256,2048,gq['generation_final_norm_output'],gq['generation_lm_head_output'],conv)[0]
            logits=(codes.astype('f8')-gq['generation_lm_head_output']['zero_point'])*gq['generation_lm_head_output']['scale'];mx=float(logits.max());nll=float(mx+np.log(np.exp(logits-mx).sum())-logits[target])
            rows.append(dict(sample_id=sample['id'],step=step,target_token=target,target_code=int(codes[target]),max_code=int(codes.max()),nll=nll,histogram=np.bincount(codes,minlength=256).tolist()))
        save(out/f'sample-{sample["id"]:03d}.json',rows);allrows.extend(rows);print('INTEGER_PPL_SAMPLES',sample['id']+1,flush=True)
    nll=sum(r['nll'] for r in allrows)/len(allrows);assert len(allrows)==2048
    save(out/'result.json',dict(scope=ds['name'],targets=len(allrows),nll=nll,ppl=math.exp(nll),dataset_sha256=sha256(DATA/'dataset.json'),package_manifest_sha256=sha256(FRONT/'manifest.json'),rows=allrows,integer_dot_audited_projections=len(oracle.audited)))
if __name__=='__main__':
    import subprocess
    subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
    torch.set_num_threads(8);p=argparse.ArgumentParser();p.add_argument('stage',choices=['teacher','integer']);globals()[p.parse_args().stage]()
