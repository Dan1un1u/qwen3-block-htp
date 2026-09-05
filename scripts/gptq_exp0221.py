#!/usr/bin/env python3
"""CPU per-output-row GPTQ; algorithm reference IST-DASLab/gptq
2d65066eeb06a5c9ff5184d8cebdf33662c67faf (Apache-2.0).
Fixed signed [-7,7] grid, no groups or clipping search. No runtime dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode
import numpy as np
import torch
from transformers import AutoTokenizer
import eval_exp0218 as ev
from rotation_exp0219 import write_json,unpack

RESULT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0221')
OUTPUT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0221')
REFERENCE='2d65066eeb06a5c9ff5184d8cebdf33662c67faf'

def sha(data):return hashlib.sha256(data).hexdigest()

def fetch(url,path):
    if path.exists():return path.read_bytes()
    # Use the user's existing Windows Clash proxy; no global proxy mutation.
    cmd=['/mnt/c/Windows/System32/curl.exe','--fail','--silent','--show-error',
         '--location','--proxy','http://127.0.0.1:7897','--max-time','45','--retry','2',url]
    data=subprocess.check_output(cmd);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data)
    return data

def calibration():
    assert not (RESULT/'calibration.json').exists()
    tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    excluded=set()
    for s in ev.dataset()['samples']:
        ids=s['prompt_ids']+s.get('target_ids',[])
        excluded.update(tuple(ids[i:i+32]) for i in range(len(ids)-31))
    sources=[];samples=[];rejected=0
    for lang,dataset,config in [('en','Salesforce/wikitext','wikitext-2-raw-v1'),('zh','wikimedia/wikipedia','20231101.zh')]:
        meta=fetch('https://huggingface.co/api/datasets/'+dataset,RESULT/'calibration_raw'/f'{lang}_metadata.json')
        revision=json.loads(meta)['sha'];selected=[];pages=[]
        for offset in range(0,5000,100):
            url='https://datasets-server.huggingface.co/rows?'+urlencode(dict(dataset=dataset,config=config,split='train',offset=offset,length=100))
            raw=fetch(url,RESULT/'calibration_raw'/f'{lang}_{offset}.json');payload=json.loads(raw)
            pages.append(dict(url=url,sha256=sha(raw),rows=len(payload['rows'])))
            for item in payload['rows']:
                # One window per source row, for greater document/paragraph diversity.
                if 'text' in item.get('truncated_cells',[]):continue
                ids=tok.encode(item['row']['text'],add_special_tokens=False)
                if len(ids)<128:continue
                window=ids[:128]
                if any(tuple(window[i:i+32]) in excluded for i in range(97)):
                    rejected+=1;continue
                selected.append(dict(language=lang,row_index=item['row_idx'],token_ids=window))
                if len(selected)==32:break
            if len(selected)==32:break
        assert len(selected)==32,(lang,len(selected))
        samples.extend(selected);sources.append(dict(language=lang,dataset=dataset,config=config,split='train',metadata_revision=revision,pages=pages))
    ids=np.asarray([s['token_ids'] for s in samples],dtype='<u4');OUTPUT.mkdir(parents=True,exist_ok=True)
    binary=OUTPUT/'calibration_ids_u32.bin'
    if binary.exists():assert binary.read_bytes()==ids.tobytes(),'Partial calibration recovery must preserve exact IDs'
    else:
        with binary.open('xb') as f:f.write(ids.tobytes())
    write_json(RESULT/'calibration.json',dict(experiment='EXP-0221',samples=samples,sources=sources,
        selection='first128 tokens of first32 eligible distinct rows per language; no padding or chat template',
        tokens=8192,sequence_length=128,ids_sha256=sha(ids.tobytes()),tokenizer_sha256=ev.digest(ev.MODEL/'qwen3-tokenizer.json'),
        overlap_32gram_rejections=rejected,evaluation_or_holdout_scoring=False))
    references()
    print('CALIBRATION_FROZEN',len(samples),sha(ids.tobytes()),flush=True)

def references():
    refs={}
    for n in ['gptq.py','quant.py','LICENSE']:
        raw=fetch(f'https://raw.githubusercontent.com/IST-DASLab/gptq/{REFERENCE}/{n}',RESULT/'references'/n);refs[n]=sha(raw)
    write_json(RESULT/'references/provenance.json',dict(repository='https://github.com/IST-DASLab/gptq',commit=REFERENCE,files=refs))

def factor(x):
    x=x.reshape(-1,x.shape[-1]).float();h=(x.T@x)*(2.0/x.shape[0])
    assert torch.isfinite(h).all()
    diag=h.diagonal().clone();dead=diag==0;h[dead,dead]=1
    perm=torch.argsort(h.diagonal(),descending=True,stable=True);h=h[perm][:,perm].double()
    damping=0.01*float(h.diagonal().mean());h.diagonal().add_(damping)
    inv=torch.cholesky_inverse(torch.linalg.cholesky(h));upper=torch.linalg.cholesky(inv,upper=True).float()
    assert torch.isfinite(upper).all()
    return dict(upper=upper,perm=perm,invperm=torch.argsort(perm),dead=dead,
        stats=dict(tokens=x.shape[0],width=x.shape[1],dead_columns=int(dead.sum()),damping=damping,
                   input_gram_sha256=sha(diag.numpy().tobytes()),factor_dtype='float64_then_float32'))

def quantize(w,f,block=128,explicit_scale=None):
    """Rows independent, factor shared. Returns codes in original input order."""
    maximum=w.abs().amax(1);scale=torch.where(maximum>0,maximum/7,torch.ones_like(maximum))
    if explicit_scale is not None:
        assert explicit_scale.shape == scale.shape
        assert torch.isfinite(explicit_scale).all() and (explicit_scale > 0).all()
        scale = explicit_scale
    work=w[:,f['perm']].clone();work[:,f['dead'][f['perm']]]=0
    qout=torch.empty_like(work,dtype=torch.int8);u=f['upper'];threads=torch.get_num_threads()
    for start in range(0,work.shape[1],block):
        end=min(start+block,work.shape[1]);part=work[:,start:end].clone();errors=torch.empty_like(part)
        # Small rank-one updates are faster without a thread-pool launch per column.
        torch.set_num_threads(1)
        try:
            for j in range(end-start):
                codes=(part[:,j]/scale).round().clamp(-7,7)
                qout[:,start+j]=codes.to(torch.int8)
                error=(part[:,j]-codes*scale)/u[start+j,start+j]
                part[:,j:]-=error[:,None]*u[start+j,start+j:end][None,:]
                errors[:,j]=error
        finally:torch.set_num_threads(threads)
        work[:,end:]-=errors@u[start:end,end:]
    return qout[:,f['invperm']],scale

def pack_codes(q):
    n,k=q.shape;assert n%32==k%32==0
    a=q.numpy().reshape(n//32,32,k//32,32).transpose(0,2,3,1)
    a=np.ascontiguousarray(a.reshape(n//32,k//32,8,4,32).transpose(0,1,2,4,3)).reshape(n//32,k//32,1024)
    a=(a.astype(np.int16)&15).astype(np.uint8);packed=a[...,::2]|(a[...,1::2]<<4)
    assert torch.equal(unpack(packed,n,k),q)
    return packed

def oracle_test():
    torch.manual_seed(221);x=torch.randn(80,32,dtype=torch.float64);x[:,1]=x[:,0]*.8+x[:,1]*.2
    w=torch.randn(32,32,dtype=torch.float32);f=factor(x.float());actual,scale=quantize(w,f,block=8)
    # Independent dense Schur-complement elimination (no Cholesky or lazy blocks).
    h=(x.float().T@x.float())*(2/len(x));p=f['perm'];h=h[p][:,p].double();h.diagonal().add_(f['stats']['damping'])
    inv=torch.linalg.inv(h);work=w[:,p].double().clone();expected=torch.empty_like(work)
    for i in range(32):
        q=(work[:,i]/scale.double()).round().clamp(-7,7);expected[:,i]=q
        error=work[:,i]-q*scale.double();work[:,i:]-=error[:,None]*(inv[0]/inv[0,0])[None,:]
        inv=inv[1:,1:]-inv[1:,0,None]*inv[None,0,1:]/inv[0,0]
    expected=expected[:,f['invperm']].to(torch.int8);assert torch.equal(actual,expected)
    pack_codes(actual)
    return dict(independent_dense_elimination_codes_equal=True,packed_roundtrip=True,seed=221)

if __name__=='__main__':
    torch.set_num_threads(16);torch.set_grad_enabled(False)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['calibration','references','oracle']);a=p.parse_args()
    if a.phase=='calibration':calibration()
    elif a.phase=='references':references()
    else:write_json(RESULT/'gptq_oracle.json',oracle_test());print('GPTQ_ORACLE_PASS')
