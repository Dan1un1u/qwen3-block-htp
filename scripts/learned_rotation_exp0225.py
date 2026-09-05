#!/usr/bin/env python3
"""SpinQuant no-online-rotation adaptation for Qwen3 per-channel W4F16.

OI convention: x'=xR1; Wv'=R2.T Wv R1; Wo'=R1.T Wo R2.
Only R1/R2 train. Original weights, including tied embedding/head, stay frozen.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from run_exp0164_semantic_gate import load_model
from rotation_exp0219 import PROJECTIONS, fwht, write_json
from eval_exp0218 import MODEL

RESULT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0225')
OUTPUT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0225')
SEED=225
PLAN=dict(steps=100,batch_size=8,sequence_length=128,seed=SEED,learning_rate=1.5,
    lr_schedule='cosine',checkpoints=[0,50,100],training_samples=800,validation_samples=32,
    quantizer='STE_RTN_absmax_per_output_row_minus7_to7_scale_FP32_dequant_FP16',
    export_quantizer='EXP0224_post_GPTQ_three_output_scales_transformer_GPTQ_absmax_head',
    rotation_compute='FP32_TF32_disabled',trainable='R1_and_per_layer_R2_only',
    checkpoint_selection='minimum_actual_export_bilingual_validation_NLL_tie_earliest',
    validation_language_guard='report_each_language_and_regressions_against_unrotated_A',
    evaluation='qbh-lite-v1_final_only_never_checkpoint_selection')

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,w):
        maximum=w.abs().amax(1,keepdim=True)
        scale=torch.where(maximum>0,maximum/7,torch.ones_like(maximum))
        return (torch.round(w/scale).clamp(-7,7)*scale).half()
    @staticmethod
    def backward(ctx,g):return g.float()

def rotated_weight(w,r1,r2=None,kind='q',gamma=None):
    w=w.float()
    if gamma is not None:w=w*gamma.float()[None,:]
    w=r1.T@w if kind in ('o','down') else w@r1
    if kind=='v':
        n,k=w.shape;w=(r2.T@w.reshape(-1,r2.shape[0],k)).reshape(n,k)
    elif kind=='o':
        n,k=w.shape;w=(w.reshape(n,-1,r2.shape[0])@r2).reshape(n,k)
    return w

class RotationLinear(nn.Module):
    def __init__(self,linear,r1,r2,kind,gamma):
        super().__init__();self.register_buffer('weight',linear.weight.detach())
        self.register_buffer('gamma',gamma.detach().clone() if gamma is not None else None)
        object.__setattr__(self,'r1',r1);object.__setattr__(self,'r2',r2)
        self.kind=kind;self.quantized=True
    def forward(self,x):
        w=rotated_weight(self.weight,self.r1,self.r2,self.kind,self.gamma)
        return F.linear(x,STE.apply(w) if self.quantized else w.to(x.dtype))

class LearnedModel(nn.Module):
    def __init__(self,model,device='cuda'):
        super().__init__();self.body=model.model;self.head=model.lm_head
        for p in model.parameters():p.requires_grad_(False)
        d=model.config.hidden_size;hd=model.config.head_dim
        gen=torch.Generator().manual_seed(SEED)
        self.r1=nn.Parameter(fwht(torch.eye(d))*torch.where(torch.rand(d,generator=gen)>.5,1.,-1.)[:,None])
        self.r2=nn.ParameterList([nn.Parameter(fwht(torch.eye(hd))*torch.where(torch.rand(hd,generator=gen)>.5,1.,-1.)[:,None]) for _ in self.body.layers])
        self.register_buffer('head_gamma',self.body.norm.weight.detach().float().clone())
        for i,layer in enumerate(self.body.layers):
            gammas={n:layer.get_submodule(n).weight.detach().float().clone() for n in ['input_layernorm','post_attention_layernorm']}
            for short,long in PROJECTIONS.items():
                gamma=gammas['input_layernorm'] if short in ('q','k','v') else gammas['post_attention_layernorm'] if short in ('gate','up') else None
                parent,attr=long.rsplit('.',1);linear=layer.get_submodule(long)
                setattr(layer.get_submodule(parent),attr,RotationLinear(linear,self.r1,self.r2[i],short,gamma))
            with torch.no_grad():
                layer.input_layernorm.weight.fill_(1);layer.post_attention_layernorm.weight.fill_(1)
        with torch.no_grad():self.body.norm.weight.fill_(1)
        self.quantized=True;self.recompute=True;self.to(device)
        # .to moves Parameters in place; non-registered references still address them.
        for i,layer in enumerate(self.body.layers):
            for name in PROJECTIONS.values():
                q=layer.get_submodule(name);object.__setattr__(q,'r1',self.r1);object.__setattr__(q,'r2',self.r2[i])
    def hidden(self,ids):
        x=(self.body.embed_tokens(ids).float()@self.r1).half()
        pos=torch.arange(ids.shape[1],device=ids.device)[None,:]
        rope=self.body.rotary_emb(x,pos)
        mask=torch.full((ids.shape[1],ids.shape[1]),torch.finfo(x.dtype).min,device=x.device,dtype=x.dtype).triu(1)[None,None]
        for layer in self.body.layers:
            def forward(h,layer=layer):return layer(h,attention_mask=mask,position_embeddings=rope)[0]
            x=checkpoint(forward,x,use_reentrant=False) if self.recompute and torch.is_grad_enabled() else forward(x)
        return self.body.norm(x)
    def head_chunk(self,h,start,end):
        w=rotated_weight(self.head.weight[start:end],self.r1,kind='head',gamma=self.head_gamma)
        return F.linear(h,STE.apply(w) if self.quantized else w.to(h.dtype)).float()
    def loss(self,ids):
        h=self.hidden(ids[:,:-1]).reshape(-1,self.r1.shape[0]);targets=ids[:,1:].reshape(-1)
        total_lse=None;target_logits=torch.zeros(len(h),device=h.device)
        for start in range(0,self.head.weight.shape[0],2048):
            end=min(start+2048,self.head.weight.shape[0])
            def part(h,start=start,end=end):
                z=self.head_chunk(h,start,end)
                target=z.gather(1,(targets-start).clamp(0,end-start-1)[:,None]).squeeze(1)
                target=target*((targets>=start)&(targets<end))
                return torch.logsumexp(z,1),target
            lse,val=checkpoint(part,h,use_reentrant=False) if self.recompute and torch.is_grad_enabled() else part(h)
            total_lse=lse if total_lse is None else torch.logaddexp(total_lse,lse)
            target_logits=target_logits+val
        return (total_lse-target_logits).mean()
    def rotations(self):return {'R1':self.r1.detach().cpu(),'R2':torch.stack([r.detach().cpu() for r in self.r2])}
    def set_quantized(self,enabled):
        self.quantized=enabled
        for layer in self.body.layers:
            for name in PROJECTIONS.values():layer.get_submodule(name).quantized=enabled

@torch.no_grad()
def fold(model,rotations):
    r1=rotations['R1'];r2=rotations['R2']
    if model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight=nn.Parameter(model.lm_head.weight.detach().clone(),requires_grad=False);model.config.tie_word_embeddings=False
    for i,layer in enumerate(model.model.layers):
        gamma=layer.input_layernorm.weight.clone();post=layer.post_attention_layernorm.weight.clone()
        for short,long in PROJECTIONS.items():
            p=layer.get_submodule(long).weight
            g=gamma if short in ('q','k','v') else post if short in ('gate','up') else None
            p.copy_(rotated_weight(p,r1,r2[i],short,g))
        layer.input_layernorm.weight.fill_(1);layer.post_attention_layernorm.weight.fill_(1)
    for start in range(0,len(model.lm_head.weight),1024):
        sl=slice(start,start+1024)
        model.model.embed_tokens.weight[sl].copy_(model.model.embed_tokens.weight[sl].float()@r1)
        w=model.lm_head.weight[sl];w.copy_(rotated_weight(w,r1,kind='head',gamma=model.model.norm.weight))
    model.model.norm.weight.fill_(1)

@torch.no_grad()
def orthogonality(rotations):
    out={}
    for k,mats in rotations.items():
        if mats.ndim==2:mats=mats[None]
        out[k]=max(float((m.double().T@m.double()-torch.eye(m.shape[0],device=m.device,dtype=torch.float64)).abs().max()) for m in mats)
    return out

def optimizer(params,lr):
    from spinquant_exp0225.optimizer import SGDG
    return SGDG(params,lr=lr,stiefel=True)

def save_checkpoint(path,model,opt,step,data_hash):
    path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    rot=model.rotations();orth=orthogonality(rot);assert max(orth.values())<.003,orth
    value=dict(rotations=rot,optimizer=opt.state_dict(),step=step,plan=PLAN,data_sha256=data_hash,
        orthogonality=orth,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate())
    torch.save(value,path);return orth

def train(smoke=False):
    from experiment_exp0220 import verify_origin
    from prepare_exp0164_generation_package import sha256_file
    assert torch.cuda.is_available(),'CUDA training environment required'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(8);random.seed(SEED);torch.manual_seed(SEED)
    data=json.loads((RESULT/'learning_data.json').read_text());data_hash=sha256_file(RESULT/'learning_data.json')
    assert data['plan']==PLAN
    origin=verify_origin();base=load_model(MODEL,torch.float16);model=LearnedModel(base);del base;gc.collect()
    opt=optimizer([model.r1,*model.r2],PLAN['learning_rate'])
    mode='smoke' if smoke else 'training';out=OUTPUT/mode;assert not out.exists()
    out.mkdir(parents=True);rows=data['train'];gen=np.random.default_rng(SEED)
    en=gen.permutation([i for i,r in enumerate(rows) if r['language']=='en']);zh=gen.permutation([i for i,r in enumerate(rows) if r['language']=='zh'])
    schedule=np.stack([np.concatenate([en[i:i+4],zh[i:i+4]]) for i in range(0,400,4)])
    orth=save_checkpoint(out/'step000.pt',model,opt,0,data_hash)
    logpath=RESULT/(mode+'.jsonl');assert not logpath.exists()
    start=time.monotonic();steps=1 if smoke else PLAN['steps']
    for step in range(steps):
        ids=torch.tensor([rows[int(i)]['token_ids'] for i in schedule[step]],device='cuda')
        opt.zero_grad(set_to_none=True);lr=PLAN['learning_rate']*.5*(1+math.cos(math.pi*step/PLAN['steps']))
        opt.param_groups[0]['lr']=lr
        loss=model.loss(ids);assert torch.isfinite(loss);loss.backward()
        grad=[model.r1.grad]+[r.grad for r in model.r2]
        assert all(g is not None and torch.isfinite(g).all() for g in grad)
        gradnorm=float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grad])))
        opt.step();torch.cuda.synchronize()
        row=dict(step=step+1,loss=float(loss.detach()),lr=lr,grad_norm=gradnorm,elapsed_s=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        if smoke or step+1 in PLAN['checkpoints']:
            row['orthogonality']=save_checkpoint(out/f'step{step+1:03d}.pt',model,opt,step+1,data_hash)
        with logpath.open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    write_json(RESULT/(mode+'_complete.json'),dict(steps=steps,origin=origin,data_sha256=data_hash,plan=PLAN,torch=torch.__version__,gpu=torch.cuda.get_device_name(),elapsed_s=time.monotonic()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','smoke']);a=p.parse_args();train(a.phase=='smoke')
