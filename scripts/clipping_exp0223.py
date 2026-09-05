#!/usr/bin/env python3
"""Fixed per-row L2.4 range selection, then unchanged explicit-scale GPTQ."""
import json
from pathlib import Path
import numpy as np
import torch
from gptq_exp0221 import factor, quantize, pack_codes, oracle_test, sha
from rotation_exp0219 import write_json

RESULT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0223')
OUTPUT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0223')
PREVIOUS=RESULT.parent/'exp0221'
PREVIOUS_MODELS=OUTPUT.parent/'exp0221'
RATIOS=[1-i/100 for i in range(80)]


def select_scale(w):
    maximum=w.abs().amax(1)
    base=torch.where(maximum>0,maximum/7,torch.ones_like(maximum))
    best=torch.full_like(base,float('inf'));choice=torch.zeros(len(w),dtype=torch.int64)
    chosen=base.clone();initial=None
    for i,p in enumerate(RATIOS):
        scale=base*p
        q=(w/scale[:,None]).round().clamp(-7,7)*scale[:,None]
        error=(q-w).abs().pow(2.4).sum(1)
        if initial is None:initial=error.clone()
        better=error<best
        best[better]=error[better];choice[better]=i;chosen[better]=scale[better]
    assert torch.isfinite(best).all() and (best<=initial).all()
    return chosen,dict(choice=choice.numpy(),scale=chosen.numpy(),absmax_scale=base.numpy(),
                       initial_objective=initial.numpy(),selected_objective=best.numpy())


def check():
    torch.manual_seed(223)
    w=torch.randn(32,32);w[:,0]*=8;w[0]=0
    scale,stats=select_scale(w)
    # Independent NumPy exhaustive row/grid scan in float64.
    reference=[]
    for row in w.numpy().astype(np.float64):
        base=np.max(np.abs(row))/7 if np.any(row) else 1.0
        errors=[np.abs(np.clip(np.rint(row/(base*p)),-7,7)*base*p-row).__pow__(2.4).sum() for p in RATIOS]
        reference.append(int(np.argmin(errors)))
    assert np.array_equal(stats['choice'],reference)
    x=torch.randn(96,32);x[:,1]=.9*x[:,0]+.1*x[:,1];f=factor(x)
    a,s=quantize(w,f);b,_=quantize(w,f,explicit_scale=s);assert torch.equal(a,b)
    codes,_=quantize(w,f,block=8,explicit_scale=scale)
    perm=f['perm'];h=(x.T@x)*(2/len(x));h=h[perm][:,perm].double();h.diagonal().add_(f['stats']['damping'])
    inv=torch.linalg.inv(h);work=w[:,perm].double().clone();expected=torch.empty_like(work)
    for i in range(32):
        q=(work[:,i]/scale.double()).round().clamp(-7,7);expected[:,i]=q
        error=work[:,i]-q*scale.double();work[:,i:]-=error[:,None]*(inv[0]/inv[0,0])[None,:]
        inv=inv[1:,1:]-inv[1:,0,None]*inv[None,0,1:]/inv[0,0]
    assert torch.equal(codes,expected[:,f['invperm']].to(torch.int8))
    pack_codes(codes)
    return dict(numpy_scale_choices_exact=True,explicit_scale_dense_GPTQ_codes_exact=True,
        absmax_scale_parity_exact=True,packed_roundtrip=True,original_oracle=oracle_test(),
        ratios=RATIOS,objective='sum_abs_weight_error_power_2.4',integer_grid='[-7,7]')


if __name__=='__main__':
    torch.set_num_threads(16);torch.set_grad_enabled(False)
    write_json(RESULT/'clipping_oracle.json',check());print('CLIPPING_ORACLES_PASS')
