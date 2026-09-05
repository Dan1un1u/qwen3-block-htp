#!/usr/bin/env python3
"""Fixed three-candidate final-GPTQ output reconstruction; no runtime change."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from clipping_exp0223 import select_scale, check as clipping_check
from gptq_exp0221 import factor, quantize, pack_codes
from rotation_exp0219 import write_json

RESULT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0224')
OUTPUT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0224')
PREVIOUS=RESULT.parent/'exp0221'
PREVIOUS_MODELS=OUTPUT.parent/'exp0221'
CLIPPED=RESULT.parent/'exp0223'
CLIPPED_MODELS=OUTPUT.parent/'exp0223'
CANDIDATES=['absmax','midpoint','weight_L2.4_clip']

def control_package(v):
    return (PREVIOUS_MODELS if v=='A' else CLIPPED_MODELS)/v

def output_error(w,codes,scale,x):
    """All calibration positions; FP16 weight export, FP32 GEMM, FP64 sum."""
    delta=(codes.float()*scale[:,None]).half().float()-w
    assert torch.isfinite(delta).all()
    total=torch.zeros(len(w),dtype=torch.float64)
    x=x.reshape(-1,w.shape[1])
    for start in range(0,len(x),512):
        error=F.linear(x[start:start+512].float(),delta)
        total+=error.double().square().sum(0)
    assert torch.isfinite(total).all()
    return total

def select_output(w,x,f,block=128):
    clipped,local=select_scale(w)
    base=torch.from_numpy(local['absmax_scale'])
    scales=[base,(base+clipped)*.5,clipped]
    scores=[];best=torch.full((len(w),),float('inf'),dtype=torch.float64)
    chosen=torch.zeros(len(w),dtype=torch.int64)
    best_codes=torch.empty_like(w,dtype=torch.int8);best_scale=base.clone()
    for i,scale in enumerate(scales):
        codes,_=quantize(w,f,block=block,explicit_scale=scale)
        error=output_error(w,codes,scale,x);scores.append(error)
        improve=error<best
        best[improve]=error[improve];chosen[improve]=i
        best_codes[improve]=codes[improve];best_scale[improve]=scale[improve]
    scores=torch.stack(scores,1)
    assert (best<=scores[:,0]).all() and (best<=scores[:,2]).all()
    assert torch.equal(best,scores.gather(1,chosen[:,None]).squeeze(1))
    return best_codes,best_scale,dict(choice=chosen.numpy(),scale=best_scale.numpy(),
        candidate_scales=torch.stack(scales,1).numpy(),candidate_output_sse=scores.numpy(),
        selected_output_sse=best.numpy(),absmax_scale=base.numpy(),
        clip_ratio=(clipped/base).numpy(),weight_clip_choice=local['choice'])

def check():
    prior=clipping_check()
    torch.manual_seed(224)
    x=torch.randn(1100,32);x[:,1]=.8*x[:,0]+.2*x[:,1];x[:,2]*=9
    w=torch.randn(32,32);w[:,0]*=5;w[0]=0
    f=factor(x);codes,scale,rows=select_output(w,x,f,block=8)
    errors=[];all_codes=[]
    for i in range(3):
        sc=torch.from_numpy(rows['candidate_scales'][:,i].copy())
        q,_=quantize(w,f,block=8,explicit_scale=sc);all_codes.append(q.numpy())
        # Independent FP64 NumPy full-input product, no torch scorer/chunking.
        delta=(q.numpy().astype(np.float32)*sc.numpy()[:,None]).astype(np.float16).astype(np.float64)-w.numpy().astype(np.float64)
        errors.append(np.square(x.numpy().astype(np.float64)@delta.T).sum(0))
    errors=np.stack(errors,1);choice=errors.argmin(1)
    np.testing.assert_allclose(rows['candidate_output_sse'],errors,rtol=2e-6,atol=1e-9)
    assert np.array_equal(rows['choice'],choice)
    for row,i in enumerate(choice):assert np.array_equal(codes[row].numpy(),all_codes[i][row])
    assert rows['choice'][0]==0 and not codes[0].any()
    pack_codes(codes)
    return dict(independent_numpy_output_scores=True,independent_numpy_choices=True,
        selected_codes_exact=True,zero_row_largest_range_tie=True,all_positions=1100,
        candidate_names=CANDIDATES,full_GPTQ_for_each_candidate=True,
        exported_weight_dtype='float16',score_accumulation='float64',prior_oracles=prior)

if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(16)
    write_json(RESULT/'output_scale_oracle.json',check());print('POST_GPTQ_OUTPUT_ORACLES_PASS')
