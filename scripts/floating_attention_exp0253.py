#!/usr/bin/env python3
"""FP32 standard attention on decoded U8 caches, with no inner quantization."""
import numpy as np
import torch

def core(q,k,v,valid,scaling):
    assert q.dtype==k.dtype==v.dtype==torch.float16
    scores=torch.matmul(q.float(),k.float().transpose(-1,-2))*scaling
    probabilities=torch.softmax(scores.masked_fill(~valid,float('-inf')),dim=-1)
    return torch.matmul(probabilities,v.float()),probabilities

def numpy_reference(q,k,v,valid,scaling,chunk=None):
    q,k,v=[x.astype(np.float64) for x in (q,k,v)]
    scores=(q@k.swapaxes(-1,-2))*scaling
    scores=np.where(valid,scores,-np.inf)
    weights=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
    probabilities=weights/weights.sum(-1,keepdims=True)
    if chunk is None:return probabilities@v,probabilities
    # Independently streamed, no globally normalized probabilities in AV.
    m=np.full(scores.shape[:-1]+(1,),-np.inf);den=np.zeros_like(m)
    out=np.zeros(scores.shape[:-1]+(v.shape[-1],),np.float64)
    for start in range(0,k.shape[-2],chunk):
        end=min(start+chunk,k.shape[-2]);z=(q@k[...,start:end,:].swapaxes(-1,-2))*scaling
        z=np.where(np.broadcast_to(valid,scores.shape)[...,start:end],z,-np.inf)
        nxt=np.maximum(m,np.max(z,axis=-1,keepdims=True));old=np.exp(m-nxt)
        w=np.exp(z-nxt);out=out*old+w@v[...,start:end,:];den=den*old+w.sum(-1,keepdims=True);m=nxt
    return out/den,probabilities
