#!/usr/bin/env python3
"""Explicit integer attention arithmetic; no device or model side effects."""
import math
import numpy as np
import torch

FIELDS = 'abi group fb dm qz kz vz pz oz vn vd ss sm avs avm'.split()

def config(fields):
    c = dict(zip(FIELDS, fields))
    assert c['abi'] == 1 and c['fb'] == 3 and c['pz'] == 0
    return c

def requant(x, mult, shift, zero):
    # Conversion order matters: saturation and rounding precede multiplier.
    div = 1 << shift
    raw = ((x + 128*div + (div//2 if shift else 0)) // div).clamp(0,255)
    return ((raw-128)*mult+zero).clamp(0,255), raw

def probability(scores, valid, division):
    top = scores.masked_fill(~valid, -100000).amax(-1, keepdim=True)
    exponent = ((top-scores+4)//8).clamp(0,15)
    weight = torch.bitwise_left_shift(torch.ones_like(exponent),15-exponent)*valid
    total = weight.sum(-1,keepdim=True)
    if division == 'exact':
        code = (weight*255+total//2)//total
    else:
        leading = total.double().log2().floor().long()
        bit = torch.bitwise_right_shift(total,(leading-1).clamp_min(0)) & 1
        coeff = torch.where((leading>0)&(bit!=0),145,209)
        shift = exponent+leading-15
        num = torch.bitwise_left_shift(255*coeff,(-shift).clamp_min(0))
        den = torch.bitwise_left_shift(torch.full_like(shift,256),shift.clamp_min(0))
        code = (num+den//2)//den
    code = torch.where(valid.sum(-1,keepdim=True)==1,255,code).clamp(0,255)*valid
    return code, exponent

def core(q, k, v, valid, c, mode='sole', real_scale=None):
    """Inputs U8 [batch,heads,rows,128], already repeated GQA K/V.

    FP32 integer dot products are exact: sum of absolute products is bounded
    below 2^24; TF32 is forbidden. All division and carrier stages use int64.
    """
    assert not torch.backends.cuda.matmul.allow_tf32
    assert q.shape[-1]*255*128 < 2**24 and k.shape[-2]*255*128 < 2**24
    kc = (k.long()-c['kz']).clamp(-128,127)
    acc = ((q.float()-c['qz']) @ kc.float().transpose(-1,-2)).long()
    score,raw = requant(acc,c['sm'],c['ss'],128)
    if mode in ['carrier','score']:
        logits = acc.double()*real_scale if mode=='carrier' else (score.double()-128)*(math.log(2)/8)
        prob = logits.masked_fill(~valid,float('-inf')).softmax(-1)
        pp = (prob*255+.5).floor().long().clamp(0,255)*valid
        ee = None
    else:
        pp,ee = probability(score,valid,'exact' if mode=='exponent' else 'sole')
    vc = v.long()-c['vz']
    vv = (vc.sign()*((vc.abs()*c['vn']+c['vd']//2)//c['vd'])).clamp(-128,127)
    avacc = (pp.float() @ vv.float()).long()
    av,_ = requant(avacc,c['avm'],c['avs'],c['oz'])
    return dict(acc=acc,raw=raw,score=score,probability=pp,exponent=ee,av=av)

def numpy_oracle(q,k,v,valid,c,mode='sole',real_scale=None):
    """Independent int64 accumulation and Python scalar division per row."""
    q,k,v = [np.asarray(x,dtype=np.int64) for x in [q,k,v]]
    acc = (q-c['qz']) @ np.clip(k-c['kz'],-128,127).swapaxes(-1,-2)
    d=2**c['ss'];raw=np.clip(np.floor_divide(acc+128*d+d//2, d),0,255)
    score=np.clip((raw-128)*c['sm']+128,0,255)
    valid=np.broadcast_to(valid,score.shape);pp=np.zeros_like(score);ee=np.zeros_like(score)
    for ix in np.ndindex(score.shape[:-1]):
        mask=valid[ix];row=score[ix][mask];assert len(row)>0
        if mode in ['carrier','score']:
            logits=acc[ix][mask]*real_scale if mode=='carrier' else (row-128)*math.log(2)/8
            w=np.exp(logits-logits.max());vals=np.floor(w/w.sum()*255+.5).astype(np.int64)
        else:
            es=[min(15,(int(row.max())-int(x)+4)//8) for x in row]
            weights=[2**(15-e) for e in es];total=sum(weights);vals=[]
            for e,w in zip(es,weights):
                if len(row)==1:p=255
                elif mode=='exponent':p=(w*255+total//2)//total
                else:
                    lead=total.bit_length()-1
                    coeff=145 if lead and (total>>(lead-1))&1 else 209
                    shift=e+lead-15
                    num=255*coeff*2**max(-shift,0);den=256*2**max(shift,0)
                    p=(num+den//2)//den
                vals.append(min(255,p))
            ee[ix][mask]=es
        pp[ix][mask]=vals
    vv=v-c['vz'];vv=np.clip(np.sign(vv)*((np.abs(vv)*c['vn']+c['vd']//2)//c['vd']),-128,127)
    avacc=pp@vv;d=2**c['avs'];tmp=np.clip((avacc+128*d+d//2)//d,0,255)
    av=np.clip((tmp-128)*c['avm']+c['oz'],0,255)
    return dict(acc=acc,raw=raw,score=score,probability=pp,exponent=ee,av=av)
