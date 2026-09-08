#!/usr/bin/env python3
"""Masked raw-score maximum, explicit int16 differences, frozen AV arithmetic."""
import math
import numpy as np
import torch
import integer_attention_exp0249 as old
from integer_attention_exp0249 import config, requant, probability

NEW = {'unbounded','wide_float','wide_exact','wide_sole','wide_nr64'}

def wide_probability(raw,valid,mult,division):
    assert 1 <= mult <= 18
    short = raw.to(torch.int16)
    maximum = short.masked_fill(~valid,-1).amax(-1,keepdim=True)
    delta = (maximum-short)*mult
    assert delta.dtype == torch.int16
    exponent = torch.bitwise_right_shift(delta+4,3).clamp(0,15)
    e = exponent.long()
    weight = torch.bitwise_left_shift(torch.ones_like(e),15-e)*valid
    total = weight.sum(-1,keepdim=True)
    if division=='exact':
        code=(weight*255+total//2)//total
    elif division=='nr64':
        leading=total.double().log2().floor().long()
        x=total << (30-leading)
        index=((x-(1<<30))>>24).clamp(0,63)
        denom=129+2*index
        r=((1<<37)+denom//2)//denom
        r=(r*((2<<30)-((x*r)>>30)))>>30
        shift=leading+30
        code=(weight*255*r+(torch.ones_like(shift)<<(shift-1)))>>shift
    else:
        leading=total.double().log2().floor().long()
        bit=torch.bitwise_right_shift(total,(leading-1).clamp_min(0)) & 1
        coeff=torch.where((leading>0)&(bit!=0),145,209)
        shift=e+leading-15
        num=torch.bitwise_left_shift(255*coeff,(-shift).clamp_min(0))
        den=torch.bitwise_left_shift(torch.full_like(shift,256),shift.clamp_min(0))
        code=(num+den//2)//den
    code=torch.where(valid.sum(-1,keepdim=True)==1,255,code).clamp(0,255)*valid
    return code,exponent,delta

def core(q,k,v,valid,c,mode='sole',real_scale=None):
    if mode not in NEW:return old.core(q,k,v,valid,c,mode,real_scale)
    assert not torch.backends.cuda.matmul.allow_tf32
    assert q.shape[-1]*255*128<2**24 and k.shape[-2]*255*128<2**24
    kc=(k.long()-c['kz']).clamp(-128,127)
    acc=((q.float()-c['qz'])@kc.float().transpose(-1,-2)).long()
    score,raw=requant(acc,c['sm'],c['ss'],128)
    delta=None;ee=None
    if mode in ['unbounded','wide_float']:
        d=1<<c['ss']
        centered=(acc+d//2)//d if mode=='unbounded' else raw-128
        logits=centered.double()*c['sm']*(math.log(2)/8)
        prob=logits.masked_fill(~valid,float('-inf')).softmax(-1)
        pp=(prob*255+.5).floor().long().clamp(0,255)*valid
    else:
        pp,ee,delta=wide_probability(raw,valid,c['sm'],'exact' if mode=='wide_exact' else 'nr64' if mode=='wide_nr64' else 'sole')
    vc=v.long()-c['vz'];vv=(vc.sign()*((vc.abs()*c['vn']+c['vd']//2)//c['vd'])).clamp(-128,127)
    avacc=(pp.float()@vv.float()).long();av,_=requant(avacc,c['avm'],c['avs'],c['oz'])
    return dict(acc=acc,raw=raw,score=score,probability=pp,exponent=ee,delta=delta,av=av)

def numpy_oracle(q,k,v,valid,c,mode='sole',real_scale=None):
    if mode not in NEW:return old.numpy_oracle(q,k,v,valid,c,mode,real_scale)
    q,k,v=[np.asarray(x,np.int64) for x in [q,k,v]]
    acc=(q-c['qz'])@np.clip(k-c['kz'],-128,127).swapaxes(-1,-2)
    div=2**c['ss'];rounded=(acc+div//2)//div
    raw=np.clip(rounded+128,0,255);score=np.clip((raw-128)*c['sm']+128,0,255)
    valid=np.broadcast_to(valid,raw.shape);pp=np.zeros_like(raw);ee=np.zeros_like(raw);delta=np.zeros_like(raw)
    for ix in np.ndindex(raw.shape[:-1]):
        mask=valid[ix];row=raw[ix][mask];assert len(row)>0
        if mode in ['unbounded','wide_float']:
            grid=rounded[ix][mask] if mode=='unbounded' else row-128
            logits=grid*c['sm']*(math.log(2)/8);ex=np.exp(logits-logits.max())
            values=np.clip(np.floor(255*ex/ex.sum()+.5),0,255)
        else:
            ds=[(int(row.max())-int(x))*c['sm'] for x in row]
            es=[min(15,(d+4)//8) for d in ds];total=sum(2**(15-e) for e in es);values=[]
            for e in es:
                if len(row)==1:val=255
                elif mode=='wide_exact':val=(255*2**(15-e)+total//2)//total
                elif mode=='wide_nr64':
                    lead=total.bit_length()-1; x=total*(2**(30-lead)); idx=(x-2**30)//2**24
                    denom=129+2*idx; r=(2**37+denom//2)//denom
                    r=r*(2**31-x*r//2**30)//2**30
                    shift=lead+30; val=(255*2**(15-e)*r+2**(shift-1))//2**shift
                else:
                    lead=total.bit_length()-1
                    coefficient=145 if lead and ((total>>(lead-1))&1) else 209
                    shift=e+lead-15;num=255*coefficient*2**max(0,-shift);den=256*2**max(0,shift)
                    val=(num+den//2)//den
                values.append(min(255,val))
            ee[ix][mask]=es;delta[ix][mask]=ds
        pp[ix][mask]=values
    vv=v-c['vz'];vv=np.clip(np.sign(vv)*((np.abs(vv)*c['vn']+c['vd']//2)//c['vd']),-128,127)
    avacc=pp@vv;div=2**c['avs'];tmp=np.clip((avacc+128*div+div//2)//div,0,255)
    av=np.clip((tmp-128)*c['avm']+c['oz'],0,255)
    return dict(acc=acc,raw=raw,score=score,probability=pp,exponent=ee,delta=delta,av=av)

def wide_safety_checks():
    from prepare_exp0042_attention import divide_probability
    count=0;peak=0
    for mult in range(1,19):
        # All possible raw-score deltas, including both int16 extreme products.
        raw=torch.arange(256,device='cuda',dtype=torch.int64)[None,:]
        valid=torch.ones_like(raw,dtype=torch.bool)
        for div in ['exact','sole']:
            p,e,d=wide_probability(raw,valid,mult,div)
            expected=(255-np.arange(256))*mult
            assert np.array_equal(d.cpu().numpy()[0],expected)
            es=np.minimum(15,(expected+4)//8);assert np.array_equal(e.cpu().numpy()[0],es)
            total=sum(1<<(15-int(x)) for x in es)
            pp=[divide_probability(int(x),total,div,256) for x in es]
            assert np.array_equal(p.cpu().numpy()[0],pp);count+=1;peak=max(peak,int(d.max()))
            # Masked future maximum must not affect a causal row, including prefix.
            rows=torch.tensor([[60,61,70,255],[60,60,60,255],[255,0,0,0]],device='cuda')
            masks=torch.tensor([[1,1,1,0],[1,1,1,0],[1,0,0,0]],device='cuda',dtype=torch.bool)
            pp,ee,dd=wide_probability(rows,masks,mult,div)
            changed=rows.clone();changed[:,3]=0
            assert torch.equal(pp,wide_probability(changed,masks,mult,div)[0])
            assert pp[2,0]==255 and torch.count_nonzero(pp[~masks])==0
            for shift in [-50,50,100]:
                shifted=rows[:2]+shift
                # Invalid keys may lie outside U8 in this invariance test; keep
                # them within U8 without changing any valid source value.
                shifted=shifted.clamp(0,255)
                assert torch.equal(pp[:2],wide_probability(shifted,masks[:2],mult,div)[0])
            count+=1
    assert peak==4590
    return dict(pass_all=True,checks=count,int16_max_difference=peak,int16_max_rounded=peak+4,
        all_multipliers_and_raw_deltas=True,masked_maximum_and_shift_invariance=True)
