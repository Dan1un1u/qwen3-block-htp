"""Batched NumPy exact-division attention for the C offline integer oracle.

Same SDK conversion and scalar integer contract as llama_u8_reference; batch
heads and vectorize exact divisions to make repeated matched PPL practical.
No device path or general reference implementation is changed.
"""
import numpy as np
from llama_u8_reference import (raw_u8s8_accumulator, power_of_two_bias_words,
    post_centered_requant, signed_round_divide)

def attention(q_u8,k_cache_u8,v_cache_u8,past_tokens,configs,converter,divide_probability):
    q=np.ascontiguousarray(q_u8,dtype='u1');k=np.ascontiguousarray(k_cache_u8,dtype='u1');v=np.ascontiguousarray(v_cache_u8,dtype='u1')
    rows,heads,dim=q.shape;groups,tokens,kdim=k.shape
    assert dim==kdim==64 and v.shape==k.shape and heads==groups*4 and tokens==past_tokens+rows
    scores=np.empty((heads,rows,tokens),dtype='u1');probs=np.zeros_like(scores);output=np.empty_like(q)
    mask=np.arange(tokens)[None,:] <= (past_tokens+np.arange(rows))[:,None]
    for group,fields in enumerate(configs):
        (abi,g,fraction_bits,division_mode,qz,kz,vz,pz,oz,vn,vd,ss,sm,avs,avm)=fields
        assert g==group and fraction_bits==3 and division_mode==1
        ks=np.clip(k[group].astype('i2')-kz,-128,127).astype('i1');ksums=ks.astype('i4').sum(1,dtype='i4')
        divisor=1<<ss;lower,_=power_of_two_bias_words(tokens,ss,128)
        upper=(-qz*ksums+128*divisor+(divisor//2 if ss else 0)).astype('<i4')
        x=q[:,group*4:group*4+4].transpose(1,0,2).reshape(4*rows,64)
        score=post_centered_requant(converter.convert(raw_u8s8_accumulator(x,ks),lower,upper),sm,128).reshape(4,rows,tokens)
        maximum=np.where(mask[None],score.astype('i4'),-1).max(-1,keepdims=True)
        exponents=np.minimum(15,(maximum-score.astype('i4')+4)>>3)
        # Invalid future columns can have a negative difference; mask BEFORE shift.
        exponents=np.where(mask[None],exponents,15)
        powers=np.left_shift(np.int64(1),15-exponents)
        powers=np.where(mask[None],powers,0)
        total=powers.sum(-1,keepdims=True,dtype='i8')
        probability=np.minimum(255,(powers*255+total//2)//total).astype('u1')
        probability=np.where(mask[None],probability,0).astype('u1')
        scores[group*4:group*4+4]=score;probs[group*4:group*4+4]=probability
        signed_v=np.clip(signed_round_divide(v[group].astype('i4')-vz,vn,vd),-128,127).astype('i1')
        lower,upper=power_of_two_bias_words(64,avs,oz if avm==1 else 128)
        av=converter.convert(raw_u8s8_accumulator(probability.reshape(4*rows,tokens),signed_v.T),lower,upper)
        if avm!=1:av=post_centered_requant(av,avm,oz)
        output[:,group*4:group*4+4]=av.reshape(4,rows,64).transpose(1,0,2)
    return output,scores,probs
