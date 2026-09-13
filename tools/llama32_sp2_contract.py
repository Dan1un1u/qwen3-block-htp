"""187-level Eq8 SP2 grid with an exact two-U8 radix257 representation.

The normalized magnitudes are {a+b}, a in {0,2^-1,...,2^-15},
b in {0,2^-1,...,2^-7}. One sign applies to the entire magnitude.
Reference: rotation-quant d9a636b scripts/phase2/down_codebooks.py.
"""
import numpy as np
RADIX=257
BIAS=32770
DENOMINATOR=32768
MODE=9

def integer_levels():
    a=[0]+[1 << (15-i) for i in range(1,16)]
    b=[0]+[1 << (15-i) for i in range(1,8)]
    values=np.array(sorted({sign*(x+y) for x in a for y in b for sign in (-1,1)}),dtype=np.int32)
    assert len(values)==187 and values[0]==-32768 and values[-1]==32768
    return values

def encode(v):
    v=np.asarray(v,dtype=np.int64)
    if not np.isin(v,integer_levels()).all():raise ValueError("Value outside SP2 codebook")
    high,low=np.divmod(v+BIAS,RADIX)
    if not ((high>=0)&(high<=255)&(low>=0)&(low<=255)).all():raise ValueError("SP2 two-plane overflow")
    return ((high<<8)|low).astype('<u2')

def decode(code):
    code=np.asarray(code,dtype=np.uint16)
    return (code&255).astype(np.int32)+RADIX*(code>>8).astype(np.int32)-BIAS

def quantize_integer(x,alpha):
    """FP32 normalize; nearest magnitude, halfway toward zero, frozen alpha."""
    alpha=np.float32(alpha)
    if not np.isfinite(alpha) or alpha<=0:raise ValueError("Invalid alpha")
    levels=integer_levels();positive=levels[levels>=0]
    normalized=np.minimum(np.abs(np.asarray(x,dtype=np.float32))/alpha,np.float32(1))
    mid=(positive[:-1]+positive[1:]).astype(np.float32)/(2*DENOMINATOR)
    indices=np.searchsorted(mid,normalized,side='left')
    return (positive[indices]*np.sign(x)).astype(np.int32)

def contract():
    return dict(mode=MODE,levels=187,radix=RADIX,bias=BIAS,denominator=DENOMINATOR,
                formula="v=low+257*high-32770",tie="nearest magnitude; ties toward zero",
                exponent_fields=[15,7],range=[-32768,32768])
