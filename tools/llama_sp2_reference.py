"""Independent SP2 integer oracle; never consumes hardware-generated tensors."""
from functools import lru_cache
import numpy as np
from llama_u8_reference import _cached_w4_projection
from prototype_llama32_sp2 import oracle

@lru_cache(None)
def table(path):
 return np.fromfile(path,dtype='<u2').reshape(256,256).astype('i4')-32768

def project_down(g,u,package,q):
 v=table(str(package/'silu_up_lut_u16.bin'))[g,u]
 w,ws=_cached_w4_projection(str(package.resolve()),'down',2048,8192)
 mult=np.floor(float(q['middle']['scale'])*ws.astype('f8')/float(q['down']['scale'])*2**31+.5).astype('i8')
 assert np.all((mult>0)&(mult<2**31))
 acc=oracle(v,w)
 assert np.max(np.abs(acc))<2**31
 y=np.clip(((acc*mult+2**30)>>31)+q['down']['zero_point'],0,255).astype('u1')
 return v,y
