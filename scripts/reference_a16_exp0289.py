#!/usr/bin/env python3
"""Independent reference honoring existing HVX FP16 intermediate boundaries.

Q/K inverse RMS, sigmoid exp/denominator/reciprocal are stored FP16 by
the existing kernels. No device outputs or fitted constants enter the oracle.
"""
import sys
import prepare_exp0289 as p
from reference_w4_exp0289 import RawW4,linear
import torch
base_rms=p.rms
def rms(x,w):
 if x.shape[-1]==128:
  inv=torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6).half()
  return (x.float()*w.float()*inv.float()).half()
 return base_rms(x,w)
def silu(x):
 e=(-x.abs()).exp().half()
 denom=(1+e.float()).half()
 inv=denom.float().reciprocal().half()
 neg=(e.float()*inv.float()).half()
 return x*torch.where(x<0,neg,inv).float()
p.rms=rms
p.F.silu=silu
p.linear=linear
p.unpack_w4_weight=RawW4
p.O=p.O.parent/'reference-a04'
p.R=p.R.parent/'reference-a04'
if __name__=='__main__':p.main(sys.argv[1])
