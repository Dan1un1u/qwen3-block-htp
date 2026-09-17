#!/usr/bin/env python3
"""Higher-accuracy independent dot products; no device-derived values."""
import sys,inspect
import prepare_exp0289 as p
from reference_w4_exp0289 import RawW4
import torch
def linear(x,w):
 if isinstance(w,RawW4):
  return (torch.nn.functional.linear(x.double(),w.codes.double())*w.scale.double()).half()
 return torch.nn.functional.linear(x.double(),w.double()).half()
src=inspect.getsource(p.layer)
src=src.replace('q.float(),k.repeat_interleave(2,0).float()', 'q.double(),k.repeat_interleave(2,0).double()')
src=src.replace('prob.float(),v.repeat_interleave(2,0).float()', 'prob.double(),v.repeat_interleave(2,0).double()')
exec(src,p.__dict__)
p.linear=linear
p.O=p.O.parent/'reference-a05'
p.R=p.R.parent/'reference-a05'
if __name__=='__main__':p.main(sys.argv[1])
