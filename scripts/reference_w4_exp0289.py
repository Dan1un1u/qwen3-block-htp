#!/usr/bin/env python3
"""Match actual FP16 HMX: raw signed4 dot, half scale at accumulator conversion."""
import prepare_exp0289 as p
from llama_u8_reference import unpack_w4_codes
import torch,numpy as np
from pathlib import Path
class RawW4:
 def __init__(self,root,name,n,k):
  self.codes=torch.from_numpy(unpack_w4_codes(root,name,n,k).copy()).half()
  self.scale=torch.from_numpy(np.fromfile(root/f'{name}_weight_w4_scale_f32.bin','<f4').copy()).half()
 def cuda(self):
  self.codes=self.codes.cuda();self.scale=self.scale.cuda();return self
def linear(x,w):
 if isinstance(w,RawW4):
  return (torch.nn.functional.linear(x.float(),w.codes.float())*w.scale.float()).half()
 return torch.nn.functional.linear(x.float(),w.float()).half()
p.unpack_w4_weight=RawW4
p.linear=linear
p.O=p.O/'reference-a03'
p.R=p.R/'reference-a03'
if __name__=='__main__':p.main('w4f16')
