"""Dense FP32 training rotations; final deployment folding remains FP64."""
import torch

def rotated_weight_fp32(self, R1=None, R2=None, transpose=False):
    if R1 is None:return self.weight
    dtype=self.weight.dtype
    with torch.autocast(device_type="cuda",enabled=False):
        w=self.weight.float();r=R1.float()
        weight=(r.T@w if transpose else w@r).to(dtype)
        if R2 is not None:
            dim=R2.shape[0]
            if transpose:
                shape=weight.shape
                weight=(weight.reshape(-1,shape[-1]//dim,dim).float()@R2.float()).reshape(shape)
            else:
                w=weight.t();shape=w.shape
                weight=(w.reshape(-1,shape[-1]//dim,dim).float()@R2.float()).reshape(shape).t()
    return weight.to(dtype)
