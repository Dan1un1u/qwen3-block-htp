"""Dense FP32 training rotations; final deployment folding remains FP64."""
import torch

def rotated_weight_fp32(self, R1=None, R2=None, transpose=False):
    if R1 is None:return self.weight
    dtype=self.weight.dtype
    with torch.autocast(device_type="cuda",enabled=False):
        precision=torch.float64 if transpose and R2 is not None else torch.float32
        w=self.weight.to(precision);r=R1.to(precision)
        weight=(r.T@w if transpose else w@r).to(dtype)
        if R2 is not None:
            dim=R2.shape[0]
            if transpose:
                shape=weight.shape
                weight=(weight.reshape(-1,shape[-1]//dim,dim).double()@R2.double()).reshape(shape)
            else:
                w=weight.t();shape=w.shape
                weight=(w.reshape(-1,shape[-1]//dim,dim).double()@R2.double()).reshape(shape).t()
    return weight.to(dtype)
