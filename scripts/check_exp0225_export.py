#!/usr/bin/env python3
"""Independent nonsymmetric-rotation, STE and Cayley update checks."""
import random
import numpy as np
import torch
import torch.nn.functional as F
from learned_rotation_exp0225 import rotated_weight,STE,optimizer,RESULT
from rotation_exp0219 import write_json

def check():
    torch.manual_seed(225);random.seed(225)
    r1=torch.linalg.qr(torch.randn(32,32,dtype=torch.float64)).Q
    r2=torch.linalg.qr(torch.randn(8,8,dtype=torch.float64)).Q
    x=torch.randn(3,32);gamma=torch.rand(32);w=torch.randn(32,32)
    r1=r1.float();r2=r2.float();xr=x@r1
    # Independent NumPy Kronecker oracle detects transpose errors hidden by symmetric H.
    z1=r1.numpy().astype(np.float64);z2=r2.numpy().astype(np.float64);v=w.numpy().astype(np.float64)
    errors={}
    for kind in ['q','v','o','down']:
        actual=rotated_weight(w,r1,r2,kind,gamma if kind in ('q','v') else None).numpy()
        expected=z1.T@v if kind in ('o','down') else (v*gamma.numpy()[None,:])@z1
        if kind=='v':expected=np.kron(np.eye(4),z2.T)@expected
        if kind=='o':expected=expected@np.kron(np.eye(4),z2)
        errors[kind]=float(np.max(np.abs(actual-expected)));np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=3e-6)
    p=torch.randn(8,32,requires_grad=True);p.data[0]=0
    y=STE.apply(p);scale=p.detach().numpy();mx=np.abs(scale).max(1,keepdims=True);sc=np.where(mx>0,mx/7,1)
    expected=(np.clip(np.rint(scale/sc),-7,7)*sc).astype(np.float16)
    assert np.array_equal(y.detach().numpy(),expected)
    y.float().sum().backward();assert torch.equal(p.grad,torch.ones_like(p))
    p=torch.nn.Parameter(r1.clone());target=torch.eye(32);opt=optimizer([p],.1)
    before=float((p-target).square().sum())
    for _ in range(10):opt.zero_grad();((p-target).square().sum()).backward();opt.step()
    after=float((p-target).square().sum());orth=float((p.T@p-torch.eye(32)).abs().max())
    assert after<before and orth<.003
    result=dict(dense_nonsymmetric_rotation_errors=errors,STE_codes_equal_numpy=True,STE_gradient=True,Cayley_descent=[before,after],Cayley_orthogonality=orth)
    write_json(RESULT/'algebra_oracle.json',result);print(result)
if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(8)
    from output_scale_exp0224 import check as output_check
    write_json(RESULT/'output_scale_oracle.json',output_check())
