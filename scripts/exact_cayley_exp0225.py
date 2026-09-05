#!/usr/bin/env python3
"""FP64 exact Cayley solve for the same SpinQuant square Stiefel update.

Adapted from Meta SpinQuant SGDG at8f47aa3f00e8662caf1a484153920a07e5281c3a;
CC BY-NC4.0, see spinquant_exp0225/LICENSE and PROVENANCE.md.
The five-step fixed-point solve and random QR are replaced by an FP64 solve.
No momentum, weight update, hyperparameter search or new objective is added.
"""
import torch
class ExactCayleySGD(torch.optim.Optimizer):
    def __init__(self,params,lr):super().__init__(params,dict(lr=lr))
    @torch.no_grad()
    def step(self,closure=None):
        assert closure is None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                r=p.double();v=-p.grad.double().T
                mx=v@r;w_hat=mx-.5*(r.T@(r@mx));w=w_hat-w_hat.T
                alpha=min(group['lr'],float(1/(w.abs().sum(0).max()+1e-8)))
                eye=torch.eye(len(r),device=r.device,dtype=torch.float64)
                updated=torch.linalg.solve(eye-.5*alpha*w,(eye+.5*alpha*w)@r.T).T
                assert torch.isfinite(updated).all()
                p.copy_(updated.to(p.dtype))
