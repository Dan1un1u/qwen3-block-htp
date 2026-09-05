import numpy as np
import torch
from exact_cayley_exp0225 import ExactCayleySGD
from learned_rotation_exp0225 import RESULT
from rotation_exp0219 import write_json
torch.manual_seed(225);torch.set_num_threads(8)
r=torch.linalg.qr(torch.randn(32,32,dtype=torch.float64)).Q
p=torch.nn.Parameter(r.clone());target=torch.eye(32,dtype=torch.float64)
g=2*(r-target);v=-g.numpy().T
# Independent square-manifold tangent formula and NumPy linear solve.
w=.5*(v@r.numpy()-r.numpy().T@v.T);a=min(.1,1/(np.abs(w).sum(0).max()+1e-8))
expected=np.linalg.solve(np.eye(32)-.5*a*w,(np.eye(32)+.5*a*w)@r.numpy().T).T
opt=ExactCayleySGD([p],lr=.1);p.grad=g.clone();opt.step()
np.testing.assert_allclose(p.detach().numpy(),expected,rtol=1e-12,atol=1e-12)
before=float((r-target).square().sum())
for _ in range(99):
 opt.zero_grad();((p-target).square().sum()).backward();opt.step()
orth=float((p.T@p-torch.eye(32)).norm().detach());after=float((p-target).square().sum().detach())
assert orth<1e-10 and after<before
write_json(RESULT/'exact_cayley_oracle.json',dict(numpy_solve_agreement=True,square_manifold_tangent_independent=True,steps=100,FP64_orthogonality_frobenius=orth,loss_before=before,loss_after=after))
print('EXACT_CAYLEY_PASS',orth,before,after)
