#!/usr/bin/env python3
"""Independent NumPy clipping and dense-Schur final-output selection oracle."""
import json,tempfile
from pathlib import Path
import numpy as np
import torch
from quantize_llama32 import factor,export
from llama_reference import sha256

def main():
 torch.set_grad_enabled(False);torch.set_num_threads(8);torch.manual_seed(320024)
 x=torch.randn(1100,32,device='cuda');x[:,1]=.8*x[:,0]+.2*x[:,1];w=torch.randn(32,32,device='cuda');w[:,0]*=5;w[0]=0
 f=factor(x)
 with tempfile.TemporaryDirectory() as temp:
  out,stats=export(w,x,f,Path(temp),'oracle')
  selected_sc=np.fromfile(Path(temp)/'oracle_weight_w4_scale_f32.bin',dtype='<f4')
 wn=w.cpu().numpy();xn=x.cpu().numpy();base=np.max(abs(wn),1)/7;base[0]=1
 errors=[]
 for ratio in [1-i/100 for i in range(80)]:
  sc=base*ratio;errors.append(np.sum(abs(np.clip(np.rint(wn/sc[:,None]),-7,7)*sc[:,None]-wn)**2.4,1))
 choice=np.argmin(np.stack(errors,1),1);clip=base*(1-choice/100).astype(np.float32);ranges=np.stack([base,(base+clip)*.5,clip])
 perm=f['perm'].cpu().numpy();hp=(xn.T@xn)*(2/len(xn));hp=hp[perm][:,perm].astype(np.float64);hp.flat[::33]+=f['damping']
 products=[];scores=[]
 for sc in ranges:
  inv=np.linalg.inv(hp);work=wn[:,perm].astype(np.float64).copy();q=np.empty_like(work)
  for j in range(32):
   c=np.clip(np.rint(work[:,j]/sc),-7,7);q[:,j]=c;e=work[:,j]-c*sc;work[:,j:]-=e[:,None]*(inv[0]/inv[0,0])[None,:]
   inv=inv[1:,1:]-inv[1:,0,None]*inv[None,0,1:]/inv[0,0]
  val=(q[:,np.argsort(perm)].astype(np.float32)*sc[:,None]).astype(np.float16)
  products.append(val);delta=val.astype(np.float64)-wn.astype(np.float64);scores.append(np.square(xn.astype(np.float64)@delta.T).sum(0))
 best=np.argmin(np.stack(scores,1),1);expected=np.stack(products)[best,np.arange(32)];expected_sc=ranges[best,np.arange(32)]
 np.testing.assert_array_equal(out.cpu().numpy(),expected);np.testing.assert_allclose(selected_sc,expected_sc,rtol=1e-6,atol=0)
 report={'pass':True,'seed':320024,'independent_numpy_clipping':True,'dense_Schur_GPTQ':True,'independent_FP64_final_output_selection':True,'selected_FP16_weights_exact':True,'quantizer_sha256':sha256(Path(__file__).with_name('quantize_llama32.py')),'stats':stats}
 dest=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0002/quantizer_independent_oracle.json')
 with dest.open('x') as stream:json.dump(report,stream,indent=2)
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
