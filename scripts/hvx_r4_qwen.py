"""EXP-0309 complete generalized Hadamard; no padding or blockwise substitute."""
import numpy as np
H12=np.array([[1,1,1,1,1,1,1,1,1,1,1,1],
[1,-1,-1,1,-1,-1,-1,1,1,1,-1,1],
[1,1,-1,-1,1,-1,-1,-1,1,1,1,-1],
[1,-1,1,-1,-1,1,-1,-1,-1,1,1,1],
[1,1,-1,1,-1,-1,1,-1,-1,-1,1,1],
[1,1,1,-1,1,-1,-1,1,-1,-1,-1,1],
[1,1,1,1,-1,1,-1,-1,1,-1,-1,-1],
[1,-1,1,1,1,-1,1,-1,-1,1,-1,-1],
[1,-1,-1,1,1,1,-1,1,-1,-1,1,-1],
[1,-1,-1,-1,1,1,1,-1,1,-1,-1,1],
[1,1,-1,-1,-1,1,1,1,-1,1,-1,-1],
[1,-1,1,-1,-1,-1,1,1,1,-1,1,-1]],dtype='i1')
assert np.array_equal(H12@H12.T,np.eye(12,dtype='i1')*12)
def transform(x):
 x=np.array(x,copy=True);n=x.shape[-1];b=n//12;assert n in [3072,6144]
 for h in [1<<j for j in range(b.bit_length()-1)]:
  v=x.reshape(-1,n//(2*h),2*h);a=v[:,:,:h].copy();c=v[:,:,h:].copy();v[:,:,:h]=a+c;v[:,:,h:]=a-c
 v=x.reshape(-1,12,b);out=np.zeros_like(v)
 for j in range(12):
  for g in range(12):
   if H12[j,g]>0:out[:,j]+=v[:,g]
   else:out[:,j]-=v[:,g]
 return (out*np.array(1/np.sqrt(n),dtype=x.dtype)).reshape(x.shape)

def pack_hmx_w4(a):
 n,k=a.shape
 a=a.reshape(n//32,32,k//32,32).transpose(0,2,3,1)
 a=np.ascontiguousarray(a.reshape(n//32,k//32,8,4,32).transpose(0,1,2,4,3)).reshape(n//32,k//32,1024)
 a=(a.astype(np.int16)&15).astype(np.uint8)
 return a[...,::2]|(a[...,1::2]<<4)
