"""Independent EXP0292 changed-boundary audit using actual device operands."""
import numpy as np,json,sys
from pathlib import Path
C=64*24*128*2;BASE=2*C+64*24*128;H=1024
def norm(x,g):
 lanes=np.zeros((len(x),32),np.float32)
 for c in range(0,H,32):
  v=x[:,c:c+32];lanes=np.add(lanes,np.multiply(v,v,dtype=np.float32),dtype=np.float32)
 sums=np.zeros(len(x),np.float32)
 for c in range(32):sums=np.add(sums,lanes[:,c],dtype=np.float32)
 inv=np.divide(np.float32(1),np.sqrt(np.add(sums/np.float32(H),np.float32(1e-6)),dtype=np.float32),dtype=np.float32)
 return ((x*inv[:,None]).astype(np.float32)*g).astype(np.float16)
def unpack(a,rows):
 # HMX FP16 native [column_tile,row_pair,column,row_in_pair].
 return a.reshape(2,H//32,16,32,2).transpose(0,2,4,1,3).reshape(64,H)[:rows]
def check(root):
 root=Path(root);out=[]
 for step in [0,1]:
  f=root/f'step{step:02d}_r3.bin';raw=f.read_bytes();rows=64 if step==0 else 1
  def slot(i,dtype,n):return np.frombuffer(raw,dtype=dtype,count=n,offset=BASE+i*C).copy()
  x=slot(5,'<f4',rows*H).reshape(rows,H)
  o=slot(7,'<f2',rows*H).astype(np.float32).reshape(rows,H)
  r=slot(8,'<f4',rows*H).reshape(rows,H)
  d=slot(10,'<f2',rows*H).astype(np.float32).reshape(rows,H)
  y=slot(4,'<f4',rows*H).reshape(rows,H)
  assert np.isfinite(x).all() and np.isfinite(y).all()
  assert np.array_equal((x+o).view('u4'),r.view('u4')),'post add'
  assert np.array_equal((r+d).view('u4'),y.view('u4')),'final add'
  norms=[]
  for ni,xi,gi in [(6,x,0),(9,r,1)]:
   actual=unpack(slot(ni,'<f2',64*H),rows)
   ref=norm(xi,slot(gi,'<f2',H).astype(np.float32))
   delta=abs(actual.astype('f4')-ref.astype('f4'))
   ulp=np.maximum(abs(np.spacing(abs(ref))),np.float16(2**-24)).astype('f4')
   maxulp=float(np.max(delta/ulp))
   norms.append(dict(slot=ni,exact=bool(np.array_equal(actual.view('u2'),ref.view('u2'))),max_abs=float(delta.max()),max_half_ulp=maxulp))
   assert maxulp<=1.,norms[-1]
  out.append(dict(step=step,rows=rows,residual_adds_bit_exact=True,norms=norms))
 (root/'fp32-components.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out))
if __name__=='__main__':check(sys.argv[1])
