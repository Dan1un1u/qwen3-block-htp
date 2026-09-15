"""Exhaustive independent exact-probability oracle for the bounded denominator domain."""
import numpy as np,json,sys,time
from pathlib import Path
r=Path(sys.argv[1]);assert not (r/'division_exhaustive.json').exists();start=time.monotonic();checked=0;N=255<<15
for base in range(1,257*32768+1,65536):
 z=np.arange(base,min(base+65536,257*32768+1),dtype=np.uint64);q=N//z;rem=N-q*z
 for c in range(16):
  old=((N>>c)+z//2)//z
  new=q+(2*rem>=z) if c==0 else (q+(1<<(c-1)))>>c
  assert np.array_equal(old,new),(base,c);checked+=len(z)
result=dict(pass_all=True,denominator_min=1,denominator_max=257*32768,codes=16,comparisons=checked,arithmetic='original integer round-half-up before byte cast; no rounded-p0 reuse',seconds=time.monotonic()-start)
with (r/'division_exhaustive.json').open('x') as f:json.dump(result,f,indent=2)
print(result)
