#!/usr/bin/env python3
import numpy as np,json
from pathlib import Path
from fractions import Fraction
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0252')
maximum=0;changed=0;count=0
# Every possible denominator for the frozen hardware scope; all16 exponent entries.
for start in range(32768,72*32768+1,65536):
 s=np.arange(start,min(start+65536,72*32768+1),dtype=np.int64);lead=np.floor(np.log2(s)).astype(np.int64);x=s<<(30-lead);idx=(x-2**30)>>24;den=129+2*idx;r=(2**37+den//2)//den;r=(r*(2**31-((x*r)>>30)))>>30;shift=lead+30
 for e in range(16):
  num=255*2**(15-e);code=(num*r+(np.ones_like(s)<<(shift-1)))>>shift;exact=(num+s//2)//s
  d=abs(code-exact);maximum=max(maximum,int(d.max()));changed+=int(np.count_nonzero(d));count+=len(s)
assert maximum<=1
lut=[(2**37+(129+2*i)//2)//(129+2*i) for i in range(64)]
for i,v in enumerate(lut):assert v==int(Fraction(2**37,129+2*i)+Fraction(1,2))
with (R/'nr64_denominator_check.json').open('x') as f:json.dump(dict(pass_all=True,denominator_min=32768,denominator_max=72*32768,entries=count,max_probability_lsb=maximum,different_from_exact=changed,table_Q30=lut),f,indent=2)
print('NR64_DENOMINATORS_PASS',maximum,changed,count)
