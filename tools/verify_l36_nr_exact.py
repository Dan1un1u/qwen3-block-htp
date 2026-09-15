import sys,re,json,time
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);S=Path('/home/daniuniu/work/llama32-htp');t=(S/'src/dsp/attention_u8_core.c').read_text();seed=np.array([int(v) for v in re.search(r'qbh_nr64_reciprocal_q30\[64\] = \{([^}]+)',t)[1].replace('U','').split(',')],dtype=np.uint64)
N=255<<15;worst=0;checked=0;start=time.monotonic()
for base in range(32768,257*32768+1,65536):
 z=np.arange(base,min(base+65536,257*32768+1),dtype=np.uint64);leading=np.floor(np.log2(z)).astype(np.uint64);x=z<<(30-leading);r=seed[((x-(1<<30))>>24).astype('i8')];r=(r*((2<<30)-((x*r)>>30)))>>30;q=(N*r)>>(leading+30);ideal=N//z;worst=max(worst,int(np.max(np.abs(q.astype('i8')-ideal.astype('i8')))));q+=(q+1)*z<=N;q-=q*z>N;assert np.array_equal(q,ideal);checked+=len(z)
with (R/'nr_exact_quotient.json').open('x') as f:json.dump(dict(pass_all=True,denominators=checked,minimum=32768,maximum=257*32768,worst_initial_quotient_lsb=worst,seconds=time.monotonic()-start,contract='correct quotient before probability LUT; sum<32768 retains exact divide'),f,indent=2)
print('NR_EXACT_PASS',checked,worst)
