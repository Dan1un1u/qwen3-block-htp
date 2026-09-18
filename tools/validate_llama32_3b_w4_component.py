import subprocess,json,hashlib,sys
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/llama32-htp');sys.path.insert(0,str(S/'tools'))
from llama_reference import pack_weight
from llama_u8_reference import unpack_w4_codes
import torch
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0047/w4-component');R.mkdir(exist_ok=False)
sdk=Path('/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0');bin=sdk/'tools/HEXAGON_Tools/19.0.07/Tools/bin'
c=R/'component.c';c.write_text("""
#include <stdio.h>
#include <stdlib.h>
#include "w4_u8_expand.h"
static unsigned char x[256*512] __attribute__((aligned(128)));
static unsigned char y[256*2048] __attribute__((aligned(128)));
int main(int argc,char**argv) {
 unsigned n=atoi(argv[3]);FILE*f=fopen(argv[1],"rb");fread(x,512,n,f);fclose(f);
 qbh_unpack_w4_to_f16_hvx_relaxed(x,y,n);
 f=fopen(argv[2],"wb");fwrite(y,2048,n,f);fclose(f);return 0;
}
""")
cmd=[str(bin/'hexagon-clang'),'-mv79','-mhvx','-mhvx-length=128B','-O2','-ffunction-sections','-fdata-sections','-I'+str(S/'include'),str(c),str(S/'src/dsp/w4_u8_expand.c'),'-Wl,--gc-sections','-o',str(R/'component.elf')]
z=subprocess.run(cmd,capture_output=True,text=True);(R/'build.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr
cases=[]
for n in [96,256]:
 d=R/str(n);d.mkdir()
 np.random.default_rng(n).integers(0,256,size=(n,512),dtype='u1').tofile(d/'test_weight_w4_hmx.bin')
 w=unpack_w4_codes(d,'test',32,n*32).astype('<f2')
 expected=pack_weight(torch.from_numpy(w)).reshape(-1);expected.tofile(d/'expected.bin')
 z=subprocess.run([str(bin/'hexagon-sim'),'-mv79',str(R/'component.elf'),'--',str(d/'test_weight_w4_hmx.bin'),str(d/'actual.bin'),str(n)],capture_output=True,text=True);(d/'sim.log').write_text(z.stdout+z.stderr);assert z.returncode==0
 a=np.fromfile(d/'actual.bin','<f2');m=int(np.count_nonzero(a.view('u2')!=expected.view('u2')));assert m==0
 cases.append(dict(k_tiles=n,elements=len(a),mismatches=m))
(R/'result.json').write_text(json.dumps(dict(pass_all=True,cases=cases,compile=cmd,source_sha256=hashlib.sha256((S/'src/dsp/w4_u8_expand.c').read_bytes()).hexdigest()),indent=2));print(cases)
