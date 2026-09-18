import sys,subprocess,json,hashlib
from pathlib import Path
import numpy as np
S=Path('/home/daniuniu/work/llama32-htp')
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0047/rope-component-v3');R.mkdir(exist_ok=False)
sdk=Path('/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0');bin=sdk/'tools/HEXAGON_Tools/19.0.07/Tools/bin'
c=R/'component.c';c.write_text("""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "hvx_fp16_ops.h"
extern void qbh_llama_rope_head(__fp16*,uint32_t,uint32_t,uint32_t,uint32_t,const __fp16*,const __fp16*);
static __fp16 x[64*3072] __attribute__((aligned(128)));
static __fp16 c[64*128] __attribute__((aligned(128)));
static __fp16 s[64*128] __attribute__((aligned(128)));
int main(int argc,char**argv) {
 FILE*f=fopen(argv[1],"rb");fread(x,2,64*3072,f);fread(c,2,64*128,f);fread(s,2,64*128,f);fclose(f);
 qbh_hvx_qk_norm_rope_f16(x,64,24,3072,128,NULL,c,s,NULL);
 f=fopen(argv[2],"wb");fwrite(x,2,64*3072,f);fclose(f);return 0;
}
""")
cmd=[str(bin/'hexagon-clang'),'-mv79','-mhvx','-mhvx-length=128B','-mllvm','-enable-xqf-gen=true','-O2','-ffunction-sections','-fdata-sections','-DQBH_MODEL_LLAMA32=1','-DQBH_LLAMA_3B=1',
'-I'+str(S/'include'),'-I'+str(sdk/'libs/qhl_hvx/inc/internal/xqf'),'-I'+str(sdk/'libs/qhl_hvx/inc/internal'),'-I'+str(sdk/'libs/qhl_hvx/inc'),str(c),str(S/'src/dsp/hvx_fp16_ops.c'),'-Wl,--gc-sections','-lm','-o',str(R/'component.elf')]
z=subprocess.run(cmd,capture_output=True,text=True);(R/'build.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr
rng=np.random.default_rng(47);x=(rng.normal(size=(64,24,128))*rng.choice([.001,1,16,128],size=(64,24,1))).astype('<f2')
angle=rng.uniform(-100,100,size=(64,64));c=np.tile(np.cos(angle),(1,2)).astype('<f2');s=np.tile(np.sin(angle),(1,2)).astype('<f2')
with (R/'input.bin').open('wb') as f:
 for a in [x,c,s]:f.write(a.tobytes())
xx=x.astype('f4');rr=np.concatenate([-xx[:,:,64:],xx[:,:,:64]],axis=-1)
expected=(xx*c[:,None].astype('f4')+rr*s[:,None].astype('f4')).astype('<f2');expected.tofile(R/'expected.bin')
cmd2=[str(bin/'hexagon-sim'),'-mv79',str(R/'component.elf'),'--',str(R/'input.bin'),str(R/'actual.bin')]
z=subprocess.run(cmd2,capture_output=True,text=True);(R/'sim.log').write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr
actual=np.fromfile(R/'actual.bin','<f2').reshape(x.shape);m=int(np.count_nonzero(actual.view('u2')!=expected.view('u2')))
record=dict(pass_all=m==0,mismatches=m,elements=x.size,finite=bool(np.isfinite(actual).all()),max_abs=float(np.max(abs(actual.astype('f4')-expected.astype('f4')))),compile=cmd,simulator=cmd2,source_sha256=hashlib.sha256((S/'src/dsp/hvx_fp16_ops.c').read_bytes()).hexdigest())
(R/'result.json').write_text(json.dumps(record,indent=2));print(record);assert m==0
