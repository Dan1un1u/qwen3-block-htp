#!/usr/bin/env python3
"""Check exact dyadic dense-dot helpers against independent FP64 arithmetic."""
import ctypes,json,subprocess
from pathlib import Path
import numpy as np
from prototype_llama32_sp2 import preflight
from probe_llama32_rotations import had,r4
ROOT=Path(__file__).resolve().parents[1]
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0024')
if __name__=='__main__':
 preflight()
 source=R/'exact-helper-check.c'; assert not source.exists()
 source.write_text('#include "'+str(ROOT/'src/dsp/llama_r4_exact.h')+'"\nvoid convert(const unsigned short*x,long long*y,int n){for(int i=0;i<n;i++)y[i]=qbh_r4_half_units(x[i]);}\nvoid rounding(const long long*x,unsigned short*y,int n,int base){for(int i=0;i<n;i++)y[i]=qbh_r4_round_units(x[i],base);}\n')
 so=R/'exact-helper-check.so';subprocess.run(['cc','-O2','-shared','-fPIC',str(source),'-o',str(so)],check=True)
 lib=ctypes.CDLL(str(so));u2=np.ctypeslib.ndpointer(dtype=np.uint16,flags='C_CONTIGUOUS');i8=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
 lib.convert.argtypes=[u2,i8,ctypes.c_int];lib.rounding.argtypes=[i8,u2,ctypes.c_int,ctypes.c_int]
 bits=np.arange(65536,dtype='u2');bits=bits[(bits&31744)!=31744];units=np.empty(len(bits),dtype='i8');lib.convert(bits,units,len(bits));assert np.array_equal(units,bits.view('f2').astype('f8')*2**24)
 rng=np.random.default_rng(24024);checks=[]
 for base in [-36,-26]:
  # Random sums, exact half-representable values and half-way ties on both signs.
  v=np.concatenate([rng.integers(-(2**57),2**57,100000,dtype='i8'),rng.integers(-(2**30),2**30,100000,dtype='i8'),units*(1<<(-24-base))])
  lo=np.arange(31743,dtype='u2').view('f2').astype('f8');hi=(np.arange(31743,dtype='u2')+1).view('f2').astype('f8');ties=np.rint((lo+hi)*.5*2**(-base)).astype('i8');v=np.concatenate([v,ties,ties-1,ties+1,-ties,-ties-1,-ties+1])
  out=np.empty(len(v),dtype='u2');lib.rounding(v,out,len(v),base)
  with np.errstate(over='ignore'): ref=(v.astype('f8')*2.**base).astype('f2').view('u2')
  assert np.array_equal(out,ref),(base,np.count_nonzero(out!=ref));checks.append(dict(base=base,cases=len(v),mismatches=0))
 # Real stage1 exact dyadic reduction of all sealed layer0 M64 inputs.
 raw=np.fromfile(R.parent/'l32-0022/validate-layer0-both-a01/actual_replay_r4_00.bin',dtype='u2',count=524288).reshape(1024,512)
 xx=np.empty(raw.size,dtype='i8');lib.convert(raw.reshape(-1),xx,len(xx));xx=xx.reshape(1024,512)
 vv=(xx@had(512).astype('i8')*181).reshape(-1);out=np.empty(len(vv),dtype='u2');lib.rounding(vv,out,len(vv),-36)
 ref=(raw.view('f2').astype('f8')@had(512)*float(np.float16(1/np.sqrt(512)))).astype('f2').view('u2').reshape(-1)
 assert np.array_equal(out,ref)
 result=dict(finite_half_conversion_cases=len(bits),rounding=checks,real_stage1_cases=len(vv),real_stage1_mismatches=0)
 (R/'exact-helper-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
