#!/usr/bin/env python3
"""Independent float64 vector oracle for the device's histogram reducer."""
import ctypes as C
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

class Score(C.Structure):
    _fields_ = [(x,C.c_float) for x in ("lse","target","nll")] + [
        (x,C.c_uint32) for x in ("count","rank","ties","max_ties","sat","nonfinite")]

def main():
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as d:
        d=Path(d)
        (d/"test.c").write_text('#include "eval_score.h"\n'
            'struct qbh_eval_score score(const uint32_t *h,int u,float s,int z,unsigned t)'
            '{return qbh_eval_reduce(h,u,s,z,t);}\n')
        subprocess.run(["gcc","-std=c11","-O2","-shared","-fPIC","-I",str(root/"include"),
            str(d/"test.c"),"-lm","-o",str(d/"test.so")],check=True)
        lib=C.CDLL(str(d/"test.so"));lib.score.restype=Score
        lib.score.argtypes=[C.POINTER(C.c_uint32),C.c_int,C.c_float,C.c_int,C.c_uint]
        rng=np.random.default_rng(218); worst=0.;cases=0
        for u8 in (False,True):
            for k in range(20):
                if u8:
                    codes=rng.integers(0,256,151936,dtype=np.uint32)
                    scale=np.float32([.01,.127,1.,2.25][k%4]);zero=129
                    values=((codes.astype(np.float32)-zero)*scale).astype(np.float64)
                else:
                    codes=rng.normal(0,10,151936).astype(np.float16).view(np.uint16).astype(np.uint32)
                    codes[:5]=[0,32768,1,32769,15360]
                    values=codes.astype(np.uint16).view(np.float16).astype(np.float64)
                    scale=np.float32(1.);zero=0
                idx=[0,1,2,1234,151935][k%5];target=int(codes[idx])
                hist=np.bincount(codes,minlength=256 if u8 else 65536).astype(np.uint32)
                r=lib.score(hist.ctypes.data_as(C.POINTER(C.c_uint32)),u8,float(scale),zero,target)
                lse=values.max()+np.log(np.exp(values-values.max()).sum())
                error=abs(r.nll-(lse-values[idx]));worst=max(worst,error)
                assert error<1e-4,(u8,k,error)
                assert r.count==151936 and r.rank==1+int((values>values[idx]).sum())
                assert r.ties==int((values==values[idx]).sum()) and r.max_ties==int((values==values.max()).sum())
                assert r.sat==(int(((codes==0)|(codes==255)).sum()) if u8 else 0)
                assert r.nonfinite==0
                cases+=1
        hist=np.zeros(65536,dtype=np.uint32);hist[31744]=1;hist[0]=151935
        r=lib.score(hist.ctypes.data_as(C.POINTER(C.c_uint32)),0,1.,0,0)
        assert r.nonfinite==1 and np.isnan(r.nll)
        print(json.dumps(dict(pass_all=True,vector_cases=cases,logits_per_case=151936,
            maximum_nll_absolute_error=worst,nonfinite_detection=True)))

if __name__=="__main__": main()
