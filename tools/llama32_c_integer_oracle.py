"""Accelerate exact integer reference dots without approximating HMX conversion.

For U8 times signed[-7,7] K<=8192, every product and partial sum is bounded
by14622720<2**24; TF32-disabled FP32 dot is integer-exact. SP2 dots combine
two such products in int64. Norm, attention, residual and libnative HMX
conversion retain the independent existing implementation.
"""
import numpy as np
import torch
import torch.nn.functional as F
from llama_u8_reference import _cached_w4_projection,projection_bias_words
from llama32_sp2_contract import encode
class NativeOracle:
    def __init__(self):
        self.weights={};self.biases={};self.audited=set();self.audited_down=set()
        torch.backends.cuda.matmul.allow_tf32=False
    def weights_for(self,package,name,n,k):
        key=(str(package.resolve()),name,n,k)
        if key not in self.weights:
            w,scale=_cached_w4_projection(*key)
            assert w.min()>=-7 and w.max()<=7 and k<=8192
            self.weights[key]=torch.tensor(w.astype('f4'),device='cuda')
        return key,self.weights[key]
    def dot(self,x,key,w):
        xx=np.asarray(x,dtype=np.uint8)
        with torch.autocast(device_type='cuda',enabled=False):y=F.linear(torch.tensor(xx.astype('f4'),device='cuda'),w).to(torch.int64).cpu().numpy()
        if key not in self.audited:
            source,_=_cached_w4_projection(*key);indices=np.linspace(0,len(source)-1,16,dtype=int)
            expected=xx[:3].astype('i8')@source[indices].astype('i8').T
            assert np.array_equal(y[:3,indices],expected)
            self.audited.add(key)
        return y
    def project(self,activation,package,name,n,k,inq,outq,converter):
        key,w=self.weights_for(package,name,n,k)
        bkey=(key,float(inq['scale']),int(inq['zero_point']),float(outq['scale']),int(outq['zero_point']))
        if bkey not in self.biases:
            codes,scales=_cached_w4_projection(*key);self.biases[bkey]=projection_bias_words(codes,scales,inq,outq)
        return converter.convert(self.dot(activation,key,w),*self.biases[bkey])
    def down(self,g,u,package,q):
        from llama_sp2_reference import table
        v=table(str(package/'silu_up_lut_u16.bin'))[g,u]
        c=encode(v);key,w=self.weights_for(package,'down',2048,8192)
        codes,scales=_cached_w4_projection(*key)
        lo=self.dot((c&255).astype('u1'),key,w);hi=self.dot((c>>8).astype('u1'),key,w)
        acc=lo+257*hi-32770*codes.astype('i4').sum(1,dtype='i4')
        if key not in self.audited_down:
            columns=np.linspace(0,len(codes)-1,16,dtype=int)
            expected=v[:3].astype('i8')@codes[columns].astype('i8').T
            assert np.array_equal(acc[:3,columns],expected)
            self.audited_down.add(key)
        multiplier=np.floor(float(q['middle']['scale'])*scales.astype('f8')/float(q['down']['scale'])*2**31+.5).astype('i8')
        assert ((multiplier>0)&(multiplier<2**31)).all() and np.abs(acc).max()<2**31
        y=np.clip(((acc*multiplier+2**30)>>31)+q['down']['zero_point'],0,255).astype('u1')
        return v,y
    def install(self):
        import export_llama32_u8,llama_sp2_reference
        export_llama32_u8.project_w4u8=self.project
        llama_sp2_reference.project_down=self.down
