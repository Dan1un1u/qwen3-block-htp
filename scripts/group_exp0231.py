#!/usr/bin/env python3
"""Static original-column group128 GPTQ, software diagnostic only.

Algorithm reference IST-DASLab/gptq 2d65066eeb06a5c9ff5184d8cebdf33662c67faf
(Apache-2.0), static_groups + actorder. No LPBQ or scale quantization.
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from gptq_exp0221 import factor, pack_codes
from clipping_exp0223 import select_scale, RATIOS
from rotation_exp0219 import unpack
from data_exp0231 import OUTPUT, RESULT, sha, write, preflight

GROUP = 128
FORMAT = 'software_only_signed_W4_static_original_column_group128_FP32_scales_v1'

def expand(scale):
    return scale.repeat_interleave(GROUP, dim=1)

def quantize(w, f, scale, block=128):
    n,k=w.shape
    assert k%GROUP==0 and scale.shape==(n,k//GROUP)
    assert torch.isfinite(scale).all() and (scale>0).all()
    work=w[:,f['perm']].clone();work[:,f['dead'][f['perm']]]=0
    qout=torch.empty_like(work,dtype=torch.int8);u=f['upper'];threads=torch.get_num_threads()
    group_ids=(f['perm']//GROUP).tolist()
    for start in range(0,k,block):
        end=min(start+block,k);part=work[:,start:end].clone();errors=torch.empty_like(part)
        torch.set_num_threads(1)
        try:
            for j in range(end-start):
                sc=scale[:,group_ids[start+j]]
                codes=(part[:,j]/sc).round().clamp(-7,7)
                qout[:,start+j]=codes.to(torch.int8)
                error=(part[:,j]-codes*sc)/u[start+j,start+j]
                part[:,j:]-=error[:,None]*u[start+j,start+j:end][None,:]
                errors[:,j]=error
        finally:torch.set_num_threads(threads)
        work[:,end:]-=errors@u[start:end,end:]
    return qout[:,f['invperm']]

def output_error(w,codes,scale,x):
    delta=(codes.float()*expand(scale)).half().float()-w
    assert torch.isfinite(delta).all()
    total=torch.zeros(len(w),dtype=torch.float64);x=x.reshape(-1,w.shape[1])
    for start in range(0,len(x),512):
        total+=F.linear(x[start:start+512].float(),delta).double().square().sum(0)
    assert torch.isfinite(total).all()
    return total

def select_output(w,x,f,block=128):
    n,k=w.shape;clipped,local=select_scale(w.reshape(-1,GROUP))
    clipped=clipped.reshape(n,k//GROUP);base=torch.from_numpy(local['absmax_scale']).reshape_as(clipped)
    scales=[base,(base+clipped)*.5,clipped]
    best=torch.full((n,),float('inf'),dtype=torch.float64);choice=torch.zeros(n,dtype=torch.int64)
    best_codes=torch.empty_like(w,dtype=torch.int8);best_scale=base.clone();scores=[]
    for i,scale in enumerate(scales):
        codes=quantize(w,f,scale,block);error=output_error(w,codes,scale,x);scores.append(error)
        better=error<best;best[better]=error[better];choice[better]=i
        best_codes[better]=codes[better];best_scale[better]=scale[better]
    scores=torch.stack(scores,1);assert torch.equal(best,scores.gather(1,choice[:,None]).squeeze(1))
    return best_codes,best_scale,dict(choice=choice.numpy(),scale=best_scale.numpy(),
        candidate_scales=torch.stack(scales,1).numpy(),candidate_output_sse=scores.numpy(),
        selected_output_sse=best.numpy(),absmax_scale=base.numpy(),
        clip_ratio=(clipped/base).numpy(),weight_clip_choice=local['choice'].reshape(n,k//GROUP))

def export(w,x,f,root,prefix):
    started=time.monotonic();n,k=w.shape;root.mkdir(parents=True,exist_ok=True)
    dequant=torch.empty((n,k),dtype=torch.float16);row_stats=[]
    cp=root/(prefix+'_group128_codes_hmx.bin');sp=root/(prefix+'_group128_scales_f32.bin')
    with cp.open('xb') as out,sp.open('xb') as scale_file:
        for first in range(0,n,1024):
            original=w[first:first+1024].float();codes,scale,rs=select_output(original,x,f);row_stats.append(rs)
            packed=pack_codes(codes);q=(codes.float()*expand(scale)).half();assert torch.isfinite(q).all()
            out.write(packed.tobytes());scale_file.write(scale.numpy().astype('<f4').tobytes())
            dequant[first:first+len(codes)]=q
    # Separate independent NumPy unpack/scale expansion on every retained projection.
    decoded=read_weight(root,prefix,n,k)
    assert np.array_equal(decoded,dequant.numpy())
    record=OUTPUT/'clip_stats'/root.relative_to(OUTPUT)/(prefix+'.npz');record.parent.mkdir(parents=True,exist_ok=True)
    rows={key:np.concatenate([r[key] for r in row_stats]) for key in row_stats[0]};np.savez_compressed(record,**rows)
    return dequant,dict(rows=n,width=k,groupsize=GROUP,format=FORMAT,act_order=True,
        factor=f['stats'],clip_stats_path=str(record),clip_stats_sha256=sha(record),
        codes_sha256=sha(cp),scales_sha256=sha(sp),packed_roundtrip=True,
        independent_numpy_export_equal=True,selected_output_sse=float(rows['selected_output_sse'].sum()),
        output_sse_by_candidate=rows['candidate_output_sse'].sum(0).tolist(),
        choice_histogram=np.bincount(rows['choice'],minlength=3).tolist(),elapsed_s=time.monotonic()-started)

def read_weight(root,name,n,k):
    raw=np.fromfile(root/(name+'_group128_codes_hmx.bin'),dtype=np.uint8)
    assert raw.size==n*k//2
    # Independent inverse of the existing HMX carrier layout; signed nibbles.
    nib=np.empty(raw.size*2,dtype=np.uint8);nib[0::2]=raw&15;nib[1::2]=raw>>4
    codes=nib.reshape(n//32,k//32,8,32,4).transpose(0,1,2,4,3).reshape(n//32,k//32,32,32).transpose(0,3,1,2).reshape(n,k).astype(np.int16)
    codes[codes>=8]-=16;assert codes.min()>=-7 and codes.max()<=7
    scales=np.fromfile(root/(name+'_group128_scales_f32.bin'),dtype='<f4').reshape(n,k//GROUP)
    assert np.isfinite(scales).all() and (scales>0).all()
    value=(codes.astype(np.float32).reshape(n,k//GROUP,GROUP)*scales[:,:,None]).reshape(n,k).astype(np.float16)
    assert np.isfinite(value).all();return value

def check():
    torch.manual_seed(231);torch.set_num_threads(16)
    x=torch.randn(1100,256);x[:,1]=.8*x[:,0]+.2*x[:,1];x[:,130]*=7;x[:,3]=0
    w=torch.randn(32,256);w[:,:128]*=3;w[0]=0;w[1,:128]=0
    f=factor(x);codes,scale,rows=select_output(w,x,f,block=32)
    assert (f['perm'][:128]>=128).any() and (f['perm'][128:]<128).any()
    # Independent per-group clipping choice, FP64 NumPy exhaustive scan.
    clip_choices=[]
    for row in w.numpy().reshape(-1,GROUP).astype(np.float64):
        base=np.abs(row).max()/7 if np.any(row) else 1.
        losses=[np.abs(np.clip(np.rint(row/(base*r)),-7,7)*base*r-row).__pow__(2.4).sum() for r in RATIOS]
        clip_choices.append(np.argmin(losses))
    assert np.array_equal(rows['weight_clip_choice'].reshape(-1),clip_choices)
    # Dense inverse/Schur elimination: no Cholesky and no blocked error propagation.
    h=(x.T@x)*(2/len(x));h[f['dead'],f['dead']]=1;p=f['perm'];h=h[p][:,p].double()
    h.diagonal().add_(f['stats']['damping']);all_codes=[];errors=[]
    for trial in range(3):
        sc=torch.from_numpy(rows['candidate_scales'][:,trial].copy())
        actual=quantize(w,f,sc,block=32);inv=torch.linalg.inv(h)
        work=w[:,p].double().clone();work[:,f['dead'][p]]=0;expected=torch.empty_like(work)
        for j in range(256):
            colscale=sc[:,int(p[j])//GROUP].double()
            q=(work[:,j]/colscale).round().clamp(-7,7);expected[:,j]=q
            error=work[:,j]-q*colscale
            work[:,j:]-=error[:,None]*(inv[0]/inv[0,0])[None,:]
            inv=inv[1:,1:]-inv[1:,0,None]*inv[None,0,1:]/inv[0,0]
        expected=expected[:,f['invperm']].to(torch.int8)
        assert torch.equal(actual,expected),(trial,int((actual!=expected).sum()))
        q=actual.numpy().astype(np.float32).reshape(32,2,GROUP)*sc.numpy()[:,:,None]
        delta=q.reshape(32,256).astype(np.float16).astype(np.float64)-w.numpy().astype(np.float64)
        errors.append(np.square(x.numpy().astype(np.float64)@delta.T).sum(0));all_codes.append(actual.numpy())
    errors=np.stack(errors,1);np.testing.assert_allclose(rows['candidate_output_sse'],errors,rtol=2e-6,atol=1e-9)
    choice=errors.argmin(1);assert np.array_equal(choice,rows['choice'])
    for i,j in enumerate(choice):assert np.array_equal(codes[i].numpy(),all_codes[j][i])
    assert rows['choice'][0]==0 and not codes[0].any();pack_codes(codes)
    # One group per row must exactly reproduce the established per-channel quantizer.
    from gptq_exp0221 import quantize as channel
    w1=w[:,:128];f1=factor(x[:,:128]);base=w1.abs().amax(1)/7;base[base==0]=1
    ref,_=channel(w1,f1,block=32,explicit_scale=base)
    assert torch.equal(ref,quantize(w1,f1,base[:,None],block=32))
    return dict(pass_all=True,seed=231,independent_dense_codes_all_three_exact=True,
        independent_numpy_group_clipping=True,independent_numpy_full_output_sse_choices=True,
        original_column_permutation_multiple_groups=True,zero_rows_and_groups=True,
        dead_columns=True,one_group_per_channel_equivalence=True,packed_roundtrip=True,format=FORMAT)

if __name__=='__main__':
    torch.set_grad_enabled(False);preflight();write('group_oracle.json',check());print('GROUP128_ORACLES_PASS')
