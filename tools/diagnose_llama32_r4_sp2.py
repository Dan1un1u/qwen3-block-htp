#!/usr/bin/env python3
"""Frozen L32-0015 native captures: local rounding/threshold/quantization diagnosis.

No fitting, hardware execution, gate relaxation or model-quality claims.
"""
import argparse, json, csv, subprocess
from pathlib import Path
import numpy as np
from llama_reference import sha256
from llama_u8_reference import load_qparams_bin, unpack_w4_codes, _residual_fixed_parameters
from prototype_llama32_sp2 import oracle, preflight
from probe_llama32_rotations import r4, encode, had, save

ROOT = Path(__file__).resolve().parents[1]
M = Path('/mnt/d/llm_exp/models/llama32-htp')
R = Path('/mnt/d/llm_exp/results/llama32-htp/l32-0015')
RUNS = {0:'layer0-r4-a05', 7:'layer7-r4-diag-a01', 15:'layer15-r4-diag-a01'}

def metrics(a,b):
    a=np.asarray(a,dtype='f8').ravel();b=np.asarray(b,dtype='f8').ravel()
    ea=float(a@a);eb=float(b@b);e=float((a-b)@(a-b))
    return dict(cosine=float(a@b/np.sqrt(ea*eb)) if ea and eb else None,
        exact=bool(np.array_equal(a,b)),actual_energy=ea,reference_energy=eb,
        error_energy=e,rmse=float(np.sqrt(e/len(a))),max_abs=float(np.max(np.abs(a-b))),
        nrmse=float(np.sqrt(e/eb)) if eb else None)

def summary_rows(rows):
    cs=[v['cosine'] for v in rows if v['cosine'] is not None]
    return dict(rows=len(rows),exact=sum(v['exact'] for v in rows),
        undefined_cosine=sum(v['cosine'] is None for v in rows),
        defined_cosine_below_0999=sum(c<.999 for c in cs),
        min_defined_cosine=min(cs) if cs else None,
        max_abs=max(v['max_abs'] for v in rows),
        changed_elements=sum(v['changed_elements'] for v in rows))

def analyze(layer,step):
    package=M/f'l32-0015/layers-a01/layer{layer}'
    pm=json.loads((package/'manifest.json').read_text())
    for n,h in pm['files'].items():assert sha256(package/n)==h['sha256'],n
    phase='prefill' if step==0 else 'decode';rows=64 if step==0 else 1
    run=R/RUNS[layer]
    ref=M/f'l32-0003/layers-a01/layer{layer}-{phase}'
    rm=json.loads((ref/'manifest.json').read_text())
    for n in ['reference_residual.npy','reference_gate.npy','reference_up.npy']:
        assert sha256(ref/n)==rm['files'][n]['sha256']
    raw=np.fromfile(run/f'actual_replay_r4_{step:02d}.bin',dtype='<f2')
    count=rows*8192
    z=raw[:count].reshape(16,rows,512).transpose(1,0,2).reshape(rows,8192).copy()
    actual_stage1=raw[524288:524288+count].reshape(16,rows,512).transpose(1,0,2).reshape(rows,8192).copy()
    y=raw[1048576:1048576+count].reshape(rows,8192).copy()
    s,ideal=r4(z)
    assert np.array_equal(ideal,np.load(package/f'ideal_r4_{step:02d}.npy'))
    q=load_qparams_bin(package/'layer0/qparams_u8.bin');alpha=q['middle']['scale']
    lut=np.fromfile(package/'layer0/silu_up_lut_u16.bin',dtype='<u2')[65536:]
    vi=lut[ideal.view('u2')].astype('i4')-32768
    va=lut[y.view('u2')].astype('i4')-32768
    levels=np.array(json.loads((R.parent/'l32-0008/codebook.json').read_text())['levels'],dtype='i4')
    for f,v in [(ideal,vi),(y,va)]:
        assert np.array_equal(encode(f.astype('f8'),levels,alpha).astype('i4')-32768,v)
    w=unpack_w4_codes(package/'layer0','down',2048,8192)
    ws=np.fromfile(package/'layer0/down_weight_w4_scale_f32.bin',dtype='<f4').astype('f8')
    mult=np.floor(alpha*ws/q['down']['scale']*2**31+.5).astype('i8')
    ai=oracle(vi,w);aa=oracle(va,w)
    assert np.array_equal(aa-ai,oracle(va-vi,w))
    def requant(a):
        return np.clip(((a*mult+2**30)>>31)+q['down']['zero_point'],0,255).astype('u1')
    di=requant(ai);da=requant(aa)
    left=np.load(ref/'reference_residual.npy').astype('i8')
    fb,lc,rc=_residual_fixed_parameters(q['post_attention_residual']['scale'],q['down']['scale'],q['block_output']['scale'])
    def residual(d):
        a=(left-q['post_attention_residual']['zero_point'])*lc+(d.astype('i8')-q['down']['zero_point'])*rc
        o=np.clip(((a+(1<<(fb-1)))>>fb)+q['block_output']['zero_point'],0,255).astype('u1')
        return a,o
    ri,oi=residual(di);ra,oa=residual(da)
    actual=np.fromfile(run/f'actual_replay_output_{step:02d}_u8.bin',dtype='u1').reshape(-1,2048)[:rows]
    expect=np.fromfile(package/('reference_w4u8_integer_attention_block_output_u8.bin' if step==0 else 'replay_decode_reference_00_u8.bin'),dtype='u1').reshape(-1,2048)[:rows]
    assert np.array_equal(oa,actual) and np.array_equal(oi,expect)
    # Matched frozen W4 operator without the two final U8 stores; not an FP16 teacher.
    left_real=(left-q['post_attention_residual']['zero_point'])*q['post_attention_residual']['scale']
    yf=ideal.astype('f8')@w.astype('f8').T*ws
    hf=y.astype('f8')@w.astype('f8').T*ws
    sp2i=ai*alpha*ws;sp2a=aa*alpha*ws
    continuous=left_real+sp2i
    unquantized_down=left_real+yf
    ui=oi.astype('f8')-q['block_output']['zero_point']
    ua=oa.astype('f8')-q['block_output']['zero_point']
    per=[]
    for t in range(rows):
        m=metrics(ua[t],ui[t]);m.update(layer=layer,phase=phase,token=t,
            changed_elements=int(np.count_nonzero(oa[t]!=oi[t])),
            reference_nonzero=int(np.count_nonzero(ui[t])),
            sp2_changed=int(np.count_nonzero(va[t]!=vi[t])),
            down_changed=int(np.count_nonzero(da[t]!=di[t])),
            down_preround_error_max_lsb=float(np.max(np.abs((aa[t]-ai[t])*mult/2**31))),
            sp2_continuous_residual_nrmse=metrics(left_real[t]+sp2a[t],continuous[t])['nrmse'],
            existing_u8_tail_nrmse=metrics(ui[t]*q['block_output']['scale'],continuous[t])['nrmse'])
        per.append(m)
    crossings=[]
    for t,k in zip(*np.where(va!=vi)):
        aidx=int(np.searchsorted(levels,va[t,k]));iidx=int(np.searchsorted(levels,vi[t,k]))
        low=min(aidx,iidx);high=max(aidx,iidx)
        boundaries=((levels[low:high].astype('f8')+levels[low+1:high+1])*alpha/2).tolist()
        crossings.append(dict(token=int(t),channel=int(k),ideal_fp16=float(ideal[t,k]),
            actual_fp16=float(y[t,k]),ideal_level=int(vi[t,k]),actual_level=int(va[t,k]),
            delta_level=int(va[t,k]-vi[t,k]),boundaries=boundaries,
            adjacent=abs(aidx-iidx)==1,
            straddles=all(min(float(y[t,k]),float(ideal[t,k]))<=b<=max(float(y[t,k]),float(ideal[t,k])) for b in boundaries)))
    assert all(v['straddles'] for v in crossings)
    changed_outputs=[]
    for t,n in zip(*np.where(oa!=oi)):
        contributions=[]
        for k in np.flatnonzero(va[t]!=vi[t]):
            delta=int((va[t,k]-vi[t,k])*int(w[n,k]))
            contributions.append(dict(input_channel=int(k),delta_dot=delta,
                delta_down_preround_lsb=float(delta*mult[n]/2**31)))
        contributions.sort(key=lambda c:abs(c['delta_dot']),reverse=True)
        assert sum(c['delta_dot'] for c in contributions)==int(aa[t,n]-ai[t,n])
        changed_outputs.append(dict(token=int(t),channel=int(n),
            ideal_output=int(oi[t,n]),actual_output=int(oa[t,n]),
            ideal_down=int(di[t,n]),actual_down=int(da[t,n]),
            ideal_down_preround_centered=float(ai[t,n]*mult[n]/2**31),
            actual_down_preround_centered=float(aa[t,n]*mult[n]/2**31),
            ideal_residual_preround_centered=float(ri[t,n]/2**fb),
            actual_residual_preround_centered=float(ra[t,n]/2**fb),
            ideal_dot=int(ai[t,n]),actual_dot=int(aa[t,n]),contributions=contributions))
    return dict(layer=layer,phase=phase,rows=rows,native_run=RUNS[layer],
        qparams=q,sp2_alpha=alpha,actual_conditional_tail_exact=True,
        stage1=metrics(actual_stage1,s),r4_output=metrics(y,ideal),
        down_float_before_sp2=metrics(hf,yf),
        down_float_after_sp2=metrics(sp2a,sp2i),
        continuous_residual=metrics(left_real+sp2a,continuous),
        output_u8=metrics(ua,ui),
        existing_u8_tail_vs_same_sp2_float=metrics(ui*q['block_output']['scale'],continuous),
        matched_sp2_vs_no_sp2_float_residual=metrics(continuous,unquantized_down),
        output_scale_over_continuous_rms=float(q['block_output']['scale']/max(np.sqrt(np.mean(continuous**2)),1e-30)),
        residual_fixed=dict(fraction_bits=fb,left=lc,right=rc),
        summary=summary_rows(per),per_token=per,sp2_crossings=crossings,
        changed_outputs=changed_outputs,down_changed_elements=int(np.count_nonzero(di!=da)),
        down_saturation_actual=int(np.count_nonzero((da==0)|(da==255))),
        output_saturation_actual=int(np.count_nonzero((oa==0)|(oa==255))),
        files={str(p):sha256(p) for p in [run/f'actual_replay_r4_{step:02d}.bin',
            run/f'actual_replay_output_{step:02d}_u8.bin',package/'manifest.json',ref/'manifest.json']})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    preflight();out=Path(args.output);out.mkdir(exist_ok=False)
    results=[];tokens=[]
    for layer in RUNS:
        for step in range(2):
            r=analyze(layer,step);save(out/f'layer{layer}-{r["phase"]}.json',r)
            results.append({k:v for k,v in r.items() if k not in ['sp2_crossings','changed_outputs','per_token','qparams','files']})
            tokens.extend(r['per_token']);print(json.dumps(results[-1]),flush=True)
    with (out/'tokens.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(tokens[0]));writer.writeheader();writer.writerows(tokens)
    repeat={}
    for step in range(2):
        for name in [f'actual_replay_output_{step:02d}_u8.bin',f'actual_replay_r4_{step:02d}.bin']:
            a=(R/'layer0-r4-a05'/name).read_bytes()
            b=(R/'layer0-r4-diag-repeat-a01'/name).read_bytes()
            if 'r4_' in name:
                length=(64 if step==0 else 1)*8192*2
                equal=all(a[offset:offset+length]==b[offset:offset+length]
                          for offset in [0,1048576,2097152])
            else:
                equal=a==b
            repeat[name]=equal
    assert all(repeat.values())
    save(out/'summary.json',dict(source_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        native_head=json.loads((R/'layer0-r4-a05/protocol.json').read_text())['source_head'],
        scope='195 frozen per-token single-layer rows, no fullmodel or quality validation',
        frozen_weights_scales_and_gates=True,layer0_repeat=repeat,all_rows=summary_rows(tokens),groups=results))
if __name__=='__main__':main()
