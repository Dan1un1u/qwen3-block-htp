#!/usr/bin/env python3
"""Independent dense arithmetic, native Q/K, cache and full-layer audit."""
from pathlib import Path
import json,struct,math,hashlib,subprocess
import numpy as np
from prepare_exp0161_segmented_cache import pack_k_segment
from prepare_exp0042_attention import CONFIG
from exp0240_export_lpbq32 import pack,unpack
from device_exp0247 import S,R,MODELS,preflight
CAP=393216
def sha(b):return hashlib.sha256(b).hexdigest()
def read_capture(tag,step):
    b=(R/tag/f'step{step:02d}_r3.bin').read_bytes();nr=64 if step==0 else 1;rows=24*nr
    assert len(b)==2*CAP+196608
    raw=np.frombuffer(b[:CAP],'<f2')[:rows*128].reshape(rows,128).copy()
    out=np.frombuffer(b[CAP:2*CAP],'<f2')[:rows*128].reshape(rows,128).copy()
    codes=np.frombuffer(b[2*CAP:],np.uint8).reshape(24,4,64,32).transpose(0,2,1,3).reshape(24,64,128)[:,:nr]
    return raw,out,codes
def ulp(x):
    # np.spacing(-16.0h) reports only the smaller, toward-zero neighbor.
    # A signed binade edge has two adjacent spacings; one ULP is the larger.
    y=x.astype('<f2');hi=np.nextafter(y,np.float16(np.inf));lo=np.nextafter(y,np.float16(-np.inf))
    return np.maximum(np.abs(hi.astype(float)-y),np.abs(y.astype(float)-lo))
def extract(text,name):
    start=text.index('static int '+name+'(');a=text.index('{',start);depth=1;i=a+1
    while depth:
        depth+=(text[i]=='{')-(text[i]=='}');i+=1
    return text[start:i]
def main():
    preflight();sign=np.asarray([[1-2*((i&j).bit_count()%2) for j in range(128)] for i in range(128)],float)
    assert np.array_equal(sign@sign.T,np.eye(128)*128)
    normalization=float(np.float16(1/math.sqrt(128)));h=sign*normalization
    package=json.loads((MODELS/'r3/manifest.json').read_text());params=package['qparams'];records=[];whole=[]
    for step in range(9):
        raw,out,codes=read_capture('audit_r3_01',step);scalar_raw,scalar,scalar_codes=read_capture('audit_scalar_01',step);identity_raw,identity,_=read_capture('audit_identity_01',step)
        assert np.array_equal(raw,scalar_raw) and np.array_equal(raw,identity_raw)
        assert np.array_equal(raw,identity),'HMX identity layout mismatch'
        expected=raw.astype(float)@h;err=np.abs(out.astype(float)-expected)
        bound=ulp(expected)+2**-14
        assert np.isfinite(out).all() and np.all(err<=bound),(step,int((err>bound).sum()))
        assert np.all(np.abs(scalar.astype(float)-expected)<=ulp(expected)+2**-14)
        logical=out.reshape(codes.shape).astype(np.float32);qerr=0
        for head in range(24):
            p=params['q_rope' if head<16 else 'k_rope']
            q=np.clip(np.floor(logical[head]*np.float32(1/p['scale'])+np.float32(p['zero_point'])+np.float32(.5)),0,255)
            qerr=max(qerr,int(np.abs(q.astype(int)-codes[head].astype(int)).max()))
        assert qerr<=1
        a=np.fromfile(R/'audit_r3_01'/f'step{step:02d}_output.bin',np.uint8).astype(float)
        b=np.fromfile(R/'audit_scalar_01'/f'step{step:02d}_output.bin',np.uint8).astype(float)
        # Cosine on dequantized output, not artificially offset unsigned codes.
        qp=params['block_output'];af=(a-qp['zero_point'])*qp['scale'];bf=(b-qp['zero_point'])*qp['scale']
        cosine=float(np.dot(af,bf)/(np.linalg.norm(af)*np.linalg.norm(bf)));difference=float(np.abs(a-b).max())
        assert difference<=2 and cosine>=.999,(step,difference,cosine)
        records.append(dict(step=step,live_rows=len(raw),dense_max_abs=float(err.max()),dense_rmse=float(np.sqrt(np.mean(err**2))),maximum_error_in_local_ulp=float((err/np.maximum(ulp(expected),2**-24)).max()),u8_max_code_error=qerr,identity_exact=True,raw_carriers_identical=True))
        whole.append(dict(step=step,max_lsb=difference,dequantized_cosine=cosine))
    configs=list(CONFIG.iter_unpack((MODELS/'r3/attention_config_all_groups.bin').read_bytes()));cache=(R/'audit_r3_01/prefill_k_cache.bin').read_bytes()
    _,_,codes=read_capture('audit_r3_01',0);expected_cache=b''.join(b''.join(pack_k_segment(codes[16+g,i:i+32],configs[g]) for i in [0,32])+bytes(4096) for g in range(8))
    assert len(cache)==len(expected_cache)==102400 and cache==expected_cache,'independent native K cache pack mismatch'
    packed_proof={}
    for name,n,k in [('q',2048,2048),('k',1024,2048),('v',1024,2048),('o',2048,2048),('gate',6144,2048),('up',6144,2048),('down',2048,6144)]:
        p=MODELS/'r3'/f'{name}_weight_w4_hmx.bin';b=np.fromfile(p,np.uint8).reshape(n//32,k//32,512);q=unpack(b,n,k)
        assert np.array_equal(pack(q),b) and q.min()>=-7 and q.max()<=7
        packed_proof[name]=dict(shape=[n,k],sha256=sha(p.read_bytes()),roundtrip_exact=True)
    parent=subprocess.check_output(['git','show','662386e6acf53f63bbadb3d1e283fc2c1e893555:src/dsp/block_imp.c'],cwd=S,text=True);current=(S/'src/dsp/block_imp.c').read_text();functions={}
    for name in ['qbh_run_projection','qbh_run_w4u8_qkv_ring']:
        assert extract(parent,name)==extract(current,name);functions[name]=sha(extract(current,name).encode())
    result=dict(pass_all=True,rotation=records,whole_layer_vs_scalar=whole,prefill_K_cache_exact=True,EOS_key_in_same_rotated_basis=True,identity_dense_exact=True,
        packed_W4_projections=packed_proof,unchanged_native_projection_functions=functions,normalization_half=normalization,
        checker_recovery='Preserved numerical_first_attempt.json. Corrected one-sided np.spacing at negative FP16 binade edges using both adjacent representable values; frozen one-ULP gate unchanged.',
        limitations='Layer0 device arithmetic and speed gate, not full-model DSP PPL; EOS is computed through the device A8 path, not an imported software warm-prefix tensor.')
    with (R/'numerical_audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('NUMERICAL_PASS',whole,flush=True)
if __name__=='__main__':main()
