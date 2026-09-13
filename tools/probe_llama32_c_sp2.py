#!/usr/bin/env python3
"""Native W4 radix257 endpoint and all-codebook integer transport proof."""
import json,shutil,subprocess
from pathlib import Path
import numpy as np
from prototype_llama32_sp2 import H,pack_w,oracle,preflight
from llama32_sp2_contract import integer_levels,encode,contract
from run_llama32_layer import ROOT,adb,windows
from llama_reference import sha256
OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0013/radix257-probe-a01')
REMOTE='/data/local/tmp/llama32-htp/l32-0013/radix257-probe-a01'
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
    preflight();OUT.mkdir(exist_ok=False)
    seal=json.loads((ROOT/'build/llama-build-seal.json').read_text());head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();assert seal['source_head']==head
    for n,h in seal['files'].items():assert sha256(Path(n))==h
    assert adb('shell','test ! -e '+REMOTE,check=False).returncode==0;adb('shell','mkdir -p '+REMOTE)
    for name,build in [('llama_sp2_cli','android_ReleaseG_aarch64'),('libqwen3_probe.so','android_ReleaseG_aarch64'),('libqwen3_probe_skel.so','hexagon_ReleaseG_toolv19_v79')]:
        p=ROOT/build/'ship'/name;shutil.copy2(p,OUT/name);adb('push',windows(p),REMOTE+'/'+name);assert adb('shell','sha256sum '+REMOTE+'/'+name).stdout.split()[0]==sha256(p)
    adb('shell','chmod 755 '+REMOTE+'/llama_sp2_cli');save(OUT/'build-seal.json',seal)
    rng=np.random.default_rng(130013);levels=integer_levels();summary=[]
    for mode,rows in [(5,4),(6,64)]:
        k=8192;n=64;v=np.resize(levels,(rows,k));v[0]=32768;v[1]=-32768;v[2]=0
        w=rng.integers(-7,8,size=(n,k),dtype=np.int8);w[0]=7;w[1]=-7
        coded=encode(v);weights=pack_w(w);sums=w.astype('i4').sum(1).astype('<i4').tobytes();xo=2048;wo=(xo+coded.nbytes+2047)&~2047;so=wo+len(weights);oo=(so+len(sums)+2047)&~2047;size=oo+rows*n*4
        blob=bytearray(size);blob[:128]=H.pack(0x3250534c,1,size,mode,rows,k,n,xo,wo,so,oo,0xffffffff,0,0,0,0,*([0]*8));blob[xo:xo+coded.nbytes]=coded.tobytes();blob[wo:wo+len(weights)]=weights;blob[so:so+len(sums)]=sums
        src=OUT/f'input-mode{mode}.bin';src.write_bytes(blob);reference=oracle(v,w);assert np.abs(reference).max()<2**31
        reference.astype('<i4').tofile(OUT/f'reference-mode{mode}.bin')
        adb('push',windows(src),REMOTE+'/input.bin');assert adb('shell','sha256sum '+REMOTE+'/input.bin').stdout.split()[0]==sha256(src)
        command=f'cd {REMOTE} && LD_LIBRARY_PATH={REMOTE} DSP_LIBRARY_PATH={REMOTE} ADSP_LIBRARY_PATH={REMOTE} ./llama_sp2_cli input.bin output.bin'
        result=adb('shell',command,check=False);(OUT/f'mode{mode}.stdout.txt').write_text(result.stdout);(OUT/f'mode{mode}.stderr.txt').write_text(result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
        dst=OUT/f'output-mode{mode}.bin';adb('pull',REMOTE+'/output.bin',windows(dst));raw=dst.read_bytes();h=H.unpack_from(raw);actual=np.frombuffer(raw,dtype='<i4',offset=oo,count=rows*n).reshape(rows,n)
        mismatches=int(np.count_nonzero(actual!=reference));assert h[11]==0 and h[12]==8388608 and h[13]<=8388608 and mismatches==0
        record=dict(mode=mode,rows=rows,k=k,n=n,elements=rows*n,mismatches=mismatches,max_abs_integer=int(np.abs(reference).max()),all187_levels_exercised=bool(set(levels).issubset(set(v.reshape(-1)))),vtcm_bytes=h[12],peak_bytes=h[13],input_sha256=sha256(src),output_sha256=sha256(dst));summary.append(record);print('RADIX257_NATIVE_PASS',json.dumps(record),flush=True)
    save(OUT/'result.json',dict(pass_gate=True,source_head=head,contract=contract(),cases=summary,processes=2,rpc_calls=4,scope='raw signed32 output before saturation; includes +/-32768 endpoints and adversarial all+7/all-7 weight rows; production retains stricter signed24 partial-dot guard'))
if __name__=='__main__':main()
