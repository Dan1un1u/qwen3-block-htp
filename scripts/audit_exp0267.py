#!/usr/bin/env python3
"""Independent integer SP2 oracle on actual Qwen3 device carriers."""
import sys,json,struct,subprocess
import numpy as np
from export_exp0267 import S,R,O,P,sha,write,preflight
from device_exp0267 import run,read,adb,win,records
from llama_u8_reference import load_qparams_bin,unpack_w4_codes
from reference_w4u8_hmx import unpack_u8_hmx_activation,exact_residual_add_u8
BASE=983040;CAP=393216
def slots(p,step):
 b=(p/f'step{step:02d}_r3.bin').read_bytes()
 return [np.frombuffer(b,'u1',count=CAP,offset=BASE+j*CAP) for j in range(11)]
def oracle(tag,layer,package=None):
 p=R/tag;d=package or O/f'layer{layer}-sp2';q=load_qparams_bin(d/'qparams_u8.bin')
 w=unpack_w4_codes(d,'down',2048,6144).astype('f8');ws=np.fromfile(d/'down_weight_w4_scale_f32.bin','<f4').astype('f8');alpha=q['middle']['scale'];mult=np.floor(alpha*ws/q['down']['scale']*2**31+.5).astype('i8');lut=np.fromfile(d/'silu_up_lut_u16.bin','<u2').reshape(256,256).astype('i4')-32768
 audit=[]
 for step in range(9):
  rows=64 if step==0 else 1;s=slots(p,step);un=lambda j,k:unpack_u8_hmx_activation(s[j][:64*k],k)[:rows]
  g=un(6,6144);u=un(7,6144);v=lut[g,u];acc=v.astype('f8')@w.T
  assert np.array_equal(acc,np.round(acc)) and np.max(np.abs(acc))<2**31
  # Integer products sum below 2**53: BLAS double sums are exact here.
  exact=v.astype('i8')@w[:8].astype('i8').T;assert np.array_equal(acc[:,:8],exact)
  expect=np.clip(((acc.astype('i8')*mult+2**30)>>31)+q['down']['zero_point'],0,255).astype('u1')
  actual=un(9,2048);assert np.array_equal(expect,actual),(tag,step,'Down',int(np.max(np.abs(expect.astype('i4')-actual))))
  residual=s[4][:rows*2048].reshape(rows,2048);out=exact_residual_add_u8(residual,q['post_attention_residual'],expect,q['down'],q['block_output'])
  actual_out=np.fromfile(p/f'step{step:02d}_output.bin','u1').reshape(rows,2048);assert np.array_equal(out,actual_out),(tag,step,'residual')
  middle=un(8,6144);assert np.array_equal(middle,((v+32768)&255).astype('u1')),(tag,step,'low')
  if step:
   native=unpack_u8_hmx_activation(s[8][:64*6144],6144);assert np.array_equal(native[4:5],((v+32768)>>8).astype('u1')),(tag,step,'high')
  audit.append(dict(step=step,rows=rows,max_abs_acc=int(np.max(np.abs(acc))),down_exact=True,residual_exact=True,low_exact=True))
 z=dict(pass_all=True,layer=layer,steps=audit,scope='changed SP2 path on actual hardware Gate/Up/residual; not floating-model quality')
 if (p/'integer_oracle.json').exists():assert read(p/'integer_oracle.json')==z
 else:write(p/'integer_oracle.json',z)
 return audit

def single():
 preflight();assert read(R/'gather_gate.json')['pass_all'];allrows=[]
 for layer in [0,14,27]:
  captures={}
  for mode in [0,4,5,8]:
   tag=f'audit-layer{layer}-m{mode}';z=run(mode,1,tag,layer=layer,dump=True);captures[mode]=z
   if mode:allrows.append(dict(layer=layer,mode=mode,steps=oracle(tag,layer)))
  for mode in [4,5,8]:
   p=R/f'audit-layer{layer}-m{mode}';c=R/f'audit-layer{layer}-m0'
   assert captures[mode]['output_hashes']==captures[4]['output_hashes']
   for step in range(9):
    a=slots(p,step);b=slots(c,step);rows=64 if step==0 else 1
    for j,k in [(2,2048),(3,2048),(5,2048),(6,6144),(7,6144)]:
     assert np.array_equal(unpack_u8_hmx_activation(a[j][:64*k],k)[:rows],unpack_u8_hmx_activation(b[j][:64*k],k)[:rows]),(layer,mode,step,j)
    assert np.array_equal(a[4][:rows*2048],b[4][:rows*2048])
   for f in c.glob('*cache*.bin'):assert sha(f)==sha(p/f.name),(layer,mode,f.name)
 write(R/'single_gate.json',dict(pass_all=True,oracle=allrows,serial_optimized_exact=True,upstream_u8_exact=True,cache_u8_exact=True))

def gather():
 preflight();state=read(R/'runtime-l1.json');root=state['remote'];d=R/'gather';d.mkdir(exist_ok=False);rows=[]
 for layer in range(28):
  lut=np.fromfile(O/f'sp2/layer{layer}/silu_up_lut_u16.bin','<u2');total=395264;b=bytearray(total);struct.pack_into('<16I8Q',b,0,0x3250534c,1,total,4,64,1024,1024,2048,0,0,133120,0,0,0,0,0,*([0]*8));b[2048:133120]=lut.tobytes();inp=d/f'layer{layer}-input.bin';out=d/f'layer{layer}-output.bin';inp.write_bytes(b)
  adb('push',win(inp),root+'/gather-input.bin');r=adb('shell',f'cd {root} && LD_LIBRARY_PATH={root} DSP_LIBRARY_PATH={root} ADSP_LIBRARY_PATH={root} ./llama_sp2_cli gather-input.bin gather-output.bin',check=False);(d/f'layer{layer}.stdout').write_text(r.stdout);(d/f'layer{layer}.stderr').write_text(r.stderr);assert r.returncode==0,(layer,r.stdout,r.stderr)
  adb('pull',root+'/gather-output.bin',win(out));raw=out.read_bytes();got=np.frombuffer(raw,'u1',offset=133120,count=262144).reshape(4,65536);low=(lut&255).astype('u1');high=(lut>>8).astype('u1');assert all(np.array_equal(got[i],v) for i,v in enumerate([low,high,low,high])),layer
  rows.append(dict(layer=layer,pairs=65536,exact=True,input_sha256=sha(inp),output_sha256=sha(out)));print('GATHER_PASS',layer,flush=True)
 write(R/'gather_gate.json',dict(pass_all=True,layers=rows))
if __name__=='__main__':
 if sys.argv[1]=='gather':gather()
 elif sys.argv[1]=='single':single()
 else:oracle(sys.argv[1],int(sys.argv[2]))
