"""Independent native-layout and full-model implementation audits, no quality scoring."""
from common_exp0285 import *
from device_exp0285 import records
from reference_w4u8_hmx import unpack_u8_hmx_activation
import numpy as np
def partial(package,tag):
 p=package_path(package);d=R/tag;assert read(d/'validated.json')['numerical_pass']
 for step in range(2):
  rows=64 if not step else 1;b=np.fromfile(d/f'step{step:02d}_r3.bin','u1');base=983040;cap=393216
  assert len(b)==base+11*cap
  for slot,name,k in [(2,'attention',2048),(5,'post',2048),(6,'gate',6144),(7,'up',6144)]:
   a=unpack_u8_hmx_activation(b[base+slot*cap:base+slot*cap+64*k],k)[:rows]
   ref=np.load(p/f'half_step{step:02d}_{name}.npy');assert np.array_equal(a,ref),(tag,step,name)
  codes=b[2*cap:base].reshape(24,4,64,32).transpose(0,2,1,3).reshape(24,64,128)[:,:rows]
  for sl,name in [(slice(0,16),'q'),(slice(16,24),'k')]:
   assert np.array_equal(codes[sl],np.load(p/f'half_step{step:02d}_{name}.npy').transpose(1,0,2)),(tag,step,name)
  mid=np.load(p/f'half_step{step:02d}_middle.npy');a=unpack_u8_hmx_activation(b[base+8*cap:base+9*cap],6144)
  assert np.array_equal(a[:rows],((mid+32768)&255).astype('u1'))
  if step:assert np.array_equal(a[4:5],((mid+32768)>>8).astype('u1'))
 write(d/'boundary_gate.json',dict(pass_all=True,reference='independent FP16 residual, FP32 add/norm',live_QK_AV_Norm_Gate_Up_SP2_exact=True))
def fullaudit(fp,tag):
 d=R/tag;z=read(d/'validated.json');assert z['pass_all'] and z['audit'];count=0
 if fp==3:
  ref=R/'frontend-reference-a02';teacher=read(ref/'reference.json')
  assert z['selected_codes']==teacher['selected_codes'],(tag,z['selected_codes'],teacher['selected_codes'])
  profiles=[v for v in records(d/'stdout.jsonl') if v.get('record')=='generation_profile']
  for step in range(16):
   a=np.fromfile(d/f'generation_hidden_step{step:02d}_f16.bin','<f2')
   b=np.fromfile(ref/f'step{step:02d}_hidden_f16.bin','<f2')
   assert np.array_equal(a.view('u2'),b.view('u2')),(tag,step,'hidden',int(np.count_nonzero(a!=b)))
   assert np.isfinite(a).all()
   n=unpack_u8_hmx_activation(np.fromfile(d/f'generation_norm_step{step:02d}_u8_native.bin','u1'),2048)[0]
   assert np.array_equal(n,np.fromfile(ref/f'step{step:02d}_norm_u8.bin','u1')),(tag,step,'norm')
   for i in range(28):
    h=profiles[step][f'slice_layer_{i}']['output_hash'];h=int(h,16) if isinstance(h,str) else h
    assert h==teacher['layer_hashes'][step][i],(step,i,hex(h),hex(teacher['layer_hashes'][step][i]))
    count+=1
 else:
  old=R.parent/'exp0284/full-SP2-fixed-audit-a02';v=read(old/'validated.json')
  assert z['selected_codes']==v['selected_codes']
  for f in d.glob('generation_*.bin'):
   if (old/f.name).exists():assert sha(f)==sha(old/f.name),f.name;count+=1
 write(d/'independent_gate.json',dict(pass_all=True,residual_bits=16 if fp==3 else 32,exact_layer_hashes=count if fp==3 else None,exact_control_files=count if fp==2 else None,full28_independent_reference=fp==3,steps=16,quality_claim=False))
 print('FULL_AUDIT_PASS',tag,count,flush=True)
if __name__=='__main__':
 import sys
 if sys.argv[1]=='partial':partial(sys.argv[2],sys.argv[3])
 else:fullaudit(int(sys.argv[2]),sys.argv[3])
