import sys,os,math
os.environ['QBH_QWEN_MODEL_SIZE']='1.7B'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S/'scripts'))
from run_exp0333_long_rotation import *
CAP=64*6144;H=2048;F=6144;ROOT=PARENTS
import numpy as np
from hvx_r4_qwen import transform,H12
from reference_w4u8_hmx import load_qparams_bin,unpack_w4_codes,unpack_u8_hmx_activation

def ulp(x):
 y=x.astype('f2');hi=np.nextafter(y,np.float16(np.inf));lo=np.nextafter(y,np.float16(-np.inf))
 return np.maximum(np.abs(hi.astype('f8')-y),np.abs(y.astype('f8')-lo))
def audit(tag):
 preflight();d=R/"runs"/tag;proto=read(d/'protocol.json');cfg=proto['configuration'];layers=proto['runtime']['layers'];li=layers-1;pkg=Path(proto['package']['package'])
 records=read(d/'records.json');profiles=[x for x in records if x.get('record')=='long_profile'];steps={x['step']:x for x in records if x.get('record')=='long_step'};checks=[];r3checks=[]
 hs=np.asarray([[1-2*((i&j).bit_count()%2) for j in range(128)] for i in range(128)],'f8')*float(np.float16(1/math.sqrt(128)))
 ix=np.arange(512,dtype='u4');hm=np.ones((512,512),'i1')
 for bit in range(9):hm*=np.where(((ix[:,None]>>bit)&(ix[None,:]>>bit)&1),-1,1).astype('i1')
 params=read(ROOT/'exp0323/frozen-qk-params.json')
 for step,profile in enumerate(profiles):
  path=d/'audit'/f'long_r0_s{step:02d}_r4.bin'
  if not path.exists():continue
  rows=steps[step]['rows'];buf=path.read_bytes();qp=load_qparams_bin(pkg/f'layer{li}/qparams_u8.bin')['middle']
  if cfg['r4']==4:
   x=np.frombuffer(buf,'<f4',count=rows*F).reshape(rows,F);z=np.frombuffer(buf,'<f4',count=rows*F,offset=CAP*4).reshape(rows,F);gold=transform(x)
   assert np.array_equal(z.view('u4'),gold.view('u4')),(tag,step,'R4 butterfly')
   rec=dict(butterfly_FP32_exact=True)
  else:
   x=np.frombuffer(buf,'<f2',count=rows*F).reshape(12,rows,512);y=np.frombuffer(buf,'<f2',count=rows*F,offset=CAP*2).reshape(12,rows,512);z=np.frombuffer(buf,'<f2',count=rows*F,offset=CAP*4).reshape(rows,F)
   r1=x.astype('f8')@hm.T*float(np.float16(1/np.sqrt(512)));r2=np.einsum('brc,ab->rac',y.astype('f8'),H12)*float(np.float16(1/np.sqrt(12)))
   r2=r2.reshape(rows,F);e1=np.abs(y-r1);e2=np.abs(z-r2);assert np.isfinite(z).all() and (e1<=ulp(r1)+2**-14).all() and (e2<=ulp(r2)+2**-14).all(),(tag,step,'R4 dense',e1.max(),e2.max())
   rec=dict(dense_FP16_within_frozen_bounds=True,stage1_max_abs=float(e1.max()),stage2_max_abs=float(e2.max()))
  low=unpack_u8_hmx_activation(np.frombuffer(buf,'u1',count=CAP,offset=F*512),F)[:rows].astype('i4');high=unpack_u8_hmx_activation(np.frombuffer(buf,'u1',count=CAP,offset=F*576),F)[:rows].astype('i4');got=low+256*(high-128)
  v=np.float32(z)*np.float32(1/qp['scale']);want=np.clip(np.trunc(v+np.copysign(np.float32(.5),v)),-32768,32767).astype('i4');assert np.array_equal(got,want),(tag,step,'INT16 codes')
  residual=np.frombuffer(buf,'<f4',count=rows*H,offset=F*640).reshape(rows,H);w=unpack_w4_codes(pkg/f'layer{li}','down',H,F).astype('f8');ws=np.fromfile(pkg/f'layer{li}/down_weight_w4_scale_f32.bin','<f4');acc=got.astype('f8')@w.T
  assert np.abs(acc).max()<2**31 and np.array_equal(acc,np.rint(acc))
  out=residual+acc.astype('f4')*(np.float32(qp['scale'])*ws);h=1469598103934665603
  for byte in out.astype('<f4').tobytes():h=((h^byte)*1099511628211)&0xffffffffffffffff
  assert profile[f'slice_layer_{li}']['output_hash']==f'{h:016x}',(tag,step,'Down residual')
  checks.append(dict(step=step,layer=li,rows=rows,INT16_exact=True,conditional_Down_residual_exact=True,**rec))
  if cfg['r3']:
   bb=(d/'audit'/f'long_r0_s{step:02d}_r3_layers.bin').read_bytes();cap=393216;base=983040;assert len(bb)==layers*base
   for il in range(layers):
    b=bb[il*base:(il+1)*base];raw=np.frombuffer(b[:cap],'<f2')[:24*rows*128].reshape(24,rows,128);y=np.frombuffer(b[cap:2*cap],'<f2')[:24*rows*128].reshape(24,rows,128);codes=np.frombuffer(b[2*cap:],'u1').reshape(24,4,64,32).transpose(0,2,1,3).reshape(24,64,128)[:,:rows]
    expected=raw.astype('f8')@hs;err=np.abs(y-expected);bound=ulp(expected)+2**-14;assert np.isfinite(raw).all() and np.isfinite(y).all() and (err<=bound).all(),(tag,step,il,'R3 bound',err.max())
    qerr=0
    for head in range(24):
     qp3=params[il]['r3']['q' if head<16 else 'k'];q=np.float32(y[head].astype('f4')*np.float32(1/qp3['scale']));q=np.float32(q+np.float32(128));q=np.clip(np.trunc(np.float32(q+np.float32(.5))),0,255);qerr=max(qerr,int(np.abs(q.astype(int)-codes[head].astype(int)).max()))
    assert qerr<=1,(tag,step,il,'R3 quantizer',qerr)
    r3checks.append(dict(step=step,layer=il,rows=rows,R3_within_frozen_FP16_bound=True,maximum_abs=float(err.max()),quantizer_max_code_error=qerr))
 assert len(checks)==4 and len(r3checks)==layers*4
 result=dict(pass_all=True,scope='Actual-input conditional reference; distinct R4 FP16/FP32 contracts retained, not ideal-model bit exact',R4_and_Down_checks=checks,R3_checks=r3checks)
 save(d/'reference-gate.json',result);print('REFERENCE_PASS',tag,len(checks),len(r3checks),flush=True)
if __name__=='__main__':audit(sys.argv[1])
