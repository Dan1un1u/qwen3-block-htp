#!/usr/bin/env python3
"""Independent3layer exactness, actual last-layer R4/Down/residual reference."""
from device_exp0265 import *
from audit_exp0261 import CAP,BASE
from audit_exp0247 import ulp
from reference_w4u8_hmx import unpack_u8_hmx_activation,project_w4u8,HmxU8Converter,exact_residual_add_u8
import sys

def equality(tag,ref,r4=True):
 checked=[]
 for step in range(9):
  rows=64 if step==0 else 1
  for suffix in ['output','r3']+(['r4'] if r4 else []):
   name=f'step{step:02d}_{suffix}.bin';a=R/tag/name;b=ref/name
   if suffix=='output':assert a.read_bytes()==b.read_bytes(),(tag,name)
   elif suffix=='r4':assert np.array_equal(np.fromfile(a,'u2').reshape(3,CAP)[:,:rows*6144],np.fromfile(b,'u2').reshape(3,CAP)[:,:rows*6144]),(tag,name)
   else:
    x=np.fromfile(a,'u1');y=np.fromfile(b,'u1');assert np.array_equal(x[:BASE+5*CAP],y[:BASE+5*CAP]),(tag,name,'preMLP')
    for j,k in [(5,2048),(6,6144),(7,6144),(8,6144),(9,2048)]:
     xx=unpack_u8_hmx_activation(x[BASE+j*CAP:BASE+j*CAP+64*k],k)[:rows];yy=unpack_u8_hmx_activation(y[BASE+j*CAP:BASE+j*CAP+64*k],k)[:rows];assert np.array_equal(xx,yy),(tag,name,j)
   checked.append(name)
 for b in ref.glob('*cache.bin'):
  a=R/tag/b.name
  if a.exists():assert a.read_bytes()==b.read_bytes(),(tag,b.name);checked.append(b.name)
 return dict(tag=tag,reference=str(ref),checked=checked,pass_all=True)

def main():
 preflight();torch.set_num_threads(6);old=R.parent/'exp0259';seal=read(old/'EVIDENCE_SHA256.json')['files']
 for p in (old/'slice_stream').glob('*.bin'):assert sha(p)==seal[str(p.relative_to(old))]['sha256']
 checks=[equality('slice_control',old/'slice_stream',False),equality('slice_fused',R/'slice_previous'),equality('slice_repeat',R/'slice_previous')]
 m=read(O/'r4/manifest.json');qp=m['layer_qparams'][2];layer=O/'r4/layer2';h12,h512=matrices();table=np.fromfile(layer/'silu_up_lut_u16.bin','<f2').reshape(256,256)
 converter=S/'build/reference/qbh_hmx_u8_reference.so';assert sha(converter)=='dd492fbdfb85b8a324cce03368f4461663b13f3d2eb0a4a2014cd483851fb5af';conv=HmxU8Converter(converter);cases=[]
 for step in range(9):
  rows=64 if step==0 else 1;p=R/'slice_fused';raw=np.fromfile(p/f'step{step:02d}_r4.bin','<f2').reshape(3,CAP);x=raw[0,:rows*6144].reshape(12,rows,512);y=raw[1,:rows*6144].reshape(12,rows,512);z=raw[2,:rows*6144].reshape(rows,6144);data=(p/f'step{step:02d}_r3.bin').read_bytes();slots=[np.frombuffer(data,'u1',count=CAP,offset=BASE+j*CAP) for j in range(11)]
  unpacked=lambda j,k:unpack_u8_hmx_activation(slots[j][:64*k],k)[:rows]
  g,u=unpacked(6,6144),unpacked(7,6144);assert np.array_equal(x.view('u2'),table[g,u].reshape(rows,12,512).transpose(1,0,2).view('u2'))
  ref1=x.astype(float)@h512*m['half_scale512'];ref2=np.einsum('brc,ba->rac',y.astype(float),h12).reshape(rows,6144)*m['half_scale12'];err1=np.abs(y.astype(float)-ref1);err2=np.abs(z.astype(float)-ref2);assert np.isfinite(z).all() and (err1<=ulp(ref1)+2**-14).all() and (err2<=ulp(ref2)+2**-14).all()
  quant=lambda v:np.clip(np.floor(v.astype('f4')*np.float32(1/qp['middle']['scale'])+np.float32(qp['middle']['zero_point'])+np.float32(.5)),0,255).astype('u1')
  codes=unpacked(8,6144);assert np.array_equal(codes,quant(z));down=project_w4u8(codes,layer,'down',2048,6144,qp['middle'],qp['down'],conv);assert np.array_equal(down,unpacked(9,2048))
  residual=slots[4][:rows*2048].reshape(rows,2048);final=exact_residual_add_u8(residual,qp['post_attention_residual'],down,qp['down'],qp['block_output']);got=np.fromfile(p/f'step{step:02d}_output.bin','u1').reshape(rows,2048);assert np.array_equal(got,final)
  ideal1=ref1.astype('<f2');ideal2=(np.einsum('brc,ba->rac',ideal1.astype(float),h12).reshape(rows,6144)*m['half_scale12']).astype('<f2');idealdown=project_w4u8(quant(ideal2),layer,'down',2048,6144,qp['middle'],qp['down'],conv);idealfinal=exact_residual_add_u8(residual,qp['post_attention_residual'],idealdown,qp['down'],qp['block_output']);md=int(np.abs(got.astype(int)-idealfinal.astype(int)).max());a=(got.astype(float).ravel()-qp['block_output']['zero_point'])*qp['block_output']['scale'];b=(idealfinal.astype(float).ravel()-qp['block_output']['zero_point'])*qp['block_output']['scale'];cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)));assert md<=2 and cos>=.999,(step,md,cos)
  cases.append(dict(step=step,stage1_max_abs=float(err1.max()),stage2_max_abs=float(err2.max()),middle_Down_residual_exact=True,conditional_ideal_R4_max_lsb=md,conditional_ideal_R4_cosine=cos));print('SLICE_COMPONENT_PASS',step,md,cos,flush=True)
 write(R/'slice_gate.json',dict(pass_all=True,exact_checks=checks,actual_layers=3,continuous_steps=9,independent_last_layer=2,component=cases,finite_FP16_device_sweep_per_layer=63488,converter_sha256=sha(converter),known_ideal_R3_failure='unchanged performance diagnostic'))
 print('SLICE_GATE_PASS',flush=True)
if __name__=='__main__':main()
