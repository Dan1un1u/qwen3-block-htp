#!/usr/bin/env python3
"""Independent full R4 factors, native Down, residual and repeat/caches."""
from export_exp0261 import *
from reference_w4u8_hmx import HmxU8Converter,project_w4u8,exact_residual_add_u8,unpack_u8_hmx_activation
from audit_exp0247 import ulp
import sys
CAP=393216;BASE=2*CAP+196608

def capture(tag,step):
 rows=64 if step==0 else 1
 p=R/tag;raw=np.fromfile(p/f'step{step:02d}_r4.bin',dtype='<f2').reshape(3,CAP)
 r3=(p/f'step{step:02d}_r3.bin').read_bytes()
 slots=[np.frombuffer(r3,np.uint8,count=CAP,offset=BASE+j*CAP).copy() for j in range(11)]
 return rows,raw[0,:rows*6144].reshape(12,rows,512),raw[1,:rows*6144].reshape(12,rows,512),raw[2,:rows*6144].reshape(rows,6144),slots

def main():
 preflight();torch.set_num_threads(6);h12,h512=matrices();m=read(O/'r4/manifest.json');qp=m['qparams'];lut=np.fromfile(O/'r4/silu_up_lut_u16.bin','<f2').reshape(256,256)
 converter=S/'build/reference/qbh_hmx_u8_reference.so';conv=HmxU8Converter(converter)
 # Verify the unchanged independent converter against previously sealed provenance.
 parent=R.parent/'exp0240';seal=read(parent/'evidence_sha256.json') if (parent/'evidence_sha256.json').exists() else read(parent/'EVIDENCE_SHA256.json')
 matched=[]
 for p in parent.glob('*integer_audit.json'):
  v=read(p)
  if v.get('converter_sha256')==sha(converter):matched.append(str(p))
 assert matched,'converter provenance'
 result=[];whole=[];component=True;downpass=True
 for step in range(9):
  rows,x,y,z,slots=capture('audit_a1',step)
  g=unpack_u8_hmx_activation(slots[6],6144)[:rows];u=unpack_u8_hmx_activation(slots[7],6144)[:rows]
  expected_x=lut[g,u].reshape(rows,12,512).transpose(1,0,2)
  assert np.array_equal(x.view('u2'),expected_x.view('u2')),'LUT/layout'+str(step)
  ref1=x.astype(float)@h512*m['half_scale512'];err1=np.abs(y.astype(float)-ref1);tol1=ulp(ref1)+2**-14
  ref2=np.einsum('brc,ba->rac',y.astype(float),h12)*m['half_scale12'];ref2=ref2.reshape(rows,6144);err2=np.abs(z.astype(float)-ref2);tol2=ulp(ref2)+2**-14
  codes=unpack_u8_hmx_activation(slots[8],6144)[:rows]
  refcode=np.clip(np.floor(z.astype('f4')*np.float32(1/qp['middle']['scale'])+np.float32(qp['middle']['zero_point'])+np.float32(.5)),0,255).astype('u1');qerr=int(np.abs(codes.astype(int)-refcode.astype(int)).max())
  pass_component=bool(np.isfinite(z).all() and (err1<=tol1).all() and (err2<=tol2).all() and qerr<=1);component&=pass_component
  # Same original native per-channel W4 semantics, independent SDK conversion.
  expected_down=project_w4u8(codes,O/'r4','down',2048,6144,qp['middle'],qp['down'],conv)
  down=unpack_u8_hmx_activation(slots[9][:131072],2048)[:rows];dmax=int(np.abs(down.astype(int)-expected_down.astype(int)).max());downpass&=dmax==0
  residual=unpack_u8_hmx_activation(slots[4][:131072],2048)[:rows]
  final=exact_residual_add_u8(residual,qp['post_attention_residual'],down,qp['down'],qp['block_output'])
  got=np.fromfile(R/'audit_a1'/f'step{step:02d}_output.bin','u1').reshape(rows,2048)
  fmax=int(np.abs(got.astype(int)-final.astype(int)).max());downpass&=fmax==0
  sr,sx,sy,sz,ss=capture('audit_a2',step)
  assert np.array_equal(x.view('u2'),sx.view('u2'))
  ideal1=ref1.astype('<f2');ideal2=(np.einsum('brc,ba->rac',ideal1.astype(float),h12)*m['half_scale12']).reshape(rows,6144).astype('<f2')
  assert np.array_equal(sy,ideal1) and np.array_equal(sz,ideal2),'scalar oracle independence'
  a=got.astype(float).ravel();b=np.fromfile(R/'audit_a2'/f'step{step:02d}_output.bin','u1').astype(float)
  af=(a-qp['block_output']['zero_point'])*qp['block_output']['scale'];bf=(b-qp['block_output']['zero_point'])*qp['block_output']['scale'];cos=float(np.dot(af,bf)/(np.linalg.norm(af)*np.linalg.norm(bf)));md=int(np.abs(a-b).max());whole.append(dict(step=step,max_lsb=md,cosine=cos,gate_pass=bool(md<=2 and cos>=.999)))
  _,_,_,_,cs=capture('audit_a0',step)
  assert all(np.array_equal(slots[j],cs[j]) for j in range(8)),'pre-R4 changed'
  result.append(dict(step=step,stage1_max_abs=float(err1.max()),stage2_max_abs=float(err2.max()),stage1_bad=int((err1>tol1).sum()),stage2_bad=int((err2>tol2).sum()),component_pass=pass_component,middle_quant_max_lsb=qerr,down_max_lsb=dmax,residual_max_lsb=fmax));print(result[-1],flush=True)
 caches={}
 for name in ['prefill_k_cache.bin','prefill_v_cache.bin','decode_k_cache.bin','decode_v_cache.bin']:
  p=R/'audit_a0'/name
  if p.exists():
   assert all((R/tag/name).read_bytes()==p.read_bytes() for tag in ['audit_a1','audit_a2']);caches[name]=sha(p)
 z=dict(component_pass=component and downpass,full_layer_ideal_R4_gate_pass=all(x['gate_pass'] for x in whole),known_R3_ideal_gate='unchanged historical failure',steps=result,whole_layer=whole,pre_R4_all_captured_stages_exact=True,cache_hashes=caches,independent_converter_sha256=sha(converter),converter_provenance=matched,model_quality='not assessed')
 write(R/'numerical_gate.json',z);print('NUMERICAL_COMPLETE',json.dumps(z),flush=True)
 assert z['component_pass'] and z['full_layer_ideal_R4_gate_pass']
if __name__=='__main__':main()
