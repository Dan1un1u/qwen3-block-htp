#!/usr/bin/env python3
"""Independent constant layout, dense Float64 component and full audit equality."""
from device_exp0259 import *
import re
preflight();P=R.parent/'exp0252';seal=read(P/'EVIDENCE_SHA256.json')
# Project-memory authority pins the inherited implementation evidence.
import yaml
entry=next(e for e in yaml.safe_load((M/'experiments/index.yaml').read_text())['experiments'] if e['id']=='EXP-0252')
assert sha(P/'EVIDENCE_SHA256.json') in yaml.safe_dump(entry)
for n in ['numerical_audit.json','device_numerical_gate.json']:
 if (P/n).exists():assert sha(P/n)==seal['files'][n]['sha256']
h=np.ones((1,1),dtype=np.int16)
for _ in range(7):h=np.block([[h,h],[h,-h]])
assert np.array_equal(h.astype(np.int32)@h.astype(np.int32).T,128*np.eye(128,dtype=np.int32))
s=(S/'src/dsp/r3_sign_matrix.inc').read_text();bits=np.array([int(v,16) for v in re.findall(r'0x[0-9a-f]+',s)],dtype=np.uint16);assert bits.size==16384
actual=bits.view(np.float16).reshape(4,4,16,32,2).transpose(0,3,1,2,4).reshape(128,128);assert np.array_equal(actual,h)
component=[];files=0;carrier=64*24*128*2
for arm,parent in [('off','nr64'),('r3','r3_nr64')]:
 for step in range(9):
  for suffix in ['output','r3']:
   q=P/f'audit_{parent}/step{step:02}_{suffix}.bin';assert sha(q)==seal['files'][str(q.relative_to(P))]['sha256'];assert q.read_bytes()==(R/f'audit_{arm}/step{step:02}_{suffix}.bin').read_bytes();files+=1
for arm in ['vector','stream','vector_fused','stream_fused']:
 for step in range(9):
  for suffix in ['output','r3']:
   assert (R/f'audit_{arm}/step{step:02}_{suffix}.bin').read_bytes()==(R/f'audit_r3/step{step:02}_{suffix}.bin').read_bytes();files+=1
for step in range(9):
 n=1536 if step==0 else 24;b=(R/f'audit_stream_fused/step{step:02}_r3.bin').read_bytes()
 raw=np.frombuffer(b,dtype='<f2',count=n*128).reshape(n,128).astype(np.float64)
 got=np.frombuffer(b,dtype='<f2',count=n*128,offset=carrier).reshape(n,128).astype(np.float64)
 ref=(raw@h.astype(np.float64))*float(np.float16(1/np.sqrt(128.0)))
 ref16=ref.astype(np.float16);tol=np.abs(np.spacing(np.abs(ref16))).astype(np.float64)+2**-14
 error=np.abs(got-ref16.astype(np.float64));assert np.isfinite(got).all() and (error<=tol).all()
 component.append(dict(step=step,rows=n,max_abs=float(error.max()),within_original_component_tolerance=True))
write(R/'numerical_gate.json',dict(pass_all=True,ABI=121,exact_comparisons=files,whole_audit_and_outputs_byte_exact=True,all_9_steps=True,independent_constant_layout_and_orthogonality=True,component=component,old_numerical_reference=P.as_posix(),old_evidence_sha256=sha(P/'EVIDENCE_SHA256.json'),ideal_R3_whole_layer_gate='known_failed_unchanged',scope='frozen layer0 replay, same-input implementation equality; not PPL acceptance'))
print('NUMERICAL_PASS',files,flush=True)
