#!/usr/bin/env python3
"""EXP0258 native package delta and independently checked rotated prefix."""
from export_exp0257 import S,M,sha,preflight,qp,qdqcode
from pathlib import Path
import json,os,shutil,copy,math
import numpy as np
import torch
from export_exp0149_vertical_slice import write_qparams,build_attention_config
from r3_exp0246 import dense,sign_matrix
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0258');O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0258');P=R.parent/'exp0246';OLD=O.parent/'exp0257'
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False);f.write('\n')
def main():
 preflight();R.mkdir(exist_ok=True);O.mkdir(exist_ok=False)
 assert sha(R.parent/'exp0257/EVIDENCE_SHA256.json')=='80b1ac2415364ef7898e118b111215279f558748035f74d26d7979ee845a3f4f'
 seal=read(R.parent/'exp0257/EVIDENCE_SHA256.json')
 for n in ['prompts.json','export_audit.json','prefix_export.json','device_package.json','ARTIFACT_PROVENANCE.json']:
  assert sha(R.parent/'exp0257'/n)==seal['files'][n]['sha256']
 old=read(OLD/'package/manifest.json');assert sha(OLD/'package/manifest.json')==read(R.parent/'exp0257/export_audit.json')['manifest_sha256']
 for n,v in old['files'].items():assert sha(OLD/'package'/n)==v['sha256'],n
 assert sha(P/'EVIDENCE_SHA256.json')=='de6b2a5bcd10369a9c0f188ae11af115e04f951fb3f970a1b0a0d5914f9e25dc';ledger=read(P/'EVIDENCE_SHA256.json')
 for n in ['parameters/A8.json','parameters/R3_ALL.json','H128_signs_i8.bin']:assert sha(P/n)==ledger['files'][n]['sha256']
 a=read(P/'parameters/A8.json')['parameters'];b=read(P/'parameters/R3_ALL.json')['parameters'];changed=[k for k in a if a[k]!=b[k]];assert set(changed)=={f'L{i:02d}.{n}' for i in range(28) for n in ['q_rope','k_cache']}
 root=O/'r3';root.mkdir();new=copy.deepcopy(old)
 for n in old['files']:
  dst=root/n;dst.parent.mkdir(parents=True,exist_ok=True);os.link(OLD/'package'/n,dst)
 for i in range(28):
  q=new['layer_qparams'][i];q['q_rope']=qp(b[f'L{i:02d}.q_rope']['mse']);q['k_rope']=qp(b[f'L{i:02d}.k_cache']['mse']);layer=root/f'layer{i}'
  (layer/'qparams_u8.bin').unlink();write_qparams(layer/'qparams_u8.bin',q)
  v=np.zeros((64,8,128),'u1');v[32:]=255;(layer/'attention_config_all_groups.bin').unlink();(layer/'attention_config_all_groups.bin').write_bytes(build_attention_config(v,q))
 for n in ['qparams_u8.bin','attention_config_all_groups.bin']:(root/n).unlink();os.link(root/'layer0'/n,root/n)
 new.update(experiment='EXP-0258',parameters_sha256=sha(P/'parameters/R3_ALL.json'),R3='plain dense HMX mode1,28layers',parent_manifest_sha256=sha(OLD/'package/manifest.json'))
 new['files']={n:dict(bytes=(root/n).stat().st_size,sha256=sha(root/n)) for n in old['files']};diff=[n for n in old['files'] if old['files'][n]!=new['files'][n]];assert set(diff)=={n for n in old['files'] if n.endswith(('qparams_u8.bin','attention_config_all_groups.bin')) and not n.startswith('generation_')}
 write(root/'manifest.json',new)
 # Raw FP16 prefix is already independently checked against unhooked frozenC64.
 rawfile=OLD/'prefix/prefix_kv_f16.bin';assert sha(rawfile)==read(R.parent/'exp0257/prefix_export.json')['fp16_sha256']
 raw=np.fromfile(rawfile,dtype='<f2').reshape(28,2,8,128);sign=sign_matrix();assert sign.tobytes()==(P/'H128_signs_i8.bin').read_bytes();assert np.array_equal(sign.astype(np.int32)@sign.astype(np.int32).T,np.eye(128,dtype=np.int32)*128)
 torch.backends.cuda.matmul.allow_tf32=False;x=torch.from_numpy(raw[:,0].copy()).cuda();ss=torch.from_numpy(sign).to('cuda',torch.float32);rot=dense(x,ss).cpu().numpy();oracle=((raw[:,0].astype(np.float64)@sign.astype(np.float64))/math.sqrt(128)).astype('<f2')
 assert np.array_equal(rot,oracle),'prefix dense FP16 versus independent Float64'
 payload=[]
 for i in range(28):
  for key,value in [('k_cache',rot[i]),('v_cache',raw[i,1])]:
   q=qp(b[f'L{i:02d}.{key}']['mse']);u=qdqcode(value,q);ref=np.clip(np.floor(value.astype(np.float32)/np.float32(q['scale'])+q['zero_point']+.5),0,255).astype('u1');assert np.array_equal(u,ref);payload.append(u.tobytes())
 seed=O/'prefix';seed.mkdir();(seed/'prefix_kv_u8.bin').write_bytes(b''.join(payload));(seed/'rotated_k_f16.bin').write_bytes(rot.tobytes())
 shutil.copy2(R.parent/'exp0257/prompts.json',R/'prompts.json')
 write(R/'export_audit.json',dict(pass_all=True,parent_manifest_sha256=sha(OLD/'package/manifest.json'),manifest_sha256=sha(root/'manifest.json'),parent_files_verified=len(old['files']),changed_files=diff,changed_parameter_sites=changed,all_weights_and_other_parameters_unchanged=True,rotated_prefix_independent_exact=True,seed_sha256=sha(seed/'prefix_kv_u8.bin'),seed_bytes=57344,prefix_V_exact=True,raw_prefix_sha256=sha(rawfile),parameters_sha256=sha(P/'parameters/R3_ALL.json'),prompts_sha256=sha(R/'prompts.json')))
 print('EXPORT_PASS',flush=True)
if __name__=='__main__':main()
