#!/usr/bin/env python3
"""Fresh Qwen3 layer14 LPBQ4/8 G32 export. No imported folded weights.
AIMET minmax symmetric grid [-8,7], max(max/7,-min/8); common row scale
max(group_scales)/16, round-to-nearest-even integer multipliers [1,16].
Pinned reference retained alongside results; not a QNN execution dependency.
"""
import ast, hashlib, json, shutil, urllib.request
from pathlib import Path
import numpy as np
from safetensors import safe_open
ROOT=Path('/mnt/d/llm_exp')
OUT=ROOT/'models/qwen3-block-htp/exp0240'
RESULT=ROOT/'results/qwen3-block-htp/exp0240'
CONTROL=ROOT/'models/qwen3-block-htp/exp0148/w4u8'
ORIGIN=ROOT/'models/Qwen3-origin'
AIMET='761ef454c39b51cd127d3104ca1d975116bedb61'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,b):
 if p.exists(): assert p.read_bytes()==b,p
 else: p.write_bytes(b)
def pack(q):
 n,k=q.shape
 physical=q.reshape(n//32,32,k//32,8,4).transpose(0,2,3,1,4).copy().reshape(n//32,k//32,1024).astype(np.uint8)&15
 return physical[...,::2] | physical[...,1::2]<<4
def unpack(b,n,k):
 a=np.empty((n//32,k//32,1024),dtype=np.int8)
 a[...,::2]=(b&15).astype(np.int8);a[...,1::2]=(b>>4).astype(np.int8)
 a[a>=8]-=16
 return a.reshape(n//32,k//32,8,32,4).transpose(0,3,1,2,4).reshape(n,k)
def main():
 OUT.mkdir(exist_ok=True);RESULT.mkdir(exist_ok=True)
 ledger=json.loads((RESULT.parent/'exp0218/original_checkpoint_sha256.json').read_text())
 for name,expected in ledger.items(): assert sha(ORIGIN/name)==expected,name
 manifest=json.loads((CONTROL/'manifest.json').read_text())
 for name,rec in manifest['files'].items():
  assert (CONTROL/name).stat().st_size==rec['bytes'] and sha(CONTROL/name)==rec['sha256'],name
 refs={}
 for name,path in [('lpbq_utils.py','TrainingExtensions/onnx/src/python/aimet_onnx/lpbq_utils.py'),('quantizer.py','TrainingExtensions/torch/src/python/aimet_torch/quantization/affine/quantizer.py'),('encoding_analyzer.py','TrainingExtensions/torch/src/python/aimet_torch/quantization/encoding_analyzer.py')]:
  u=f'https://raw.githubusercontent.com/qualcomm/aimet/{AIMET}/{path}'
  b=urllib.request.urlopen(u).read();write(RESULT/('aimet_'+name),b);refs[name]={'url':u,'sha256':hashlib.sha256(b).hexdigest()}
 tree=ast.parse((RESULT/'aimet_lpbq_utils.py').read_text())
 keep={'_split_blocks','_get_per_group_scale_factor','grouped_dynamic_quantize'}
 tree.body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in keep]
 ns={'np':np,'Sequence':__import__('typing').Sequence,'Tuple':__import__('typing').Tuple}
 exec(compile(tree,'pinned_aimet_lpbq_utils','exec'),ns)
 index=json.loads((ORIGIN/'model.safetensors.index.json').read_text())['weight_map']
 destinations={x:OUT/x for x in ['lpbq32','unit_multiplier']}
 for d in destinations.values():
  d.mkdir(exist_ok=True)
  for name in manifest['files']:
   if not (d/name).exists(): shutil.copy2(CONTROL/name,d/name)
 diagnostics={}
 for name in ['q','k','v','o','gate','up','down']:
  key=f'model.layers.14.{"self_attn" if name in ["q","k","v","o"] else "mlp"}.{name}_proj.weight'
  with safe_open(ORIGIN/index[key],framework='pt',device='cpu') as f: w=f.get_tensor(key).float().numpy()
  n,k=w.shape;g=w.reshape(n,k//32,32)
  raw=np.maximum(np.maximum(g.max(-1),0)/np.float32(7),-np.minimum(g.min(-1),0)/np.float32(8))
  assert np.all(raw>0)
  base=raw.max(1)/np.float32(16)
  mult=np.clip(np.rint(raw/base[:,None]),1,16).astype(np.uint8)
  official,officialbase=ns['grouped_dynamic_quantize'](raw,[1,-1],4)
  assert np.array_equal(mult,official) and np.array_equal(base,officialbase.reshape(n))
  scales=(mult.astype(np.float32)*base[:,None])
  q=np.clip(np.rint(g/scales[:,:,None]),-8,7).astype(np.int8).reshape(n,k)
  packed=pack(q);assert np.array_equal(unpack(packed,n,k),q)
  old=np.fromfile(CONTROL/f'{name}_weight_w4_hmx.bin',dtype=np.uint8).reshape(n//32,k//32,512)
  for variant,codes,mm,bs in [('lpbq32',packed,mult,base),('unit_multiplier',old,np.ones_like(mult),np.fromfile(CONTROL/f'{name}_weight_w4_scale_f32.bin',dtype='<f4'))]:
   mm=(mm-1).reshape(n//32,32,k//32).transpose(0,2,1).copy()
   metadata=mm[...,::2] | mm[...,1::2]<<4
   payload=np.concatenate([codes.reshape(n//32,-1),metadata.reshape(n//32,-1)],axis=1)
   dense=unpack(codes,n,k)
   reconstructed=(dense.reshape(n,k//32,32).astype(np.int16)*(mm.transpose(0,2,1).reshape(n,k//32).astype(np.int16)+1)[:,:,None]).reshape(n,k)
   assert reconstructed.min()>=-128 and reconstructed.max()<=127
   d=destinations[variant]
   # These are owned new packages; original/control remain immutable.
   (d/f'{name}_weight_w4_hmx.bin').write_bytes(codes.tobytes())
   (d/f'{name}_weight_w4_scale_f32.bin').write_bytes(bs.astype('<f4').tobytes())
   write(d/f'{name}_lpbq32.bin',payload.tobytes())
   if variant=='lpbq32':
    np.save(RESULT/f'{name}_reference_s8.npy',reconstructed.astype(np.int8))
  diagnostics[name]={'shape':[n,k],'multiplier_min':int(mult.min()),'multiplier_max':int(mult.max()),'mse':float(np.mean((w-q.reshape(n,k//32,32).astype(np.float32).reshape(n,k)*np.repeat(scales,32,axis=1))**2)),'packed_bytes':packed.size,'lpbq_bytes':payload.size}
 for variant,d in destinations.items():
  x={'experiment':'EXP-0240','variant':variant,'original_checkpoint_sha256':ledger,'source_control_manifest_sha256':sha(CONTROL/'manifest.json'),'reference':refs,'grid':'signed [-8,7], nearest-even minmax, group K32, multiplier 1..16 stored m-1','files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(d.iterdir()) if p.name!='manifest.json'}}
  (d/'manifest.json').write_text(json.dumps(x,indent=2)+'\n')
 (RESULT/'export_audit.json').write_text(json.dumps({'original_hashes_verified':True,'control_files_verified':True,'official_group_scale_exact':True,'packing_roundtrip_exact':True,'projections':diagnostics,'reference':refs},indent=2)+'\n')
 print(json.dumps(diagnostics,indent=2),flush=True)
if __name__=='__main__': main()
