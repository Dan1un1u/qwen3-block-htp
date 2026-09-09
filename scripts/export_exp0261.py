#!/usr/bin/env python3
"""EXP0261 full-width orthogonal R4 fixture, from original Down tensors."""
from export_exp0257 import S,M,sha,write,preflight
from pathlib import Path
import os,json,math,subprocess
import numpy as np
import torch
from safetensors import safe_open
from prepare_exp0164_generation_package import pack_w4_chunk
from rotation_exp0219 import unpack
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0261')
O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0261')
P=O.parent/'exp0247/r3'
def read(p):return json.loads(Path(p).read_text())
def matrices():
 a=np.arange(512,dtype=np.uint32)
 b=np.bitwise_and(a[:,None],a[None,:]); parity=np.zeros_like(b)
 for i in range(9):parity^=(b>>i)&1
 h512=(1-2*parity.astype(np.int32)).astype(np.float64)
 residues={i*i%11 for i in range(1,11)};h12=np.ones((12,12),np.float64)
 for i in range(11):
  for j in range(11):h12[i+1,j+1]=-1 if i==j else (1 if (i-j)%11 in residues else -1)
 return h12,h512

def main():
 preflight();torch.set_num_threads(6)
 parent=R.parent/'exp0259';assert sha(parent/'EVIDENCE_SHA256.json')=='2f891f9aa35e7fa5bece5180e8e3a81fe4fd1253dc3749565dd226b462541a07'
 seal=read(parent/'EVIDENCE_SHA256.json')
 for n in ['package_provenance.json','numerical_gate.json']:
  assert sha(parent/n)==seal['files'][n]['sha256']
 pm=read(P/'manifest.json');assert sha(P/'manifest.json')==read(parent/'package_provenance.json')['r3']['manifest_sha256']
 for n,v in pm['files'].items():assert sha(P/n)==v['sha256'],n
 origin=Path('/mnt/d/llm_exp/models/Qwen3-origin')
 pins={'model-00001-of-00002.safetensors':'169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5','model-00002-of-00002.safetensors':'912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9'}
 for n,h in pins.items():assert sha(origin/n)==h,n
 h12,h512=matrices();assert np.array_equal(h12.T@h12,12*np.eye(12)) and np.array_equal(h512.T@h512,512*np.eye(512))
 full=np.kron(h12,h512)/math.sqrt(6144)
 rng=np.random.default_rng(261);x=rng.normal(size=(4,6144));fact=np.einsum('rbc,ba->rac',(x.reshape(-1,12,512)@h512)/math.sqrt(512),h12)/math.sqrt(12)
 assert np.max(np.abs(x@full-fact.reshape(-1,6144)))<1e-12
 name='model.layers.0.mlp.down_proj.weight';index=read(origin/'model.safetensors.index.json')
 with safe_open(origin/index['weight_map'][name],framework='pt',device='cpu') as f:w=f.get_tensor(name).double()
 assert tuple(w.shape)==(2048,6144)
 rotated=torch.einsum('obc,ba->oac',w.reshape(2048,12,512)@torch.from_numpy(h512)/math.sqrt(512),torch.from_numpy(h12))/math.sqrt(12);rotated=rotated.reshape(2048,6144)
 origout=x@w.numpy().T;rotout=fact.reshape(-1,6144)@rotated.numpy().T
 invariant=float(np.max(np.abs(origout-rotout)));assert invariant<1e-10
 packed,scales=pack_w4_chunk(rotated.float());q=unpack(packed,2048,6144).numpy()
 assert np.array_equal(q,np.clip(np.rint(rotated.float().numpy()/scales[:,None]),-7,7))
 O.mkdir(exist_ok=True);d=O/'r4';d.mkdir(exist_ok=True);assert not (d/'manifest.json').exists()
 changed=['down_weight_w4_hmx.bin','down_weight_w4_scale_f32.bin','silu_up_lut_u16.bin']
 for n in pm['files']:
  dst=d/n;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.name not in changed:
   if dst.exists():assert sha(dst)==pm['files'][n]['sha256']
   else:os.link(P/n,dst)
 qp=pm['qparams'];g=(np.arange(256,dtype=float)-qp['gate']['zero_point'])*qp['gate']['scale'];u=(np.arange(256,dtype=float)-qp['up']['zero_point'])*qp['up']['scale']
 # Float64 exp implementations can straddle a half midpoint. Generate correctly
 # rounded half values with Decimal80; independently certify rounding intervals
 # with mpmath100 instead of accepting backend-dependent one-ULP mismatches.
 from decimal import Decimal,localcontext
 import mpmath as mp
 mp.mp.dps=100;lut=np.empty((256,256),'<f2');tie_repairs=0
 with localcontext() as ctx:
  ctx.prec=80
  for i,gv in enumerate(g):
   dg=Decimal.from_float(float(gv));sg=dg/(1+(-dg).exp());mg=mp.mpf(float(gv));ms=mg/(1+mp.exp(-mg))
   for j,uv in enumerate(u):
    exact=sg*Decimal.from_float(float(uv));center=np.float16(float(exact))
    choices=[center,np.nextafter(center,np.float16(-np.inf)),np.nextafter(center,np.float16(np.inf))]
    chosen=min(choices,key=lambda z:(abs(Decimal.from_float(float(z))-exact),int(np.asarray(z).view('u2'))&1))
    lut[i,j]=chosen;tie_repairs+=int(np.asarray(center).view('u2')!=np.asarray(chosen).view('u2'))
    mex=ms*mp.mpf(float(uv));lo=np.nextafter(chosen,np.float16(-np.inf));hi=np.nextafter(chosen,np.float16(np.inf));midlo=(mp.mpf(float(lo))+mp.mpf(float(chosen)))/2;midhi=(mp.mpf(float(hi))+mp.mpf(float(chosen)))/2
    assert midlo<=mex<=midhi,(i,j)
 assert np.isfinite(lut).all()

 for base in [d,d/'layer0']:
  packed.tofile(base/'down_weight_w4_hmx.bin');scales.tofile(base/'down_weight_w4_scale_f32.bin');lut.tofile(base/'silu_up_lut_u16.bin')
 manifest=dict(experiment='EXP-0261',parent_manifest_sha256=sha(P/'manifest.json'),original_shards=pins,qparams=qp,rotation='full6144 H12 tensor H512',half_scale512=float(np.float16(1/math.sqrt(512))),half_scale12=float(np.float16(1/math.sqrt(12))),quantizer='Down only fresh original W R, per-output-channel RTN[-7,7], FP32scale',files={str(p.relative_to(d)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(d.rglob('*')) if p.is_file()})
 write(d/'manifest.json',manifest)
 write(R/'export_audit.json',dict(pass_all=True,original_shards=pins,parent_manifest_sha256=sha(P/'manifest.json'),manifest_sha256=sha(d/'manifest.json'),parent_files_verified=len(pm['files']),full_rotation_dimension=6144,factor_dimensions=[12,512],matrix_orthogonality=True,dense_factor_equivalence=True,invariance_max_abs=invariant,independent_W4_unpack=True,LUT_all65536_halfwords_exact=True,LUT_correct_rounding_backend="Decimal80 independently verified by mpmath100",double_rounding_repairs=tie_repairs,changed_files=[n for n in pm['files'] if manifest['files'][n]!=pm['files'][n]],weights_semantics='one FP32scale per output channel; no groups; actual W4 HMX carrier',quality='not assessed; middle static range unchanged; new Down RTN affects accuracy',source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()))
 print('EXPORT_PASS',invariant,flush=True)
if __name__=='__main__':main()
