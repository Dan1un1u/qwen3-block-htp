#!/usr/bin/env python3
"""Fresh original-weight R4 folding for all28layers, with exact LUT audit."""
from export_exp0261 import matrices,sha,write,preflight,S,M,read
from pathlib import Path
import numpy as np,torch,os,math,copy,json,subprocess
from safetensors import safe_open
from prepare_exp0164_generation_package import pack_w4_chunk
from rotation_exp0219 import unpack
from decimal import Decimal,localcontext
import mpmath as mp
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0265')
O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0265')
P=O.parent/'exp0258/r3'
ORIGIN=Path('/mnt/d/llm_exp/models/Qwen3-origin')
PINS={'model-00001-of-00002.safetensors':'169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5','model-00002-of-00002.safetensors':'912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9'}
def sealed(exp,names):
 import yaml
 e=next(e for e in yaml.safe_load((M/'experiments/index.yaml').read_text())['experiments'] if e['id']==f'EXP-{exp:04d}')
 root=R.parent/f'exp{exp:04d}';seal=read(root/'EVIDENCE_SHA256.json');assert sha(root/'EVIDENCE_SHA256.json')==e['evidence']['evidence_ledger_sha256']
 for n in names:assert sha(root/n)==seal['files'][n]['sha256'],(exp,n)
 return dict(experiment=exp,seal_sha256=sha(root/'EVIDENCE_SHA256.json'),checked={n:seal['files'][n] for n in names})
def lut(qp):
 g=(np.arange(256,dtype=float)-qp['gate']['zero_point'])*qp['gate']['scale'];u=(np.arange(256,dtype=float)-qp['up']['zero_point'])*qp['up']['scale'];out=np.empty((256,256),'<f2');repairs=0;mp.mp.dps=100
 with localcontext() as ctx:
  ctx.prec=80
  for i,gv in enumerate(g):
   dg=Decimal.from_float(float(gv));sg=dg/(1+(-dg).exp());mg=mp.mpf(float(gv));ms=mg/(1+mp.exp(-mg))
   for j,uv in enumerate(u):
    exact=sg*Decimal.from_float(float(uv));center=np.float16(float(exact));choices=[center,np.nextafter(center,np.float16(-np.inf)),np.nextafter(center,np.float16(np.inf))]
    chosen=min(choices,key=lambda z:(abs(Decimal.from_float(float(z))-exact),int(np.asarray(z).view('u2'))&1));out[i,j]=chosen;repairs+=int(np.asarray(center).view('u2')!=np.asarray(chosen).view('u2'))
    mex=ms*mp.mpf(float(uv));lo=np.nextafter(chosen,np.float16(-np.inf));hi=np.nextafter(chosen,np.float16(np.inf));assert (mp.mpf(float(lo))+mp.mpf(float(chosen)))/2<=mex<=(mp.mpf(float(hi))+mp.mpf(float(chosen)))/2
 assert np.isfinite(out).all();return out,repairs

def main():
 preflight();R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);d=O/'r4';d.mkdir(exist_ok=True);assert not (d/'manifest.json').exists();torch.set_num_threads(6)
 refs=[sealed(264,['summary.json','numerical_gate.json','ARTIFACT_PROVENANCE.json']),sealed(261,['export_audit.json']),sealed(259,['ARTIFACT_PROVENANCE.json','slice_stream/step00_output.bin','smoke_a1/validated.json','smoke_a1/stdout.jsonl','historical_provenance.json']),sealed(258,['export_audit.json','prompts.json','device_package.json'])]
 z=read(R.parent/'exp0264/summary.json');assert all(z['performance'][f'r10_{mode}_ns']['ci95'][1]<=1.1 for mode in ['prefill','decode']);write(R/'inherited_evidence.json',dict(pass_all=True,refs=refs,repeat10_escalation_eligible=True,repeat1_auxiliary_PC079=True))
 old=read(P/'manifest.json');assert sha(P/'manifest.json')==read(R.parent/'exp0258/export_audit.json')['manifest_sha256']
 for n,v in old['files'].items():assert sha(P/n)==v['sha256'],n
 for n,h in PINS.items():assert sha(ORIGIN/n)==h,n
 h12,h512=matrices();assert np.array_equal(h12.T@h12,12*np.eye(12)) and np.array_equal(h512.T@h512,512*np.eye(512));a=torch.from_numpy(h12);b=torch.from_numpy(h512)
 changed_names={'down_weight_w4_hmx.bin','down_weight_w4_scale_f32.bin','silu_up_lut_u16.bin'}
 for n,v in old['files'].items():
  p=d/n;p.parent.mkdir(parents=True,exist_ok=True)
  if p.name not in changed_names:
   if p.exists():assert sha(p)==v['sha256']
   else:os.link(P/n,p)
 index=read(ORIGIN/'model.safetensors.index.json');rng=np.random.default_rng(265);x=rng.normal(size=(4,6144));xr=np.einsum('rbc,ba->rac',x.reshape(4,12,512)@h512/math.sqrt(512),h12).reshape(4,6144)/math.sqrt(12)
 results=[]
 for i in range(28):
  record=R/f'export_layer{i:02d}.json';layer=d/f'layer{i}'
  if record.exists():
   rec=read(record);assert rec['pass_all']
   for n,h in rec['file_sha256'].items():assert sha(layer/n)==h
   results.append(rec);continue
  name=f'model.layers.{i}.mlp.down_proj.weight'
  with safe_open(ORIGIN/index['weight_map'][name],framework='pt',device='cpu') as f:w=f.get_tensor(name).double()
  rotated=(torch.einsum('obc,ba->oac',w.reshape(2048,12,512)@b/math.sqrt(512),a)/math.sqrt(12)).reshape(2048,6144)
  err=float(np.abs(x@w.numpy().T-xr@rotated.numpy().T).max());assert err<1e-10
  packed,scales=pack_w4_chunk(rotated.float());q=unpack(packed,2048,6144).numpy();assert np.array_equal(q,np.clip(np.rint(rotated.float().numpy()/scales[:,None]),-7,7))
  table,repairs=lut(old['layer_qparams'][i]);values={'down_weight_w4_hmx.bin':packed,'down_weight_w4_scale_f32.bin':scales,'silu_up_lut_u16.bin':table}
  for n,v in values.items():
   target=layer/n;assert not target.exists();v.tofile(target)
   if i==0:assert target.read_bytes()==(O.parent/'exp0261/r4'/n).read_bytes(),('layer0 reproduction',n)
  rec=dict(layer=i,pass_all=True,invariance_max_abs=err,independent_W4_unpack=True,LUT_all65536_halfwords_exact=True,LUT_backend='Decimal80 independent mpmath100',tie_repairs=repairs,file_sha256={n:sha(layer/n) for n in values});write(record,rec);results.append(rec);print('LAYER_EXPORT_PASS',i,err,repairs,flush=True)
 for n in changed_names:assert not (d/n).exists();os.link(d/'layer0'/n,d/n)
 new=copy.deepcopy(old);new.update(experiment='EXP-0265',parent_manifest_sha256=sha(P/'manifest.json'),original_shards=PINS,R4='full6144 H12 tensor H512 denseHMX OPT6 all28layers',half_scale512=float(np.float16(1/math.sqrt(512))),half_scale12=float(np.float16(1/math.sqrt(12))),quantizer='Down fresh original W R then RTN[-7,7] oneFP32scale/output; allotherweightsC64 frozen')
 new['files']={n:dict(bytes=(d/n).stat().st_size,sha256=sha(d/n)) for n in old['files']};diff=[n for n in old['files'] if old['files'][n]!=new['files'][n]];assert set(diff)=={n for n in old['files'] if Path(n).name in changed_names};write(d/'manifest.json',new)
 import shutil
 shutil.copyfile(R.parent/'exp0258/prompts.json',R/'prompts.json')
 prefix=O.parent/'exp0258/prefix/prefix_kv_u8.bin';expected=read(R.parent/'exp0258/export_audit.json')['seed_sha256'];assert sha(prefix)==expected
 write(R/'export_audit.json',dict(pass_all=True,original_shards=PINS,parent_manifest_sha256=sha(P/'manifest.json'),manifest_sha256=sha(d/'manifest.json'),files=len(new['files']),changed_files=diff,layer_results=results,all_other_files_unchanged=True,layer0_reproduced_EXP0261=True,frozen_prefix_sha256=expected,prefix_policy='unchanged frozen R3 diagnostic seed; not regenerated through R4',source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()))
 print('EXPORT_ALL_PASS',len(diff),flush=True)
if __name__=='__main__':main()
