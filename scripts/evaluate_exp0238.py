#!/usr/bin/env python3
"""Common packed per-channel reference suite, independent short-context PPL."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,math,sys,time,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from reference_exp0238 import S,M,R,O,sha,write,preflight
from data_exp0238 import frozen
from data_exp0229 import CELLS,verified
import evaluate_exp0230 as old
import rotation_exp0219 as rot
from autoround_exp0233 import read_weight

def settings():
 torch.set_grad_enabled(False);torch.set_num_threads(8);torch.manual_seed(238)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
 torch.use_deterministic_algorithms(True)
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def package(v):
 if v=='AR-P':p=verified('exp0233','AR-P/package.json')
 elif v=='Qronos':p=R/'Qronos/package.json'
 elif v=='OmniQuant':p=R.parent/'exp0239/OmniQuant/package.json'
 else:raise ValueError(v)
 info=json.loads(p.read_text());root=Path(info['root']);assert sha(root/'manifest.json')==info['manifest_sha256']
 return root,json.loads((root/'manifest.json').read_text()),info['manifest_sha256']
def load(v):
 if v=='F':return old.load('F','cuda')
 base=json.loads(verified('exp0230','C64/package.json').read_text());assert base['manifest_sha256']=='7de4f0758d83f2ba3b58696c695bfbfed72a25dd3bf308475abee0a0f0575a89'
 model,h=old.load('C64','cpu')
 if v=='C64':return model.cuda(),h
 root,manifest,h=package(v);assert manifest['grid']==[-7,7] and manifest['group_size']==-1
 for n,e in manifest['files'].items():assert sha(root/n)==e['sha256'],n
 changed={f'model.layers.{i}.{long}.weight' for i in range(28) for long in rot.PROJECTIONS.values()}
 before={n:digest(t) for n,t in list(model.named_parameters())+list(model.named_buffers()) if n not in changed}
 for i,layer in enumerate(model.model.layers):
  for n,long in rot.PROJECTIONS.items():
   mod=layer.get_submodule(long);sc=np.load(root/f'layer{i}/{n}_scales.npy');assert sc.shape==(mod.weight.shape[0],1) and sc.dtype==np.float32 and (sc>0).all()
   raw=np.fromfile(root/f'layer{i}/{n}_codes_linear_u4.bin',dtype=np.uint8);assert not np.any((raw&15)==8) and not np.any((raw>>4)==8)
   mod.weight.copy_(torch.from_numpy(read_weight(root/f'layer{i}',n,tuple(mod.weight.shape))))
 after={n:digest(t) for n,t in list(model.named_parameters())+list(model.named_buffers()) if n not in changed}
 assert after==before
 if v=='Qronos':assert after==manifest['frozen_nontransformer']
 return model.cuda().eval(),h

def evaluate(v):
 preflight();data=frozen();assert json.loads((R/'independent_data_audit.json').read_text())['pass_all'];settings()
 destination=R if v!='OmniQuant' else R.parent/'exp0239';outpath=destination/'software'/f'final_{v}.json';assert not outpath.exists()
 rows=data['samples'];assert len(rows)==1024;started=time.monotonic();model,mh=load(v)
 def forward(batch):
  x=torch.tensor([r['token_ids'] for r in batch],device='cuda');logits=model(input_ids=x[:,:79],use_cache=False).logits[:,63:79,:].float()
  nll=torch.logsumexp(logits,-1)-logits.gather(-1,x[:,64:80,None]).squeeze(-1)
  return x,logits,nll
 with torch.inference_mode():
  x,logits,nll=forward(rows[:4]);_,again,rep=forward(rows[:4]);assert torch.equal(logits,again) and torch.equal(nll,rep)
  ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,64:80].reshape(-1),reduction='none').reshape(4,16);error=float((ce-nll).abs().max());assert error<5e-6
  changed=x[:,:79].clone();changed[:,70:]=123;other=model(input_ids=changed,use_cache=False).logits[:,63:70,:].float();assert torch.equal(other,logits[:,:7])
  output=[]
  for i in range(0,len(rows),4):
   batch=rows[i:i+4];_,logits,nll=forward(batch);assert torch.isfinite(logits).all() and torch.isfinite(nll).all()
   output.extend(dict(id=r['id'],cell=r['cell'],nll=ns,top1=ts) for r,ns,ts in zip(batch,nll.cpu().tolist(),logits.argmax(-1).cpu().tolist()))
   if i%128==0:print('SUITE_EVAL_PROGRESS',v,i+4,len(rows),flush=True)
 means={c:math.fsum(x for r in output if r['cell']==c for x in r['nll'])/(256*16) for c in CELLS}
 mean=math.fsum(x for r in output for x in r['nll'])/16384
 result=dict(variant=v,dataset_sha256=sha(R/'dataset.json'),manifest_sha256=mh,samples=output,mean_nll=mean,ppl=math.exp(mean),cell_nll=means,repeat_exact=True,causal_mask_exact=True,independent_CE_max_abs=error,all_finite=True,role='packed_FP16_software_not_DSP',head_policy='F16_for_F_identical_C64_W4_for_all_quantized_variants',torch_version=torch.__version__,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),elapsed_s=time.monotonic()-started)
 outpath.parent.mkdir(parents=True,exist_ok=True)
 with outpath.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('SUITE_EVAL_COMPLETE',v,mean,math.exp(mean),flush=True)

def report(include_omni=False):
 preflight();data=frozen();variants=['F','C64','AR-P','Qronos']+(['OmniQuant'] if include_omni else [])
 destination=R if not include_omni else R.parent/'exp0239';runs={};arrays={};audit=[]
 for v in variants:
  p=(R.parent/'exp0239' if v=='OmniQuant' else R)/'software'/f'final_{v}.json';d=json.loads(p.read_text());runs[v]=d
  assert d['dataset_sha256']==sha(R/'dataset.json') and d['repeat_exact'] and d['causal_mask_exact'] and d['independent_CE_max_abs']<5e-6
  assert [r['id'] for r in d['samples']]==[r['id'] for r in data['samples']]
  a=np.array([r['nll'] for r in d['samples']],dtype=np.float64);assert a.shape==(1024,16) and np.isfinite(a).all()
  assert abs(a.mean()-d['mean_nll'])<1e-12 and abs(float(np.exp(a.mean()))-d['ppl'])<1e-10
  arrays[v]=a.mean(1);audit.append(dict(variant=v,independent_numpy_raw_token_reduction=True,source_sha256=sha(p)))
 masks={c:np.array([r['cell']==c for r in data['samples']]) for c in CELLS};masks['overall']=np.ones(1024,dtype=bool)
 result={}
 for cell,mask in masks.items():
  local={v:arrays[v][mask] for v in variants};n=len(local['F']);rng=np.random.default_rng(238);ix=rng.integers(0,n,size=(5000,n))
  row={'ppl':{v:math.exp(math.fsum(local[v])/n) for v in variants},'vs_F16':{},'vs_C64':{}}
  for v in variants:
   if v=='F':continue
   delta=local[v]-local['F'];ratio=math.exp(math.fsum(delta)/n);ci=np.exp(np.quantile(delta[ix].mean(1),[.025,.975]));limit=1.05 if cell=='overall' else 1.10
   row['vs_F16'][v]=dict(ppl_ratio=ratio,ratio_ci95=ci.tolist(),limit=limit,point_pass=ratio<=limit,confident_pass=float(ci[1])<=limit)
   if v!='C64':
    diff=local[v]-local['C64'];row['vs_C64'][v]=dict(ppl_ratio=math.exp(math.fsum(diff)/n),ratio_ci95=np.exp(np.quantile(diff[ix].mean(1),[.025,.975])).tolist())
  result[cell]=row
 status={v:('pass' if all(result[c]['vs_F16'][v]['confident_pass'] for c in masks) else 'fail' if any(not result[c]['vs_F16'][v]['point_pass'] for c in masks) else 'inconclusive') for v in variants if v!='F'}
 payload=dict(dataset_sha256=sha(R/'dataset.json'),results=result,quality_status=status,independent_reductions=audit,bootstrap='5000 paired document resamples; overall pooled balanced panel',baseline_promoted=False,device_speed='N/A_software_only')
 with (destination/'closure.json').open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
 print('SUITE_RESULT',json.dumps(result['overall']),status,flush=True)
if __name__=='__main__':
 report(len(sys.argv)>2) if sys.argv[1]=='report' else evaluate(sys.argv[1])
