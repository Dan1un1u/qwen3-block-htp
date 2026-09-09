#!/usr/bin/env python3
"""Prespecified joint restoration, then independent conditional attribution."""
import argparse,json,math,time,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
import data_exp0254 as data
import r3_exp0246 as r
import floating_attention_exp0253 as fa
ROOT=data.RESULT
BASE_NAMES=['F','C64','B','J']
BRANCHES={'attention':['F','C64','B','J','K','QK','V','C'],
          'downstream':['F','C64','B','J','R','S','RS','JR','JS','JRS']}
GROUPS={'J':['q_out','k_out','v_out','q_rope','k_cache','v_cache','attn_context'],
 'K':['k_out','k_cache'],'QK':['q_out','k_out','q_rope','k_cache'],
 'V':['v_out','v_cache'],'C':['attn_context'], 'R':['residual_mid','residual_out'],'S':['swiglu']}
HOOKS=['q_out','k_out','v_out','attn_context','residual_mid','residual_out','swiglu']
ALL_NAMES=list(dict.fromkeys(BASE_NAMES+sum(BRANCHES.values(),[])))
def write(n,d):data.write(n,d)
def read(n):return json.loads((ROOT/n).read_text())
def preflight():
 data.preflight();assert subprocess.check_output(['git','branch','--show-current'],cwd=data.SOURCE,text=True).strip()=='codex/exp-0254-joint-attention'
def mask(name):
 if name in ['F','C64','B']:return []
 groups=[name] if name in GROUPS else list(name)
 shorts=set(x for g in groups for x in GROUPS[g])
 return sorted(f'L{i:02d}.'+n for i in range(28) for n in shorts)
def freeze():
 data.frozen();p=data.verified('exp0253','inputs.json');old=json.loads(p.read_text())
 write('inputs.json',dict(names=ALL_NAMES,base_names=BASE_NAMES,branches=BRANCHES,masks={n:mask(n) for n in ALL_NAMES},parameters=old['parameters'],development=old['development'],prefix=[151645],source_head=r.b.head()))
 ref=data.verified('exp0253','checks/numerical.json');core=data.SOURCE/'scripts/floating_attention_exp0253.py'
 # The numerical core itself must be the exact archived parent source version.
 archived=subprocess.check_output(['git','show','6d130035cf089ba099c8f161799952bf1d70a385:scripts/floating_attention_exp0253.py'],cwd=data.SOURCE)
 assert hashlib.sha256(archived).hexdigest()==data.sha(core)
 write('freeze.json',dict(files={n:data.sha(ROOT/n) for n in ['inputs.json','dataset.json']},references={str(x):data.sha(x) for x in [p,ref,core]},protocol_sha256=data.sha(data.MEMORY/'docs/experiments/EXP-0254.md'),frozen_before_inference=True))
def frozen():
 data.frozen();f=read('freeze.json')
 assert f['protocol_sha256']==data.sha(data.MEMORY/'docs/experiments/EXP-0254.md')
 for n,h in f['files'].items():assert data.sha(ROOT/n)==h
 for n,h in f['references'].items():assert data.sha(n)==h
 assert read('inputs.json')['masks']=={n:mask(n) for n in ALL_NAMES}
class Cache(r.DenseCache):
 def update(self,k,v,i,cache_kwargs=None):
  out=super().update(k,v,i,cache_kwargs);self.ins.current_values[i]=(self.key_cache[i],self.value_cache[i]);return out
class OracleCache(r.DenseFloatCache):
 def update(self,k,v,i,cache_kwargs=None):
  out=super().update(k,v,i,cache_kwargs);self.ins.current_values[i]=(self.key_cache[i],self.value_cache[i]);return out
class Instrument(r.Instrument):
 def __init__(self,model):
  self.current_values={};self.counts={};self.hook_counts={};self.is_oracle=False
  super().__init__(model);self.cache_type=Cache
 def select(self,name):
  self.name=name;self.params=read('inputs.json')['parameters'][name if name in ['F','C64'] else 'R3_ALL']
  self.restored=set(mask(name));self.rotation=set() if name in ['F','C64'] else set(range(28))
  self.configure(None if name in ['F','C64'] else 'mse',r.b.a.FAMILIES);self.reset_counts()
 def reset_counts(self):self.rotation_counts={};self.counts={};self.hook_counts={}
 def count(self,key):self.counts[key]=self.counts.get(key,0)+1
 def apply(self,name,family,x,layout='last',skip_quant=False):
  if not self.warm and self.policy:
   short=name.split('.')[-1]
   if short=='attention_prob':raise AssertionError('unexpected probability QDQ')
   if short in HOOKS:
    assert not skip_quant
    key=name+('=restored' if name in self.restored else '=qdq');self.hook_counts[key]=self.hook_counts.get(key,0)+1
  return super().apply(name,family,x,layout,skip_quant)
 def attention(self,module,query,key,value,attention_mask,scaling,dropout=0.,**kw):
  if self.warm or not self.policy:return super().attention(module,query,key,value,attention_mask,scaling,dropout,**kw)
  assert dropout==0;i=module.layer_idx;p=f'L{i:02d}.'
  if i in self.rotation:
   query=r.dense(query,self.s);self.rotation_counts['Q']=self.rotation_counts.get('Q',0)+1
  if self.enabled(p+'q_rope'):
   query=r.b.p.decode(r.b.p.codes(query,self.params[p+'q_rope']['mse']),self.params[p+'q_rope']['mse'],query.dtype);self.count('q_qdq')
  else:self.count('q_restored')
  k,v=self.current_values[i];decoded=[]
  for x,site in [(k,'k_cache'),(v,'v_out')]:
   enabled=self.enabled(p+site)
   assert x.dtype==(torch.float16 if self.is_oracle or not enabled else torch.uint8)
   decoded.append(r.b.p.decode(x,self.params[p+site]['mse'],query.dtype) if x.dtype==torch.uint8 else x)
   self.count(site+('=qdq' if enabled else '=restored'))
  k,v=[r.b.a.qwen.repeat_kv(x,module.num_key_value_groups) for x in decoded]
  nq,nk=query.shape[-2],k.shape[-2]
  valid=(torch.arange(nk,device=query.device)[None,:]<=torch.arange(nk-nq,nk,device=query.device)[:,None])[None,None]
  if attention_mask is not None:valid=valid&(attention_mask[:,:,:,:nk]==0)
  out,prob=fa.core(query,k,v,valid,scaling);self.count('core')
  return out.to(query.dtype).transpose(1,2).contiguous(),prob
 def verify_counts(self,n):
  batches=n//4;steps=16*batches
  rotation={'prefix_K':28*batches,'body_K':28*steps,'Q':28*steps} if self.rotation else {}
  assert self.rotation_counts==rotation,(self.name,self.rotation_counts,rotation)
  if not self.policy:assert not self.counts and not self.hook_counts;return
  expected={'core':28*steps}
  for site,a,b in [('q_rope','q_qdq','q_restored'),('k_cache','k_cache=qdq','k_cache=restored'),('v_out','v_out=qdq','v_out=restored')]:expected[a if self.enabled('L00.'+site) else b]=28*steps
  assert self.counts==expected,(self.name,self.counts,expected)
  hooks={f'L{i:02d}.'+short+('=qdq' if self.enabled(f'L{i:02d}.'+short) else '=restored'):steps for i in range(28) for short in HOOKS}
  assert self.hook_counts==hooks,(self.name,self.hook_counts,hooks)
def numerical():
 old=data.verified('exp0253','checks/numerical.json');assert json.loads(old.read_text())['pass_all'];rng=np.random.default_rng(254);cases=[]
 for size in [(64,65),(1,80),(7,13)]:
  for magnitude in [0.1,1.,16.,128.]:
   nq,nk=size;q,k,v=[(rng.normal(size=(1,2,n,128))*magnitude).astype(np.float16) for n in [nq,nk,nk]]
   valid=(np.arange(nk)[None,:]<=np.arange(nk-nq,nk)[:,None])[None,None]
   out,prob=fa.core(*[torch.from_numpy(x).cuda() for x in [q,k,v,valid]],1/math.sqrt(128));ref,_=fa.numpy_reference(q,k,v,valid,1/math.sqrt(128));stream,_=fa.numpy_reference(q,k,v,valid,1/math.sqrt(128),chunk=7)
   assert np.allclose(ref,stream,rtol=1e-12,atol=1e-12)
   e=out.cpu().numpy()-ref;nrms=float(np.sqrt(np.mean(e*e))/max(1e-12,np.sqrt(np.mean(ref*ref))));maximum=float(np.max(abs(e)));row=float((prob.sum(-1)-1).abs().max())
   assert nrms<=1e-5 and maximum<=2e-5*max(1,float(np.max(abs(ref)))) and row<=2e-6
   assert (prob.cpu().numpy()[~np.broadcast_to(valid,prob.shape)]==0).all()
   cases.append(dict(nq=nq,nk=nk,magnitude=magnitude,normalized_rms=nrms,max_abs=maximum,row_mass_error=row))
 write('checks/numerical.json',dict(pass_all=True,parent_280_cases_sha256=data.sha(old),new_FP16_cases=cases,core_sha256=data.sha(data.SOURCE/'scripts/floating_attention_exp0253.py')))
 print('NUMERICAL_PASS',flush=True)
def score(model,ins,name,samples,phase):
 path=f'scores/{phase}_{name}.json'
 if (ROOT/path).exists():
  d=read(path);assert d['freeze_sha256']==data.sha(ROOT/'freeze.json') and d['mask']==mask(name);return d
 ins.select(name);before=r.b.prefix_digest(ins);start=time.monotonic()
 # Direct hook sentinel on all quantized/restored projection/context/MLP sites.
 if ins.policy:
  x=torch.linspace(-77,83,128,device='cuda',dtype=torch.float16).reshape(1,1,128)
  for i in range(28):
   for short in HOOKS:
    site=f'L{i:02d}.'+short;got=ins.apply(site,'attention' if short=='attn_context' else 'residual' if short.startswith('residual') else 'swiglu' if short=='swiglu' else 'qkv_output',x)
    expect=r.b.a.qdq(x,ins.params[site]['mse']) if ins.enabled(site) else x
    assert torch.equal(got,expect),(name,site,'hook sentinel')
 logits,nll,_=r.forward(model,ins,samples[:4]);again,rn,_=r.forward(model,ins,samples[:4]);assert torch.equal(logits,again) and torch.equal(nll,rn)
 labels=torch.tensor([x['target_ids'] for x in samples[:4]],device='cuda');ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.flatten(),reduction='none').reshape(4,16);error=(ce-nll).abs().max().item();assert error<5e-6
 changed=[dict(x,token_ids=x['token_ids'][:70]+[123]*10) for x in samples[:4]];future,_,_=r.forward(model,ins,changed);assert torch.equal(logits[:,:7],future[:,:7])
 ins.cache_type=OracleCache;ins.is_oracle=True
 try:oracle,onll,_=r.forward(model,ins,samples[:4]);assert torch.equal(logits,oracle) and torch.equal(nll,onll),(name,'float cache oracle')
 finally:ins.cache_type=Cache;ins.is_oracle=False
 rows=[];ins.reset_counts()
 for j in range(0,len(samples),4):
  logits,nll,cache=r.forward(model,ins,samples[j:j+4]);assert torch.isfinite(nll).all()
  for i,(k,v) in enumerate(zip(cache.key_cache,cache.value_cache)):
   assert k.dtype==(torch.uint8 if ins.enabled(f'L{i:02d}.k_cache') else torch.float16)
   assert v.dtype==(torch.uint8 if ins.enabled(f'L{i:02d}.v_out') else torch.float16)
   assert k.shape[-2]==v.shape[-2]==80
  rows.extend(dict(id=x['id'],cell=x['cell'],nll=loss,top1=top) for x,loss,top in zip(samples[j:j+4],nll.cpu().tolist(),logits.argmax(-1).cpu().tolist()))
  if j%64==0:print('PROGRESS',phase,name,j+4,len(samples),flush=True)
 ins.verify_counts(len(samples));assert before==r.b.prefix_digest(ins)
 cells={c:float(np.mean([x['nll'] for x in rows if x['cell']==c])) for c in data.CELLS};mean=float(np.mean(list(cells.values())))
 d=dict(name=name,phase=phase,ppl=math.exp(mean),mean_nll=mean,cell_nll=cells,samples=rows,mask=mask(name),source_head=r.b.head(),freeze_sha256=data.sha(ROOT/'freeze.json'),route_sha256=data.sha(ROOT/'route.json') if (ROOT/'route.json').exists() else None,
 checks=dict(repeat_exact=True,causal_exact=True,CE_max_abs=error,cache_oracle_exact=True,hook_sentinel_exact=True,prefix_immutable=True,rotation_counts=ins.rotation_counts,boundary_counts=ins.counts,hook_counts=ins.hook_counts),elapsed_s=time.monotonic()-start,role='conditional_software_restoration_not_deployable_A8_or_device_PPL')
 write(path,d);print('SCORE',phase,name,d['ppl'],round(d['elapsed_s'],1),flush=True);return d

def route():
 assert read('triage_complete.json')['pass_all']
 c=read('scores/triage_C64.json');j=read('scores/triage_J.json')
 ratio=j['ppl']/c['ppl'];cells={k:math.exp(j['cell_nll'][k]-c['cell_nll'][k]) for k in data.CELLS}
 close=ratio<=1.05 and all(v<=1.10 for v in cells.values());branch='attention' if close else 'downstream'
 assert not list((ROOT/'scores').glob('confirmation_*.json'))
 write('route.json',dict(branch=branch,names=BRANCHES[branch],J_over_C64=ratio,cell_ratios=cells,close_point=close,rule='overall<=1.05_and_each_cell<=1.10_vs_C64_route_only_not_acceptance',frozen_before_confirmation=True,freeze_sha256=data.sha(ROOT/'freeze.json'),triage_sha256={n:data.sha(ROOT/f'scores/triage_{n}.json') for n in BASE_NAMES}))
 print('ROUTE',branch,ratio,cells,flush=True)
def selected():
 z=read('route.json');assert z['freeze_sha256']==data.sha(ROOT/'freeze.json') and z['names']==BRANCHES[z['branch']]
 for n,h in z['triage_sha256'].items():assert data.sha(ROOT/f'scores/triage_{n}.json')==h
 return z['names']
def run(phase):
 assert read('checks/numerical.json')['pass_all'] and read('independent_data_audit.json')['pass_all']
 if phase=='triage':assert read('development_complete.json')['pass_all']
 if phase=='confirmation':assert read('branch_development_complete.json')['pass_all']
 names=BASE_NAMES if phase in ['development','triage'] else selected()
 if phase=='branch_development':names=[n for n in names if n not in BASE_NAMES]
 samples=read('inputs.json')['development'] if 'development' in phase else [x for x in read('dataset.json')['samples'] if x['split']==phase]
 for recipe in ['C64','F']:
  todo=[n for n in names if (n=='F')==(recipe=='F')]
  if not todo:continue
  proof=f'checks/{phase}_{recipe}_weights.json'
  if (ROOT/proof).exists():continue
  model,before,mh=r.b.model_session(recipe);ins=Instrument(model);ins.build_prefix([151645])
  for name in todo:
   d=score(model,ins,name,samples,phase)
   if phase=='development' and name in ['F','C64','B']:
    oldname='R3_float_core' if name=='B' else name;p=data.verified('exp0253',f'scores/development_{oldname}.json');old=json.loads(p.read_text())
    assert d['samples']==old['samples'] and d['ppl']==old['ppl'],('parent reproduction',name)
    write(f'checks/reproduction_{name}.json',dict(exact=True,reference_sha256=data.sha(p)))
  assert before==r.b.a.state_digest(model);write(proof,dict(unchanged=True,digest=before,manifest_sha256=mh,source_head=r.b.head()))
  ins.close();del model,ins;torch.cuda.empty_cache()
 write(phase+'_complete.json',dict(pass_all=True,names=names,panel_documents=len(samples)))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','numerical','development','triage','route','branch_development','confirmation']);a=parser.parse_args();preflight();r.b.a.settings()
 with torch.inference_mode():
  if a.phase=='freeze':freeze()
  else:
   frozen()
   if a.phase=='numerical':numerical()
   elif a.phase=='route':route()
   else:run(a.phase)
if __name__=='__main__':main()
