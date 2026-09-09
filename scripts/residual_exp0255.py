#!/usr/bin/env python3
"""PC069 bounded static-U8 residual calibration on frozen quantized trajectory."""
import argparse,copy,json,math,subprocess,time
import numpy as np
import torch
import data_exp0255 as data
import attention_exp0253 as old
r=old.r;a=r.b.a
ROOT=data.RESULT
MODES=['float_core','wide_nr64']
CANDIDATES=['B','R_MSE','R_FAN','S_MSE']
POSITIONS=[0,9,18,27,36,45,63,78]
SITES=[f'L{i:02d}.{s}' for i in range(28) for s in ['residual_mid','residual_out','swiglu']]
def read(n):return json.loads((ROOT/n).read_text())
def write(n,d):data.write(n,d)
def bind():old.ROOT=ROOT;old.data=data

def preflight():
 data.preflight()
 assert subprocess.check_output(['git','branch','--show-current'],cwd=data.SOURCE,text=True).strip()=='codex/exp-0255-residual-calibration'

def freeze():
 data.frozen();p=data.verified('exp0253','inputs.json');inputs=json.loads(p.read_text())
 cal=data.verified('exp0246','inputs.json');inputs['calibration']=json.loads(cal.read_text())['calibration']
 inputs['names']=CANDIDATES;write('inputs.json',inputs)
 refs=[p,cal,data.verified('exp0253','checks/numerical.json'),data.verified('exp0252','checks/numerical.json')]
 scripts=['floating_attention_exp0253.py','integer_attention_exp0252.py','attention_exp0253.py','r3_exp0246.py','ablate_exp0244.py','prefix_exp0243.py','a8_exp0242.py']
 refs += [data.SOURCE/'scripts'/s for s in scripts]
 write('freeze.json',dict(files={n:data.sha(ROOT/n) for n in ['inputs.json','dataset.json']},references={str(p):data.sha(p) for p in refs},protocol_sha256=data.sha(data.MEMORY/'docs/experiments/EXP-0255.md'),before_calibration=True))

def frozen():
 data.frozen();f=read('freeze.json');assert f['protocol_sha256']==data.sha(data.MEMORY/'docs/experiments/EXP-0255.md')
 for n,h in f['files'].items():assert data.sha(ROOT/n)==h
 for n,h in f['references'].items():assert data.sha(n)==h

def norm_meta(model,name):
 i=int(name[1:3])
 if name.endswith('residual_mid'):mod=model.model.layers[i].post_attention_layernorm
 elif name.endswith('residual_out'):mod=model.model.layers[i+1].input_layernorm if i<27 else model.model.norm
 else:return None
 return mod.weight.detach().float(),mod.variance_epsilon

def norm(x,meta):
 w,eps=meta;x=x.float();return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*w

def errors(x,y,meta):
 x=x.float();y=y.float();e=(x-y).square().mean().double();power=x.square().mean().double().clamp_min(1e-30)
 n=torch.tensor(0.,device=x.device,dtype=torch.float64)
 if meta is not None:
  ref=norm(x,meta);n=(norm(y,meta)-ref).square().mean().double()/ref.square().mean().double().clamp_min(1e-30)
 return float(e),float(e/power),float(n)

class Instrument(old.Instrument):
 def __init__(self,model):
  self.gather=False;self.parts={};self.extrema={};self.diagnostics=None;self.calls={};self.shared_mid={};self.shared_norm=0
  super().__init__(model)
  for i,layer in enumerate(model.model.layers):
   def check(module,args,_i=i):
    if not self.warm and self.policy:
     assert args[0] is self.shared_mid[_i],('shared fanout',_i)
     self.shared_norm+=1
   self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(check))
 def select(self,name):
  if name in ['F','C64']:super().select(name);return
  candidate,mode=name.split('__');super().select('R3_'+mode)
  if candidate!='B':self.params=read(f'parameters/{candidate}.json')['parameters']
 def apply(self,name,family,x,layout='last',skip_quant=False):
  active=not self.warm and bool(self.policy) and name in SITES
  if active:
   self.calls[name]=self.calls.get(name,0)+1
   if self.gather:
    self.parts.setdefault(name,[]).append(x.detach().cpu())
    rec=self.extrema.setdefault(name,dict(minimum=0.,maximum=0.,count=0))
    rec['minimum']=min(rec['minimum'],x.min().item());rec['maximum']=max(rec['maximum'],x.max().item());rec['count']+=x.numel()
  y=super().apply(name,family,x,layout,skip_quant)
  if active:
   if name.endswith('residual_mid'):self.shared_mid[int(name[1:3])]=y
   if self.diagnostics is not None:
    mse,rel,nerr=errors(x,y,norm_meta(self.model,name));v=self.diagnostics.setdefault(name,dict(count=0,squared_error=0.,energy=0.,normalized_norm_error_weighted=0.,clipped=0))
    count=x.numel();v['count']+=count;v['squared_error']+=mse*count;v['energy']+=float(x.float().square().sum());v['normalized_norm_error_weighted']+=nerr*count
    p=self.params[name]['mse'];v['clipped']+=int(((x<p['lo'])|(x>p['hi'])).sum())
  return y

def grid(x,original):
 vals=x.cpu().numpy().astype(np.float32).reshape(-1);negative=-vals[vals<0];positive=vals[vals>0];out=[copy.deepcopy(original)]
 for ql in [.99,.999,.9999,1.]:
  for qh in [.99,.999,.9999,1.]:
   lo=-float(np.quantile(negative,ql)) if len(negative) else 0.;hi=float(np.quantile(positive,qh)) if len(positive) else 0.
   p=a.qparams(lo,hi)
   for dz in [-2,0,2]:
    v=copy.deepcopy(p);v['zero']=max(0,min(255,p['zero']+dz));v['lo']=-v['zero']*v['scale'];v['hi']=(255-v['zero'])*v['scale'];v['grid']=dict(negative_coverage=ql,positive_coverage=qh,zero_offset=dz)
    if not any(all(t[k]==v[k] for k in ['scale','inv_scale','zero']) for t in out):out.append(v)
 return out

def calibrate():
 model,before,mh=r.b.model_session('C64');ins=Instrument(model);ins.build_prefix([151645]);ins.select('B__wide_nr64');ins.gather=True
 rows=read('inputs.json')['calibration'];vectors={n:[] for n in SITES};prefix=r.b.prefix_digest(ins)
 for j in range(0,128,4):
  x=torch.tensor([q['token_ids'] for q in rows[j:j+4]],device='cuda');cache=ins.new_cache(4)
  model.model(input_ids=x[:,:64],past_key_values=cache,use_cache=True)
  for t in range(64,79):model.model(input_ids=x[:,t:t+1],past_key_values=cache,use_cache=True)
  assert cache.get_seq_length()==80 and len(cache.appended)==28*16
  for n in SITES:
   v=torch.cat(ins.parts[n],1);assert v.shape[1]==79;vectors[n].append(v[:,POSITIONS].reshape(-1,v.shape[-1]))
  ins.parts={};print('CALIBRATION',j+4,128,flush=True)
 ins.gather=False;assert prefix==r.b.prefix_digest(ins)
 expected=32*16;assert ins.calls=={n:expected for n in SITES};assert ins.shared_norm==28*expected
 base=read('inputs.json')['parameters']['R3_ALL'];params={n:copy.deepcopy(base) for n in CANDIDATES[1:]};fits={};arrays={}
 for name in SITES:
  x=torch.cat(vectors[name]);assert x.shape[0]==1024;arrays[name]=x.numpy();x=x.cuda();meta=norm_meta(model,name);candidates=grid(x,base[name]['mse']);metrics=[]
  for p in candidates:
   mse,rel,ne=errors(x,a.qdq(x,p),meta);metrics.append(dict(mse=mse,relative_mse=rel,norm_relative_mse=ne,fanout=rel+ne))
  m=min(range(len(candidates)),key=lambda i:(metrics[i]['mse'],i));f=min(range(len(candidates)),key=lambda i:(metrics[i]['fanout'],i))
  if name.endswith('swiglu'):params['S_MSE'][name]['mse']=candidates[m]
  else:params['R_MSE'][name]['mse']=candidates[m];params['R_FAN'][name]['mse']=candidates[f]
  fits[name]=dict(candidates=candidates,errors=metrics,mse_index=m,fanout_index=f,full_trajectory=ins.extrema[name])
  if meta is not None:arrays[name+'/gamma']=meta[0].cpu().numpy();fits[name]['epsilon']=meta[1]
  print('FIT',name,len(candidates),m,f,flush=True)
 path=ROOT/'calibration_vectors.npz';assert not path.exists();np.savez_compressed(path,**arrays)
 write('calibration.json',dict(sites=fits,ids=[x['id'] for x in rows],positions=POSITIONS,vectors_per_site=1024,all_trajectory_tokens=128*79,attention='wide_nr64',parameters_fixed_during_collection=True,vectors_sha256=data.sha(path),source_head=r.b.head()))
 for candidate,p in params.items():
  allowed={n for n in SITES if n.endswith('swiglu')==(candidate=='S_MSE')};changed=[n for n in base if p[n]!=base[n]];assert set(changed)<=allowed
  write(f'parameters/{candidate}.json',dict(parameters=p,changed_sites=changed,allowed_sites=sorted(allowed),original_other_sites_exact=True))
 assert before==a.state_digest(model);write('checks/calibration_weights.json',dict(unchanged=True,digest=before,manifest_sha256=mh,prefix_immutable=True,shared_norm_calls=ins.shared_norm,site_calls=ins.calls))
 ins.close();write('parameters_freeze.json',dict(files={str(p.relative_to(ROOT)):data.sha(p) for p in [ROOT/'calibration.json',ROOT/'calibration_vectors.npz']+[ROOT/f'parameters/{n}.json' for n in CANDIDATES[1:]]},before_development=True))

def parameter_check():
 for n,h in read('parameters_freeze.json')['files'].items():assert data.sha(ROOT/n)==h

def numerical():
 parameter_check();cal=read('calibration.json');vectors=np.load(ROOT/'calibration_vectors.npz');proof=[]
 for name,fit in cal['sites'].items():
  x=vectors[name].astype(np.float32)
  for idx in set([0,fit['mse_index'],fit['fanout_index']]):
   p=fit['candidates'][idx];y=a.qdq_np(x,p).astype(np.float16).astype(np.float32);mse=float(np.mean((x-y)**2,dtype=np.float64));rel=mse/max(float(np.mean(x*x,dtype=np.float64)),1e-30);ne=0.
   if not name.endswith('swiglu'):
    gamma=vectors[name+'/gamma'];eps=fit['epsilon']
    def nn(v):return v/np.sqrt(np.mean(v*v,axis=-1,keepdims=True)+eps)*gamma
    ref=nn(x);ne=float(np.mean((nn(y)-ref)**2,dtype=np.float64)/max(np.mean(ref*ref,dtype=np.float64),1e-30))
   for k,v in [('mse',mse),('relative_mse',rel),('norm_relative_mse',ne)]:assert abs(v-fit['errors'][idx][k])<=max(1e-10,abs(v)*1e-5),(name,idx,k,v,fit['errors'][idx][k])
   # FP16 input halfway neighbours, exact unchanged FP32 arithmetic oracle.
   half=(np.arange(256,dtype=np.float32)-p['zero']+.5)*np.float32(p['scale']);h=half.astype(np.float16)
   inp=np.concatenate([h,np.nextafter(h,np.float16(-np.inf)),np.nextafter(h,np.float16(np.inf)),np.array([-65504,0,65504],np.float16)])
   expected=a.qdq_np(inp,p).astype(np.float16);actual=a.qdq(torch.from_numpy(inp).cuda(),p).cpu().numpy();assert np.array_equal(actual,expected),(name,idx,'U8 oracle')
   proof.append(dict(site=name,index=idx,mse=mse,relative_mse=rel,norm_relative_mse=ne,U8_FP16_exact=True))
 write('checks/numerical.json',dict(pass_all=True,objective_and_rounding=proof,retained_cores=read('freeze.json')['references']))
 print('NUMERICAL PASS',len(proof),flush=True)

def score(model,ins,name,rows,phase):
 file=f'scores/{phase}_{name}.json'
 if (ROOT/file).exists():return old.score(model,ins,name,rows,phase)
 ins.calls={};ins.shared_norm=0;result=old.score(model,ins,name,rows,phase)
 if name not in ['F','C64']:
  expected=(len(rows)//4+3)*16
  assert ins.calls=={n:expected for n in SITES},(phase,name,'site counts');assert ins.shared_norm==expected*28
 write(f'checks/{phase}_{name}_bindings.json',dict(parameters_sha256=data.sha(ROOT/'parameters_freeze.json'),selection_sha256=data.sha(ROOT/'selection.json') if phase=='final' else None,site_calls=ins.calls,shared_norm_calls=ins.shared_norm,shared_fanout_exact=True))
 return result

def run(phase):
 parameter_check();assert read('checks/numerical.json')['pass_all']
 if phase=='reproduction':rows=read('inputs.json')['development'];names=['F','C64','B__float_core','B__wide_nr64']
 else:
  assert read('reproduction_complete.json')['pass_all'];rows=[x for x in read('dataset.json')['samples'] if x['split']==phase]
  if phase=='development':names=[c+'__'+m for c in CANDIDATES for m in MODES]
  else:
   selected=read('selection.json')['selected'];names=['F','C64']+list(dict.fromkeys(c+'__'+m for c in ['B',selected,'S_MSE'] for m in MODES))
 for recipe in ['C64','F']:
  if not any((n=='F')==(recipe=='F') for n in names):continue
  proof=f'checks/{phase}_{recipe}_weights.json'
  if (ROOT/proof).exists():continue
  model,before,mh=r.b.model_session(recipe);ins=Instrument(model);ins.build_prefix([151645])
  for name in names:
   if (name=='F')!=(recipe=='F'):continue
   d=score(model,ins,name,rows,phase)
   if phase=='reproduction':
    pn='R3_'+name.split('__')[1] if '__' in name else name;p=data.verified('exp0253',f'scores/development_{pn}.json');ref=json.loads(p.read_text());assert d['samples']==ref['samples'] and d['ppl']==ref['ppl'],name
    write(f'checks/reproduction_{name}.json',dict(exact=True,parent_sha256=data.sha(p)))
   if phase=='development':
    ins.select(name);ins.diagnostics={};r.forward(model,ins,rows[:4]);write(f'diagnostics/{name}.json',dict(sites=ins.diagnostics,own_candidate_trajectory=True,first_four_development_ids=[x['id'] for x in rows[:4]],no_fitting=True));ins.diagnostics=None
  assert before==a.state_digest(model);write(proof,dict(unchanged=True,digest=before,manifest_sha256=mh,source_head=r.b.head()));ins.close();del model,ins;torch.cuda.empty_cache()
 write(f'{phase}_complete.json',dict(pass_all=True,names=names))
 if phase=='development':
  scores={c:{m:read(f'scores/development_{c}__{m}.json')['mean_nll'] for m in MODES} for c in CANDIDATES};eligible=CANDIDATES[:3]
  worst={c:max(scores[c][m]-scores['B'][m] for m in MODES) for c in eligible};selected=min(eligible,key=lambda c:(round(worst[c],6),eligible.index(c)))
  write('selection.json',dict(selected=selected,worst_mode_deltaNLL=worst,development_scores=scores,rule='minimum_worst_mode_deltaNLL_tie_B_then_R_MSE_then_R_FAN',parameters_sha256=data.sha(ROOT/'parameters_freeze.json'),final_not_used=True,source_head=r.b.head()))
  print('SELECTED',selected,worst,flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','calibrate','numerical','reproduction','development','final']);args=parser.parse_args();preflight();a.settings();bind()
 with torch.inference_mode():
  if args.phase=='freeze':freeze()
  else:
   frozen()
   if args.phase=='calibrate':calibrate()
   elif args.phase=='numerical':numerical()
   else:run(args.phase)
if __name__=='__main__':main()
