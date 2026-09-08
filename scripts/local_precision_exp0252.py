from pathlib import Path
import json,hashlib,math
import numpy as np
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0252');P=R.parent/'exp0249'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ledger=P/'EVIDENCE_SHA256.json';assert sha(ledger)=='559391a5579a572e5ef2bca17017b4d2bd69e3a20036720160aab6696034e689'
hashes=json.loads(ledger.read_text())['files'];ins=json.loads((R/'inputs.json').read_text());refs={str(ledger):sha(ledger)};results={}
for arm,param in [('OFF','A8'),('R3','R3_ALL')]:
 layers=[]
 for layer in range(28):
  p=P/f'trace/{arm}_carrier/L{layer:02d}.npz';assert sha(p)==hashes[str(p.relative_to(P))]['sha256'];refs[str(p)]=sha(p)
  z=np.load(p);raw=z['raw'];mask=z['valid'];mult=ins['configs'][param][layer]['sm']
  logits=np.where(mask,(raw-128)*mult*math.log(2)/8,-np.inf);exp=np.exp(logits-logits.max(-1,keepdims=True));ideal=exp/exp.sum(-1,keepdims=True)
  maximum=np.where(mask,raw,-1).max(-1,keepdims=True);delta=(maximum-raw)*mult;ex=np.clip((delta+4)//8,0,15)
  weight=np.where(mask,np.left_shift(1,15-ex),0);total=weight.sum(-1,keepdims=True)
  exact=(weight*255+total//2)//total
  lead=np.floor(np.log2(total)).astype(np.int64);bit=np.right_shift(total,np.maximum(0,lead-1))&1;coefficient=np.where((lead>0)&(bit!=0),145,209);shift=ex+lead-15
  numerator=np.left_shift(255*coefficient,np.maximum(0,-shift));denominator=np.left_shift(np.full_like(shift,256),np.maximum(0,shift))
  sole=(numerator+denominator//2)//denominator
  xx=total<<(30-lead);idx=(xx-2**30)>>24;dd=129+2*idx;rr=(2**37+dd//2)//dd;rr=(rr*(2**31-((xx*rr)>>30)))>>30
  nr64=(weight*255*rr+(np.ones_like(total)<<(lead+29)))>>(lead+30)
  single=mask.sum(-1,keepdims=True)==1
  methods={'wide_float':np.floor(ideal*255+.5),'wide_exact':np.where(single,255,exact),'wide_sole':np.where(single,255,sole),'wide_nr64':np.where(single,255,nr64)};metrics={}
  for name,code in methods.items():
   prob=np.clip(code,0,255)*mask/255;err=prob-ideal;mass=prob.sum(-1)
   metrics[name]=dict(mean_row_L1=float(np.abs(err).sum(-1).mean()),valid_RMSE=float(np.sqrt(np.mean(err[mask]**2))),row_mass_mean=float(mass.mean()),row_mass_min=float(mass.min()),row_mass_max=float(mass.max()),row_mass_p01_p50_p99=np.quantile(mass,[.01,.5,.99]).tolist())
  layers.append(dict(layer=layer,rows=int(raw.shape[0]*raw.shape[1]*raw.shape[2]),methods=metrics))
 results[arm]=layers
out=dict(scope='CPU-only replay of all28 retained EXP0249 exposed-dev4 carrier prefill trajectories; identical inputs within each comparison, no new model run or tuning. Probability normalized by code255 here; inherited AV scale discrepancy remains in PPL.',references=refs,variants=results)
with (R/'local_nr64_precision.json').open('x') as f:json.dump(out,f,indent=2)
for arm,layers in results.items():
 print(arm,{m:dict(mean_L1=float(np.mean([l['methods'][m]['mean_row_L1'] for l in layers])),mean_mass=float(np.mean([l['methods'][m]['row_mass_mean'] for l in layers]))) for m in ['wide_float','wide_exact','wide_sole','wide_nr64']})
