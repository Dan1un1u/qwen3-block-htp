#!/usr/bin/env python3
"""Frozen head-only comparisons, exact control regression and whole-state audits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc,hashlib,json,math,subprocess,time
import numpy as np
import torch
from data_exp0236 import RESULT,OUTPUT,SOURCE,MODEL,BASE,CELLS,write,sha,verified,preflight,frozen,GROOT
from head_exp0236 import head_read
from evaluate_exp0234 import load as old_load
from evaluate_exp0235 import digest,tensors

def main():
 preflight();data=frozen();assert json.loads((RESULT/'independent_data_audit.json').read_text())['pass_all']
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
 torch.set_num_threads(8);torch.set_grad_enabled(False);torch.manual_seed(236)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.use_deterministic_algorithms(True)
 packages={}
 for v in ['P64','H64']:
  m=json.loads((RESULT/v/'package.json').read_text());root=OUTPUT/v;assert sha(root/'manifest.json')==m['manifest_sha256']
  for n,e in m['files'].items():assert sha(root/n)==e['sha256']
  assert m['frozen_G64_manifest_sha256']==sha(GROOT/'manifest.json') and m['only_changed_tensor']=='lm_head.weight' and m['W4'] and not m['mixed_precision']
  packages[v]=m
 write('exports_frozen_before_final.json',dict(manifests={v:m['manifest_sha256'] for v,m in packages.items()},source_head=source,no_further_tuning=True))
 dev=json.loads(verified('exp0230','dataset.json').read_text());old=json.loads(verified('exp0233','dataset.json').read_text())
 datasets={'development':[r for r in dev['samples'] if r['split']=='development'],'final':data['samples'],'PC052':old['samples']}
 hashes={'development':sha(BASE/'exp0230/dataset.json'),'final':sha(RESULT/'dataset.json'),'PC052':sha(BASE/'exp0233/dataset.json')}
 audits=[]
 def score(phase,v,sentinel=False):
  rows=datasets[phase];name=f'software/{phase}_{v}.json' if not sentinel else 'G64_final_sentinel.json';assert not (RESULT/name).exists()
  def forward(batch):
   x=torch.tensor([r['prompt_ids']+r['target_ids'] for r in batch],device='cuda')
   z=model(input_ids=x[:,:79],use_cache=False).logits[:,63:79,:].float();assert torch.isfinite(z).all()
   nll=torch.logsumexp(z,-1)-z.gather(-1,x[:,64:80,None]).squeeze(-1);return x,z,nll
  started=time.monotonic()
  with torch.inference_mode():
   x,z,nll=forward(rows[:4]);_,z2,nll2=forward(rows[:4]);assert torch.equal(z,z2) and torch.equal(nll,nll2)
   ce=torch.nn.functional.cross_entropy(z.reshape(-1,z.shape[-1]),x[:,64:80].reshape(-1),reduction='none').reshape(4,16);error=float((ce-nll).abs().max());assert error<5e-6,error
   changed=x[:,:79].clone();changed[:,70:]=123;other=model(input_ids=changed,use_cache=False).logits[:,63:70,:].float();assert torch.equal(z[:,:7],other)
   output=[]
   for start in range(0,len(rows),4):
    batch=rows[start:start+4];_,z,nll=forward(batch);assert torch.isfinite(nll).all()
    for row,loss,top in zip(batch,nll.cpu().tolist(),z.argmax(-1).cpu().tolist()):output.append(dict(id=row['id'],cell=row['cell'],nll=loss,top1=top))
    if start%256==0:print('HEAD_SCORE_PROGRESS',phase,v,start+4,len(rows),flush=True)
  regression=None
  if phase=='development' and v in ['F','G64']:
   exp='exp0230' if v=='F' else 'exp0234';prev=json.loads(verified(exp,f'software/development_{v}.json').read_text());assert output==prev['samples'],('control regression',v);regression=True
  if sentinel:assert output==json.loads((RESULT/'software/development_G64.json').read_text())['samples']
  means={c:math.fsum(x for row in output if row['cell']==c for x in row['nll'])/(16*sum(row['cell']==c for row in output)) for c in CELLS};nll=math.fsum(x for row in output for x in row['nll'])/(16*len(rows))
  write(name,dict(variant=v,phase=phase,role='canonical_FP16_software_W4_head_only',samples=output,cell_nll=means,mean_nll=nll,ppl=math.exp(nll),repeat_exact=True,causal_mask_exact=True,independent_CE_max_abs=error,all_logits_finite=True,exact_prior_development_regression=regression,dataset_sha256=hashes[phase],torch_version=torch.__version__,source_head=source,elapsed_s=time.monotonic()-started))
  print('HEAD_SCORE_COMPLETE',phase,v,math.exp(nll),'sentinel',sentinel,flush=True)
 # Hash-verified same-token/backend PC052 controls, no rerun or altered old files.
 for v in ['F','G64']:
  samples=[];refs={};ce_values=[]
  for phase in ['primary','reserve']:
   p=verified('exp0234',f'software/{phase}_{v}.json');r=json.loads(p.read_text());assert r['repeat_exact'] and r['causal_mask_exact'] and r['independent_CE_max_abs']<5e-6 and r['dataset_sha256']==hashes['PC052'];samples+=r['samples'];refs[str(p)]=sha(p);ce_values.append(r['independent_CE_max_abs'])
  assert [r['id'] for r in samples]==[r['id'] for r in old['samples']]
  write(f'software/PC052_{v}.json',dict(variant=v,phase='PC052',samples=samples,dataset_sha256=hashes['PC052'],reused_verified_controls=refs,repeat_exact=True,causal_mask_exact=True,independent_CE_max_abs=max(ce_values),note='CE maximum from retained original runs; no new CE measurement'))
 model,fh=old_load('F','cuda');score('development','F');score('final','F');del model;gc.collect();torch.cuda.empty_cache()
 model,gh=old_load('G64','cpu');baseline={n:digest(t) for n,t in tensors(model).items()};original_head=model.lm_head.weight.detach().clone();assert model.lm_head.weight.data_ptr()!=model.model.embed_tokens.weight.data_ptr()
 write('G64_state_manifest.json',dict(manifest_sha256=gh,state_hashes=baseline,only_mutable_tensor='lm_head.weight',source_head=source))
 model=model.to('cuda').eval()
 def audit(v):
  actual={n:digest(t) for n,t in tensors(model).items()};expected=dict(baseline)
  if v!='G64':expected['lm_head.weight']=packages[v]['dequant_FP16_sha256']
  assert actual==expected,('head-only whole-state mismatch',v)
  audits.append(dict(variant=v,all_parameter_and_buffer_hashes_match=True,state_hashes=actual))
 audit('G64');score('development','G64');score('final','G64')
 for v in ['P64','H64']:
  q=head_read(OUTPUT/v,v=='H64');assert hashlib.sha256(q.tobytes()).hexdigest()==packages[v]['dequant_FP16_sha256'];model.lm_head.weight.copy_(torch.from_numpy(q));del q
  audit(v)
  for phase in ['development','final','PC052']:score(phase,v)
 model.lm_head.weight.copy_(original_head);audit('G64');score('development','G64',sentinel=True)
 write('head_state_audits.json',dict(pass_all=True,audits=audits,final_G64_sentinel_exact=True,only_head_changed=True))
 write('environment.json',dict(torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),source_head=source,TF32=False,FP16_reduced_precision_reduction=False,deterministic=True,CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],batch_size=4,use_cache=False))
 print('EXP236_ALL_HEAD_EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()
