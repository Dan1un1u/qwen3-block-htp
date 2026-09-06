#!/usr/bin/env python3
"""Independent full-model check of official calibration teacher/cache and exported student."""
import json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch,numpy as np
from pathlib import Path
from data_exp0233 import RESULT,OUTPUT,write,verified,preflight,sha
from autoround_exp0233 import frozen
from evaluate_exp0233 import load
from experiment_exp0220 import metrics

def main():
 preflight();frozen();torch.set_num_threads(8);torch.set_grad_enabled(False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.use_deterministic_algorithms(True)
 rows=[r for r in json.loads(verified('exp0230','dataset.json').read_text())['samples'] if r['split']=='calibration'][:4]
 ids=torch.tensor([r['token_ids'] for r in rows],device='cuda');roots={v:Path(json.loads((RESULT/v/'package.json').read_text())['root']) for v in ['AR-P','AR-G']}
 checks=[]
 for v in ['F','AR-P','AR-G']:
  model,h=load(v,'cuda');handles=[]
  def capture(i):
   def hook(mod,args,out):
    actual=out[0] if isinstance(out,tuple) else out
    target=roots['AR-P'] if v=='F' else roots[v];label='teacher' if v=='F' else 'student'
    stored=np.load(target/f'layer{i}/{label}_first4.npy')
    if v=='F':assert np.array_equal(stored,np.load(roots['AR-G']/f'layer{i}/teacher_first4.npy'))
    m=metrics(actual,torch.from_numpy(stored).to(actual.device))
    assert m['finite'] and m['nrmse']<=.003 and m['cosine']>=.99999,(v,i,m)
    checks.append(dict(variant=v,layer=i,**m))
   return hook
  for i,layer in enumerate(model.model.layers):handles.append(layer.register_forward_hook(capture(i)))
  with torch.inference_mode():model(input_ids=ids,use_cache=False)
  for h in handles:h.remove()
  del model;torch.cuda.empty_cache();print('FULL_MODEL_CALIBRATION_AUDIT',v,flush=True)
 assert len(checks)==84
 write('full_model_calibration_audit.json',dict(pass_all=True,checks=checks,teacher_branches_exact=True,independent_full_model_not_cached_blocks=True,threshold_nrmse=.003,threshold_cosine=.99999))
if __name__=='__main__':main()
