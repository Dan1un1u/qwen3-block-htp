#!/usr/bin/env python3
import sys,json,math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
import attention_exp0249 as a
R=a.ROOT

def trace(model,ins):
 records={};origin=a.ia.core
 for name in ['OFF_carrier','R3_carrier']:
  ins.select(name);calls=[0];rows=[];directory=R/'trace'/name;directory.mkdir(parents=True,exist_ok=False)
  def wrapped(q,k,v,valid,c,mode='sole',real_scale=None):
   result=origin(q,k,v,valid,c,mode,real_scale)
   call=calls[0];calls[0]+=1;layer=call%28;step=call//28
   acc=result['acc'].cpu().numpy();score=result['score'].cpu().numpy();raw=result['raw'].cpu().numpy();mask=np.broadcast_to(valid.cpu().numpy(),acc.shape)
   logits=acc*real_scale;compressed=(score-128)*(math.log(2)/8)
   def softmax(x):
    x=np.where(mask,x,-np.inf);e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
   p=softmax(logits);z=softmax(compressed);diff=p-z
   before=(raw-128)*c['sm']+128
   row=dict(layer=layer,step=step,query_rows=q.shape[0]*q.shape[1]*q.shape[2],valid_entries=int(mask.sum()),
    raw_endpoint_entries=int((((raw==0)|(raw==255))&mask).sum()),scaled_clipped_entries=int((((before<0)|(before>255))&mask).sum()),
    row_max_clipped=int(((np.where(mask,before,-np.inf).max(-1)>255)).sum()),probability_L1_sum=float(np.abs(diff).sum()),probability_squared_error=float(np.square(diff).sum()),
    max_probability_abs=float(np.abs(diff).max()),raw_score_multiplier=c['sm'],raw_score_natural_logit_step=c['sm']*math.log(2)/8,
    logits_min=float(logits[mask].min()),logits_max=float(logits[mask].max()),compressed_min=float(compressed[mask].min()),compressed_max=float(compressed[mask].max()))
   rows.append(row)
   if step==0:
    np.savez_compressed(directory/f'L{layer:02d}.npz',acc=acc,raw=raw,score=score,valid=mask,real_scale=real_scale)
   return result
  a.ia.core=wrapped
  before=a.r.b.prefix_digest(ins)
  try:logits,nll,_=a.r.forward(model,ins,a.read('inputs.json')['development'][:4])
  finally:a.ia.core=origin
  assert calls[0]==448 and before==a.r.b.prefix_digest(ins)
  old=a.read('scores/development_'+name+'.json')['samples'][:4]
  assert nll.cpu().tolist()==[x['nll'] for x in old] and logits.argmax(-1).cpu().tolist()==[x['top1'] for x in old]
  summaries=[]
  for i in range(28):
   xs=[x for x in rows if x['layer']==i];nr=sum(x['query_rows'] for x in xs);ne=sum(x['valid_entries'] for x in xs)
   summaries.append(dict(layer=i,raw_endpoint_fraction=sum(x['raw_endpoint_entries'] for x in xs)/ne,
    scaled_clipped_fraction=sum(x['scaled_clipped_entries'] for x in xs)/ne,row_max_clipped_fraction=sum(x['row_max_clipped'] for x in xs)/nr,
    mean_row_probability_L1=sum(x['probability_L1_sum'] for x in xs)/nr,probability_RMSE=math.sqrt(sum(x['probability_squared_error'] for x in xs)/ne),
    max_probability_abs=max(x['max_probability_abs'] for x in xs),raw_score_natural_logit_step=xs[0]['raw_score_natural_logit_step']))
  records[name]=dict(rows=rows,layers=summaries,trace_did_not_change_NLL_or_top1=True)
  print('TRACE',name,'highest probability L1',sorted(summaries,key=lambda x:x['mean_row_probability_L1'],reverse=True)[:3],flush=True)
 a.write('score_grid_trace.json',dict(scope='first four exposed development documents on uncompressed carrier trajectories; paired same-input score maps, no candidate or final selection',variants=records,
  source_head=a.r.b.head(),freeze_sha256=a.data.sha(R/'freeze.json')))

def main():
 a.preflight();a.frozen();a.r.b.a.settings()
 assert a.read('development_complete.json')['pass_all']
 with torch.inference_mode():
  model,before,mh=a.r.b.model_session('C64');ins=a.Instrument(model);ins.build_prefix([151645]);trace(model,ins)
  assert before==a.r.b.a.state_digest(model)
  a.write('checks/trace_weights.json',dict(unchanged=True,digest=before,manifest_sha256=mh));ins.close()
if __name__=='__main__':main()
