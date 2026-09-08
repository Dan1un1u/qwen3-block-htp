"""Same-shape storage oracle, separate from FP16 bulk/sequential sensitivity."""
import sys
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
import prefix_exp0243 as p
from transformers import DynamicCache
import torch,types

class FloatOracleCache(DynamicCache):
 def __init__(self,ins):super().__init__();self.ins=ins;self.appended=[]
 def update(self,k,v,layer,cache_kwargs=None):
  # Independent expression + ordinary floating DynamicCache; identical quantization points.
  if not (cache_kwargs and cache_kwargs.get('prefix_seed')):self.appended.append((layer,int(k.shape[-2])))
  k=p.a.qdq(k,self.ins.params[f'L{layer:02d}.k_cache'][self.ins.policy])
  v=p.a.qdq(v,self.ins.params[f'L{layer:02d}.v_out'][self.ins.policy])
  return super().update(k,v,layer,cache_kwargs)

def main():
 p.preflight();p.a.settings()
 with torch.inference_mode():
  model,_=p.a.canonical.load('C64','cuda');before=p.a.state_digest(model);ins=p.Instrument(model)
  ins.params=p.read('calibration/eos.json')['parameters'];ins.configure('mse',p.a.FAMILIES);ins.build_prefix([151645])
  batch=p.rows('development')[:4]
  actual,actual_nll,integer=p.forward(model,ins,batch,'sequential')
  body=torch.tensor([r['token_ids'] for r in batch],device='cuda')
  cache=FloatOracleCache(ins)
  for i,(k,v) in enumerate(ins.prefix_raw):cache.update(k.expand(4,-1,-1,-1).clone(),v.expand(4,-1,-1,-1).clone(),i,{'prefix_seed':True})
  parts=[model(input_ids=body[:,:64],past_key_values=cache,use_cache=True).logits[:,-1:].float()]
  for j in range(15):parts.append(model(input_ids=body[:,64+j:65+j],past_key_values=cache,use_cache=True).logits[:,-1:].float())
  oracle=torch.cat(parts,1)
  assert torch.equal(actual,oracle),(actual-oracle).abs().max().item()
  for i,(k,v) in enumerate(zip(integer.key_cache,integer.value_cache)):
   assert torch.equal(p.decode(k,ins.params[f'L{i:02d}.k_cache']['mse'],torch.float16),cache.key_cache[i])
   assert torch.equal(p.decode(v,ins.params[f'L{i:02d}.v_out']['mse'],torch.float16),cache.value_cache[i])
  assert before==p.a.state_digest(model)
  p.write('checks/cache_model_oracle.json',dict(same_shape_logits_exact=True,all28_decoded_KV_exact=True,weights_unchanged=True,documents=4,targets=64,prefix='eos',policy='mse',actual_storage='uint8',oracle_storage='float16_QDQ_reference_only',interpretation='verifies_U8_storage_implementation_not_bulk_sequential_FP16_invariance'))
  print('FULL_MODEL_CACHE_STORAGE_ORACLE_PASS',flush=True)
if __name__=='__main__':main()
