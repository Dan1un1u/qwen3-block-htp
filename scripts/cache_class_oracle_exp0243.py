from pathlib import Path
import sys
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
import prefix_exp0243 as p
from types import SimpleNamespace
import torch
p.preflight()
q=p.a.qparams(-4,4)
ins=SimpleNamespace(policy='minmax',params={'L00.k_cache':{'minmax':q},'L00.v_out':{'minmax':q}},observe_kv=lambda *args:None)
torch.manual_seed(243)
k=torch.randn(2,2,12,8);v=torch.randn_like(k)
cache=p.CarrierCache(ins)
cache.update(k[:,:,:3],v[:,:,:3],0,{'prefix_seed':True})
seed=cache.key_cache[0].clone()
for i in range(3,12):kd,vd=cache.update(k[:,:,i:i+1],v[:,:,i:i+1],0)
assert cache.get_seq_length()==12
assert len(cache.appended)==9 and cache.seed_lengths==[(0,3)]
assert torch.equal(cache.key_cache[0],p.codes(k,q))
assert torch.equal(cache.value_cache[0],p.codes(v,q))
assert torch.equal(cache.key_cache[0][:,:,:3],seed)
assert torch.equal(kd,p.decode(p.codes(k,q),q,k.dtype))
assert torch.equal(vd,p.decode(p.codes(v,q),q,v.dtype))
assert cache.key_cache[0].dtype==cache.value_cache[0].dtype==torch.uint8
p.write('checks/cache_class_oracle.json',dict(seed_unchanged=True,append_codes_exact=True,decoded_reads_exact=True,storage='uint8',seed_length=3,final_length=12,append_calls=9))
print('CACHE_CLASS_ORACLE_PASS')
