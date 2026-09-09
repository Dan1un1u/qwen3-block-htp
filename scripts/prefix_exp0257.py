#!/usr/bin/env python3
"""Export existing frozen offline EOS prefix semantics; no fitting."""
from export_exp0257 import *
import ablate_exp0244 as b
from transformers.cache_utils import DynamicCache

def main():
 preflight();model,before,mh=b.model_session('C64');ins=b.Instrument(model);raw=ins.build_prefix([151645]);again=ins.build_prefix([151645])
 assert all(torch.equal(x,y) for p,q in zip(raw,again) for x,y in zip(p,q))
 raw=[(k.detach().clone(),v.detach().clone()) for k,v in raw];ins.close()
 # Independent original, unhooked model pass confirms all stored FP16 K/V.
 cache=DynamicCache()
 with torch.no_grad():model.model(input_ids=torch.tensor([[151645]],device='cuda'),past_key_values=cache,use_cache=True)
 assert all(torch.equal(x,y) for i,p in enumerate(raw) for x,y in zip(p,[cache.key_cache[i],cache.value_cache[i]]))
 params=json.loads((P/'parameters/A8.json').read_text())['parameters'];payload=[];fp=[]
 for i,(k,v) in enumerate(raw):
  for short,x in [('k_cache',k),('v_cache',v)]:
   pp=params[f'L{i:02d}.{short}']['mse'];u=torch.floor(x.float()*pp['inv_scale']+pp['zero']+.5).clamp(0,255).to(torch.uint8).cpu().numpy().reshape(8,128)
   oracle=qdqcode(x.cpu().numpy().reshape(8,128),qp(pp));assert np.array_equal(u,oracle)
   payload.append(u.tobytes());fp.append(x.cpu().numpy().tobytes())
 assert before==b.a.state_digest(model)
 d=O/'prefix';d.mkdir(exist_ok=False);(d/'prefix_kv_u8.bin').write_bytes(b''.join(payload));(d/'prefix_kv_f16.bin').write_bytes(b''.join(fp))
 write(R/'prefix_export.json',dict(pass_all=True,prefix_token=151645,source='frozen C64 W4A16 offline prefix; original model unhooked pass exact',bytes=57344,sha256=sha(d/'prefix_kv_u8.bin'),fp16_sha256=sha(d/'prefix_kv_f16.bin'),weights_digest=before,manifest_sha256=mh))
 print('PREFIX_EXPORT_PASS',flush=True)
if __name__=='__main__':main()
