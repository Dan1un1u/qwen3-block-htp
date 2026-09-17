#!/usr/bin/env python3
"""Add independently generated step1 cache evidence in a fresh fixture revision."""
from prepare_exp0289 import *
import shutil
@torch.inference_mode()
def main(recipe):
 preflight();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 base=O/recipe;manifest=json.loads((base/'manifest.json').read_text())
 for n,v in manifest['files'].items():assert sha256(base/n)==v['sha256'],n
 ids=np.fromfile(base/'generation_prompt_token_ids_u32.bin','<u4')
 token=json.loads((R/(recipe+'-reference.json')).read_text())['fixed_input_tokens'][0]
 emb=tensor('model.embed_tokens.weight').half().cuda()
 x=F.embedding(torch.tensor(ids.astype('i8'),device='cuda'),emb);dx=emb[token][None];caches=[]
 for i in range(28):
  w={}
  for n,k in PROJECTIONS.items():
   shape=tuple(tensor(f'model.layers.{i}.{k}.weight').shape)
   w[n]=(unpack_w4_weight(base/f'layer{i}',n,*shape) if recipe=='w4f16' else tensor(f'model.layers.{i}.{k}.weight').half()).cuda()
  for n,k in [('input','input_layernorm'),('post','post_attention_layernorm'),('qn','self_attn.q_norm'),('kn','self_attn.k_norm')]:w[n]=tensor(f'model.layers.{i}.{k}.weight').half().cuda()
  x,past=layer(x,w,0);dx,cache=layer(dx,w,64,past);caches.append(tuple(t.cpu() for t in cache))
 for name,first,count in [('layer0',0,1),('layer14',14,1),('layer27',27,1),('chain3',0,3)]:
  src=O/(recipe+'-'+name);dst=O/(recipe+'-'+name+'-a02')
  m=json.loads((src/'manifest.json').read_text());dst.mkdir(exist_ok=False)
  for n,v in m['files'].items():
   assert sha256(src/n)==v['sha256'];p=dst/n;p.parent.mkdir(parents=True,exist_ok=True);os.link(src/n,p)
  for j in range(count):
   for k,n in enumerate(['k','v']):put(dst/f'layer{j}/reference_kv_cache_{n}_hmx_f16_step01.bin',cache_carrier(caches[first+j][k],n))
  seal(dst,recipe=recipe,layers=count,first_original_layer=first,cache_capacity=128,decode_steps=1,reference='independent two-step floating reference; added missing journal boundary; prior immutable package retained')
 print('REPLAY_A02_COMPLETE',recipe,flush=True)
if __name__=='__main__':main(sys.argv[1])
