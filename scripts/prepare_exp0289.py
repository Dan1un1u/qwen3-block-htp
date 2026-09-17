#!/usr/bin/env python3
"""Original-derived Qwen0.6 A16 fixtures; independent floating reference."""
import sys,os,json,struct,argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from llama_reference import pack_weight,sha256,PROJECTIONS
from prepare_exp0288 import tensor,original,rope,MODEL
from prepare_exp0042_block import unpack_w4_weight
from common_exp0289 import preflight,R,O,write
H=1024;A=2048;D=128;KH=8;NH=16;CAP=128
Q=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0288')
SP=Q/'frontend64-a03'
def put(p,a):
 if p.exists():raise FileExistsError(p)
 if isinstance(a,torch.Tensor):a=a.detach().cpu().numpy()
 np.asarray(a).tofile(p)
def padded(x):
 a=np.zeros((64,H),'<f2');a[:x.shape[0]]=x.detach().cpu().numpy();return a
def rms(x,w):
 return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*w.float()).half()
def linear(x,w):return F.linear(x.float(),w.float()).half()
def rot(x,c,s):
 y=x.float();half=y.shape[-1]//2
 return (y*c.float()[:,None,:]+torch.cat((-y[...,half:],y[...,:half]),-1)*s.float()[:,None,:]).half()
def layer(x,w,pos,past=None):
 n=rms(x,w['input']);m=len(x)
 c,s=rope({'head_dim':D,'rope_theta':1000000.},torch.arange(pos,pos+m,device=x.device)[None],torch.float16);c=c[0];s=s[0]
 q=rot(rms(linear(n,w['q']).reshape(m,NH,D),w['qn']),c,s).transpose(0,1)
 k=rot(rms(linear(n,w['k']).reshape(m,KH,D),w['kn']),c,s).transpose(0,1)
 v=linear(n,w['v']).reshape(m,KH,D).transpose(0,1)
 if past is not None:k=torch.cat((past[0],k),1);v=torch.cat((past[1],v),1)
 scores=torch.matmul(q.float(),k.repeat_interleave(2,0).float().transpose(-1,-2)).half()
 if pos==0:
  scores=(scores.float()*np.float16(D**-.5).item()).half().float()
  mask=torch.arange(len(k[0]),device=x.device)[None,:]>torch.arange(m,device=x.device)[:,None]
  scores.masked_fill_(mask,-float('inf'))
  exp=(scores-scores.max(-1,keepdim=True).values).half().float().exp().half()
  prob=(exp.float()/exp.float().sum(-1,keepdim=True)).half()
 else:
  scores=scores.float()*(D**-.5);exp=(scores-scores.max(-1,keepdim=True).values).exp()
  prob=(exp.half().float()/exp.sum(-1,keepdim=True)).half()
 av=torch.matmul(prob.float(),v.repeat_interleave(2,0).float()).half().transpose(0,1).reshape(m,A)
 res=(x+linear(av,w['o'])).half();post=rms(res,w['post'])
 mid=(F.silu(linear(post,w['gate']).float())*linear(post,w['up']).float()).half()
 return (res+linear(mid,w['down'])).half(),(k,v)
def seal(p,**extra):
 write(p/'manifest.json',dict(experiment='EXP-0289',model='Qwen3-0.6B',**extra,files={str(f.relative_to(p)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in p.rglob('*') if f.is_file()}))
def cache_carrier(k,kind):
 arr=np.zeros((KH,CAP*D),'<f2')
 for h in range(KH):
  base=k[h,:64] if kind=='k' else k[h,:64].T
  arr[h,:64*D]=pack_weight(base).reshape(-1)
  if len(k[h])>64:arr[h,64*D:len(k[h])*D]=k[h,64:].cpu().numpy().reshape(-1)
 return arr
@torch.inference_mode()
def main(recipe):
 preflight();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 prov=original();cfg=prov['config'];dst=O/recipe;dst.mkdir(parents=True,exist_ok=False)
 w4=recipe=='w4f16';weights=[]
 qm=json.loads((Q/'manifest.json').read_text())['files']
 def link(src,p):
  rel=str(src.relative_to(Q));assert sha256(src)==qm[rel],rel;os.link(src,p)
 for i in range(28):
  out=dst/f'layer{i}';out.mkdir();w={}
  for n,k in PROJECTIONS.items():
   if w4:
    for suff in ['hmx','scale_f32']:link(Q/f'layer{i}/{n}_weight_w4_{suff}.bin',out/f'{n}_weight_w4_{suff}.bin')
    w[n]=unpack_w4_weight(out,n,*tensor(f'model.layers.{i}.{k}.weight').shape).cuda()
   else:
    w[n]=tensor(f'model.layers.{i}.{k}.weight').half().cuda();put(out/f'{n}_weight_f16_hmx.bin',pack_weight(w[n]))
  for n,k in [('input','input_layernorm'),('post','post_attention_layernorm'),('qn','self_attn.q_norm'),('kn','self_attn.k_norm')]:
   w[n]=tensor(f'model.layers.{i}.{k}.weight').half().cuda()
   put(out/f'{dict(qn="q",kn="k").get(n,n)}_norm_weight_f16.bin',w[n])
  os.link(SP/f'layer{i}/qparams_u8.bin',out/'qparams_u8.bin')
  for kind in ['k','v']:put(out/f'kv_cache_{kind}_hmx_f16.bin',np.zeros(KH*CAP*D,'<f2'))
  weights.append(w);print('WEIGHTS',recipe,i,flush=True)
 for n in ['generation_embedding_weight_f16.bin','generation_final_norm_weight_f16.bin','generation_prompt_token_ids_u32.bin']:
  os.link(SP/n,dst/n)
 if w4:
  for suff in ['hmx','scale_f32']:link(Q/f'head/generation_lm_head_weight_w4_{suff}.bin',dst/f'generation_lm_head_weight_w4_{suff}.bin')
  head=unpack_w4_weight(dst,'generation_lm_head',151936,H).cuda()
 else:
  head=tensor('model.embed_tokens.weight').half().cuda();put(dst/'generation_lm_head_weight_f16_hmx.bin',pack_weight(head))
 emb=tensor('model.embed_tokens.weight').half().cuda();gamma=tensor('model.norm.weight').half().cuda()
 ids=np.fromfile(dst/'generation_prompt_token_ids_u32.bin','<u4')
 fixed=np.fromfile(SP/'generation_expected_token_ids_u32.bin','<u4')[:43]
 x=F.embedding(torch.tensor(ids.astype('i8'),device='cuda'),emb);put(dst/'block_input_f16.bin',x)
 caches=[None]*28;tokens=[];pref=[];dec=[]
 for step in range(43):
  pos=0 if not step else 63+step
  if step:x=emb[int(fixed[step-1])][None]
  c,s=rope(cfg,torch.arange(pos,pos+64)[None],torch.float16)
  for n,v in [('cos',c),('sin',s)]:
   put(dst/(f'rope_{n}_f16.bin' if not step else f'generation_decode_rope_{n}_{step-1:02d}_f16.bin'),v[0])
  ins=[]
  for i,w in enumerate(weights):
   ins.append(x.clone());x,caches[i]=layer(x,w,pos,caches[i])
   if not step:
    pref.append((ins[-1].clone(),x.clone()))
    for j,n in enumerate(['k','v']):put(dst/f'layer{i}/reference_kv_cache_{n}_hmx_f16_step00.bin',cache_carrier(caches[i][j],n))
   if step==1:
    dec.append((ins[-1].clone(),x.clone()))
    for j,n in enumerate(['k','v']):put(dst/f'layer{i}/reference_kv_cache_{n}_hmx_f16_step01.bin',cache_carrier(caches[i][j],n))
  put(dst/f'audit_hidden_{step:02d}_f16.bin',x[-1])
  norm=rms(x[-1:],gamma);tok=int(linear(norm,head)[0].argmax());tokens.append(tok)
  if not step:put(dst/f'reference_{recipe}_block_output_f16.bin',x)
  print('REFERENCE',recipe,step,tok,flush=True)
 put(dst/'generation_expected_token_ids_u32.bin',np.asarray(tokens,'<u4'))
 # Selected single-layer and three-layer replay fixtures share immutable weights.
 for name,first,count in [('layer0',0,1),('layer14',14,1),('layer27',27,1),('chain3',0,3)]:
  out=O/(recipe+'-'+name);out.mkdir()
  for j in range(count):
   dd=out/f'layer{j}';dd.mkdir()
   for f in (dst/f'layer{first+j}').iterdir():
    if f.is_file():os.link(f,dd/f.name)
  put(out/'block_input_f16.bin',padded(pref[first][0]))
  put(out/f'reference_{recipe}_block_output_f16.bin',padded(pref[first+count-1][1]))
  put(out/'replay_decode_input_00_f16.bin',padded(dec[first][0]))
  put(out/'replay_decode_reference_00_f16.bin',padded(dec[first+count-1][1]))
  for n in ['cos','sin']:
   os.link(dst/f'rope_{n}_f16.bin',out/f'rope_{n}_f16.bin')
   os.link(dst/f'generation_decode_rope_{n}_00_f16.bin',out/f'replay_decode_rope_{n}_00_f16.bin')
  seal(out,recipe=recipe,layers=count,first_original_layer=first,cache_capacity=128,decode_steps=1,reference='independent FP32 matmul with declared FP16 boundaries')
 write(R/(recipe+'-reference.json'),dict(recipe=recipe,prompt_ids=ids.tolist(),fixed_input_tokens=fixed.tolist(),selected_tokens=tokens,reference='independent floating math, not teacher quality; no device tensor inputs',original=prov))
 seal(dst,recipe=recipe,layers=28,cache_capacity=128,generation_tokens=43,prompt_tokens=64,reference_sha256=sha256(R/(recipe+'-reference.json')))
 print('EXPORT_DONE',recipe,flush=True)
if __name__=='__main__':main(sys.argv[1])
