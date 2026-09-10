#!/usr/bin/env python3
"""Fresh signed per-channel GPTQ, act-order, 1% damping, final-output 3-range selection.
Method follows retained EXP0221/0224/0230 source; no historical model/data imports.
Torch GPU implementation is checked against independent dense elimination.
"""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file,load_file
from llama_reference import PROJECTIONS,sha256,provenance,rms,rope,rotate
ROOT=Path(__file__).resolve().parents[1]
MODEL=Path('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin')
DATA=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0002/data')
OUT=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0002/quant-a01')

def factor(x):
 x=x.reshape(-1,x.shape[-1]).float();h=(x.T@x)*(2.0/len(x));dead=h.diagonal()==0;h[dead,dead]=1
 perm=torch.argsort(h.diagonal(),descending=True,stable=True);h=h[perm][:,perm].double()
 damping=.01*float(h.diagonal().mean());h.diagonal().add_(damping)
 u=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)),upper=True).float()
 return {'upper':u,'perm':perm,'invperm':torch.argsort(perm),'dead':dead,'damping':damping}

def codes(w,f,scale,block=128):
 work=w[:,f['perm']].clone();work[:,f['dead'][f['perm']]]=0;q=torch.empty_like(work,dtype=torch.int8);u=f['upper']
 for start in range(0,work.shape[1],block):
  end=min(start+block,work.shape[1]);part=work[:,start:end].clone();errors=torch.empty_like(part)
  for j in range(end-start):
   c=(part[:,j]/scale).round().clamp(-7,7);q[:,start+j]=c.to(torch.int8)
   err=(part[:,j]-c*scale)/u[start+j,start+j];errors[:,j]=err
   part[:,j:]-=err[:,None]*u[start+j,start+j:end][None,:]
  work[:,end:]-=errors@u[start:end,end:]
 return q[:,f['invperm']]

def pack(q):
 a=q.cpu().numpy();n,k=a.shape
 a=a.reshape(n//32,32,k//32,32).transpose(0,2,3,1)
 a=np.ascontiguousarray(a.reshape(n//32,k//32,8,4,32).transpose(0,1,2,4,3)).reshape(n//32,k//32,1024)
 a=(a.astype(np.int16)&15).astype(np.uint8);packed=a[...,::2]|(a[...,1::2]<<4)
 # Independent element-address inverse packing.
 b=np.empty((*packed.shape[:-1],1024),dtype=np.uint8);b[...,::2]=packed&15;b[...,1::2]=packed>>4
 b=b.reshape(n//32,k//32,8,32,4).transpose(0,1,2,4,3).reshape(n//32,k//32,32,32).transpose(0,3,1,2).reshape(n,k).astype(np.int8)
 b[b>=8]-=16;assert np.array_equal(b,q.cpu().numpy())
 return packed

def export(w,x,f,path,name,three=True):
 x=x.reshape(-1,x.shape[-1]);out=torch.empty_like(w,dtype=torch.float16);hist=[0,0,0];begin=time.monotonic()
 cp=path/(name+'_weight_w4_hmx.bin');sp=path/(name+'_weight_w4_scale_f32.bin')
 with cp.open('xb') as cf,sp.open('xb') as sf:
  for first in range(0,len(w),2048):
   orig=w[first:first+2048].float();maximum=orig.abs().amax(1);base=torch.where(maximum>0,maximum/7,torch.ones_like(maximum))
   clipped=base.clone();best=torch.full_like(base,float('inf'))
   if three:
    for i in range(80):
     sc=base*(1-i/100);err=((orig/sc[:,None]).round().clamp(-7,7)*sc[:,None]-orig).abs().pow(2.4).sum(1)
     better=err<best;best[better]=err[better];clipped[better]=sc[better]
   scales=torch.cat([base,(base+clipped)*.5,clipped]) if three else base
   allq=codes(orig.repeat(3 if three else 1,1),f,scales);best=torch.full((len(orig),),float('inf'),dtype=torch.float64,device=w.device)
   chosen=torch.zeros(len(orig),device=w.device,dtype=torch.int64);bq=allq[:len(orig)].clone();bs=base.clone()
   for i in range(3 if three else 1):
    q=allq[i*len(orig):(i+1)*len(orig)];sc=scales[i*len(orig):(i+1)*len(orig)];delta=(q.float()*sc[:,None]).half().float()-orig
    err=torch.zeros_like(best)
    for j in range(0,len(x),512):err+=F.linear(x[j:j+512].float(),delta).double().square().sum(0)
    better=err<best;best[better]=err[better];chosen[better]=i;bq[better]=q[better];bs[better]=sc[better]
   cf.write(pack(bq).tobytes());sf.write(bs.cpu().numpy().astype('<f4').tobytes())
   out[first:first+len(orig)]=(bq.float()*bs[:,None]).half()
   for i in range(3):hist[i]+=int((chosen==i).sum())
   if len(w)>20000:print('HEAD_ROWS',first+len(orig),flush=True)
 return out,{'rows':len(w),'columns':w.shape[1],'choice_histogram':hist,'damping':f['damping'],'elapsed_s':time.monotonic()-begin,'groupsize':-1,'grid':[-7,7],'packed_roundtrip':True}

def oracle():
 torch.manual_seed(320002);x=torch.randn(96,32,device='cuda');x[:,1]=.8*x[:,0]+.2*x[:,1];w=torch.randn(32,32,device='cuda');w[0]=0
 f=factor(x);sc=w.abs().amax(1)/7;sc[0]=1;a=codes(w,f,sc,8)
 h=((x.T@x)*(2/len(x)))[f['perm']][:,f['perm']].double();h.diagonal().add_(f['damping']);inv=torch.linalg.inv(h)
 work=w[:,f['perm']].double().clone();expected=torch.empty_like(work)
 for j in range(32):
  c=(work[:,j]/sc.double()).round().clamp(-7,7);expected[:,j]=c;err=work[:,j]-c*sc.double()
  work[:,j:]-=err[:,None]*(inv[0]/inv[0,0])[None,:];inv=inv[1:,1:]-inv[1:,0,None]*inv[None,0,1:]/inv[0,0]
 assert torch.equal(a,expected[:,f['invperm']].to(torch.int8));pack(a)
 return {'dense_elimination_equal':True,'pack_roundtrip':True,'seed':320002}

def attention(x,w,cfg):
 b,m,_=x.shape;norm=rms(x,w['input_layernorm.weight'],cfg['rms_norm_eps']);c,s=rope(cfg,torch.arange(m,device=x.device)[None],x.dtype)
 q=rotate(F.linear(norm,w['self_attn.q_proj.weight']).view(b,m,32,64).transpose(1,2),c,s)
 k=rotate(F.linear(norm,w['self_attn.k_proj.weight']).view(b,m,8,64).transpose(1,2),c,s).repeat_interleave(4,1)
 v=F.linear(norm,w['self_attn.v_proj.weight']).view(b,m,8,64).transpose(1,2).repeat_interleave(4,1)
 score=(q@k.transpose(-1,-2))*.125;mask=torch.ones(m,m,dtype=torch.bool,device=x.device).triu(1)
 prob=score.masked_fill(mask,torch.finfo(x.dtype).min).float().softmax(-1).half()
 return (prob@v).transpose(1,2).contiguous().view(b,m,2048)

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 for n,h in json.loads((DATA/'freeze.json').read_text()).items():assert sha256(DATA/n)==h
 prov=provenance(MODEL);checks=oracle()
 if not args.resume:OUT.mkdir(parents=True,exist_ok=False)
 else:assert OUT.is_dir()
 (OUT/'oracle.json').write_text(json.dumps(checks,indent=2))
 cfg=json.loads((MODEL/'config.json').read_text());ids=torch.tensor(np.fromfile(DATA/'calibration.bin',dtype='<u4').astype(np.int64).reshape(-1,128),device='cuda')
 model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,attn_implementation='eager',local_files_only=True).cuda().eval()
 hidden=model.model.embed_tokens(ids);stats={};started=time.monotonic()
 for index,mod in enumerate(model.model.layers):
  path=OUT/f'layer{index}'
  if (path/'complete.json').exists():
   done=json.loads((path/'complete.json').read_text())
   for n,h in done['files'].items():assert sha256(path/n)==h
   mod.load_state_dict(load_file(path/'dequant.safetensors',device='cuda'));hidden=torch.from_numpy(np.load(path/'hidden.npy')).cuda();print('RESUME_LAYER',index,flush=True);continue
  path.mkdir(exist_ok=False);w=mod.state_dict();original={n:w[k+'.weight'].float().clone() for n,k in PROJECTIONS.items()}
  norm=rms(hidden,w['input_layernorm.weight'],cfg['rms_norm_eps']);f=factor(norm)
  for n in ['q','k','v']:
   value,st=export(original[n],norm,f,path,n);w[PROJECTIONS[n]+'.weight'].copy_(value);stats[n]=st
  del norm,f
  av=torch.cat([attention(hidden[j:j+8],w,cfg) for j in range(0,len(hidden),8)])
  f=factor(av);value,stats['o']=export(original['o'],av,f,path,'o');w[PROJECTIONS['o']+'.weight'].copy_(value);del f
  residual=hidden+F.linear(av,value);del av
  post=rms(residual,w['post_attention_layernorm.weight'],cfg['rms_norm_eps']);f=factor(post)
  for n in ['gate','up']:
   value,st=export(original[n],post,f,path,n);w[PROJECTIONS[n]+'.weight'].copy_(value);stats[n]=st
  del f
  middle=F.silu(F.linear(post,w['mlp.gate_proj.weight']))*F.linear(post,w['mlp.up_proj.weight']);del post
  f=factor(middle);value,stats['down']=export(original['down'],middle,f,path,'down');w['mlp.down_proj.weight'].copy_(value);hidden=residual+F.linear(middle,value)
  assert torch.isfinite(hidden).all();del middle,residual,f,original,value
  save_file({k:v.cpu().contiguous() for k,v in w.items()},path/'dequant.safetensors');np.save(path/'hidden.npy',hidden.cpu().numpy())
  (path/'complete.json').write_text(json.dumps({'layer':index,'stats':stats,'files':{p.name:sha256(p) for p in path.iterdir() if p.is_file()}},indent=2)+'\n')
  print('QUANTIZED_LAYER',index,'elapsed_s',round(time.monotonic()-started,1),flush=True)
 head=OUT/'head';head.mkdir(exist_ok=False);x=model.model.norm(hidden);f=factor(x)
 value,st=export(model.model.embed_tokens.weight.float(),x,f,head,'generation_lm_head',three=False)
 save_file({'weight':value.cpu()},head/'dequant.safetensors');(head/'stats.json').write_text(json.dumps(st,indent=2))
 (OUT/'manifest.json').write_text(json.dumps({'experiment':'L32-0002','recipe':'W4A16','original':prov,'data_freeze_sha256':sha256(DATA/'freeze.json'),'method':'C64 GPTQ true sequential, act-order, damping1%, 3 final-output per-row ranges; head absmax GPTQ; signed[-7,7], no groups','files':{str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file()}},indent=2)+'\n')
 print('QUANTIZATION_COMPLETE',flush=True)
if __name__=='__main__':main()
