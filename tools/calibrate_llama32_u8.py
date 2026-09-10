#!/usr/bin/env python3
"""Fresh Llama static affine A8 ranges from the sealed Llama C64 calibration."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from llama_reference import provenance,sha256,rms,rope,rotate,PROJECTIONS

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 model=Path('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin');quant=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0002/quant-a01');data=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0002/data')
 prov=provenance(model);manifest=json.loads((quant/'manifest.json').read_text());assert manifest['experiment']=='L32-0002'
 for n,h in manifest['files'].items():assert sha256(quant/n)==h,n
 a.output.mkdir(parents=True);cfg=json.loads((model/'config.json').read_text());ids=np.fromfile(data/'calibration.bin',dtype='<u4').reshape(512,128)
 with safe_open(model/'model.safetensors',framework='pt',device='cpu') as f:embed=f.get_tensor('model.embed_tokens.weight').half().cuda();norm=f.get_tensor('model.norm.weight').half().cuda()
 ranges=[];torch.set_num_threads(8);started=time.monotonic()
 def update(r,n,x):
  assert torch.isfinite(x).all(),n
  lo,hi=float(x.min()),float(x.max());r[n]=[min(r.get(n,[0,0])[0],lo),max(r.get(n,[0,0])[1],hi)]
 for index in range(16):
  w=load_file(quant/f'layer{index}/dequant.safetensors',device='cuda');prior=None if index==0 else np.load(quant/f'layer{index-1}/hidden.npy',mmap_mode='r');r={}
  for start in range(0,512,8):
   x=F.embedding(torch.tensor(ids[start:start+8].astype('int64'),device='cuda'),embed) if index==0 else torch.tensor(np.array(prior[start:start+8]),device='cuda')
   update(r,'block_input',x);n=rms(x,w['input_layernorm.weight'],1e-5);update(r,'input_norm',n)
   q=F.linear(n,w['self_attn.q_proj.weight']);k=F.linear(n,w['self_attn.k_proj.weight']);v=F.linear(n,w['self_attn.v_proj.weight'])
   for name,t in [('q_projection',q),('k_projection',k),('v',v)]:update(r,name,t)
   cos,sin=rope(cfg,torch.arange(128,device='cuda')[None],torch.float16)
   q=rotate(q.reshape(-1,128,32,64).transpose(1,2),cos,sin);k=rotate(k.reshape(-1,128,8,64).transpose(1,2),cos,sin)
   update(r,'q_rope',q);update(r,'k_rope',k)
   kk=k.repeat_interleave(4,1);vv=v.reshape(-1,128,8,64).transpose(1,2).repeat_interleave(4,1)
   scores=(q@kk.transpose(-1,-2))*.125;scores.masked_fill_(torch.ones(128,128,device='cuda',dtype=torch.bool).triu(1),torch.finfo(torch.float16).min)
   prob=scores.float().softmax(-1).half();av=(prob@vv).transpose(1,2).reshape(-1,128,2048);update(r,'attention_concat',av)
   o=F.linear(av,w['self_attn.o_proj.weight']);update(r,'attention_projection',o);res=x+o;update(r,'post_attention_residual',res)
   post=rms(res,w['post_attention_layernorm.weight'],1e-5);update(r,'post_attention_norm',post)
   g=F.linear(post,w['mlp.gate_proj.weight']);u=F.linear(post,w['mlp.up_proj.weight']);mid=F.silu(g)*u
   for name,t in [('gate',g),('up',u),('middle',mid)]:update(r,name,t)
   down=F.linear(mid,w['mlp.down_proj.weight']);update(r,'down',down);update(r,'block_output',res+down)
  ranges.append(r);print('CALIBRATED',index,round(time.monotonic()-started,1),flush=True)
  del w,prior
 def qp(lo,hi):
  lo=min(0.,lo);hi=max(0.,hi);scale=np.float32(max(hi-lo,1e-8)/255);zp=int(np.clip(np.floor(-lo/scale+.5),0,255));return dict(scale=float(scale),zero_point=zp,minimum=lo,maximum=hi)
 qparams=[]
 for i,r in enumerate(ranges):
  q={n:qp(*v) for n,v in r.items()};q['attention_probability']=qp(0,1)
  if i:q['block_input']=qparams[-1]['block_output'].copy()
  qparams.append(q)
 head=load_file(quant/'head/dequant.safetensors',device='cuda')['weight'];last=np.load(quant/'layer15/hidden.npy',mmap_mode='r');hr={}
 for start in range(0,512,16):
  x=torch.tensor(np.array(last[start:start+16,-1]),device='cuda');n=rms(x,norm,1e-5);update(hr,'generation_final_norm_output',n);update(hr,'generation_lm_head_output',F.linear(n,head))
 result=dict(experiment='L32-0003',recipe='W4A8',rotation='OFF',method='affine static minmax, zero included; all C64 positions for transformer, last position of all512 calibration documents for head; no clipping search',original=prov,weight_manifest_sha256=sha256(quant/'manifest.json'),data_freeze_sha256=sha256(data/'freeze.json'),ranges=ranges,qparams=qparams,generation_qparams={n:qp(*v) for n,v in hr.items()},quality_gate=None)
 (a.output/'calibration.json').write_text(json.dumps(result,indent=2)+'\n');print('CALIBRATION_COMPLETE',flush=True)
if __name__=='__main__':main()
