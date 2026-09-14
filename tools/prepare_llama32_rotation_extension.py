#!/usr/bin/env python3
"""L32-0015 original-weight full8192 R4 folding and frozen-train SP2 calibration."""
import json,os,argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from llama_reference import provenance,sha256,rms
from quantize_llama32 import attention,pack
from llama_u8_reference import load_qparams_bin,unpack_w4_codes
from export_llama32_u8 import write_qparams
from probe_llama32_rotations import had,encode,save,OUT
from prototype_llama32_sp2 import preflight
ROOT=Path(__file__).resolve().parents[1]
ORIG=Path('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin')
BASE=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0010/frontend-a01')
OLD=BASE.parent.parent/'l32-0002/quant-a01'
NEW=BASE.parent.parent/'l32-0021/rotated-down-extension-a01'
DATA=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0002/data')
@torch.inference_mode()
def main():
 preflight();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 prov=provenance(ORIG);freeze=json.loads((DATA/'freeze.json').read_text())
 for n,h in freeze.items():assert sha256(DATA/n)==h
 bm=json.loads((BASE/'manifest.json').read_text());qm=json.loads((OLD/'manifest.json').read_text())
 NEW.mkdir(parents=True,exist_ok=False)
 ids=np.fromfile(DATA/'calibration.bin',dtype='<u4').astype('i8').reshape(-1,128)[:8]
 cfg=json.loads((ORIG/'config.json').read_text());levels=np.array(json.loads((OUT.parent/'l32-0008/codebook.json').read_text())['levels'],dtype='i2');assert len(levels)==241
 hs=torch.from_numpy(had(512)).cuda();hg=torch.from_numpy(had(16)).cuda();report=[]
 with safe_open(ORIG/'model.safetensors',framework='pt',device='cpu') as original:
  for i in selected_layers:
   dst=NEW/f'layer{i}';dst.mkdir();old=OLD/f'layer{i}'
   for name in ['dequant.safetensors']:
    assert sha256(old/name)==qm['files'][f'layer{i}/{name}']
   w=load_file(old/'dequant.safetensors',device='cuda')
   if i==0:hidden=original.get_tensor('model.embed_tokens.weight')[torch.from_numpy(ids)].half().cuda()
   else:
    pp=OLD/f'layer{i-1}/hidden.npy';assert sha256(pp)==qm['files'][f'layer{i-1}/hidden.npy'];hidden=torch.from_numpy(np.array(np.load(pp,mmap_mode='r')[:8])).cuda()
   av=attention(hidden,w,cfg);res=hidden+F.linear(av,w['self_attn.o_proj.weight']);post=rms(res,w['post_attention_layernorm.weight'],cfg['rms_norm_eps'])
   q=load_qparams_bin(BASE/f'layer{i}/qparams_u8.bin')
   assert sha256(BASE/f'layer{i}/qparams_u8.bin')==bm['files'][f'layer{i}/qparams_u8.bin']['sha256']
   def fq(x,p):return ((x.float()/p['scale']+p['zero_point']+.5).floor().clamp(0,255)-p['zero_point'])*p['scale']
   post=fq(post,q['post_attention_norm']).half();g=fq(F.linear(post,w['mlp.gate_proj.weight']),q['gate']);u=fq(F.linear(post,w['mlp.up_proj.weight']),q['up'])
   z=(F.silu(g)*u).half().reshape(-1,16,512)
   a=(z.double()@hs*float(np.float16(1/np.sqrt(512)))).half()
   zr=(torch.einsum('ag,tgc->tac',hg,a.double())*.25).half().float().cpu().numpy().reshape(-1,8192)
   maximum=float(np.abs(zr).max());grid=[]
   for ratio in [.5,.625,.75,.875,1.]:
    alpha=float(np.float32(maximum*ratio/24576));codes=encode(zr.astype('f8'),levels,alpha);recon=(codes.astype('i4')-32768)*alpha;grid.append((float(np.mean((recon-zr)**2)),alpha,ratio))
   best=min(grid);alpha=best[1]
   source=original.get_tensor(f'model.layers.{i}.mlp.down_proj.weight').double().cuda();wr=(source.reshape(2048,16,512)@hs)/np.sqrt(512);wr=(torch.einsum('ag,tgc->tac',hg,wr)/4).reshape(2048,8192)
   # Separate ideal orthogonal transform check, before FP16 casts / W4 quantization.
   torch.manual_seed(15015+i);x=torch.randn(2,16,512,dtype=torch.float64,device='cuda');xr=torch.einsum('ag,tgc->tac',hg,x@hs/np.sqrt(512))/4
   rel=float(torch.linalg.norm(x.reshape(2,8192)@source.T-xr.reshape(2,8192)@wr.T)/torch.linalg.norm(x.reshape(2,8192)@source.T));assert rel<2e-12
   wf=wr.float();sc=wf.abs().amax(1).clamp_min(1e-8)/7;code=(wf/sc[:,None]).round().clamp(-7,7).to(torch.int8)
   pack(code).tofile(dst/'down_weight_w4_hmx.bin');sc.cpu().numpy().astype('<f4').tofile(dst/'down_weight_w4_scale_f32.bin')
   assert np.array_equal(unpack_w4_codes(dst,'down',2048,8192),code.cpu().numpy())
   q['middle'].update(scale=alpha,zero_point=0,minimum=-24576*alpha,maximum=24576*alpha);write_qparams(dst/'qparams_u8.bin',q)
   gate=(np.arange(256,dtype='f8')-q['gate']['zero_point'])*q['gate']['scale'];up=(np.arange(256,dtype='f8')-q['up']['zero_point'])*q['up']['scale'];sw=(gate/(1+np.exp(-np.clip(gate,-700,700))))[:,None]*up[None,:]
   assert np.isfinite(sw.astype('f2')).all()
   vals=np.arange(65536,dtype='<u2').view('<f2').astype('f8');vals[~np.isfinite(vals)]=0
   with (dst/'silu_up_lut_u16.bin').open('xb') as f:f.write(sw.astype('<f2').tobytes());f.write(encode(vals,levels,alpha).tobytes())
   wc=code.cpu().numpy().astype('i4');lo=int(255*np.minimum(wc,0).sum(1).min());hi=int(255*np.maximum(wc,0).sum(1).max());assert lo>=-8388608 and hi<=8388607
   mult=np.floor(alpha*sc.cpu().numpy().astype('f8')/q['down']['scale']*2**31+.5);assert ((mult>0)&(mult<2**31)).all()
   stat=dict(layer=i,alpha=alpha,grid=grid,calibration_tokens=int(ids.size),calibration_scope='eight frozen training windows; old W4 trajectory, fakequant Gate/Up; not heldout and not native model-quality proof',transform_relative_l2=rel,partial_dot_bounds=[lo,hi],weight_method='original BF16 Down -> FP64 normalized H16 tensor H512 -> FP32 per-output absmax RTN[-7,7]',files={p.name:sha256(p) for p in dst.iterdir() if p.is_file()})
   save(dst/'manifest.json',stat);report.append(stat);print('PREPARED',i,alpha,rel,flush=True)
   del w,hidden,av,res,post,g,u,z,a,zr,source,wr,wf,code,sc;torch.cuda.empty_cache()
 save(NEW/'manifest.json',dict(experiment=NEW.parent.name.upper(),original=prov,baseline_manifest_sha256=sha256(BASE/'manifest.json'),calibration_freeze_sha256=sha256(DATA/'freeze.json'),layers=report))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--result-experiment',choices=['l32-0021','l32-0023'],default='l32-0021');ap.add_argument('--layers',required=True);ap.add_argument('--attempt',required=True);a=ap.parse_args();selected_layers=[int(x) for x in a.layers.split(',')];assert len(set(selected_layers))==len(selected_layers) and all(0<=i<16 for i in selected_layers);NEW=NEW.parent.parent/a.result_experiment/a.attempt;main()
