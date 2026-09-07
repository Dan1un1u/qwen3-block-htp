#!/usr/bin/env python3
"""Official OmniQuant LWC quantizer/scaler; native Qwen3 block-loop adapter."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,subprocess,importlib.util,hashlib,time,copy
from pathlib import Path
import numpy as np
import torch
from reference_exp0238 import S,M,sha,preflight,V
from qronos_exp0238 import pack,settings
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0239')
O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0239')
UP=Path('/home/daniuniu/.cache/qwen3-block-htp-omniquant-pinned')
def write(n,d):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2,default=str);f.write('\n')
def setup():
 preflight();R.mkdir(parents=True,exist_ok=True)
 if not UP.exists():subprocess.run(['git','clone','--depth','1','https://github.com/OpenGVLab/OmniQuant.git',str(UP)],check=True)
 assert subprocess.check_output(['git','status','--porcelain'],cwd=UP,text=True)==''
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()
 if importlib.util.find_spec('termcolor') is None:subprocess.run([sys.executable,'-m','pip','install','--no-deps','termcolor==2.4.0'],check=True)
 subprocess.run(['git','archive','--format=tar','-o',str(R/'omniquant-upstream.tar'),head],cwd=UP,check=True)
 write('upstream.json',dict(repository='https://github.com/OpenGVLab/OmniQuant',commit=head,archive_sha256=sha(R/'omniquant-upstream.tar'),files={str(p.relative_to(UP)):sha(p) for p in sorted(UP.rglob('*.py'))},adapter='native_Qwen3_loop_official_LWC_quantizer_and_NativeScaler',solver_policy='twenty_epochs_no_LET_no_aug_loss'))
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(R/'environment_freeze.txt').open('x'),check=True)
 load_upstream();print('OMNI_SETUP_COMPLETE',head,flush=True)
def load_upstream():
 d=json.loads((R/'upstream.json').read_text());assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()==d['commit']
 for n,h in d['files'].items():assert sha(UP/n)==h,n
 sys.dont_write_bytecode=True;mods=[]
 for n,p in [('omni_pinned_quantizer',UP/'quantize/quantizer.py'),('omni_pinned_utils',UP/'utils.py')]:
  spec=importlib.util.spec_from_file_location(n,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mods.append(mod)
 return mods[0].UniformAffineQuantizer,mods[1].NativeScalerWithGradNormCount,d

def make_quant(cls,w):
 q=cls(n_bits=4,symmetric=True,per_channel_axes=[0],dynamic_method='per_channel',group_size=None,shape=w.shape,lwc=True,disable_zero_point=True).to(w.device)
 q.qmin=-7;q.qmax=7
 return q
class LWCLinear(torch.nn.Module):
 def __init__(self,module,cls):
  super().__init__();self.in_features=module.in_features;self.out_features=module.out_features
  self.register_buffer('weight',module.weight.detach().float().clone())
  self.register_buffer('bias',None if module.bias is None else module.bias.detach().float().clone())
  self.weight_quantizer=make_quant(cls,self.weight)
 def forward(self,x):return torch.nn.functional.linear(x,self.weight_quantizer(self.weight),self.bias)
def replace(parent,path,mod):
 a,_,b=path.rpartition('.');setattr(parent.get_submodule(a) if a else parent,b,mod)
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def oracle():
 preflight();settings(239);cls,_,_=load_upstream();torch.manual_seed(239);checks=[]
 for kind in ['mixed','zero_and_one_sided']:
  w=torch.randn(6,10,device='cuda')
  if kind=='zero_and_one_sided':w[0]=0;w[1]=w[1].abs();w[2]=-w[2].abs()
  q=make_quant(cls,w)
  with torch.no_grad():q.upbound_factor.fill_(2.2);q.lowbound_factor.fill_(1.7)
  y=q(w);a=w.cpu().numpy();hi=a.max(-1,keepdims=True)/(1+np.exp(-np.float32(2.2)));lo=a.min(-1,keepdims=True)/(1+np.exp(-np.float32(1.7)));sc=np.clip(np.maximum(abs(lo),abs(hi))/7,1e-5,1e4).astype(np.float32);codes=np.clip(np.rint(a/sc),-7,7).astype(np.int8);ref=codes.astype(np.float32)*sc
  assert np.allclose(q.scale.detach().cpu().numpy(),sc,rtol=2e-6,atol=1e-8)
  assert np.allclose(y.detach().cpu().numpy(),ref,rtol=2e-6,atol=1e-6)
  (y.square().mean()).backward();grads=[p.grad for p in q.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
  record=pack(O/'oracle'/kind,'toy',(w/q.scale.detach()).round().clamp(-7,7).to(torch.int8),q.scale.detach())
  checks.append(dict(case=kind,independent_numpy=True,STE_gradient_finite_nonzero=True,packing=record))
 native=torch.nn.Linear(10,6,bias=False,device='cuda',dtype=torch.float16);wrap=LWCLinear(native,cls);wrap.weight_quantizer.enable=False;x=torch.randn(8,10,device='cuda',dtype=torch.float16)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(native(x),wrap(x))
 write('quantization_oracle.json',dict(pass_all=True,cases=checks,disabled_quantizer_linear_exact=True,official_quantizer_unmodified_except_pretraining_qmin_minus7=True));print('OMNI_ORACLE_PASS',flush=True)

def export():
 preflight();settings(239);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip();cls,Scaler,up=load_upstream()
 from data_exp0238 import frozen
 from data_exp0229 import verified
 import evaluate_exp0230 as old
 from experiment_exp0220 import load_packed
 from autoround_exp0233 import read_weight
 import rotation_exp0219 as rot
 frozen();assert json.loads((R/'quantization_oracle.json').read_text())['pass_all']
 assert json.loads(verified('exp0238','independent_data_audit.json').read_text())['pass_all']
 root=O/'OmniQuant';assert not root.exists();root.mkdir(parents=True)
 ids=np.fromfile(verified('exp0230','inputs/C64_calibration_u32.bin'),dtype='<u4').reshape(512,128)
 with torch.no_grad():
  model,fh=old.load('F','cpu');load_packed(model.lm_head.weight,O.parent/'exp0230/C64','generation_lm_head')
 model.requires_grad_(False)
 original=json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())
 assert all(sha(Path('/mnt/d/llm_exp/models/Qwen3-origin')/n)==h for n,h in original.items())
 changed={f'model.layers.{i}.{long}.weight' for i in range(28) for long in rot.PROJECTIONS.values()};others={n:digest(t) for n,t in list(model.named_parameters())+list(model.named_buffers()) if n not in changed}
 with torch.no_grad():hidden=model.model.embed_tokens(torch.tensor(ids.astype(np.int64))).half();fp_hidden=hidden.clone()
 rotary=model.model.rotary_emb.to('cuda');pos=torch.arange(128,device='cuda')[None]
 with torch.no_grad():pe=rotary(torch.empty(1,128,2048,device='cuda',dtype=torch.float16),pos)
 mask=torch.full((1,1,128,128),torch.finfo(torch.float16).min,device='cuda',dtype=torch.float16).triu(1);kwargs=dict(attention_mask=mask,position_ids=pos,cache_position=pos[0],position_embeddings=pe,use_cache=False)
 start=time.monotonic();blocks=[];records={}
 for i,layer in enumerate(model.model.layers):
  layer.cuda();fp_next=torch.empty_like(fp_hidden)
  with torch.no_grad():
   native_probe=layer(hidden[:4].cuda(),**kwargs)[0].cpu()
   for j in range(0,512,8):fp_next[j:j+8]=layer(fp_hidden[j:j+8].cuda(),**kwargs)[0].cpu()
  modules={}
  for name,long in rot.PROJECTIONS.items():
   mod=LWCLinear(layer.get_submodule(long),cls);modules[name]=mod;replace(layer,long,mod);mod.weight_quantizer.enable=False
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(layer(hidden[:4].cuda(),**kwargs)[0].cpu(),native_probe),(i,'disabled_adapter_parity')
  for mod in modules.values():mod.weight_quantizer.enable=True
  params=[p for mod in modules.values() for p in mod.weight_quantizer.parameters()]
  assert {id(p) for p in layer.parameters() if p.requires_grad}=={id(p) for p in params}
  opt=torch.optim.AdamW(params,lr=.01,weight_decay=0);scaler=Scaler();epochs=[];steps=0;skipped=0
  for epoch in range(20):
   losses=[];epoch_skips=0
   for j in range(0,512,8):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
     output=layer(hidden[j:j+8].cuda(),**kwargs)[0];loss=torch.nn.functional.mse_loss(output.float(),fp_next[j:j+8].cuda().float())
    assert torch.isfinite(loss), (i,epoch,j)
    scale_before=scaler._scaler.get_scale();norm=scaler(loss,opt,parameters=params);scale_after=scaler._scaler.get_scale()
    if scale_after<scale_before:skipped+=1;epoch_skips+=1
    else:steps+=1
    losses.append(float(loss.detach()));del output,loss
   epochs.append(dict(epoch=epoch+1,mean_MSE=float(np.mean(losses)),overflow_skipped_steps=epoch_skips))
   if epoch in [0,4,9,14,19]:print('OMNI_EPOCH',i,epoch+1,epochs[-1],'seconds',round(time.monotonic()-start,1),flush=True)
  assert steps>0
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=layer(hidden[:4].cuda(),**kwargs)[0].cpu()
  with torch.no_grad():
   for name,long in rot.PROJECTIONS.items():
    mod=modules[name];w=mod.weight;mod.weight_quantizer(w);scale=mod.weight_quantizer.scale.detach().float();q=(w/scale).round().clamp(-7,7).to(torch.int8)
    records[f'layer{i}/{name}']=pack(root/f'layer{i}',name,q,scale)
    tuning=root/f'layer{i}/{name}_lwc_factors.npy'
    with tuning.open('xb') as f:np.save(f,torch.stack([mod.weight_quantizer.lowbound_factor.detach(),mod.weight_quantizer.upbound_factor.detach()]).cpu().numpy())
    native=torch.nn.Linear(mod.in_features,mod.out_features,bias=mod.bias is not None,device='cuda',dtype=torch.float16).requires_grad_(False)
    native.weight.copy_(torch.from_numpy(read_weight(root/f'layer{i}',name,tuple(w.shape))).cuda())
    if mod.bias is not None:native.bias.copy_(mod.bias.half())
    replace(layer,long,native)
   assert torch.equal(layer(hidden[:4].cuda(),**kwargs)[0].cpu(),expected),(i,'packed_replay')
   next_hidden=torch.empty_like(hidden)
   for j in range(0,512,8):next_hidden[j:j+8]=layer(hidden[j:j+8].cuda(),**kwargs)[0].cpu()
   assert torch.isfinite(next_hidden).all() and torch.isfinite(fp_next).all();hidden=next_hidden;fp_hidden=fp_next
  layer.cpu();blocks.append(dict(layer=i,epochs=epochs,successful_steps=steps,overflow_skipped_steps=skipped,disabled_adapter_exact=True,packed_replay_exact=True,elapsed_s=time.monotonic()-start));write(f'blocks/layer{i}.json',blocks[-1])
  del opt,scaler,modules,params,mod,native,w,q,scale,norm;torch.cuda.empty_cache();print('OMNI_BLOCK_COMPLETE',i,flush=True)
 assert others=={n:digest(t) for n,t in list(model.named_parameters())+list(model.named_buffers()) if n not in changed}
 files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file()}
 manifest=dict(experiment='EXP-0239',variant='OmniQuant',format='software_linear_nibbles_FP32_scales',grid=[-7,7],group_size=-1,files=files,projections=records,frozen_nontransformer=others,upstream_commit=up['commit'],base_C64_manifest_sha256=sha(O.parent/'exp0230/C64/manifest.json'),head_policy='identical_frozen_C64_W4_head',calibration_sha256=sha(verified('exp0230','inputs/C64_calibration_u32.bin')),learning=dict(epochs=20,batch_size=8,lr=.01,weight_decay=0,seed=239,LET=False,aug_loss=False,checkpoint='final_epoch'))
 with (root/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
 write('OmniQuant/package.json',dict(root=str(root),manifest_sha256=sha(root/'manifest.json'),source_head=head,blocks=blocks,elapsed_s=time.monotonic()-start,whole_frozen_state_exact=True));print('OMNI_EXPORT_COMPLETE',sha(root/'manifest.json'),flush=True)
if __name__=='__main__':{'setup':setup,'oracle':oracle,'export':export}[sys.argv[1]]()
