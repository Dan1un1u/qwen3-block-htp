#!/usr/bin/env python3
"""Official Qronos solver with a fixed signed per-output-channel grid adapter."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,json,math,time,warnings,types,gc,sys
from pathlib import Path
import numpy as np
import torch
from brevitas.graph.qronos import Qronos
from brevitas.graph.gpxq import LayerHandler
from brevitas.graph.utils import power_iteration
from brevitas.utils.torch_utils import StopFwdException
from reference_exp0238 import S,M,R,O,sha,write,preflight,upstream_verified

class FixedGrid(Qronos):
 def __init__(self,w,name='oracle',num_blocks=100,dtype=torch.float32):
  layer=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,device=w.device,dtype=dtype)
  layer.weight=torch.nn.Parameter(w.detach().to(dtype).clone(),requires_grad=False)
  layer.weight_quant=types.SimpleNamespace(requires_quant_input=False,is_quant_enabled=True)
  super().__init__(layer,name,True,1,True,num_blocks=num_blocks,alpha=1e-6,device=str(w.device),dtype=dtype)
  self.scale=w.float().abs().amax(-1,keepdim=True).div(7).clamp(min=1e-5)
 def get_quant_weights(self,i,i1,permutation_list,with_quant_history=False):
  idx=permutation_list[0][:i+i1] if with_quant_history else permutation_list[0][i+i1]
  w=self.layer.weight[:,idx]
  scale=self.scale if with_quant_history else self.scale[:,0]
  return ((w/scale).round().clamp(-7,7)*scale).unsqueeze(0)
 def pair(self,xq,xf):
  for flag,x in [(True,xq),(False,xf)]:
   self.layer.weight_quant.is_quant_enabled=flag
   try:super().update_batch(self.layer,(x,),LayerHandler())
   except StopFwdException:pass
 def codes(self):return (self.layer.weight.detach()/self.scale).round().clamp(-7,7).to(torch.int8)

def settings(seed=238):
 torch.set_num_threads(8);torch.manual_seed(seed)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
 torch.use_deterministic_algorithms(True)

def pack(root,name,q,scale):
 root.mkdir(parents=True,exist_ok=True)
 a=q.detach().cpu().numpy();sc=scale.detach().cpu().float().numpy()
 assert a.dtype==np.int8 and sc.shape==(a.shape[0],1) and np.isfinite(sc).all() and (sc>0).all()
 assert a.min()>=-7 and a.max()<=7
 flat=a.reshape(-1).astype(np.uint8)&15;assert len(flat)%2==0
 cp=root/(name+'_codes_linear_u4.bin');sp=root/(name+'_scales.npy')
 with cp.open('xb') as f:f.write((flat[0::2]|(flat[1::2]<<4)).tobytes())
 with sp.open('xb') as f:np.save(f,sc)
 raw=np.fromfile(cp,dtype=np.uint8);u=np.stack([raw&15,raw>>4],axis=1).reshape(a.shape).astype(np.int16);u=np.where(u>=8,u-16,u)
 assert np.array_equal(u,a)
 recon=(u.astype(np.float32)*np.load(sp)).astype(np.float16)
 expected=(q.float()*scale.float()).half().cpu().numpy();assert np.array_equal(recon,expected)
 return dict(shape=list(a.shape),codes_min=int(a.min()),codes_max=int(a.max()),scale_shape=list(sc.shape),codes_sha256=sha(cp),scales_sha256=sha(sp),independent_roundtrip=True)

def oracle():
 preflight();upstream_verified();settings();torch.set_grad_enabled(False);checks=[]
 for kind in ['correlated_mismatch','correlated_same','diagonal']:
  xq=torch.randn(256,10,device='cuda');xf=xq+.25*torch.randn_like(xq)
  if kind=='correlated_same':xf=xq.clone()
  if kind=='diagonal':xq=torch.diag(torch.arange(1,11,device='cuda').float());xf=xq.clone()
  w=torch.randn(6,10,device='cuda');w[0]=0;w[1]=w[1].abs();w[2]=-w[2].abs()
  obj=FixedGrid(w,num_blocks=3)
  for start in range(0,len(xq),32):obj.pair(xq[start:start+32],xf[start:start+32])
  h=xq.T@xq/len(xq);g=xf.T@xq/len(xq)
  err=max(float((obj.H[0]-h).abs().max()),float((obj.G[0]-g).abs().max()));assert err<1e-5
  perm=torch.argsort(obj.H[0].diag(),descending=True);hp=obj.H[0][perm][:,perm];gp=obj.G[0][perm][:,perm]
  torch.manual_seed(123);damp=float((1e-6*power_iteration(hp,30)).float());hp=hp.double().cpu().numpy();gp=gp.double().cpu().numpy();wp=w[:,perm].double().cpu().numpy();sc=obj.scale.cpu().numpy().astype(np.float64)
  result=np.zeros_like(wp);arg=(wp@gp[:,0]-wp[:,1:]@hp[0,1:])/hp[0,0];result[:,0]=np.clip(np.rint(arg/sc[:,0]),-7,7)*sc[:,0]
  eye=np.eye(10)
  for i in range(1,10):
   rhs=wp@(gp[:,i:]+damp*eye[:,i:])-result[:,:i]@hp[:i,i:]
   cond=np.linalg.solve(hp[i:,i:]+damp*np.eye(10-i),rhs.T).T
   result[:,i]=np.clip(np.rint(cond[:,0]/sc[:,0]),-7,7)*sc[:,0]
  torch.manual_seed(123)
  with warnings.catch_warnings():
   warnings.simplefilter('error');obj.single_layer_update()
  q=obj.codes()[:,perm].cpu().numpy();expected=np.rint(result/sc).astype(np.int8);assert np.array_equal(q,expected),(kind,q,expected)
  if kind=='diagonal':assert torch.equal(obj.codes(),(w/obj.scale).round().clamp(-7,7).to(torch.int8))
  root=O/'oracle'/kind;record=pack(root,'toy',obj.codes(),obj.scale)
  checks.append(dict(kind=kind,paired_moment_max_abs=err,independent_conditional_solve_codes_exact=True,packing=record))
 write('quantization_oracle.json',dict(pass_all=True,checks=checks,upstream_algorithm_unmodified=True,grid=[-7,7],runtime_group_size=-1))
 print('QRONOS_ORACLE_PASS',flush=True)

class CaptureStop(Exception):pass

def capture(block,x,kwargs,long):
 saved=[]
 def hook(module,args):saved.append(args[0].detach());raise CaptureStop()
 handle=block.get_submodule(long).register_forward_pre_hook(hook)
 try:
  try:block(x,**kwargs)
  except CaptureStop:pass
 finally:handle.remove()
 assert len(saved)==1
 return saved[0]

def export():
 preflight();up=upstream_verified();settings();torch.set_grad_enabled(False)
 from data_exp0238 import frozen
 from data_exp0229 import verified
 import evaluate_exp0230 as old
 import rotation_exp0219 as rot
 frozen();assert json.loads((R/'independent_data_audit.json').read_text())['pass_all'];assert json.loads((R/'quantization_oracle.json').read_text())['pass_all']
 root=O/'Qronos';assert not root.exists();root.mkdir(parents=True)
 ids=np.fromfile(verified('exp0230','inputs/C64_calibration_u32.bin'),dtype='<u4').reshape(512,128)
 model,fh=old.load('F','cpu');original={n:sha(Path('/mnt/d/llm_exp/models/Qwen3-origin')/n) for n in json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())}
 assert original==json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())
 def digest(t):return __import__('hashlib').sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
 changed={f'model.layers.{i}.{long}.weight' for i in range(28) for long in rot.PROJECTIONS.values()}
 others={n:digest(t) for n,t in list(model.named_parameters())+list(model.named_buffers()) if n not in changed}
 with torch.no_grad():
  hidden=model.model.embed_tokens(torch.tensor(ids.astype(np.int64))).half()
 rotary=model.model.rotary_emb.to('cuda');pos=torch.arange(128,device='cuda')[None];pe=rotary(torch.empty(1,128,2048,device='cuda',dtype=torch.float16),pos)
 mask=torch.full((1,1,128,128),torch.finfo(torch.float16).min,device='cuda',dtype=torch.float16).triu(1)
 kwargs=dict(attention_mask=mask,position_ids=pos,cache_position=pos[0],position_embeddings=pe,use_cache=False)
 # Verify standalone block inputs/mask/RoPE against the canonical full-model hook.
 first=model.model.layers[0].to('cuda');expected=first(hidden[:4].cuda(),**kwargs)[0].cpu();first.cpu();saved=[]
 def sentinel(mod,inp,out):saved.append(out[0].detach().cpu());raise CaptureStop()
 handle=first.register_forward_hook(sentinel);model.cuda()
 try:
  try:model(input_ids=torch.tensor(ids[:4].astype(np.int64),device='cuda'),use_cache=False)
  except CaptureStop:pass
 finally:handle.remove();model.cpu()
 assert len(saved)==1 and torch.equal(saved[0],expected)
 write('block_adapter_parity.json',dict(exact=True,rows=4,context=128))
 records={};blocks=[];start=time.monotonic()
 for i,layer in enumerate(model.model.layers):
  layer=layer.cuda();teacher=copy.deepcopy(layer);repairs=[]
  for names in [['q','k','v'],['o'],['gate','up'],['down']]:
   long=rot.PROJECTIONS[names[0]];width=layer.get_submodule(long).weight.shape[1]
   moments=FixedGrid(torch.zeros(1,width,device='cuda'),name='moments')
   for j in range(0,512,8):
    x=hidden[j:j+8].cuda();xq=capture(layer,x,kwargs,long);xf=capture(teacher,x,kwargs,long);moments.pair(xq,xf)
   assert moments.nsamples==65536 and moments.quant_input is None
   h=moments.H[0].clone();g=moments.G[0].clone();del moments,x,xq,xf
   for name in names:
    mod=layer.get_submodule(rot.PROJECTIONS[name]);w=teacher.get_submodule(rot.PROJECTIONS[name]).weight.detach().float()
    for dtype in [torch.float32,torch.float64]:
     solver=FixedGrid(w,name=f'layer{i}/{name}',dtype=dtype);solver.H[0].copy_(h);solver.G[0].copy_(g)
     try:
      with warnings.catch_warnings():warnings.simplefilter('error');solver.single_layer_update()
      q=solver.codes();scale=solver.scale;assert torch.isfinite(solver.layer.weight).all();break
     except (UserWarning,torch.linalg.LinAlgError) as exc:
      if dtype==torch.float64:raise
      repairs.append(dict(projection=name,cause=str(exc),repair='same_data_grid_alpha_FP64_arithmetic'));del solver;torch.cuda.empty_cache()
    mod.weight.copy_((q.float()*scale).half());rec=pack(root/f'layer{i}',name,q,scale);rec['solver_dtype']=str(dtype);records[f'layer{i}/{name}']=rec
    del solver,w,q,scale
   del h,g
   print('QRONOS_STAGE',i,names,'seconds',round(time.monotonic()-start,1),flush=True)
  # Packed reload must reproduce the same entire-block calibration output exactly.
  expected=layer(hidden[:4].cuda(),**kwargs)[0].cpu()
  from autoround_exp0233 import read_weight
  for name,long in rot.PROJECTIONS.items():
   mod=layer.get_submodule(long);mod.weight.copy_(torch.from_numpy(read_weight(root/f'layer{i}',name,tuple(mod.weight.shape))).to(mod.weight.device))
  assert torch.equal(layer(hidden[:4].cuda(),**kwargs)[0].cpu(),expected)
  next_hidden=torch.empty_like(hidden)
  for j in range(0,512,8):next_hidden[j:j+8]=layer(hidden[j:j+8].cuda(),**kwargs)[0].cpu()
  assert torch.isfinite(next_hidden).all();hidden=next_hidden;layer.cpu();del teacher;torch.cuda.empty_cache()
  blocks.append(dict(layer=i,paired_calibration_tokens=65536,packed_replay_exact=True,repairs=repairs,elapsed_s=time.monotonic()-start))
  write(f'blocks/layer{i}.json',blocks[-1]);print('QRONOS_BLOCK_COMPLETE',i,flush=True)
 assert others=={n:digest(t) for n,t in list(model.named_parameters())+list(model.named_buffers()) if n not in changed}
 files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file()}
 manifest=dict(experiment='EXP-0238',variant='Qronos',format='software_linear_nibbles_FP32_scales',grid=[-7,7],group_size=-1,files=files,projections=records,upstream_commit=up['commit'],frozen_nontransformer=others,base_C64_manifest_sha256=sha(O.parent/'exp0230/C64/manifest.json'),head_policy='identical_frozen_C64_W4_head',calibration_sha256=sha(verified('exp0230','inputs/C64_calibration_u32.bin')))
 with (root/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
 write('Qronos/package.json',dict(root=str(root),manifest_sha256=sha(root/'manifest.json'),blocks=blocks,elapsed_s=time.monotonic()-start,source_head=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()))
 print('QRONOS_EXPORT_COMPLETE',sha(root/'manifest.json'),flush=True)
if __name__=='__main__':oracle() if sys.argv[1]=='oracle' else export()
