#!/usr/bin/env python3
"""Pinned official AutoRound reference setup; numerical stages appended separately."""
import argparse,json,os,subprocess,sys,time,urllib.request,hashlib
from pathlib import Path
S=Path(__file__).resolve().parents[1]
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0233')
O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0233')
V=Path('/home/daniuniu/.cache/qwen3-block-htp-autoround-py')
UP=Path('/home/daniuniu/.cache/qwen3-block-htp-autoround-v0.5.1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def write(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,default=str);f.write('\n')
def preflight():
 subprocess.run(['python3','/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(S)],check=True)
def setup():
 preflight()
 if not UP.exists():subprocess.run(['git','clone','--depth','1','--branch','v0.5.1','https://github.com/intel/auto-round.git',str(UP)],check=True)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()
 assert subprocess.check_output(['git','status','--porcelain'],cwd=UP,text=True)==''
 if not V.exists():subprocess.run(['python3','-m','venv',str(V)],check=True)
 site=next((V/'lib').glob('python*/site-packages'))
 (site/'frozen_gpu_parent.pth').write_text('/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/lib/python3.10/site-packages\n/home/daniuniu/.cache/qwen3-block-htp-py/lib/python3.10/site-packages\n')
 subprocess.run([str(V/'bin/python'),'-m','pip','install','--no-deps',str(UP),'numpy==1.26.4','py-cpuinfo==9.0.0','pillow==11.3.0','threadpoolctl==3.6.0'],check=True)
 subprocess.run([str(V/'bin/python'),'-m','pip','install','pandas==2.2.3','datasets==3.5.0','sentencepiece==0.2.0','numpy==1.26.4'],check=True)
 subprocess.run([str(V/'bin/python'),'-c','import auto_round,torch,transformers;print(auto_round.__version__,torch.__version__,transformers.__version__)'],check=True)
 archive=O/'upstream';archive.mkdir(parents=True,exist_ok=True)
 subprocess.run(['git','archive','--format=tar','-o',str(archive/'auto-round-v0.5.1.tar'),head],cwd=UP,check=True)
 write('upstream.json',dict(repository='https://github.com/intel/auto-round',tag='v0.5.1',commit=head,archive_sha256=sha(archive/'auto-round-v0.5.1.tar'),license_sha256=sha(UP/'LICENSE'),files={str(p.relative_to(UP)):sha(p) for p in sorted((UP/'auto_round').rglob('*.py'))}))
 subprocess.run([str(V/'bin/python'),'-m','pip','freeze'],stdout=(R/'environment_freeze.txt').open('x'),check=True)


def frozen():
 from data_exp0233 import MEMORY
 assert sha(R/'dataset_freeze.json')=='d86c6511728de21dcf2732c6dd7f890c06c6bc95ef2b8c377182169b51ea395b'
 f=json.loads((R/'dataset_freeze.json').read_text())
 assert sha(MEMORY/'docs/experiments/EXP-0233.md')==f['protocol_sha256']
 for n,h in f['files'].items():assert sha(R/n)==h,n
 assert json.loads((R/'independent_data_audit.json').read_text())['pass_all']
 return json.loads((R/'dataset.json').read_text())

def install_grid():
 import torch
 from auto_round.data_type.register import register_dtype
 from auto_round.data_type.utils import round_ste
 @register_dtype('qbh_int_sym7')
 def sym7(tensor,bits=4,group_size=-1,v=0,min_scale=1.,max_scale=1.,scale_dtype=torch.float32,tensor_min=None,tensor_max=None,q_scale_thresh=1e-5,**kwargs):
  assert bits==4 and group_size==-1 and tensor.ndim==2
  w=tensor.float()
  lo=w.amin(-1).clamp(max=0) if tensor_min is None else tensor_min.float()
  hi=w.amax(-1).clamp(min=0) if tensor_max is None else tensor_max.float()
  scale=torch.maximum((-lo)*min_scale,hi*max_scale).div(7).clamp(min=q_scale_thresh).to(scale_dtype).reshape(-1,1)
  q=round_ste(w/scale+v).clamp(-7,7)
  return (q*scale).to(tensor.dtype),scale,torch.zeros_like(scale)
 return sym7

def oracle():
 preflight();frozen()
 import numpy as np,torch
 from auto_round.data_type.int import quant_tensor_sym
 f=install_grid();torch.manual_seed(233);checks=[]
 for typ,group,fn in [('AR-P',-1,f),('AR-G',128,quant_tensor_sym)]:
  for dtype in [torch.float32,torch.float16]:
   w=torch.randn(6,256,dtype=dtype);w[0]=0;w[1]=w[1].abs();w[2]=-w[2].abs()
   cols=256 if group==-1 else group;shape=(-1,cols);a=w.float().numpy().reshape(shape)
   v=torch.rand(a.shape)*.7-.35;mn=torch.full((a.shape[0],),.87);mx=torch.full_like(mn,.93)
   q,scale,zp=fn(w,group_size=group,v=v if group!=-1 else v.reshape(w.shape),min_scale=mn,max_scale=mx,scale_dtype=torch.float32)
   # Native extrema are computed in the input dtype, matching upstream semantics.
   lo=np.minimum(a.min(-1),0);hi=np.maximum(a.max(-1),0)
   if typ=='AR-P':sc=np.maximum(-lo*.87,hi*.93)/7;sc=np.maximum(sc,1e-5);low,high=-7,7
   else:
    # Multiplication promotes because tuning scale parameters here are FP32.
    neg=-lo*.87;pos=hi*.93;sc=(2*(pos<neg).astype(np.int32)-1)*np.maximum(pos,neg)/8
    sc=np.where(sc<0,np.minimum(sc,-1e-5),np.maximum(sc,1e-5)).astype(np.float32);low,high=-8,7
   codes=np.clip(np.rint(a/sc[:,None]+v.numpy()),low,high)
   ref=(codes*sc[:,None]).reshape(w.shape).astype(np.float16 if dtype==torch.float16 else np.float32)
   assert np.allclose(scale.numpy().reshape(-1),sc,rtol=2e-6,atol=1e-8)
   assert np.allclose(q.numpy(),ref,rtol=2e-6,atol=1e-6),typ
   checks.append(dict(variant=typ,dtype=str(dtype),zero_and_one_sided=True))
 value=torch.zeros((4,256),requires_grad=True);a=torch.randn(4,256)
 out,_,_=f(a,v=value);out.square().sum().backward();assert value.grad.isfinite().all() and value.grad.abs().sum()>0
 write('quantization_oracle.json',dict(pass_all=True,independent_numpy=checks,STE_gradient_finite_nonzero=True))
 print('AUTOROUND_QUANTIZATION_ORACLE_PASS',flush=True)

def read_weight(root,name,shape):
 import numpy as np
 root=Path(root);raw=np.fromfile(root/(name+'_codes_linear_u4.bin'),dtype=np.uint8)
 u=np.empty(raw.size*2,dtype=np.int8);u[0::2]=(raw&15).astype(np.int8);u[1::2]=(raw>>4).astype(np.int8)
 u[u>=8]-=16;q=u.reshape(shape)
 sc=np.load(root/(name+'_scales.npy'));assert sc.dtype==np.float32
 width=shape[1]//sc.shape[1]
 result=(q.reshape(shape[0],-1,width).astype(np.float32)*sc[:,:,None]).reshape(shape).astype(np.float16)
 assert np.isfinite(result).all()
 return result

def export(v,attempt):
 actual_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()
 preflight();frozen()
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 import numpy as np,torch,auto_round,inspect
 from data_exp0233 import verified,MODEL
 import rotation_exp0219 as rot
 import evaluate_exp0230 as old
 from experiment_exp0220 import load_packed
 assert json.loads((R/'quantization_oracle.json').read_text())['pass_all']
 up=json.loads((R/'upstream.json').read_text());assert up['commit']=='73669aa50871bca9f244ad99faa9979790f7c729'
 installed=Path(auto_round.__file__).parent
 for n,h in up['files'].items():
  local=installed/Path(n).relative_to('auto_round');assert sha(local)==h,n
 ledger=json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())
 for n,h in ledger.items():assert sha(MODEL/n)==(h['sha256'] if isinstance(h,dict) else h)
 root=O/v/f'attempt{attempt}';assert not root.exists() and not (R/v/'package.json').exists()
 root.mkdir(parents=True);(root/'SOFTWARE_ONLY.txt').write_text('Linear nibble codes, original-column groups, FP32 scales. Not a DSP package.\n')
 torch.set_num_threads(8);torch.manual_seed(233)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
 torch.use_deterministic_algorithms(True)
 with torch.no_grad():
  model,fh=old.load('F','cpu')
  # Head is excluded from training and exactly inherited from C64.
  head_root=O.parent/'exp0230/C64';load_packed(model.lm_head.weight,head_root,'generation_lm_head')
 changed={f'model.layers.{i}.{long}.weight' for i in range(28) for long in rot.PROJECTIONS.values()}
 def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
 others={n:digest(t) for n,t in list(model.named_parameters())+list(model.named_buffers()) if n not in changed}
 d=json.loads(verified('exp0230','dataset.json').read_text());rows=[r for r in d['samples'] if r['split']=='calibration']
 ids=torch.tensor([r['token_ids'] for r in rows]);assert ids.shape==(512,128)
 cal=verified('exp0230','inputs/C64_calibration_u32.bin');assert np.array_equal(np.frombuffer(cal.read_bytes(),dtype='<u4').reshape(512,128),ids.numpy())
 batches=[ids[i:i+8] for i in range(0,512,8)]
 install_grid();kwargs=dict(bits=4,group_size=-1 if v=='AR-P' else 128,sym=True,layer_config={'lm_head':{'bits':16}},batch_size=8,amp=True,device='cuda:0',
  lr=.005,minmax_lr=.005,enable_quanted_input=True,enable_minmax_tuning=True,low_gpu_mem_usage=True,low_cpu_mem_usage=False,iters=200,seqlen=128,nsamples=512,
  sampler='rand',seed=233,nblocks=1,gradient_accumulate_steps=1,not_use_best_mse=False,dynamic_max_gap=-1,
  data_type='qbh_int_sym7' if v=='AR-P' else 'int_sym',scale_dtype='fp32',act_bits=16,enable_norm_bias_tuning=False,enable_torch_compile=False)
 started=time.monotonic();records={};block_checks=[]
 def retain_block(block,i):
  for name,long in rot.PROJECTIONS.items():
   mod=block.get_submodule(long);w=mod.weight.detach().cpu().half().numpy();sc=mod.scale.detach().cpu().float().numpy()
   assert sc.dtype==np.float32 and sc.shape[0]==w.shape[0] and np.isfinite(sc).all() and (sc!=0).all()
   g=w.shape[1]//sc.shape[1];q=np.rint(w.astype(np.float32).reshape(w.shape[0],-1,g)/sc[:,:,None]).reshape(w.shape).astype(np.int8)
   low=-7 if v=='AR-P' else -8;assert q.min()>=low and q.max()<=7
   deq=(q.reshape(w.shape[0],-1,g).astype(np.float32)*sc[:,:,None]).reshape(w.shape).astype(np.float16)
   assert np.array_equal(deq,w),(i,name,'integer reconstruction')
   target=root/f'layer{i}';target.mkdir(exist_ok=True);u=q.reshape(-1).astype(np.uint8)&15
   (target/(name+'_codes_linear_u4.bin')).write_bytes((u[0::2]|(u[1::2]<<4)).tobytes());np.save(target/(name+'_scales.npy'),sc)
   assert np.array_equal(read_weight(target,name,w.shape),w)
   records[f'layer{i}/{name}']=dict(shape=list(w.shape),group_size=g,codes_min=int(q.min()),codes_max=int(q.max()),negative_scale_count=int((sc<0).sum()),
    codes_sha256=sha(target/(name+'_codes_linear_u4.bin')),scales_sha256=sha(target/(name+'_scales.npy')),effective_FP16_sha256=hashlib.sha256(w.tobytes()).hexdigest(),independent_roundtrip=True)
 class RetainedAutoRound(auto_round.AutoRound):
  def quant_block(self,block,input_ids,input_others,q_input=None,device=torch.device('cpu')):
   i=len(block_checks);assert len(input_ids)==512
   if q_input is not None:assert len(q_input)==512
   outputs=super().quant_block(block,input_ids,input_others,q_input,device)
   retain_block(block,i)
   for label,vals in zip(['student','teacher'],outputs):
    assert len(vals)==512 and all(torch.isfinite(t).all() for t in vals)
    sample=torch.cat([t.detach().cpu() for t in vals[:4]],dim=0)
    np.save(root/f'layer{i}/{label}_first4.npy',sample.numpy())
   item=dict(layer=i,calibration_samples=512,finite=True,elapsed_s=time.monotonic()-started)
   block_checks.append(item);write(f'{v}/attempt{attempt}/layer{i}.json',item)
   print('AUTOROUND_LAYER_COMPLETE',v,i,round(time.monotonic()-started,1),flush=True)
   return outputs
 runner=RetainedAutoRound(model,None,dataset=batches,**kwargs)
 write(f'{v}/attempt{attempt}/arguments.json',dict(kwargs=kwargs,upstream_commit=up['commit'],calibration_sha256=sha(cal),source_head=actual_head,
  actual_model_dtype=str(runner.model.dtype),amp_dtype=str(runner.amp_dtype),tool_version=auto_round.__version__,input_weight_policy='original FP16 rounded by official AMP construction',unique_calibration_tokens=65536))
 model,config=runner.quantize()
 assert len(block_checks)==28 and len(records)==196
 assert all(digest(t)==others[n] for n,t in list(model.named_parameters())+list(model.named_buffers()) if n not in changed)
 write(f'{v}/attempt{attempt}/projection_checks.json',records)
 files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file()}
 manifest=dict(experiment='EXP-0233',variant=v,format='software_linear_nibbles_FP32_scales',grid=[-7,7] if v=='AR-P' else [-8,7],group_size=kwargs['group_size'],
  files=files,projections=records,frozen_nontransformer=others,base_C64_manifest_sha256=sha(head_root/'manifest.json'),upstream=up['commit'],calibration_sha256=sha(cal))
 with (root/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
 write(f'{v}/package.json',dict(root=str(root),manifest_sha256=sha(root/'manifest.json'),elapsed_s=time.monotonic()-started,blocks=block_checks,source_head=actual_head))
 print('AUTOROUND_PACKAGE_COMPLETE',v,round(time.monotonic()-started,1),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['setup','oracle','export']);p.add_argument('variant',nargs='?',choices=['AR-P','AR-G']);p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
 if a.phase=='setup':setup()
 elif a.phase=='oracle':oracle()
 else:export(a.variant,a.attempt)
