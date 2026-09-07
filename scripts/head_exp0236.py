#!/usr/bin/env python3
"""Head-only CPU GPTQ export from frozen G64 64K inputs."""
import argparse,gc,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from data_exp0236 import RESULT,OUTPUT,SOURCE,MODEL,GROOT,CHECKPOINT,write,sha,verified,preflight,frozen
from gptq_exp0221 import factor,pack_codes
import group_exp0234 as group
import output_scale_exp0224 as channel
from experiment_exp0220 import metrics
from evaluate_exp0234 import load as load_g64

def head_read(root,grouped):
 n,k=151936,2048
 raw=np.fromfile(root/'codes.bin',dtype=np.uint8);assert raw.size==n*k//2
 nib=np.empty(raw.size*2,dtype=np.uint8);nib[::2]=raw&15;nib[1::2]=raw>>4
 codes=nib.reshape(n//32,k//32,8,32,4).transpose(0,1,2,4,3).reshape(n//32,k//32,32,32).transpose(0,3,1,2).reshape(n,k).astype(np.int8)
 codes[codes>=8]-=16;assert codes.min()>=-7 and codes.max()<=7
 scale=np.fromfile(root/'scales.bin',dtype='<f4').reshape(n,16 if grouped else 1)
 assert np.isfinite(scale).all() and (scale>0).all()
 value=(codes.astype(np.float32).reshape(n,scale.shape[1],-1)*scale[:,:,None]).reshape(n,k).astype(np.float16)
 assert np.isfinite(value).all();return value

def prepare_inputs():
 preflight();frozen();assert json.loads((RESULT/'independent_data_audit.json').read_text())['pass_all']
 assert not (OUTPUT/'head_inputs.npy').exists()
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
 origin=json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())
 for n,h in origin.items():assert sha(MODEL/n)==(h['sha256'] if isinstance(h,dict) else h),n
 write('quantizer_oracles.json',dict(channel=channel.check(),group=group.check(),pass_all=True))
 print('HEAD_QUANTIZER_ORACLES_PASS',flush=True)
 model,mh=load_g64('G64','cpu');assert sha(GROOT/'manifest.json')==mh
 hidden=torch.from_numpy(np.load(CHECKPOINT));assert hidden.shape==(512,128,2048) and torch.isfinite(hidden).all()
 x=model.model.norm(hidden);assert x.dtype==torch.float16 and torch.isfinite(x).all()
 a=hidden.numpy().astype(np.float32);normal=a/np.sqrt(np.mean(a*a,axis=-1,keepdims=True,dtype=np.float32)+model.model.norm.variance_epsilon)
 ref=(normal.astype(np.float16)*model.model.norm.weight.numpy()).astype(np.float16)
 check=metrics(x,torch.from_numpy(ref));assert check['finite'] and check['nrmse']<=.003 and check['cosine']>=.99999,check
 del a,normal,ref,hidden;gc.collect()
 raw=verified('exp0230','inputs/C64_calibration_u32.bin');ids=torch.from_numpy(np.fromfile(raw,dtype='<u4').astype(np.int64).reshape(512,128));selected=[0,129,258,387]
 replay=model.model(input_ids=ids[selected],use_cache=False).last_hidden_state
 checks=[]
 for j,i in enumerate(selected):
  c=dict(sample=i,**metrics(x[i:i+1],replay[j:j+1]));checks.append(c)
 write('head_input_replay.json',dict(source_head=source,checkpoint_sha256=sha(CHECKPOINT),G64_manifest_sha256=mh,calibration_sha256=sha(raw),independent_numpy_final_norm=check,canonical_CPU_replay=checks))
 assert all(c['finite'] and c['nrmse']<=.003 and c['cosine']>=.99999 for c in checks),checks
 np.save(OUTPUT/'head_inputs.npy',x.numpy())
 write('head_inputs.json',dict(path=str(OUTPUT/'head_inputs.npy'),sha256=sha(OUTPUT/'head_inputs.npy'),shape=list(x.shape),dtype='float16',head_input_replay_sha256=sha(RESULT/'head_input_replay.json'),source_head=source))
 print('HEAD_INPUTS_PASS',checks,flush=True)

def export(v):
 preflight();frozen();assert json.loads((RESULT/'quantizer_oracles.json').read_text())['pass_all']
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip();start=time.monotonic();root=OUTPUT/v;assert not root.exists()
 rec=json.loads((RESULT/'head_inputs.json').read_text());assert sha(rec['path'])==rec['sha256'];x=torch.from_numpy(np.load(rec['path']));f=factor(x)
 origin=json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())
 for n,h in origin.items():assert sha(MODEL/n)==(h['sha256'] if isinstance(h,dict) else h),n
 index=json.loads((MODEL/'model.safetensors.index.json').read_text());shard=MODEL/index['weight_map']['lm_head.weight']
 with safe_open(shard,framework='pt',device='cpu') as sf:w=sf.get_tensor('lm_head.weight').float()
 assert w.shape==(151936,2048) and torch.isfinite(w).all();original_hash=hashlib.sha256(w.numpy().tobytes()).hexdigest()
 root.mkdir();stats=[];reconstructed=hashlib.sha256();select=group.select_output if v=='H64' else channel.select_output
 with (root/'codes.bin').open('xb') as cp,(root/'scales.bin').open('xb') as sp:
  for first in range(0,len(w),1024):
   codes,scale,r=select(w[first:first+1024],x,f);stats.append(r)
   q=(codes.float()*(group.expand(scale) if v=='H64' else scale[:,None])).half()
   assert torch.isfinite(q).all();reconstructed.update(q.numpy().tobytes())
   cp.write(pack_codes(codes).tobytes());sp.write(scale.numpy().astype('<f4').tobytes());cp.flush();sp.flush()
   if first%4096==0 or first+1024>=len(w):print('HEAD_EXPORT',v,min(first+1024,len(w)),len(w),'elapsed_s',round(time.monotonic()-start,1),flush=True)
 rows={k:np.concatenate([r[k] for r in stats]) for k in stats[0]};np.savez_compressed(root/'selection.npz',**rows)
 decoded=head_read(root,v=='H64');assert hashlib.sha256(decoded.tobytes()).hexdigest()==reconstructed.hexdigest()
 assert hashlib.sha256(w.numpy().tobytes()).hexdigest()==original_hash
 assert np.all(rows['selected_output_sse']<=rows['candidate_output_sse'][:,0])
 assert np.array_equal(rows['choice'],rows['candidate_output_sse'].argmin(1))
 manifest=dict(experiment='EXP-0236',variant=v,only_changed_tensor='lm_head.weight',W4=True,mixed_precision=False,groupsize=128 if v=='H64' else -1,grid=[-7,7],scale_dtype='float32',activation_dtype='float16',frozen_G64_manifest_sha256=sha(GROOT/'manifest.json'),head_inputs_sha256=rec['sha256'],original_shard=str(shard),original_shard_sha256=sha(shard),original_FP32_head_sha256=original_hash,source_head=source,factor=f['stats'],files={n:dict(sha256=sha(root/n),bytes=(root/n).stat().st_size) for n in ['codes.bin','scales.bin','selection.npz']},dequant_FP16_sha256=reconstructed.hexdigest(),independent_numpy_all_rows_exact=True,choice_histogram=np.bincount(rows['choice'],minlength=3).tolist(),selected_output_sse=float(rows['selected_output_sse'].sum()),candidate_output_sse=rows['candidate_output_sse'].sum(0).tolist(),elapsed_s=time.monotonic()-start,quantizer_source={n:sha(SOURCE/'scripts'/n) for n in ['head_exp0236.py','gptq_exp0221.py','clipping_exp0223.py','group_exp0234.py','output_scale_exp0224.py']})
 with (root/'manifest.json').open('x') as out:json.dump(manifest,out,indent=2);out.write('\n')
 write(v+'/package.json',dict(manifest_sha256=sha(root/'manifest.json'),**manifest));print('HEAD_PACKAGE_COMPLETE',v,manifest['elapsed_s'],flush=True)
if __name__=='__main__':
 torch.set_num_threads(16);torch.set_grad_enabled(False)
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['inputs','P64','H64']);a=p.parse_args()
 prepare_inputs() if a.phase=='inputs' else export(a.phase)
