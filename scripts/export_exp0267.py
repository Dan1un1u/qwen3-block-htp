#!/usr/bin/env python3
"""Frozen Qwen3 C64 SP2 metadata and genuine BF16-teacher replay inputs."""
import json,os,subprocess,sys,hashlib,struct,contextlib
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from export_exp0257 import sha,write,preflight,qdqcode
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from llama_u8_reference import load_qparams_bin,unpack_w4_codes,QPARAM_RECORD
S=Path('/home/daniuniu/work/qwen3-block-htp');R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0267');O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0267')
P=O.parent/'exp0257/package';C=O.parent/'exp0230/C64';OLD=O.parent/'exp0247/control'
def link(a,b):b.parent.mkdir(parents=True,exist_ok=True);os.link(a,b)
def main():
 preflight();torch.set_grad_enabled(False);torch.set_num_threads(8);O.mkdir(exist_ok=False)
 # Existing authority evidence pins both the C64 weights and A8 package.
 assert sha(C/'manifest.json')=='7de4f0758d83f2ba3b58696c695bfbfed72a25dd3bf308475abee0a0f0575a89'
 assert sha(P/'manifest.json')=='ab2442027f315797dc7ff8f9c49b0d6bf27a3de3c958e831eddf4e1953f915bb'
 pm=json.loads((P/'manifest.json').read_text());cm=json.loads((C/'manifest.json').read_text())
 for n,v in pm['files'].items():assert sha(P/n)==v['sha256'],n
 old=json.loads((OLD/'manifest.json').read_text())
 for n,v in old['files'].items():assert sha(OLD/n)==v['sha256'],n
 original=Path('/mnt/d/llm_exp/models/Qwen3-origin')
 for n,h in cm['original_shards'].items():assert sha(original/n)==h,n
 ids=old['token_ids'];assert len(ids)==72
 import export_exp0022_block as base
 base.M=72
 index=json.loads((original/'model.safetensors.index.json').read_text());captured={}
 with contextlib.ExitStack() as stack:
  shards={n:stack.enter_context(safe_open(original/n,framework='pt',device='cpu')) for n in cm['original_shards']}
  def weight(k):return shards[index['weight_map'][k]].get_tensor(k)
  hidden=torch.nn.functional.embedding(torch.tensor([ids]),weight('model.embed_tokens.weight'))
  cos,sin=base.rope_tables(torch.bfloat16)
  for layer in range(28):
   if layer in [0,14,27]:captured[layer]=hidden.float().numpy().copy()[0]
   if layer<27:hidden,_=base.layer_forward_bf16(hidden,weight,layer,cos,sin,False)
   print('TEACHER_LAYER',layer,flush=True)
 (R/'teacher-inputs').mkdir(exist_ok=False)
 for layer,x in captured.items():np.save(R/'teacher-inputs'/f'layer{layer}.npy',x)
 write(R/'teacher-inputs/manifest.json',dict(original_shards=cm['original_shards'],token_ids=ids,layers=[0,14,27],scope='BF16 complete preceding stack supplies inputs only; each device layer computes its own persistent KV; not model-quality scoring',files={p.name:sha(p) for p in (R/'teacher-inputs').glob('*.npy')}))
 levels=np.array(sorted({0}|{s*2**p for s in [-1,1] for p in range(15)}|{s*(2**p+2**q) for s in [-1,1] for p in range(15) for q in range(p)}),dtype='i4');assert len(levels)==241
 write(R/'codebook.json',dict(levels=levels.tolist(),rule='nearest; ties lower numeric; alpha=f32(maxabs frozen middle/24576)',frozen_before_device=True))
 full=O/'sp2';full.mkdir();audit=[];changed=[]
 for n in pm['files']:
  if Path(n).name in ['qparams_u8.bin','silu_up_lut_u16.bin']:continue
  link(P/n,full/n)
 for layer in range(28):
  d=full/f'layer{layer}';q=load_qparams_bin(P/f'layer{layer}/qparams_u8.bin');alpha=float(np.float32(max(abs(q['middle']['minimum']),abs(q['middle']['maximum']))/24576))
  g=(np.arange(256,dtype='f8')-q['gate']['zero_point'])*q['gate']['scale'];u=(np.arange(256,dtype='f8')-q['up']['zero_point'])*q['up']['scale'];x=(g/(1+np.exp(-np.clip(g,-700,700))))[:,None]*u[None,:];grid=levels.astype('f8')*alpha
  ix=np.searchsorted(grid,x).clip(1,len(grid)-1);ix-=np.abs(x-grid[ix-1])<=np.abs(x-grid[ix]);lut=levels[ix]
  # Independent exhaustive nearest-code enumeration; lower numeric tie by argmin.
  for first in range(0,256,8):assert np.array_equal(lut[first:first+8],levels[np.abs(x[first:first+8,:,None]-grid).argmin(-1)])
  (lut+32768).astype('<u2').tofile(d/'silu_up_lut_u16.bin')
  raw=(P/f'layer{layer}/qparams_u8.bin').read_bytes();recs=[]
  for off in range(0,len(raw),QPARAM_RECORD.size):
   rec=list(QPARAM_RECORD.unpack_from(raw,off))
   if rec[0].split(b'\0')[0]==b'middle':rec[1]=alpha;rec[2]=0
   recs.append(QPARAM_RECORD.pack(*rec))
  (d/'qparams_u8.bin').write_bytes(b''.join(recs))
  w=unpack_w4_codes(P/f'layer{layer}','down',2048,6144).astype('i8');ws=np.fromfile(P/f'layer{layer}/down_weight_w4_scale_f32.bin',dtype='<f4').astype('f8');mult=np.floor(alpha*ws/q['down']['scale']*2**31+.5).astype('i8')
  pos=np.maximum(w,0).sum(1)*255;neg=np.minimum(w,0).sum(1)*255;bound=np.abs(w).sum(1)*24576
  assert pos.max()<2**23 and neg.min()>=-2**23 and bound.max()<2**31 and np.all((mult>0)&(mult<2**31))
  audit.append(dict(layer=layer,alpha=alpha,mult_min=int(mult.min()),mult_max=int(mult.max()),partial_positive_max=int(pos.max()),partial_negative_min=int(neg.min()),acc_abs_bound=int(bound.max()),lut_pairs=65536,lut_sha256=sha(d/'silu_up_lut_u16.bin')))
  changed += [f'layer{layer}/qparams_u8.bin',f'layer{layer}/silu_up_lut_u16.bin'];print('SP2_LAYER',layer,flush=True)
 for n in ['qparams_u8.bin','silu_up_lut_u16.bin']:link(full/'layer0'/n,full/n);changed.append(n)
 write(full/'manifest.json',dict(experiment='EXP-0267',layers=28,sp2=True,parent_manifest_sha256=sha(P/'manifest.json'),changed_files=changed,files={str(p.relative_to(full)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(full.rglob('*')) if p.is_file()}))
 # Isolated selected-layer packages retain original allocation/reference placeholders;
 # changed output validation is independently performed on untimed real carriers.
 for layer in [0,14,27]:
  for arm in ['u8','sp2']:
   d=O/f'layer{layer}-{arm}';d.mkdir();ld=d/'layer0';ld.mkdir();src=(P if arm=='u8' else full)/f'layer{layer}'
   for n in old['files']:
    if '/' in n:continue
    if 'weight' in n or n in ['qparams_u8.bin','silu_up_lut_u16.bin','attention_config_all_groups.bin']:a=src/n
    else:a=OLD/n
    if n=='reference_w4u8_block_input_u8.bin' or n.startswith('replay_decode_input_'):continue
    link(a,d/n)
   q=load_qparams_bin(src/'qparams_u8.bin')['block_input'];enc=qdqcode(captured[layer],q);enc[:64].tofile(d/'reference_w4u8_block_input_u8.bin')
   for i in range(8):
    z=np.full((64,2048),q['zero_point'],dtype='u1');z[0]=enc[64+i];z.tofile(d/f'replay_decode_input_{i:02d}_u8.bin')
   for f in list(d.iterdir()):
    if f.is_file():link(f,ld/f.name)
   for n in pm['files']:
    if n.startswith('layer0/') and 'kv_cache' in n:link(P/n,ld/Path(n).name)
   write(d/'manifest.json',dict(experiment='EXP-0267',actual_layer=layer,sp2=arm=='sp2',teacher_input_sha256=sha(R/'teacher-inputs'/f'layer{layer}.npy'),token_ids=ids,files={str(p.relative_to(d)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(d.rglob('*')) if p.is_file()}))
 write(R/'export_audit.json',dict(pass_all=True,parent_manifest_sha256=sha(P/'manifest.json'),sp2_manifest_sha256=sha(full/'manifest.json'),changed_files=changed,layer_bounds=audit,weight_changes=0,non_middle_qparam_changes=0,models=str(O),teacher_input_reference='original BF16 Qwen3 preceding stack',prefix='existing C64 OFF prefix frozen for both fullmodel arms; not regenerated through SP2'))
 print('EXPORT_PASS',flush=True)
if __name__=='__main__':main()
