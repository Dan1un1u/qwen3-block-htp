#!/usr/bin/env python3
"""Frozen C64/static-OFF A8 native package. No quantizer fitting."""
from pathlib import Path
import hashlib,json,subprocess,shutil,os,struct
import numpy as np
import torch
from transformers import AutoTokenizer
from export_exp0149_vertical_slice import write_qparams,build_silu_lut,build_attention_config
import prepare_exp0167_generation_package as head
from prepare_exp0169_long_generation_overlay import segmented_cache_bytes
S=Path('/home/daniuniu/work/qwen3-block-htp'); M=S.parent/'qwen3-block-htp-project-memory'
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0257'); O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0257')
C=O.parent/'exp0230/C64';P=R.parent/'exp0246';OLD=O.parent/'exp0247/control'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n')
def link(a,b):
 b.parent.mkdir(parents=True,exist_ok=True)
 os.link(a,b)
def preflight():subprocess.run(['python3',str(M/'scripts/project_memory.py'),'preflight','--source-worktree',str(S)],check=True)
def qp(x):return dict(scale=x['scale'],zero_point=x['zero'],minimum=x['lo'],maximum=x['hi'])
def qdqcode(x,q):return np.clip(np.floor(x.astype(np.float32)*np.float32(1/q['scale'])+np.float32(q['zero_point'])+np.float32(.5)),0,255).astype('u1')
def main():
 preflight();R.mkdir(exist_ok=True);O.mkdir(exist_ok=False)
 assert sha(C/'manifest.json')=='7de4f0758d83f2ba3b58696c695bfbfed72a25dd3bf308475abee0a0f0575a89'
 cm=json.loads((C/'manifest.json').read_text())
 for n,d in cm['files'].items():assert sha(C/n.replace('\\','/'))==d['sha256'],n
 assert sha(P/'EVIDENCE_SHA256.json')=='de6b2a5bcd10369a9c0f188ae11af115e04f951fb3f970a1b0a0d5914f9e25dc'
 seal=json.loads((P/'EVIDENCE_SHA256.json').read_text());n='parameters/A8.json';assert sha(P/n)==seal['files'][n]['sha256']
 params=json.loads((P/n).read_text())['parameters'];old=json.loads((OLD/'manifest.json').read_text())
 for n,d in old['files'].items():assert sha(OLD/n)==d['sha256'],n
 root=O/'package';root.mkdir()
 # Host allocation-only placeholders are explicitly not numerical references.
 for p in OLD.iterdir():
  if p.is_file() and p.name!='manifest.json':link(p,root/p.name)
 mapping=dict(input_norm='norm_qkv',q_projection='q_out',k_projection='k_out',v='v_out',q_rope='q_rope',k_rope='k_cache',attention_probability='attention_prob',attention_concat='attn_context',attention_projection='o_out',post_attention_residual='residual_mid',post_attention_norm='norm_mlp',gate='gate_out',up='up_out',middle='swiglu',down='down_out',block_output='residual_out')
 qs=[]
 for layer in range(28):
  d=root/f'layer{layer}';d.mkdir();base=f'L{layer:02d}.'
  q={name:qp(params[base+site]['mse']) for name,site in mapping.items()}
  q['block_input']=qp(params['L00.embedding_out' if layer==0 else f'L{layer-1:02d}.residual_out']['mse'])
  assert params[base+'v_out']['mse']==params[base+'v_cache']['mse'];qs.append(q)
  for p in (C/f'layer{layer}').iterdir():
   if p.is_file() and ('weight' in p.name):link(p,d/p.name)
  write_qparams(d/'qparams_u8.bin',q);build_silu_lut(q).tofile(d/'silu_up_lut_u16.bin')
  v=np.zeros((64,8,128),'u1');v[32:]=255
  (d/'attention_config_all_groups.bin').write_bytes(build_attention_config(v,q))
  for kind in ['k','v']:
   for pre,post in [('', ''),('reference_','_step00')]:
    (d/f'{pre}kv_cache_{kind}_hmx_u8_segmented{post}.bin').write_bytes(bytes(segmented_cache_bytes(128,kind)))
 # Compare layer0 exporter byte-for-byte with independently retained EXP0247.
 for n in ['qparams_u8.bin','silu_up_lut_u16.bin','attention_config_all_groups.bin']:
  assert (root/'layer0'/n).read_bytes()==(OLD/n).read_bytes(),n
 # Only own hardlink directory entries removed; never write into retained parents.
 for p in (root/'layer0').iterdir():
  if p.is_file() and ('weight' in p.name or p.name in ['qparams_u8.bin','silu_up_lut_u16.bin','attention_config_all_groups.bin']):
   target=root/p.name
   if target.exists():target.unlink()
   link(p,target)
 for n in ['generation_final_norm_weight_f16.bin','generation_lm_head_weight_w4_hmx.bin','generation_lm_head_weight_w4_scale_f32.bin']:
  link(C/n,root/n)
 head.FINAL_NORM_QPARAM=qp(params['head_input']['mse'])
 # Existing native-U8 logit grid is declared; software PPL does not model this head-output QDQ.
 head.export_generation_qparams(root/'generation_qparams_u8.bin')
 head.export_lm_head_bias(root/'generation_lm_head_weight_w4_hmx.bin',root/'generation_lm_head_weight_w4_scale_f32.bin',root/'generation_lm_head_bias_u32.bin',64)
 emb=np.memmap(C/'generation_embedding_weight_f16.bin',dtype='<f2',mode='r',shape=(151936,2048))
 with (root/'generation_embedding_weight_u8.bin').open('xb') as f:
  for i in range(0,151936,2048):f.write(qdqcode(emb[i:i+2048],qs[0]['block_input']).tobytes())
 tok=AutoTokenizer.from_pretrained('/mnt/d/llm_exp/models/Qwen3-origin',local_files_only=True)
 questions=[('en_fact','What is the capital of France? Answer briefly.'),('en_math','What is 12 plus 9? Answer with the number.'),('zh_fact','中国的首都是哪里？请简短回答。'),('zh_math','12加9等于多少？请只回答数字。'),('en_prose','Explain in two simple sentences why plants need sunlight.'),('zh_prose','请用两句话解释植物为什么需要阳光。')]
 # Choose a fixed fluent instruction whose rendered body is exactly63 tokens; all searches precede outputs.
 prefixes=['Please give a clear and direct answer to the following question. ','This is a short question. Please answer it clearly and directly. ','请直接回答下面的问题，不要重复问题。','']
 padding=['Please be concise. ','Use simple words. ','Answer directly. ','请简洁回答。','请回答。',' ']
 samples=[]
 import itertools
 for name,question in questions:
  found=None
  for prefix in prefixes:
   for count in range(7):
    for extra in itertools.product(padding,repeat=count):
     prompt=prefix+''.join(extra)+question
     text=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
     ids=tok.encode(text,add_special_tokens=False)
     if len(ids)==63:found=(prompt,text,[151645]+ids);break
    if found:break
   if found:break
  assert found,name
  samples.append(dict(id=name,prompt=found[0],rendered=found[1],token_ids=found[2],prefix_token_id=151645))
  print('PROMPT_FROZEN',name,found[0],flush=True)
 write(R/'prompts.json',dict(samples=samples,generation_budget=64,speed_generated_steps=16,cache_capacity=128,EOS_stop='semantic trim at first151645/151643; raw fixed-budget trace retained'))
 np.array(samples[0]['token_ids'],'<u4').tofile(root/'generation_prompt_token_ids_u32.bin')
 np.zeros(16,'<u4').tofile(root/'generation_expected_token_ids_u32.bin')
 inv=1/(1000000.0**(torch.arange(0,128,2,dtype=torch.float32)/128.0));phase=torch.arange(128,dtype=torch.float32)[:,None]*inv[None,:];phase=torch.cat([phase,phase],-1)
 for kind,array in [('cos',phase.cos().half().numpy()),('sin',phase.sin().half().numpy())]:
  p=root/f'rope_{kind}_f16.bin';p.unlink();array[:64].tofile(p)
  for step in range(64):
   z=np.zeros((64,128),'<f2');z[0]=array[64+step];z.tofile(root/f'generation_decode_rope_{kind}_{step:02d}_f16.bin')
 dm=dict(experiment='EXP-0257',capacity=128,C64_manifest_sha256=sha(C/'manifest.json'),parameters_sha256=sha(P/'parameters/A8.json'),layer_qparams=qs,head_input=head.FINAL_NORM_QPARAM,head_output=head.LM_HEAD_QPARAM,head_output_vs_software='existing native U8 logits grid; not included in software PPL',files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()})
 write(root/'manifest.json',dm)
 write(R/'export_audit.json',dict(pass_all=True,C64_files_verified=len(cm['files']),layer0_exact_parent=True,weights_unchanged=True,residual_transition_qparams_exact=True,manifest_sha256=sha(root/'manifest.json'),prompts_sha256=sha(R/'prompts.json'),head_output=head.LM_HEAD_QPARAM))
 print('EXPORT_PASS',len(dm['files']),flush=True)
if __name__=='__main__':main()
