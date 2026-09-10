#!/usr/bin/env python3
"""Fresh W4 software PPL and post-EOS diagnostic, separate from original teacher."""
import json,math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from llama_quantized import load_quantized
from llama_reference import sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 root=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0002');ref=root/'frontend-reference-a01'
 ds=json.loads((ref/'dataset.json').read_text());teacher=json.loads((ref/'teacher.json').read_text())
 m=AutoModelForCausalLM.from_pretrained('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin',torch_dtype=torch.float16,attn_implementation='eager',local_files_only=True).cuda().eval()
 qm=load_quantized(m,Path('/mnt/d/llm_exp/models/llama32-htp/l32-0002/quant-a01'));rows=[]
 for s in ds['samples']:
  ids=torch.tensor([s['prompt_ids']+s['target_ids'][:-1]],device='cuda');t=torch.tensor(s['target_ids'],device='cuda');l=m(ids,use_cache=False).logits[0,63:79].float();nll=l.logsumexp(-1)-l.gather(-1,t[:,None]).squeeze(-1);rows.append({'id':s['id'],'language':s['language'],'nll':nll.tolist()})
 out={'recipe':'W4A16','reference':'FP16 software executing freshly exported W4 weights','nll':rows,'ppl':{}}
 for lang in ['all','en','zh']:
  vals=[v for s in rows if lang=='all' or s['language']==lang for v in s['nll']];out['ppl'][lang]=math.exp(sum(vals)/len(vals))
 ids=teacher['prompt_ids']+teacher['fp16_generated_ids'][:12];l=m(torch.tensor([ids],device='cuda'),use_cache=False).logits[0,-1].float();v,k=l.topk(10)
 out['post_eos_step12_diagnostic']={'top_ids':k.tolist(),'logits':v.tolist(),'top2_margin':float(v[0]-v[1]),'scope':'after first EOS at generation step7; the retained16-step greedy replay is still failed'}
 with (root/'software_quality.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps({'ppl':out['ppl'],'post_eos':out['post_eos_step12_diagnostic']}),flush=True)
if __name__=='__main__':main()
