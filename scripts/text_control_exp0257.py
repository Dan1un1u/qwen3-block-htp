#!/usr/bin/env python3
"""Same fixed prompts, frozen W4A16 software control; no device-speed claim."""
from export_exp0257 import *
import ablate_exp0244 as b
from transformers.cache_utils import DynamicCache

def main():
 preflight();torch.set_grad_enabled(False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
 model,before,mh=b.model_session('C64');tok=AutoTokenizer.from_pretrained('/mnt/d/llm_exp/models/Qwen3-origin',local_files_only=True);rows=[]
 for sample in json.loads((R/'prompts.json').read_text())['samples']:
  cache=DynamicCache();x=torch.tensor([sample['token_ids']],device='cuda');tokens=[]
  for step in range(64):
   y=model(input_ids=x,past_key_values=cache,use_cache=True).logits[:,-1];token=int(y.argmax());tokens.append(token)
   if token in [151645,151643]:break
   x=torch.tensor([[token]],device='cuda')
  row=dict(id=sample['id'],tokens=tokens,text=tok.decode(tokens,skip_special_tokens=False));rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
 assert before==b.a.state_digest(model)
 write(R/'W4A16_prompt_control.json',dict(scope='frozen W4A16 software, same64 inputtokens, greedy up to64tokens, no speed',weights_unchanged=True,manifest_sha256=mh,prompts_sha256=sha(R/'prompts.json'),samples=rows))
if __name__=='__main__':main()
