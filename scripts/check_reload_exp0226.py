#!/usr/bin/env python3
"""Reproduce preserved RoPE reload recovery and CPU worker parity oracle."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from export_exp0226 import load_package,RESULT,OUTPUT
from rotation_exp0219 import write_json

def main():
    torch.set_grad_enabled(False);torch.set_num_threads(8);model=load_package(OUTPUT/'step000')
    data=json.loads((RESULT/'learning_data.json').read_text());old=json.loads((RESULT.parent/'exp0225/step000/validation.json').read_text());records=[]
    for i in [0,16]:
        ids=torch.tensor([data['validation'][i]['token_ids']]);scores=[]
        for threads in [8,16]:
            torch.set_num_threads(threads);h=model.model(ids[:,:-1],use_cache=False).last_hidden_state[0];loss=[]
            for start in range(0,len(h),16):
                logits=model.lm_head(h[start:start+16]).float();loss.extend(F.cross_entropy(logits,ids[0,1+start:1+start+len(logits)],reduction='none').tolist())
            scores.append(float(np.mean(loss)))
        record=dict(sample=i,threads8_nll=scores[0],threads16_nll=scores[1],original_export_nll=old['samples'][i]['nll']);records.append(record)
        assert scores[0]==scores[1]==old['samples'][i]['nll'],record
    result=dict(checks=records,rope_buffer_dtype=str(model.model.rotary_emb.inv_freq.dtype),passed=True)
    path=RESULT/'reload_oracle.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:write_json(path,result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
