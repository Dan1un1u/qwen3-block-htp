#!/usr/bin/env python3
"""Additional fixed-weight controls and prefix-vs-body boundary statistics."""
import json,math
import torch
from transformers import AutoTokenizer
from llama32_smooth_accuracy import *
@torch.no_grad()
def diagnostics():
    model=load_quant();q=json.loads((MOD/'carriers.json').read_text());model.qparams=q
    tok=AutoTokenizer.from_pretrained(ORIG,local_files_only=True)
    ids=tok.apply_chat_template([dict(role='user',content='What is the capital of France? Answer briefly.')],tokenize=True,add_generation_prompt=True,date_string='14 Sep 2026')
    # Same first validation M64 input across isolated boundary diagnostic.
    sample=json.loads((OLD/'dataset.json').read_text())['samples'][0];prompt=torch.tensor([sample['prompt_ids']],device='cuda')
    rows=[]
    def capture(name,x):
        if not name.endswith(('.middle_pre_r4','down.input','.down','block_output','post_attention_residual')):return
        if x.ndim!=3:return
        t=x.float().reshape(-1,x.shape[-1]);row=dict(name=name,first_rms=float(t[0].square().mean().sqrt()),body_rms=float(t[1:].square().mean().sqrt()),absmax=float(t.abs().max()))
        if name in q:
            y=uq(t,q[name]);row.update(body_zero_fraction=float((y[1:]==0).float().mean()),body_nrmse=float((y[1:]-t[1:]).norm()/t[1:].norm().clamp_min(1e-30)),step=q[name]['scale'])
        if name.endswith('down.input'):
            step=model.steps[name.removesuffix('.input')];y=aq(t,step);row.update(input_step=step,body_nrmse=float((y[1:]-t[1:]).norm()/t[1:].norm().clamp_min(1e-30)),body_zero_fraction=float((y[1:]==0).float().mean()))
        rows.append(row)
    model.observer=capture;model(prompt);model.observer=None
    save(OUT/'boundary-diagnostic.json',dict(scope='isolated U8 on same input-A8 control trajectory, first frozen validation bridge window',rows=rows))
    generated=[];cache=None
    for i in range(24):
        x=torch.tensor([ids if i==0 else [generated[-1]]],device='cuda');logits,cache=model(x,cache,select=slice(-1,None));generated.append(int(logits[0,-1].argmax()))
        if generated[-1] in [128001,128008,128009]:break
    save(OUT/'input-generation.json',dict(ids=generated,text=tok.decode(generated,skip_special_tokens=True),scope='BF16 input-only A8 control'))
    log('INPUT_TEXT',tok.decode(generated))
if __name__=='__main__':
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    for weights,mode,scope in [('smooth','float','bridge'),('smooth','float','full'),('gptq','down_a16','full'),('gptq','except_residual','bridge')]:evaluate(weights,mode,scope)
    diagnostics()
