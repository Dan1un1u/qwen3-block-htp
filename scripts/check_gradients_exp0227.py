#!/usr/bin/env python3
"""Compare loss-scaled gradients on identical real block inputs/FP16 forwards."""
import copy,json
import torch
from block_reconstruction_exp0227 import *


def main():
    preflight();torch.set_num_threads(16);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    train,_=data();idx=batch_schedule()[0];ids=torch.tensor([train[i]['token_ids'] for i in idx])
    model=load_model(ev.MODEL,torch.float16).half()
    for p in model.parameters():p.requires_grad_(False)
    with torch.no_grad():
        a=model.model.embed_tokens(ids).cuda();b=a.clone()
        rope=model.model.rotary_emb.cuda()(a,torch.arange(128,device='cuda')[None,:])
        mask=torch.full((128,128),torch.finfo(torch.float16).min,dtype=torch.float16,device='cuda').triu(1)[None,None]
    for i in range(4):
        teacher=model.model.layers[i].cuda();student=copy.deepcopy(teacher);modules={}
        for short,long in rot.PROJECTIONS.items():
            codes,scales=read_codes(CONTROLS['A']/f'layer{i}',short,student.get_submodule(long).weight.shape)
            m=ScaleLinear(codes,scales).cuda();parent,name=long.rsplit('.',1)
            setattr(student.get_submodule(parent),name,m);modules[short]=m
        with torch.no_grad():target=layer_forward(teacher,a,rope,mask)
        if i==3:
            rows={};gradients={}
            for scale in [1.,1024.,65536.]:
                student.zero_grad(set_to_none=True)
                pred=layer_forward(student,b,rope,mask);loss=rel_mse(pred,target)
                (loss*scale).backward()
                gradients[scale]={n:m.theta.grad.detach().float().clone()/scale for n,m in modules.items()}
                rows[str(scale)]={n:dict(norm=float(g.norm()),nonzero=int(torch.count_nonzero(g)),total=g.numel(),finite=bool(torch.isfinite(g).all())) for n,g in gradients[scale].items()}
            comparisons={}
            for scale in [1.,1024.]:
                x=torch.cat(list(gradients[scale].values()));y=torch.cat(list(gradients[65536.].values()))
                comparisons[str(scale)]=dict(relative_difference=float((x-y).norm()/y.norm()),cosine=float(torch.nn.functional.cosine_similarity(x,y,dim=0)))
            rot.write_json(RESULT/'gradient_precision_audit.json',dict(layer=3,loss=float(loss.detach()),same_forward=True,gradient_stats=rows,comparisons_to_65536=comparisons))
            print(json.dumps(dict(stats=rows,comparisons=comparisons),indent=2));return
        with torch.no_grad():b=layer_forward(student,b,rope,mask);a=target
        teacher.cpu();del student,modules


if __name__=='__main__':main()
