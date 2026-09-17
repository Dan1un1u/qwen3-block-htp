#!/usr/bin/env python3
"""L32-0041 fresh 3B per-channel GPTQ/SP2 preparation; no 1B weights/ranges."""
import argparse, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from llama_reference import PROJECTIONS, sha256, rms, rope, rotate
from quantize_llama32 import factor, export, oracle
ROOT=Path(__file__).resolve().parents[1]
MODEL=Path('/mnt/d/llm_exp/models/llama3.2-3B-Instruct-origin')
OUT=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0041')
RES=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0041')
DATA=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0002/data')
PY=Path('/home/daniuniu/.cache/qwen3-block-htp-spinquant-py/bin/python')
def save(p,z):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(z,f,indent=2,allow_nan=False)
def preflight():
    subprocess.run(['python3',str(ROOT)+'-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
def original():
    z=json.loads((RES/'original-provenance.json').read_text())
    for n,h in z['files'].items(): assert sha256(MODEL/n)==h['sha256'],n
    return z
def tensor(name):
    idx=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(MODEL/idx[name],framework='pt',device='cpu') as f:return f.get_tensor(name)
def attention(x,w,cfg,collect=None):
    b,m,_=x.shape;nh=cfg['num_attention_heads'];kh=cfg['num_key_value_heads'];hd=cfg['head_dim']
    n=rms(x,w['input_layernorm.weight'],cfg['rms_norm_eps']);c,s=rope(cfg,torch.arange(m,device=x.device)[None],x.dtype)
    q=F.linear(n,w['self_attn.q_proj.weight']);k=F.linear(n,w['self_attn.k_proj.weight']);v=F.linear(n,w['self_attn.v_proj.weight'])
    if collect:
        for name,t in [('block_input',x),('input_norm',n),('q_projection',q),('k_projection',k),('v',v)]:collect(name,t)
    q=rotate(q.view(b,m,nh,hd).transpose(1,2),c,s);k=rotate(k.view(b,m,kh,hd).transpose(1,2),c,s)
    if collect:collect('q_rope',q);collect('k_rope',k)
    kk=k.repeat_interleave(nh//kh,1);vv=v.view(b,m,kh,hd).transpose(1,2).repeat_interleave(nh//kh,1)
    scores=(q@kk.transpose(-1,-2))*(hd**-.5)
    scores.masked_fill_(torch.ones(m,m,dtype=torch.bool,device=x.device).triu(1),torch.finfo(x.dtype).min)
    av=(scores.float().softmax(-1).half()@vv).transpose(1,2).contiguous().view(b,m,cfg['hidden_size'])
    if collect:collect('attention_concat',av)
    return av
def qp(lo,hi):
    lo=min(0.,lo);hi=max(0.,hi);s=np.float32(max(hi-lo,1e-8)/255);z=int(np.clip(np.floor(-lo/s+.5),0,255))
    return dict(scale=float(s),zero_point=z,minimum=lo,maximum=hi)
@torch.inference_mode()
def quantize(resume=False):
    preflight();prov=original();cfg=prov['config'];torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    # Same immutable calibration token corpus only; tokenizer identity checked.
    old=json.loads((ROOT/'tools/llama_reference.py').read_text().split('EXPECTED = ',1)[1].split('\nPROJECTIONS',1)[0].replace("'",'"')) if False else None
    from llama_reference import EXPECTED
    assert prov['files']['tokenizer.json']['sha256']==EXPECTED['tokenizer.json']
    for n,h in json.loads((DATA/'freeze.json').read_text()).items():assert sha256(DATA/n)==h
    if not resume:OUT.mkdir(parents=True,exist_ok=False)
    save(RES/'quantizer-oracle.json',oracle()) if not (RES/'quantizer-oracle.json').exists() else None
    ids=np.fromfile(DATA/'calibration.bin',dtype='<u4').reshape(512,128)
    embed=tensor('model.embed_tokens.weight').half()
    hidden=F.embedding(torch.from_numpy(ids.astype('i8')),embed).numpy()
    started=time.monotonic()
    for i in range(28):
        path=OUT/f'layer{i}'
        if (path/'complete.json').exists():
            done=json.loads((path/'complete.json').read_text())
            for n,h in done['files'].items():assert sha256(path/n)==h,n
            hidden=np.load(path/'hidden.npy');print('RESUMED',i,flush=True);continue
        path.mkdir(exist_ok=False)
        w={k:tensor(f'model.layers.{i}.'+k).half().cuda() for k in [v+'.weight' for v in PROJECTIONS.values()]+['input_layernorm.weight','post_attention_layernorm.weight']}
        orig={n:w[k+'.weight'].float().clone() for n,k in PROJECTIONS.items()}
        x=torch.from_numpy(hidden).cuda();norm=rms(x,w['input_layernorm.weight'],cfg['rms_norm_eps']);fac=factor(norm);stats={}
        for n in ['q','k','v']:
            val,stats[n]=export(orig[n],norm,fac,path,n);w[PROJECTIONS[n]+'.weight'].copy_(val)
        del norm,fac,val
        av=torch.cat([attention(x[j:j+4],w,cfg) for j in range(0,512,4)])
        fac=factor(av);val,stats['o']=export(orig['o'],av,fac,path,'o');w['self_attn.o_proj.weight'].copy_(val);del fac
        res=x+F.linear(av,val);del av,val
        post=rms(res,w['post_attention_layernorm.weight'],cfg['rms_norm_eps']);fac=factor(post)
        for n in ['gate','up']:
            val,stats[n]=export(orig[n],post,fac,path,n);w[PROJECTIONS[n]+'.weight'].copy_(val)
        del fac,val
        middle=F.silu(F.linear(post,w['mlp.gate_proj.weight']))*F.linear(post,w['mlp.up_proj.weight']);del post
        fac=factor(middle);val,stats['down']=export(orig['down'],middle,fac,path,'down');w['mlp.down_proj.weight'].copy_(val)
        next_hidden=res+F.linear(middle,val);assert torch.isfinite(next_hidden).all()
        del middle,res,fac,orig,val
        ranges={}
        def collect(n,t):
            assert torch.isfinite(t).all(),(i,n)
            lo,hi=float(t.min()),float(t.max());prior=ranges.get(n,[0.,0.]);ranges[n]=[min(prior[0],lo),max(prior[1],hi)]
        for j in range(0,512,4):
            xx=x[j:j+4];av=attention(xx,w,cfg,collect)
            o=F.linear(av,w['self_attn.o_proj.weight']);rr=xx+o;post=rms(rr,w['post_attention_layernorm.weight'],cfg['rms_norm_eps'])
            g=F.linear(post,w['mlp.gate_proj.weight']);u=F.linear(post,w['mlp.up_proj.weight']);mid=F.silu(g)*u;down=F.linear(mid,w['mlp.down_proj.weight'])
            for n,t in [('attention_projection',o),('post_attention_residual',rr),('post_attention_norm',post),('gate',g),('up',u),('middle',mid),('down',down),('block_output',rr+down)]:collect(n,t)
        for n,k in [('input','input_layernorm'),('post','post_attention_layernorm')]:w[k+'.weight'].cpu().numpy().astype('<f2').tofile(path/f'{n}_norm_weight_f16.bin')
        hidden=next_hidden.cpu().numpy();np.save(path/'hidden.npy',hidden)
        save(path/'ranges.json',ranges);save(path/'complete.json',dict(layer=i,stats=stats,files={p.name:sha256(p) for p in path.iterdir() if p.is_file()}))
        print('QUANTIZED_CALIBRATED',i,'seconds',round(time.monotonic()-started,1),flush=True)
        del w,x,next_hidden,av,o,rr,post,g,u,mid,down,xx;t=None;torch.cuda.empty_cache()
    head=OUT/'head'
    if not (head/'complete.json').exists():
        head.mkdir(exist_ok=False);gamma=tensor('model.norm.weight').half().cuda()
        x=rms(torch.from_numpy(hidden).cuda(),gamma,cfg['rms_norm_eps']);fac=factor(x)
        value,stats=export(embed.cuda().float(),x,fac,head,'generation_lm_head',three=False)
        hr={}
        last=x[:,-1]
        for n,t in [('generation_final_norm_output',last),('generation_lm_head_output',F.linear(last,value))]:hr[n]=[float(t.min()),float(t.max())]
        save(head/'ranges.json',hr);save(head/'complete.json',dict(stats=stats,files={p.name:sha256(p) for p in head.iterdir() if p.is_file()}))
    qparams=[]
    for i in range(28):
        q={n:qp(*v) for n,v in json.loads((OUT/f'layer{i}/ranges.json').read_text()).items()};q['attention_probability']=qp(0,1)
        if i:q['block_input']=qparams[-1]['block_output'].copy()
        qparams.append(q)
    save(OUT/'calibration.json',dict(experiment='L32-0041',qparams=qparams,generation_qparams={n:qp(*v) for n,v in json.loads((head/'ranges.json').read_text()).items()},data_freeze_sha256=sha256(DATA/'freeze.json'),original_provenance_sha256=sha256(RES/'original-provenance.json'),method='Same C64 true-sequential GPTQ act-order damping1%, three final-output per-channel ranges; head absmax GPTQ. Fresh static affine minmax A8. Shared tokenizer-identical calibration tokens only.'))
    save(OUT/'manifest.json',dict(experiment='L32-0041',model='Llama-3.2-3B-Instruct',files={str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file()}))
    print('QUANTIZATION_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args();quantize(a.resume)
