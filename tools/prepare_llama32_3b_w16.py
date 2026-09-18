#!/usr/bin/env python3
"""L32-0049 original-derived FP16 export and independent composition oracle."""
import sys,os,json
from pathlib import Path
import numpy as np
import torch
from prepare_llama32_3b import ROOT,OUT,RES,preflight,save,original,tensor
from llama_reference import sha256,PROJECTIONS,layer,rms,pack_weight
from llama_u8_reference import unpack_w4_codes
R=RES.parent/'l32-0049';M=OUT.parent/'l32-0049';LONG=OUT.parent/'l32-0044/frontend64-a01'
H=3072;D=128;V=128256
def read(p):return json.loads(p.read_text())
def write(p,a):
    if p.exists():p.unlink() # unlink own hardlink, never overwrite parent payload
    if isinstance(a,torch.Tensor):a=a.detach().cpu().numpy()
    np.asarray(a).tofile(p)
def padded(p,a):
    t=np.zeros((64,H),'<f2');t[:a.shape[1]]=a[0].detach().cpu().numpy();write(p,t)
def clone(src,dst):
    dst.mkdir(parents=True,exist_ok=False)
    for n,h in read(src/'manifest.json')['files'].items():
        if n.endswith('.npy') or '_weight_w4' in n:continue
        assert sha256(src/n)==h['sha256'],n
        q=dst/n;q.parent.mkdir(parents=True,exist_ok=True);os.link(src/n,q)
def weights(p,index,original_index=None):
    original_index=index if original_index is None else original_index
    w={}
    for n,key in PROJECTIONS.items():
        v=tensor(f'model.layers.{original_index}.{key}.weight').half()
        w[f'model.layers.{index}.{key}.weight']=v.cuda()
        canonical=M/'weights'/f'layer{original_index}'/f'{n}_weight_f16_hmx.bin'
        q=p/canonical.name
        if not q.exists():os.link(canonical,q)
    for n,key in [('input','input_layernorm'),('post','post_attention_layernorm')]:
        v=tensor(f'model.layers.{original_index}.{key}.weight').half()
        w[f'model.layers.{index}.{key}.weight']=v.cuda()
        write(p/f'{n}_norm_weight_f16.bin',v)
    return w
def export_weights():
    root=M/'weights';root.mkdir(exist_ok=False)
    for i in range(28):
        p=root/f'layer{i}';p.mkdir()
        for n,key in PROJECTIONS.items():
            v=tensor(f'model.layers.{i}.{key}.weight').half()
            packed=pack_weight(v)
            # Independent inverse tile permutation: shape N32,K32,Kpair,Nlane,Khalf.
            inv=packed.reshape(v.shape[0]//32,v.shape[1]//32,16,32,2).transpose(0,3,1,2,4).reshape(v.shape)
            assert np.array_equal(inv,v.numpy()),(i,n)
            write(p/f'{n}_weight_f16_hmx.bin',packed)
        print('PACKED',i,flush=True)
    e=tensor('model.embed_tokens.weight').half()
    write(root/'generation_embedding_weight_f16.bin',e)
    write(root/'generation_lm_head_weight_f16_hmx.bin',pack_weight(e))
    manifest(root,kind='original_fp16_weights',layers=28)
def native_cache(p,kv,cap):
    # Match established hybrid cache: tiled 64-token prefix + row-major decode tail.
    rows=[]
    for a in kv[0]:
        head=pack_weight(a if p=='k' else a.T).reshape(-1)
        rows.append(np.concatenate([head,np.zeros((cap-64)*D,'<f2')]))
    return np.stack(rows).astype('<f2')
def manifest(p,**kw):
    save(p/'manifest.json',dict(experiment='L32-0049',recipe='W16A16',model='Llama-3.2-3B-Instruct',rotation=False,sp2=False,
        reference='Independent sequential FP16 mathematical composition; not a bit-exact HMX oracle',
        **kw,files={str(f.relative_to(p)):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in p.rglob('*') if f.is_file() and f.name!='manifest.json'}))
@torch.inference_mode()
def selected(first,count,cfg):
    name=f'layer{first}-a01' if count==1 else 'chain3-a01'
    src=OUT/name;p=M/name;clone(src,p)
    x=torch.from_numpy(np.fromfile(p/'reference_w4u8_block_input_f32.bin','<f4').reshape(1,64,H)).cuda().half()
    dx=torch.from_numpy(np.fromfile(p/'replay_decode_input_00_f32.bin','<f4').reshape(1,64,H)[:,:1].copy()).cuda().half()
    padded(p/'block_input_f16.bin',x);padded(p/'replay_decode_input_00_f16.bin',dx)
    for j in range(count):
        q=p/f'layer{j}';w=weights(q,j,first+j)
        x,cache=layer(x,w,j,cfg,torch.arange(64,device='cuda')[None])
        dx,full=layer(dx,w,j,cfg,torch.tensor([[64]],device='cuda'),cache)
        for n,v in zip(['k','v'],full):
            a=np.zeros((8,80,D),'<f2');write(q/f'kv_cache_{n}_f16.bin',a)
            a[:,:65]=v[0].cpu().numpy();write(q/f'reference_kv_cache_{n}_f16.bin',a)
        del w
    padded(p/'reference_f16f16_block_output_f16.bin',x);padded(p/'replay_decode_reference_00_f16.bin',dx)
    manifest(p,layers=count,source_layer=first,cache_capacity=80);print('EXPORTED',name,flush=True)
@torch.inference_mode()
def full(cfg):
    p=M/'frontend64-fixed';clone(LONG,p)
    ws=[weights(p/f'layer{i}',i) for i in range(28)]
    embed=torch.from_numpy(np.fromfile(p/'generation_embedding_weight_f16.bin','<f2').reshape(V,H)).cuda()
    head=embed
    os.link(M/'weights/generation_lm_head_weight_f16_hmx.bin',p/'generation_lm_head_weight_f16_hmx.bin')
    norm=torch.from_numpy(np.fromfile(p/'generation_final_norm_weight_f16.bin','<f2')).cuda()
    ids=np.fromfile(p/'generation_prompt_token_ids_u32.bin','<u4')
    old=read(RES.parent/'l32-0044/frontend64-a01-teacher.json')
    caches=[None]*28;tokens=[];values=[]
    for step in range(43):
        ii=ids.tolist() if not step else [old['u8_generated_ids'][step-1]]
        x=embed[torch.tensor([ii],device='cuda')]
        pos=torch.arange(64,device='cuda')[None] if not step else torch.tensor([[63+step]],device='cuda')
        if not step:padded(p/'block_input_f16.bin',x)
        for i in range(28):
            x,caches[i]=layer(x,ws[i],i,cfg,pos,caches[i])
            if not step:
                for n,v in zip(['k','v'],caches[i]):
                    a=native_cache(n,v,128);q=p/f'layer{i}'
                    write(q/f'kv_cache_{n}_hmx_f16.bin',np.zeros_like(a))
                    write(q/f'reference_kv_cache_{n}_hmx_f16.bin',a)
                    write(q/f'reference_kv_cache_{n}_hmx_f16_step00.bin',a)
        assert torch.isfinite(x).all()
        if not step:padded(p/'reference_f16f16_block_output_f16.bin',x)
        write(p/f'audit_hidden_{step:02d}_f16.bin',x[0,-1])
        z=rms(x[:,-1:],norm,cfg['rms_norm_eps']);write(p/f'audit_norm_{step:02d}_f16.bin',z)
        logits=torch.nn.functional.linear(z,head);t=int(logits.argmax());tokens.append(t);values.append(float(logits.flatten()[t]))
        print('FLOAT_REFERENCE',step,t,values[-1],flush=True)
    write(p/'generation_expected_token_ids_u32.bin',np.array(tokens+[0]*21,'<u4'))
    save(R/'frontend64-fixed-teacher.json',dict(prompt_ids=ids.tolist(),reference_ids=tokens,reference_logits=values,input_generated_ids=old['u8_generated_ids'][:43],quality_accepted=False))
    manifest(p,layers=28,cache_capacity=128,prompt_tokens=64,generation_tokens=64,validated_steps=43)
if __name__=='__main__':
    preflight();R.mkdir(exist_ok=True);M.mkdir(exist_ok=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    prov=original();cfg=prov['config']
    if sys.argv[1]=='weights':export_weights()
    elif sys.argv[1]=='selected':
        for first,count in [(0,1),(13,1),(27,1),(0,3)]:selected(first,count,cfg)
    elif sys.argv[1]=='full':full(cfg)
    else:raise ValueError(sys.argv[1])
