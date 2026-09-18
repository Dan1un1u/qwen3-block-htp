"""L32-0046 immutable ordinary-A8 packages and independent references."""
import os,json,sys,functools
from pathlib import Path
import numpy as np
import torch
from llama32_3b_a8_reference import *
from llama_u8_reference import load_qparams_bin
R=RES.parent/'l32-0046';M=OUT.parent/'l32-0046'
LONG=OUT.parent/'l32-0044'
def read(p):return json.loads(p.read_text())
def replace(p,a,dtype=None):
    if p.exists():p.unlink()
    np.asarray(a,dtype=dtype).tofile(p)
def clone(src,dst):
    dst.mkdir(parents=True,exist_ok=False)
    for n,h in read(src/'manifest.json')['files'].items():
        assert sha256(src/n)==h['sha256'],n
        p=dst/n;p.parent.mkdir(exist_ok=True,parents=True);os.link(src/n,p)
def change(p,original_layer):
    q=load_qparams_bin(p/'qparams_u8.bin')
    q['middle']=read(OUT/'calibration.json')['qparams'][original_layer]['middle']
    g=(np.arange(256,dtype='f8')-q['gate']['zero_point'])*q['gate']['scale']
    u=(np.arange(256,dtype='f8')-q['up']['zero_point'])*q['up']['scale']
    z=(g/(1+np.exp(-np.clip(g,-700,700))))[:,None]*u[None,:]
    lut=np.clip(np.floor(z/q['middle']['scale']+q['middle']['zero_point']+.5),0,255)
    replace(p/'silu_up_lut_u16.bin',lut,'<u2')
    (p/'qparams_u8.bin').unlink();write_qparams(p/'qparams_u8.bin',q)
    return q
def manifest(dst,**kw):
    save(dst/'manifest.json',dict(experiment='L32-0046',model='Llama-3.2-3B-Instruct',recipe='ordinary W4A8',fp32_residual=True,rotation='OFF',**kw,files={str(p.relative_to(dst)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in dst.rglob('*') if p.is_file()}))
def padded(p,x):
    a=np.zeros((64,H),'<f4');a[:len(x)]=x;replace(p,a)
def selected(first,count):
    preflight();name=f'layer{first}-a01' if count==1 else 'chain3-a01';src=OUT/name;dst=M/name
    clone(src,dst);qs=[change(dst/f'layer{i}',first+i) for i in range(count)]
    x=np.fromfile(dst/'reference_w4u8_block_input_f32.bin','<f4').reshape(64,H)
    dx=np.fromfile(dst/'replay_decode_input_00_f32.bin','<f4').reshape(64,H)[:1]
    c=np.fromfile(dst/'rope_cos_f16.bin','<f2').reshape(64,D);ss=np.fromfile(dst/'rope_sin_f16.bin','<f2').reshape(64,D)
    dc=np.fromfile(dst/'replay_decode_rope_cos_00_f16.bin','<f2').reshape(64,D);ds=np.fromfile(dst/'replay_decode_rope_sin_00_f16.bin','<f2').reshape(64,D)
    for i,q in enumerate(qs):
        pp=dst/f'layer{i}';x,cache,diag=layer(x,pp,q,c,ss);dx,cache,ddiag=layer(dx,pp,q,dc,ds,cache)
        for phase,d in [('prefill',diag),('decode',ddiag)]:
            for n,v in d.items():
                p=dst/f'l{i}-{phase}-{n}.npy'
                if p.exists():p.unlink()
                np.save(p,v)
        for j,n in enumerate(['k','v']):
            ref=np.full((KH,80,D),q['k_rope' if n=='k' else 'v']['zero_point'],'u1');ref[:,:65]=cache[j];replace(pp/f'reference_kv_cache_{n}_u8.bin',ref)
        for phase,v in [('prefill',x),('decode',dx)]:
            p=dst/f'l{i}-{phase}-hidden.npy'
            if p.exists():p.unlink()
            np.save(p,v)
    padded(dst/'reference_w4u8_block_output_f32.bin',x);padded(dst/'replay_decode_reference_00_f32.bin',dx)
    manifest(dst,layers=count,source_layer=first,cache_capacity=80)
    print('SELECTED_REFERENCE',name,flush=True)
def full(greedy):
    preflight();torch.set_num_threads(8);name='frontend64-'+('greedy' if greedy else 'fixed');dst=M/name
    clone(LONG/'frontend64-a01',dst);qs=[change(dst/f'layer{i}',i) for i in range(28)]
    ids=np.fromfile(dst/'generation_prompt_token_ids_u32.bin','<u4')
    embed=np.memmap(dst/'generation_embedding_weight_f16.bin',dtype='<f2',mode='r',shape=(128256,H))
    gamma=np.fromfile(dst/'generation_final_norm_weight_f16.bin','<f2');gq=load_qparams_bin(dst/'generation_qparams_u8.bin')
    old=read(RES.parent/'l32-0044/frontend64-a01-teacher.json')
    cv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');caches=[None]*28;tokens=[];codes=[]
    for step in range(43):
        inp=ids if not step else [tokens[-1] if greedy else old['u8_generated_ids'][step-1]]
        x=np.array(embed[inp],dtype='f4')
        cn='rope_cos_f16.bin' if not step else f'generation_decode_rope_cos_{step-1:02d}_f16.bin'
        sn=cn.replace('_cos_','_sin_')
        c=np.fromfile(dst/cn,'<f2').reshape(64,D);ss=np.fromfile(dst/sn,'<f2').reshape(64,D)
        for i in range(28):
            x,caches[i],_=layer(x,dst/f'layer{i}',qs[i],c,ss,caches[i])
            if not step:
                for j,n in enumerate(['k','v']):
                    ref=np.full((KH,128,D),qs[i]['k_rope' if n=='k' else 'v']['zero_point'],'u1');ref[:,:64]=caches[i][j]
                    replace(dst/f'layer{i}/reference_kv_cache_{n}_u8.bin',ref)
        if not step:padded(dst/'reference_w4u8_block_output_f32.bin',x)
        replace(dst/f'audit_hidden_{step:02d}_f32.bin',x,'<f4')
        act=norm(x[-1:],gamma,gq['generation_final_norm_output'])
        logits=project_w4u8(act,dst,'generation_lm_head',128256,H,gq['generation_final_norm_output'],gq['generation_lm_head_output'],cv)[0]
        tok=int(logits.argmax());tokens.append(tok);codes.append(int(logits[tok]))
        print('ORACLE_STEP',name,step,tok,codes[-1],flush=True)
    # Loader reserves 64 entries; only 43 requested/evaluated here.
    replace(dst/'generation_expected_token_ids_u32.bin',tokens+[0]*21,'<u4')
    save(R/(name+'-teacher.json'),dict(prompt_ids=ids.tolist(),u8_generated_ids=tokens,u8_selected_codes=codes,input_generated_ids=tokens if greedy else old['u8_generated_ids'][:43],quality_accepted=False,greedy=greedy))
    manifest(dst,layers=28,generation_tokens=64,validated_steps=43,prompt_tokens=64,cache_capacity=128,teacher_sha256=sha256(R/(name+'-teacher.json')))
    print('FULL_REFERENCE',name,flush=True)
if __name__=='__main__':
    preflight();R.mkdir(exist_ok=True);M.mkdir(exist_ok=True)
    if sys.argv[1]=='selected':
        for first,count in [(0,1),(13,1),(27,1),(0,3)]:selected(first,count)
    else:full(sys.argv[1]=='greedy')
