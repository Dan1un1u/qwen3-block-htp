#!/usr/bin/env python3
"""Frozen conditional PPL ladder, with actual integer attention boundaries."""
import argparse, json, math, time, subprocess
from pathlib import Path
import numpy as np
import torch
import r3_exp0246 as r
import data_exp0253 as data
import floating_attention_exp0253 as fa
import integer_attention_exp0252 as ia
from prepare_exp0042_attention import CONFIG
from export_exp0149_vertical_slice import build_attention_config

ROOT=data.RESULT
MODES=['legacy','wide_nr64','float_core']
NAMES=['F','C64']+[x+'_'+m for x in ['OFF','R3'] for m in MODES]
def write(n,d):data.write(n,d)
def read(n):return json.loads((ROOT/n).read_text())
def parent(n):return data.verified('exp0246',n)
def preflight():
    data.preflight()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=data.SOURCE,text=True).strip()=='codex/exp-0253-float-attention'
def freeze():
    data.frozen();parameters={};configs={};refs={}
    for name in ['F','C64','A8','R3_ALL']:
        p=parent(f'parameters/{name}.json');refs[str(p)]=data.sha(p)
        parameters[name]=json.loads(p.read_text())['parameters']
    for name in ['A8','R3_ALL']:
        configs[name]=[];p=parameters[name]
        for i in range(28):
            mapping=dict(q_rope='q_rope',k_rope='k_cache',v='v_out',attention_probability='attention_prob',attention_concat='attn_context')
            q={n:dict(scale=p[f'L{i:02d}.{site}']['mse']['scale'],zero_point=p[f'L{i:02d}.{site}']['mse']['zero']) for n,site in mapping.items()}
            assert q['attention_probability']['zero_point']==0
            v=np.zeros((64,8,128),np.uint8);v[32:]=255
            fields=list(CONFIG.iter_unpack(build_attention_config(v,q)))
            assert all(f[:1]+f[2:]==fields[0][:1]+fields[0][2:] for f in fields)
            configs[name].append(ia.config(fields[0]))
    p=parent('inputs.json');refs[str(p)]=data.sha(p)
    write('inputs.json',dict(names=NAMES,parameters=parameters,configs=configs,development=json.loads(p.read_text())['development'],prefix=[151645],source_head=r.b.head()))
    write('freeze.json',dict(files={'inputs.json':data.sha(ROOT/'inputs.json'),'dataset.json':data.sha(ROOT/'dataset.json')},references=refs,
        protocol_sha256=data.sha(data.MEMORY/'docs/experiments/EXP-0253.md'),frozen_before_inference=True))
def frozen():
    data.frozen();f=read('freeze.json')
    assert f['protocol_sha256']==data.sha(data.MEMORY/'docs/experiments/EXP-0253.md')
    for n,h in f['files'].items():assert data.sha(ROOT/n)==h
    for n,h in f['references'].items():assert data.sha(n)==h
    assert read('inputs.json')['names']==NAMES

class Cache(r.DenseCache):
    def update(self,k,v,i,cache_kwargs=None):
        values=super().update(k,v,i,cache_kwargs)
        self.ins.current_codes[i]=(self.key_cache[i],self.value_cache[i])
        return values

class Instrument(r.Instrument):
    def __init__(self,model):
        self.mode='legacy';self.current_codes={};self.frozen=read('inputs.json')
        super().__init__(model);self.cache_type=Cache
    def select(self,name):
        self.mode=name.split('_',1)[1] if '_' in name else 'legacy'
        self.param_name='R3_ALL' if name.startswith('R3_') else 'A8' if name.startswith('OFF_') else name
        self.rotation=set(range(28)) if name.startswith('R3_') else set()
        self.params=self.frozen['parameters'][self.param_name];self.restored=set()
        self.configure(None if name in ['F','C64'] else 'mse',r.b.a.FAMILIES);self.rotation_counts={};self.boundary_counts={}
    def apply(self,name,family,x,layout='last',skip_quant=False):
        if not self.warm and self.mode=='wide_nr64' and name.endswith('.attn_context'):
            return x  # Output already went through native integer AV conversion.
        if not self.warm and self.mode=='float_core':
            if name.endswith('.attention_prob'):raise AssertionError('floating core must bypass probability QDQ')
            if name.endswith('.attn_context'):
                assert self.policy=='mse' and not skip_quant and name not in self.restored
                self.boundary_counts['context_qdq']=self.boundary_counts.get('context_qdq',0)+1
        return super().apply(name,family,x,layout,skip_quant)
    def attention(self,module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        if self.warm or self.mode=='legacy':
            return super().attention(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
        i=module.layer_idx;p=f'L{i:02d}.';assert dropout==0
        if i in self.rotation:
            query=r.dense(query,self.s);self.rotation_counts['Q']=self.rotation_counts.get('Q',0)+1
        q=r.b.p.codes(query,self.params[p+'q_rope']['mse'])
        k,v=self.current_codes[i];assert k.dtype==v.dtype==torch.uint8
        k=r.b.a.qwen.repeat_kv(k,module.num_key_value_groups);v=r.b.a.qwen.repeat_kv(v,module.num_key_value_groups)
        nq,nk=q.shape[-2],k.shape[-2]
        valid=torch.arange(nk,device=q.device)[None,:] <= torch.arange(nk-nq,nk,device=q.device)[:,None]
        valid=valid[None,None]
        if attention_mask is not None:valid=valid & (attention_mask[:,:,:,:nk]==0)
        if self.mode=='float_core':
            self.boundary_counts['float_core']=self.boundary_counts.get('float_core',0)+1
            q,k,v=[r.b.p.decode(x,self.params[p+site]['mse'],query.dtype) for x,site in [(q,'q_rope'),(k,'k_cache'),(v,'v_out')]]
            output,probability=fa.core(q,k,v,valid,scaling)
            return output.to(query.dtype).transpose(1,2).contiguous(),probability
        c=self.frozen['configs'][self.param_name][i]
        scale=self.params[p+'q_rope']['mse']['scale']*self.params[p+'k_cache']['mse']['scale']*scaling
        result=ia.core(q,k,v,valid,c,self.mode,scale)
        output=r.b.p.decode(result['av'],self.params[p+'attn_context']['mse'],query.dtype)
        return output.transpose(1,2).contiguous(),(result['probability'].float()/255).to(query.dtype)

def numerical():
    rng=np.random.default_rng(253);proofs=[]
    for recipe in ['A8','R3_ALL']:
        for layer in range(28):
            params=read('inputs.json')['parameters'][recipe];p=f'L{layer:02d}.'
            for nq,nk in [(64,65),(1,80),(1,1),(7,13),(3,128)]:
                codes=[rng.integers(0,256,(1,2,n,128),dtype=np.uint8) for n in [nq,nk,nk]]
                if nk==1:codes[0][:]=128;codes[1][:]=128
                gpu=[];cpu=[]
                for x,site in zip(codes,['q_rope','k_cache','v_out']):
                    par=params[p+site]['mse'];gpu.append(r.b.p.decode(torch.from_numpy(x).cuda(),par,torch.float16))
                    cpu.append(((x.astype(np.float32)-np.float32(par['zero']))*np.float32(par['scale'])).astype(np.float16))
                    assert np.array_equal(gpu[-1].cpu().numpy(),cpu[-1])
                valid=(np.arange(nk)[None,:]<=np.arange(nk-nq,nk)[:,None])[None,None]
                if nq==3:valid[...,1::3]=False
                out,prob=fa.core(*gpu,torch.from_numpy(valid).cuda(),1/math.sqrt(128))
                ref,rp=fa.numpy_reference(*cpu,valid,1/math.sqrt(128))
                online,_=fa.numpy_reference(*cpu,valid,1/math.sqrt(128),chunk=7)
                assert np.allclose(online,ref,rtol=1e-12,atol=1e-12)
                got=out.cpu().numpy();probs=prob.cpu().numpy();err=got-ref
                scale=max(1.,float(np.max(np.abs(ref))));maximum=float(np.max(np.abs(err)))
                rms=float(np.sqrt(np.mean(err**2))/max(1e-12,np.sqrt(np.mean(ref**2))))
                row=float(np.max(np.abs(probs.sum(-1)-1)))
                assert np.isfinite(got).all() and np.isfinite(probs).all()
                assert maximum<=2e-5*scale and rms<=1e-5 and row<=2e-6,(recipe,layer,nq,nk,maximum,rms,row)
                assert (probs[~np.broadcast_to(valid,probs.shape)]==0).all()
                proofs.append(dict(recipe=recipe,layer=layer,nq=nq,nk=nk,max_abs=maximum,normalized_rms=rms,row_mass_error=row))
    # Explicit flat and sharp probability cases independent of frozen scales.
    for sharp in [False,True]:
        q=torch.zeros((1,1,4,128),device='cuda',dtype=torch.float16);k=q.clone();v=torch.arange(512,device='cuda',dtype=torch.float16).reshape_as(q)/512
        if sharp:q[...,0]=100;k[...,0]=torch.tensor([100.,-100.,0.,50.],device='cuda')
        mask=torch.ones((4,4),device='cuda',dtype=torch.bool);mask[:,-1]=False
        out,pr=fa.core(q,k,v,mask,1/math.sqrt(128));ref,_=fa.numpy_reference(*[z.cpu().numpy() for z in [q,k,v]],mask.cpu().numpy(),1/math.sqrt(128))
        assert np.allclose(out.cpu().numpy(),ref,rtol=1e-5,atol=2e-5)
    reference=data.verified('exp0252','checks/numerical.json');assert json.loads(reference.read_text())['pass_all']
    write('checks/numerical.json',dict(pass_all=True,cases=proofs,all56_layer_configs_checked=True,flat_and_sharp=True,
        float64_dense_and_online_oracles=True,FP16_U8_decode_exact=True,TF32=False,
        unchanged_integer_numerical_reference=dict(path=str(reference),sha256=data.sha(reference))))
    print('NUMERICAL PASS',len(proofs),flush=True)

def score(model,ins,name,samples,phase):
    path=f'scores/{phase}_{name}.json'
    if (ROOT/path).exists():
        d=read(path);assert d['freeze_sha256']==data.sha(ROOT/'freeze.json');return d
    ins.select(name);before=r.b.prefix_digest(ins);start=time.monotonic()
    logits,nll,_=r.forward(model,ins,samples[:4]);again,rn,_=r.forward(model,ins,samples[:4]);assert torch.equal(logits,again) and torch.equal(nll,rn)
    labels=torch.tensor([x['target_ids'] for x in samples[:4]],device='cuda')
    ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.flatten(),reduction='none').reshape(4,16)
    error=(ce-nll).abs().max().item();assert error<5e-6
    changed=[dict(x,token_ids=x['token_ids'][:70]+[123]*10) for x in samples[:4]]
    future,_,_=r.forward(model,ins,changed);assert torch.equal(logits[:,:7],future[:,:7])
    rows=[];ins.rotation_counts={};ins.boundary_counts={}
    for j in range(0,len(samples),4):
        logits,nll,cache=r.forward(model,ins,samples[j:j+4]);assert torch.isfinite(nll).all()
        assert all(k.dtype==v.dtype==(torch.uint8 if ins.policy else torch.float16) for k,v in zip(cache.key_cache,cache.value_cache))
        rows.extend(dict(id=x['id'],cell=x['cell'],nll=loss,top1=top) for x,loss,top in zip(samples[j:j+4],nll.cpu().tolist(),logits.argmax(-1).cpu().tolist()))
        if j%64==0:print('PROGRESS',phase,name,j+4,len(samples),flush=True)
    expected=len(ins.rotation)*len(samples)//4
    assert ins.rotation_counts==({'prefix_K':expected,'body_K':expected*16,'Q':expected*16} if expected else {})
    if ins.mode=='float_core':assert ins.boundary_counts==dict(float_core=28*16*len(samples)//4,context_qdq=28*16*len(samples)//4)
    assert before==r.b.prefix_digest(ins)
    means={c:float(np.mean([x['nll'] for x in rows if x['cell']==c])) for c in data.CELLS};mean=float(np.mean(list(means.values())))
    d=dict(name=name,phase=phase,ppl=math.exp(mean),mean_nll=mean,cell_nll=means,samples=rows,source_head=r.b.head(),freeze_sha256=data.sha(ROOT/'freeze.json'),
        checks=dict(repeat_exact=True,causal_exact=True,CE_max_abs=error,prefix_immutable=True,rotation_counts=ins.rotation_counts,boundary_counts=ins.boundary_counts),elapsed_s=time.monotonic()-start,
        role='conditional_attention_software_PPL_not_full_DSP_PPL_or_throughput')
    write(path,d);print('SCORE',phase,name,d['ppl'],round(d['elapsed_s'],1),flush=True);return d

def run(phase):
    assert read('checks/numerical.json')['pass_all']
    if phase=='final':assert read('development_complete.json')['pass_all']
    samples=read('inputs.json')['development'] if phase=='development' else read('dataset.json')['samples']
    for recipe in ['C64','F']:
        proof=f'checks/{phase}_{recipe}_weights.json'
        if (ROOT/proof).exists():continue
        model,before,mh=r.b.model_session(recipe);ins=Instrument(model);ins.build_prefix([151645])
        for name in NAMES:
            if (name=='F')!=(recipe=='F'):continue
            d=score(model,ins,name,samples,phase)
            if phase=='development' and not name.endswith('_float_core'):
                p=data.verified('exp0249' if name.endswith('_legacy') else 'exp0252',f'scores/development_{name}.json');old=json.loads(p.read_text())
                assert d['samples']==old['samples'] and d['ppl']==old['ppl'],('parent reproduction',name)
                write(f'checks/reproduction_{name}.json',dict(exact=True,reference_sha256=data.sha(p)))
        assert before==r.b.a.state_digest(model);write(proof,dict(unchanged=True,digest=before,manifest_sha256=mh,source_head=r.b.head()))
        ins.close();del model,ins;torch.cuda.empty_cache()
    if phase=='development':write('development_complete.json',dict(pass_all=True,names=NAMES,final_not_used=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','numerical','development','final']);args=parser.parse_args()
    preflight();r.b.a.settings()
    with torch.inference_mode():
        if args.phase=='freeze':freeze()
        else:
            frozen()
            if args.phase=='numerical':numerical()
            else:run(args.phase)
if __name__=='__main__':main()
