#!/usr/bin/env python3
"""PC059 fixed-prefix, immutable-W4 boundary restoration diagnostics."""
import argparse, collections, hashlib, json, math, subprocess, time, types
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
import prefix_exp0243 as p
import a8_exp0242 as a
import data_exp0244 as data

ROOT = data.RESULT
PARENT = ROOT.parent/'exp0243'
SOURCE = a.SOURCE

def read(n): return json.loads((ROOT/n).read_text())
def write(n,x): data.write(n,x)
def head(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
def preflight():
    data.preflight()
    state=json.loads(subprocess.check_output(['python3','-c',
        'import yaml,json;print(json.dumps(yaml.safe_load(open("'+str(a.MEMORY/'PROJECT_STATUS.yaml')+'"))["governance"]))'],text=True))
    assert state['active_experiment']=='EXP-0244'
@lru_cache(maxsize=1)
def params(): return json.loads(data.verified('exp0243','calibration/eos.json').read_text())['parameters']
def rows(split): return [r for r in read('dataset.json')['samples'] if r['split']==split]
def logical(s): return len([x for x in s if not x.endswith('.v_cache')])

@lru_cache(maxsize=1)
def masks():
    sites=set(params());groups={
        'norm_inputs':['norm_qkv','norm_mlp'], 'qk_projection':['q_out','k_out'],
        'kv_and_qrope':['v_out','v_cache','k_cache','q_rope'],
        'attention':['attention_prob','attn_context'], 'o_projection':['o_out'],
        'gate_up':['gate_out','up_out'], 'swiglu':['swiglu'], 'down':['down_out'],
        'residual':['residual_mid','residual_out'], 'embedding':['embedding_out'],
        'head_input':['head_input']}
    m={f'global_{k}':sorted(n for n in sites if n.split('.')[-1] in v) for k,v in groups.items()}
    assert set().union(*map(set,m.values()))==sites and sum(map(len,m.values()))==len(sites)==478
    m['union']=sorted(set(m['global_swiglu']+m['global_down']+m['global_residual']))
    for i in range(28):
        layer=f'L{i:02d}'
        m[f'layer_{layer}']=sorted(n for n in sites if n.startswith(layer+'.') and not n.endswith('embedding_out'))
        assert len(m[f'layer_{layer}'])==17
        for group in ['swiglu','down','residual']:
            m[f'{group}_{layer}']=sorted(n for n in m[f'global_{group}'] if n.startswith(layer+'.'))
    for j in range(4):m[f'segment_{j}']=sorted(n for i in range(j*7,j*7+7) for n in m[f'layer_L{i:02d}'])
    m['all_restored']=sorted(sites)
    return m

class SelectiveCache(DynamicCache):
    """Store U8 where enabled; FP16 storage only at explicitly restored sites."""
    def __init__(self,ins):
        super().__init__();self.ins=ins;self.appended=[];self.seed_lengths=[]
    def update(self,k,v,i,cache_kwargs=None):
        seed=bool(cache_kwargs and cache_kwargs.get('prefix_seed'))
        dtype=k.dtype;kp=self.ins.params[f'L{i:02d}.k_cache']['mse'];vp=self.ins.params[f'L{i:02d}.v_out']['mse']
        ek=self.ins.enabled(f'L{i:02d}.k_cache');ev=self.ins.enabled(f'L{i:02d}.v_out')
        if ek:k=p.codes(k,kp)
        if ev:v=p.codes(v,vp)
        k,v=super().update(k,v,i,cache_kwargs)
        (self.seed_lengths if seed else self.appended).append((i,int(k.shape[-2]) if seed else int(cache_kwargs.get('cache_position',torch.empty(0)).numel()) if cache_kwargs else 0))
        return p.decode(k,kp,dtype) if ek else k,p.decode(v,vp,dtype) if ev else v

class FloatOracleCache(DynamicCache):
    """Independent append of decoded QDQ tensors, same attention call shapes."""
    def __init__(self,ins): super().__init__();self.ins=ins;self.appended=[];self.seed_lengths=[]
    def update(self,k,v,i,cache_kwargs=None):
        def scalar_tensor(x,name):
            if not self.ins.enabled(name):return x
            q=self.ins.params[name]['mse']
            code=(x.float()*q['inv_scale']+q['zero']+0.5).floor().clamp(0,255)
            return ((code-q['zero'])*q['scale']).to(x.dtype)
        k=scalar_tensor(k,f'L{i:02d}.k_cache');v=scalar_tensor(v,f'L{i:02d}.v_out')
        (self.seed_lengths if cache_kwargs and cache_kwargs.get('prefix_seed') else self.appended).append((i,k.shape[-2]))
        return super().update(k,v,i,cache_kwargs)

class Instrument(p.Instrument):
    def __init__(self,model,shared=True):
        self.restored=set();self.shared=shared;self.original_layers=[];self.cache_type=SelectiveCache
        super().__init__(model);self.params=params()
    def enabled(self,name): return bool(self.policy) and not self.warm and name not in self.restored
    def apply(self,name,family,x,layout='last',skip_quant=False):
        if name in self.restored:return x
        return super().apply(name,family,x,layout,skip_quant)
    def install(self):
        super().install()
        if not self.shared:return
        for i,layer in enumerate(self.model.model.layers):
            owned=[h for h in self.handles if h.id in layer.post_attention_layernorm._forward_pre_hooks]
            assert len(owned)==1
            owned[0].remove()
            original=layer.forward;self.original_layers.append((layer,original))
            def forward(block,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
                        output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None,_i=i,**kwargs):
                residual=hidden_states
                x=block.input_layernorm(hidden_states)
                x,attn=block.self_attn(hidden_states=x,attention_mask=attention_mask,position_ids=position_ids,
                    past_key_value=past_key_value,output_attentions=output_attentions,use_cache=use_cache,
                    cache_position=cache_position,position_embeddings=position_embeddings,**kwargs)
                x=self.apply(f'L{_i:02d}.residual_mid','residual',residual+x)
                residual=x
                x=block.mlp(block.post_attention_layernorm(x))
                outputs=(residual+x,)
                return outputs+(attn,) if output_attentions else outputs
            layer.forward=types.MethodType(forward,layer)
    def new_cache(self,batch):
        cache=self.cache_type(self)
        for i,(k,v) in enumerate(self.prefix_raw):
            cache.update(k.expand(batch,-1,-1,-1).clone(),v.expand(batch,-1,-1,-1).clone(),i,{'prefix_seed':True})
        return cache
    def select(self,name):
        self.restored=set(masks()[name]) if name in masks() else set()
        self.configure(None if name in ['C64','F'] else 'mse',a.FAMILIES)
    def close(self):
        for layer,original in self.original_layers:layer.forward=original
        super().close()

def prefix_digest(ins):
    h=hashlib.sha256()
    for pair in ins.prefix_raw:
        for x in pair:h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()

def forward(model,ins,batch,mode='bulk'):
    x=torch.tensor([r['token_ids'] for r in batch],device='cuda');cache=ins.new_cache(len(batch))
    assert cache.get_seq_length()==1
    if mode=='bulk':logits=model(input_ids=x[:,:79],past_key_values=cache,use_cache=True).logits[:,63:79].float()
    else:
        parts=[model(input_ids=x[:,:64],past_key_values=cache,use_cache=True).logits[:,-1:].float()]
        for j in range(15):parts.append(model(input_ids=x[:,64+j:65+j],past_key_values=cache,use_cache=True).logits[:,-1:].float())
        logits=torch.cat(parts,1)
    assert cache.get_seq_length()==80 and len(cache.appended)==28*(1 if mode=='bulk' else 16)
    for i,(k,v) in enumerate(zip(cache.key_cache,cache.value_cache)):
        assert k.shape[-2]==v.shape[-2]==80
        if ins.cache_type is SelectiveCache:
            assert k.dtype==(torch.uint8 if ins.enabled(f'L{i:02d}.k_cache') else torch.float16)
            assert v.dtype==(torch.uint8 if ins.enabled(f'L{i:02d}.v_out') else torch.float16)
    labels=x[:,64:80];nll=torch.logsumexp(logits,-1)-logits.gather(-1,labels[:,:,None]).squeeze(-1)
    return logits,nll,cache

def score(model,ins,name,samples,phase,mode):
    path=f'scores/{phase}_{name}_{mode}.json'
    if (ROOT/path).exists():
        d=read(path);assert d['dataset_sha256']==a.sha(ROOT/'dataset.json');assert d['mask']==masks().get(name,[])
        print('RETAINED',phase,name,d['ppl'],flush=True);return d
    ins.select(name);before=prefix_digest(ins);start=time.monotonic()
    # Every final/rerank variant gets its own deterministic, causal and independent-CE check.
    check={}
    if mode=='sequential':
        logits,nll,_=forward(model,ins,samples[:4],mode);again,rn,_=forward(model,ins,samples[:4],mode)
        assert torch.equal(logits,again) and torch.equal(nll,rn)
        labels=torch.tensor([r['target_ids'] for r in samples[:4]],device='cuda')
        ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.flatten(),reduction='none').reshape(4,16)
        err=(ce-nll).abs().max().item();assert err<5e-6
        changed=[dict(r,token_ids=r['token_ids'][:70]+[123]*10) for r in samples[:4]]
        fl,_,_=forward(model,ins,changed,mode);assert torch.equal(logits[:,:7],fl[:,:7])
        check=dict(repeat_exact=True,causal_exact=True,independent_CE_error=err)
    output=[]
    for j in range(0,len(samples),4):
        logits,nll,cache=forward(model,ins,samples[j:j+4],mode);assert torch.isfinite(nll).all()
        output.extend(dict(id=r['id'],cell=r['cell'],nll=v,top1=t) for r,v,t in zip(samples[j:j+4],nll.cpu().tolist(),logits.argmax(-1).cpu().tolist()))
    means={c:float(np.mean([r['nll'] for r in output if r['cell']==c])) for c in a.CELLS};mean=float(np.mean(list(means.values())))
    assert before==prefix_digest(ins)
    d=dict(name=name,phase=phase,mode=mode,mask=masks().get(name,[]),shared_midpoint=ins.shared,ppl=math.exp(mean),mean_nll=mean,
        cell_nll=means,samples=output,checks=check,prefix_immutable=True,prefix_digest=before,source_head=head(),
        dataset_sha256=a.sha(ROOT/'dataset.json'),parameters_sha256=a.sha(data.verified('exp0243','calibration/eos.json')),
        elapsed_s=time.monotonic()-start,role='software_conditional_restoration_diagnostic_not_deployable_or_speed')
    write(path,d);print('SCORE',phase,name,mode,round(d['ppl'],6),round(d['elapsed_s'],1),flush=True);return d

def model_session(variant):
    model,mh=a.canonical.load(variant,'cuda');before=a.state_digest(model)
    if variant=='C64':assert mh==a.C64_HASH and before=='1a798eca2237d2e31158c0574c44255484cb2890b6ef3cbba8f71d975ed62d3d'
    return model,before,mh

def check_model():
    model,before,mh=model_session('C64');samples=rows('development')[:4]
    legacy=Instrument(model,shared=False);legacy.build_prefix([151645]);legacy.select('C64')
    original={mode:forward(model,legacy,samples,mode)[0].clone() for mode in ['bulk','sequential']};legacy.close()
    ins=Instrument(model);ins.build_prefix([151645]);checks={}
    for mode in ['bulk','sequential']:
        ins.select('C64');x,_,_=forward(model,ins,samples,mode);assert torch.equal(x,original[mode])
        ins.select('all_restored');y,_,_=forward(model,ins,samples,mode);assert torch.equal(x,y)
        ins.select('A8');ins.cache_type=SelectiveCache;x,_,_=forward(model,ins,samples,mode)
        ins.cache_type=FloatOracleCache;y,_,_=forward(model,ins,samples,mode);assert torch.equal(x,y)
        ins.cache_type=SelectiveCache
        ins.select('layer_L27');x,_,_=forward(model,ins,samples,mode)
        ins.cache_type=FloatOracleCache;y,_,_=forward(model,ins,samples,mode);assert torch.equal(x,y)
        ins.cache_type=SelectiveCache;checks[mode]=dict(disabled_exact=True,all_restored_exact=True,full_and_selective_float_cache_oracle_exact=True)
    # Isolated real decoder wrapper: zero MLP exposes its skip; norm captures fanout input.
    layer=model.model.layers[0];ins.select('A8');seen=[]
    class Identity(torch.nn.Module):
        def forward(self,x):return x
    class Attention(torch.nn.Module):
        def forward(self,hidden_states,**kw):return hidden_states*0.031,None
    class Norm(torch.nn.Module):
        def forward(self,x):seen.append(x.clone());return x
    class Zero(torch.nn.Module):
        def forward(self,x):return torch.zeros_like(x)
    mock=types.SimpleNamespace(input_layernorm=Identity(),self_attn=Attention(),post_attention_layernorm=Norm(),mlp=Zero())
    x=torch.linspace(-2,2,1024,device='cuda',dtype=torch.float16).reshape(1,1,-1)
    mid=x+x*0.031;expected=a.qdq(mid,ins.params['L00.residual_mid']['mse'])
    y=layer.forward.__func__(mock,x)[0];assert torch.equal(y,expected) and torch.equal(seen[0],expected) and not torch.equal(mid,expected)
    checks['shared_fanout_sentinel']=True;ins.close()
    legacy=Instrument(model,shared=False);legacy.build_prefix([151645]);legacy.params=params()
    old=json.loads(data.verified('exp0243','inputs.json').read_text())['development']
    # Full exposed old development panel reproduces frozen legacy bulk result.
    d=score(model,legacy,'legacy',old,'reproduction','bulk')
    archived=json.loads(data.verified('exp0243','scores/development_C64_eos_mse_bulk.json').read_text())
    assert d['samples']==archived['samples'] and d['ppl']==archived['ppl']
    checks['legacy_full_128docs_exact']=True;legacy.close()
    assert before==a.state_digest(model)
    checks.update(weight_digest=before,manifest_sha256=mh,unchanged=True,scalar_oracle=a.scalar_oracle())
    write('checks/model_oracles.json',checks);print('MODEL_ORACLES_PASS',flush=True)

def develop():
    model,before,mh=model_session('C64');ins=Instrument(model);ins.build_prefix([151645]);samples=rows('development')
    names=['C64','A8']+[n for n in masks() if n!='all_restored']
    for name in names:score(model,ins,name,samples,'development','bulk')
    ins.close();legacy=Instrument(model,shared=False);legacy.build_prefix([151645]);score(model,legacy,'legacy',samples,'development','bulk');legacy.close()
    assert before==a.state_digest(model);write('checks/C64_bulk_weights.json',dict(digest=before,unchanged=True));del legacy,ins,model;torch.cuda.empty_cache()
    model,before,mh=model_session('F');ins=Instrument(model);ins.build_prefix([151645]);score(model,ins,'F',samples,'development','bulk');ins.close()
    assert before==a.state_digest(model);write('checks/F_bulk_weights.json',dict(digest=before,unchanged=True))
    def key(n):return (round(read(f'scores/development_{n}_bulk.json')['mean_nll'],6),logical(masks().get(n,[])),n)
    local=[n for group in ['swiglu','down','residual'] for n in sorted([k for k in masks() if k.startswith(group+'_L')],key=key)[:2]]
    layer=sorted([n for n in masks() if n.startswith('layer_')],key=key)[:2]
    selected=['C64','A8','legacy','F']+[n for n in masks() if n.startswith('global_')]+['union']+local+layer
    write('rerank_selection.json',dict(names=selected,local=local,layers=layer,final_used=False,rule='frozen_PC059_bulk_top2_per_focus_and_layer'))

def rerank():
    selection=read('rerank_selection.json')
    samples=[r for i in range(8) for c in a.CELLS for r in [([v for v in rows('development') if v['cell']==c])[i]]]
    for variant in ['C64','F']:
        model,before,mh=model_session(variant);ins=Instrument(model);ins.build_prefix([151645])
        for n in selection['names']:
            if n=='legacy' or (n=='F')!=(variant=='F'):continue
            score(model,ins,n,samples,'rerank','sequential')
        ins.close()
        if variant=='C64':
            legacy=Instrument(model,shared=False);legacy.build_prefix([151645]);score(model,legacy,'legacy',samples,'rerank','sequential');legacy.close();del legacy
        assert before==a.state_digest(model);write(f'checks/{variant}_rerank_weights.json',dict(digest=before,unchanged=True));del ins,model;torch.cuda.empty_cache()
    def key(n):return (round(read(f'scores/rerank_{n}_sequential.json')['mean_nll'],6),logical(masks().get(n,[])),n)
    chosen=dict(local=min(selection['local'],key=key),layer=min(selection['layers'],key=key),global_group=min([n for n in selection['names'] if n.startswith('global_')],key=key))
    names=list(dict.fromkeys(['F','C64','A8','global_swiglu','global_down','global_residual','union']+list(chosen.values())))
    write('final_selection.json',dict(names=names,chosen=chosen,final_used=False,selected_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          scores_sha256={n:a.sha(ROOT/f'scores/rerank_{n}_sequential.json') for n in selection['names']}))
    print('FINAL_SELECTION',chosen,names,flush=True)

def final():
    selection=read('final_selection.json');samples=rows('final')
    for variant in ['C64','F']:
        model,before,mh=model_session(variant);ins=Instrument(model);ins.build_prefix([151645])
        for n in selection['names']:
            if (n=='F')==(variant=='F'):score(model,ins,n,samples,'final','sequential')
        assert before==a.state_digest(model);write(f'checks/{variant}_final_weights.json',dict(digest=before,unchanged=True,manifest_sha256=mh));ins.close();del ins,model;torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','check','develop','rerank','final']);args=parser.parse_args()
    preflight();data.frozen();a.settings()
    if args.phase!='freeze':
        frozen=read('masks.json');assert frozen['masks']==masks()
        assert frozen['parameters_sha256']==a.sha(data.verified('exp0243','calibration/eos.json'))
        assert frozen['protocol_sha256']==a.sha(a.MEMORY/'docs/experiments/EXP-0244.md')
    with torch.inference_mode():
        if args.phase=='freeze':write('masks.json',dict(masks=masks(),parameters_sha256=a.sha(data.verified('exp0243','calibration/eos.json')),protocol_sha256=a.sha(a.MEMORY/'docs/experiments/EXP-0244.md'),source_head=head()))
        else:{'check':check_model,'develop':develop,'rerank':rerank,'final':final}[args.phase]()

if __name__=='__main__':main()
