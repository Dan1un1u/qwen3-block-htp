#!/usr/bin/env python3
"""PC058 fixed-weight prefix attribution with real uint8 software KV storage."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, collections, hashlib, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, DynamicCache
import a8_exp0242 as a
import data_exp0243 as final_data

ROOT=a.RESULT.parent/'exp0243'
SOURCE=a.SOURCE

def read(n): return json.loads((ROOT/n).read_text())
def write(n,x):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def preflight():
    subprocess.run(['python3',str(a.MEMORY/'scripts/project_memory.py'),'preflight','--source-worktree',str(SOURCE)],check=True)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip()=='codex/exp-0243-prefix-a8-causality'
def frozen():
    d=read('inputs.json');f=read('inputs_freeze.json')
    assert a.sha(ROOT/'inputs.json')==f['inputs_sha256']
    assert a.sha(ROOT/'dataset.json')==f['final_dataset_sha256']
    return d
def rows(split):
    if split=='final':return read('dataset.json')['samples']
    d=frozen();return d[split]
def prefix_list():return frozen()['prefixes']
def codes(x,p):return torch.floor(x.float()*p['inv_scale']+p['zero']+.5).clamp(0,255).to(torch.uint8)
def decode(x,p,dtype):return ((x.float()-p['zero'])*p['scale']).to(dtype)


def prepare():
    final_data.frozen()
    tok=AutoTokenizer.from_pretrained(a.canonical.MODEL,local_files_only=True)
    cal=read_old_cal()
    data=a.dataset();selected=set(cal['calibration_ids'])
    calibration=[r for r in data if r['id'] in selected]
    development=[r for r in data if r['split']=='development']
    marker='PREFIX_CONTENT_MARKER_0243'
    rendered=tok.apply_chat_template([dict(role='user',content=marker)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert rendered.count(marker)==1
    header,suffix=rendered.split(marker)
    mode=min(collections.Counter(r['token_ids'][0] for r in calibration),key=lambda t:(-sum(r['token_ids'][0]==t for r in calibration),t))
    dot=tok.encode('.',add_special_tokens=False);assert len(dot)==1
    proposals=[('empty',[]),('dot',dot),('eos',[tok.eos_token_id]),('mode_first',[mode]),('chat_header',tok.encode(header,add_special_tokens=False))]
    prefixes=[];seen=set()
    for name,ids in proposals:
        assert all(isinstance(i,int) for i in ids)
        if tuple(ids) in seen:continue
        seen.add(tuple(ids));prefixes.append(dict(name=name,ids=ids,text=tok.decode(ids),eligible=len(ids)<=16))
    diagnostics=[]
    for cell in a.CELLS:
        for r in [x for x in calibration if x['cell']==cell][:4]:
            text=tok.decode(r['token_ids'],skip_special_tokens=False)
            chat=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            diagnostics.append(dict(id=r['id'],cell=cell,body=r['token_ids'],chat=chat))
    write('inputs.json',dict(calibration=calibration,development=development,prefixes=prefixes,diagnostics=diagnostics,
          chat_header=header,chat_suffix=suffix,tokenizer_sha256=a.sha(a.canonical.MODEL/'qwen3-tokenizer.json'),
          calibration_tokens=16384,development_targets=2048,final_targets=8192))
    write('inputs_freeze.json',dict(inputs_sha256=a.sha(ROOT/'inputs.json'),final_dataset_sha256=a.sha(ROOT/'dataset.json'),
          protocol_sha256=a.sha(a.MEMORY/'docs/experiments/EXP-0243.md'),frozen_before_inference=True))
    print('INPUTS_FROZEN',prefixes,flush=True)


def read_old_cal():
    p=a.RESULT/'EVIDENCE_SHA256.json';assert a.sha(p)=='5c6e1f92088a9e7ff49aa27cf1ab6a59f13ace3b46ef34f997474ec022d92977'
    ledger=json.loads(p.read_text())
    for name in ['calibration.json','summary.json']:
        assert a.sha(a.RESULT/name)==ledger['files'][name]['sha256']
    return json.loads((a.RESULT/'calibration.json').read_text())


class CarrierCache(DynamicCache):
    def __init__(self,ins):
        super().__init__();self.ins=ins;self.quantized=bool(ins.policy);self.appended=[];self.seed_lengths=[]
    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        seed=bool(cache_kwargs and cache_kwargs.get('prefix_seed'))
        if not seed:self.ins.observe_kv(layer_idx,key_states,value_states)
        dtype=key_states.dtype
        if self.quantized:
            kp=self.ins.params[f'L{layer_idx:02d}.k_cache'][self.ins.policy]
            vp=self.ins.params[f'L{layer_idx:02d}.v_out'][self.ins.policy]
            key_states=codes(key_states,kp);value_states=codes(value_states,vp)
            assert key_states.dtype==value_states.dtype==torch.uint8
        keys,values=super().update(key_states,value_states,layer_idx,cache_kwargs)
        (self.seed_lengths if seed else self.appended).append((layer_idx,int(key_states.shape[-2])))
        if self.quantized:return decode(keys,kp,dtype),decode(values,vp,dtype)
        return keys,values


class Instrument(a.Instrument):
    def __init__(self,model):
        self.warm=False;self.trace=None;self.prefix_raw=[]
        super().__init__(model)
    def record_trace(self,name,x,layout):
        if self.trace is None:return
        if not any(name.endswith(t) for t in ['norm_qkv','norm_mlp','swiglu','down_out','residual_out']):return
        z=x.float()
        if layout=='heads':z=z.transpose(1,2).reshape(z.shape[0],z.shape[2],-1)
        v=self.trace.setdefault(name,[])
        v.append(dict(token_max=z.abs().amax(-1).cpu().tolist(),token_rms=z.square().mean(-1).sqrt().cpu().tolist(),
                      zero_fraction=(z==0).float().mean(-1).cpu().tolist()))
    def apply(self,name,family,x,layout='last',skip_quant=False):
        if self.warm:return x
        # Cache K/V were quantized before append; avoid quantizing or recollecting all history.
        if name.endswith('.k_cache') or name.endswith('.v_cache'):return x
        self.record_trace(name,x,layout)
        return super().apply(name,family,x,layout,skip_quant)
    def observe_kv(self,layer,k,v):
        if self.warm:return
        if self.collect:
            for short,x in [('k_cache',k),('v_cache',v)]:
                name=f'L{layer:02d}.{short}'
                if name not in self.stats:self.stats[name]=a.Stats('qk_kv')
                self.stats[name].add(x,'heads')
    def build_prefix(self,ids):
        old=self.policy;collect=self.collect
        self.configure();self.collect=False;self.warm=True
        cache=CarrierCache(self)
        if ids:self.model.model(input_ids=torch.tensor([ids],device='cuda'),past_key_values=cache,use_cache=True)
        self.prefix_raw=[(k.clone(),v.clone()) for k,v in zip(cache.key_cache,cache.value_cache)]
        self.warm=False;self.collect=collect;self.configure(old,a.FAMILIES)
        return self.prefix_raw
    def new_cache(self,batch):
        cache=CarrierCache(self)
        for i,(k,v) in enumerate(self.prefix_raw):
            cache.update(k.expand(batch,-1,-1,-1).clone(),v.expand(batch,-1,-1,-1).clone(),i,{'prefix_seed':True})
        return cache


def forward(model,ins,batch,mode='bulk'):
    x=torch.tensor([r['token_ids'] for r in batch],device='cuda')
    cache=ins.new_cache(len(batch));prefix_len=cache.get_seq_length()
    if mode=='calibration':
        model(input_ids=x,past_key_values=cache,use_cache=True)
        return cache
    if mode=='bulk':
        logits=model(input_ids=x[:,:79],past_key_values=cache,use_cache=True).logits[:,63:79].float()
    else:
        parts=[model(input_ids=x[:,:64],past_key_values=cache,use_cache=True).logits[:,-1:].float()]
        for j in range(15):
            parts.append(model(input_ids=x[:,64+j:65+j],past_key_values=cache,use_cache=True).logits[:,-1:].float())
        logits=torch.cat(parts,1)
    assert cache.get_seq_length()==prefix_len+79
    assert len(cache.appended)==28*(1 if mode=='bulk' else 16)
    if ins.policy:assert all(k.dtype==v.dtype==torch.uint8 for k,v in zip(cache.key_cache,cache.value_cache))
    labels=x[:,64:80]
    nll=torch.logsumexp(logits,-1)-logits.gather(-1,labels[:,:,None]).squeeze(-1)
    return logits,nll,cache


def score(model,ins,variant,prefix,policy,phase,mode='bulk'):
    name=f'scores/{phase}_{variant}_{prefix["name"]}_{policy or "a16"}_{mode}.json'
    if (ROOT/name).exists():
        d=read(name);assert d['inputs_sha256']==a.sha(ROOT/'inputs.json');print('RETAINED',name,d['ppl'],flush=True);return d
    ins.configure(policy,a.FAMILIES);ins.build_prefix(prefix['ids']);samples=rows(phase)
    start_time=time.monotonic();logits,nll,_=forward(model,ins,samples[:4],mode);again,repeated,_=forward(model,ins,samples[:4],mode)
    assert torch.equal(logits,again) and torch.equal(nll,repeated)
    labels=torch.tensor([r['target_ids'] for r in samples[:4]],device='cuda')
    ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none').reshape(4,16)
    error=(ce-nll).abs().max().item();assert error<5e-6,error
    changed=[dict(r,token_ids=r['token_ids'][:70]+[123]*10) for r in samples[:4]]
    future,_,_=forward(model,ins,changed,mode);assert torch.equal(logits[:,:7],future[:,:7])
    output=[]
    for j in range(0,len(samples),4):
        logits,nll,cache=forward(model,ins,samples[j:j+4],mode)
        assert torch.isfinite(nll).all(),name
        for r,loss,top in zip(samples[j:j+4],nll.cpu().tolist(),logits.argmax(-1).cpu().tolist()):
            output.append(dict(id=r['id'],cell=r['cell'],nll=loss,top1=top))
    means={c:float(np.mean([r['nll'] for r in output if r['cell']==c])) for c in a.CELLS};mean=float(np.mean(list(means.values())))
    result=dict(variant=variant,prefix=prefix,policy=policy,phase=phase,mode=mode,ppl=math.exp(mean),mean_nll=mean,cell_nll=means,samples=output,
                repeat_exact=True,causal_exact=True,independent_CE_error=error,cache_storage='uint8' if policy else 'float16',
                cache_length=cache.get_seq_length(),cache_append_calls=len(cache.appended),inputs_sha256=a.sha(ROOT/'inputs.json'),
                parameters_sha256=a.sha(ROOT/f'calibration/{prefix["name"]}.json') if policy else None,
                source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),elapsed_s=time.monotonic()-start_time,
                role='software_U8_cache_QDQ_not_DSP_or_speed_measurement')
    write(name,result);print('SCORE',phase,variant,prefix['name'],policy or 'a16',mode,result['ppl'],flush=True);return result


def diagnose():
    d=frozen();dot=next(p['ids'][0] for p in d['prefixes'] if p['name']=='dot')
    for variant in ['F','C64']:
        model,mh=a.canonical.load(variant,'cuda');before=a.state_digest(model)
        # Reproduce canonical EXP0242 sentinel before and after instrumentation.
        original=a.forward(model,d['development'][:4])[1];ins=Instrument(model)
        current=a.forward(model,d['development'][:4])[1];assert torch.equal(original,current)
        result=[]
        for row in d['diagnostics']:
            body=row['body'];variants={'bare128':body,'bare64':body[:64],'shift1_64':body[1:65],'replace_first':[dot]+body[1:],'chat':row['chat']}
            for name,ids in variants.items():
                ins.trace={};cache=CarrierCache(ins);model(input_ids=torch.tensor([ids],device='cuda'),past_key_values=cache,use_cache=True)
                result.append(dict(id=row['id'],cell=row['cell'],case=name,input_ids=ids,trace=ins.trace))
            ins.trace={};cache=CarrierCache(ins)
            model(input_ids=torch.tensor([body[:64]],device='cuda'),past_key_values=cache,use_cache=True)
            for token in body[64:]:model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True)
            result.append(dict(id=row['id'],cell=row['cell'],case='cached64_then64steps',input_ids=body,trace=ins.trace))
            ins.trace={};cache=CarrierCache(ins)
            model(input_ids=torch.tensor([body[64:]],device='cuda'),past_key_values=cache,use_cache=True)
            result.append(dict(id=row['id'],cell=row['cell'],case='restart_second64',input_ids=body[64:],trace=ins.trace))
            print('DIAG',variant,row['id'],flush=True)
        assert before==a.state_digest(model);ins.close()
        write(f'diagnostics/{variant}.json',dict(variant=variant,manifest_sha256=mh,weight_digest=before,unchanged=True,disabled_exact=True,cases=result))
        del ins,model;torch.cuda.empty_cache()


def calibration_and_development():
    model,mh=a.canonical.load('C64','cuda');assert mh==a.C64_HASH;before=a.state_digest(model);assert before==read_old_cal()['state_digest']
    ins=Instrument(model)
    for prefix in prefix_list():
        if not prefix['eligible']:continue
        name=f'calibration/{prefix["name"]}.json'
        if not (ROOT/name).exists():
            ins.configure();ins.stats={};ins.build_prefix(prefix['ids']);ins.collect=True
            for j in range(0,128,4):forward(model,ins,rows('calibration')[j:j+4],'calibration')
            ins.collect=False
            # Expand shared stored-K/V scale domains to include offline prefix values.
            for i,(k,v) in enumerate(ins.prefix_raw):
                for short,value in [('k_cache',k),('v_cache',v),('v_out',v)]:
                    st=ins.stats[f'L{i:02d}.{short}'];st.lo=min(st.lo,value.min().item());st.hi=max(st.hi,value.max().item())
            records={n:st.record() for n,st in ins.stats.items()}
            params={n:a.choose_params(st,records[n]) for n,st in ins.stats.items()}
            # Cache endpoints must include prefix extrema even when online tensors are clipped.
            for i,(k,v) in enumerate(ins.prefix_raw):
                for short,value in [('k_cache',k),('v_out',v)]:
                    for method in ['minmax','mse','percentile']:
                        p=params[f'L{i:02d}.{short}'][method]
                        params[f'L{i:02d}.{short}'][method]=a.qparams(min(p['requested_lo'],value.min().item()),max(p['requested_hi'],value.max().item()))
            for i in range(28):params[f'L{i:02d}.v_cache']=params[f'L{i:02d}.v_out']
            arrays={}
            for n,st in ins.stats.items():
                if any(n.endswith(x) for x in ['swiglu','down_out','residual_out','norm_qkv','norm_mlp']):
                    arrays[n+'/token_absmax']=np.asarray(st.token_absmax,np.float32)
                    arrays[n+'/token_rms']=np.asarray(st.token_rms,np.float32)
            p=ROOT/f'calibration/{prefix["name"]}_arrays.npz';p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,**arrays)
            write(name,dict(prefix=prefix,sites=records,parameters=params,manifest_sha256=mh,arrays_sha256=a.sha(p),
                           prefix_KV_extrema_included=True,V_cache_scale_shared=True,inputs_sha256=a.sha(ROOT/'inputs.json')))
            print('CALIBRATED',prefix['name'],len(records),flush=True)
        ins.params=read(name)['parameters']
        score(model,ins,'C64',prefix,None,'development')
        for policy in ['minmax','mse','percentile']:score(model,ins,'C64',prefix,policy,'development')
        # Actual U8 serialized seed roundtrip and sequential/bulk diagnostics on four body rows.
        checks={}
        for policy in [None,'minmax','mse','percentile']:
            ins.configure(policy,a.FAMILIES);ins.build_prefix(prefix['ids'])
            bulk,bn,_=forward(model,ins,rows('development')[:4],'bulk');seq,sn,cache=forward(model,ins,rows('development')[:4],'sequential')
            checks[policy or 'a16']=dict(max_logit_abs=(bulk-seq).abs().max().item(),max_nll_abs=(bn-sn).abs().max().item(),
                     prefix_length=len(prefix['ids']),final_length=cache.get_seq_length(),append_calls=len(cache.appended),
                     cache_dtypes=sorted(set(str(k.dtype) for k in cache.key_cache)),same_prefix_body_positions=True)
            if policy is None:
                ids=torch.tensor([prefix['ids']+r['token_ids'][:79] for r in rows('development')[:4]],device='cuda')
                direct=model(input_ids=ids,use_cache=False).logits[:,len(prefix['ids'])+63:].float()
                checks['a16']['direct_full_prefix_max_logit_abs']=(direct-bulk).abs().max().item()
            else:
                seed=ins.new_cache(1)
                arrays={f'L{i:02d}.{kind}':value.cpu().numpy() for i,pair in enumerate(zip(seed.key_cache,seed.value_cache)) for kind,value in zip(['K','V'],pair)}
                path=ROOT/f'calibration/{prefix["name"]}_{policy}_prefix_u8.npz'
                with path.open('xb') as f:np.savez_compressed(f,**arrays)
                with np.load(path) as retained:
                    assert all(np.array_equal(value,retained[key]) for key,value in arrays.items())
                checks[policy]['prefix_artifact_sha256']=a.sha(path)
                checks[policy]['serialized_U8_seed_exact']=True
        write(f'checks/{prefix["name"]}_cache_paths.json',checks)
    assert before==a.state_digest(model);write('checks/C64_development_weights.json',dict(unchanged=True,digest=before))
    ins.close();del ins,model;torch.cuda.empty_cache()
    model,mh=a.canonical.load('F','cuda');before=a.state_digest(model);ins=Instrument(model)
    for prefix in prefix_list():
        if prefix['eligible']:score(model,ins,'F',prefix,None,'development')
    assert before==a.state_digest(model);write('checks/F_development_weights.json',dict(unchanged=True,digest=before));ins.close()
    candidates=[]
    for prefix in prefix_list():
        if prefix['eligible']:
            for policy in ['minmax','mse','percentile']:
                p=f'scores/development_C64_{prefix["name"]}_{policy}_bulk.json';q=read(p)
                candidates.append(dict(prefix=prefix,policy=policy,mean_nll=q['mean_nll'],ppl=q['ppl'],score_sha256=a.sha(ROOT/p)))
    def key(c):return (round(c['mean_nll'],6),len(c['prefix']['ids']),[p['name'] for p in prefix_list()].index(c['prefix']['name']),['minmax','mse','percentile'].index(c['policy']))
    selected=min(candidates,key=key);empty=min([c for c in candidates if not c['prefix']['ids']],key=key)
    write('selection.json',dict(selected=selected,empty=empty,candidates=candidates,final_not_used=True,inputs_sha256=a.sha(ROOT/'inputs.json'),
          selected_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('SELECTION',selected,flush=True)


def final():
    selection=read('selection.json');chosen=selection['selected'];empty=next(p for p in prefix_list() if not p['ids'])
    for variant in ['C64','F']:
        model,mh=a.canonical.load(variant,'cuda');before=a.state_digest(model);ins=Instrument(model)
        prefixes={p['name']:p for p in [empty,chosen['prefix']]}
        for prefix in prefixes.values():score(model,ins,variant,prefix,None,'final','sequential')
        if variant=='C64':
            configs=[(chosen['prefix'],chosen['policy']),(empty,'minmax'),(empty,selection['empty']['policy'])]
            for prefix,policy in configs:
                ins.params=read(f'calibration/{prefix["name"]}.json')['parameters']
                score(model,ins,variant,prefix,policy,'final','sequential')
        assert before==a.state_digest(model);write(f'checks/{variant}_final_weights.json',dict(unchanged=True,digest=before));ins.close()
        del ins,model;torch.cuda.empty_cache()


def oracles():
    checks=a.scalar_oracle();torch.manual_seed(243)
    q,k,v=[torch.randn(2,2,12,8,device='cuda') for _ in range(3)]
    p=a.qparams(-4,4);kc=codes(k,p);vc=codes(v,p)
    assert torch.equal(codes(decode(kc,p,torch.float32),p),kc)
    assert torch.equal(codes(decode(vc,p,torch.float32),p),vc)
    kd,vd=decode(kc,p,torch.float32),decode(vc,p,torch.float32)
    weights=q@kd.transpose(-1,-2)/math.sqrt(8)
    weights=weights.masked_fill(torch.ones(12,12,device='cuda',dtype=torch.bool).triu(1),float('-inf'))
    full=weights.softmax(-1)@vd
    steps=torch.cat([(q[:,:,i:i+1]@kd[:,:,:i+1].transpose(-1,-2)/math.sqrt(8)).softmax(-1)@vd[:,:,:i+1] for i in range(12)],2)
    err=(full-steps).abs().max().item();assert torch.allclose(full,steps,rtol=1e-5,atol=1e-5)
    checks.update(uint8_roundtrip_exact=True,FP32_cached_attention_max_abs=err,FP32_tolerance=1e-5)
    write('checks/numerical_oracles.json',checks);print('ORACLES_PASS',checks,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','oracles','diagnose','develop','final']);args=parser.parse_args()
    preflight();a.settings();ROOT.mkdir(exist_ok=True)
    with torch.inference_mode():
        {'prepare':prepare,'oracles':oracles,'diagnose':diagnose,'develop':calibration_and_development,'final':final}[args.phase]()

if __name__=='__main__':main()
