#!/usr/bin/env python3
"""Frozen conditional PPL ladder, with actual integer attention boundaries."""
import argparse, json, math, time, subprocess
from pathlib import Path
import numpy as np
import torch
import r3_exp0246 as r
import data_exp0249 as data
import integer_attention_exp0249 as ia
from prepare_exp0042_attention import CONFIG
from export_exp0149_vertical_slice import build_attention_config

ROOT=data.RESULT
MODES=['legacy','carrier','score','exponent','sole']
NAMES=['F','C64']+[x+'_'+m for x in ['OFF','R3'] for m in MODES]
def write(n,d):data.write(n,d)
def read(n):return json.loads((ROOT/n).read_text())
def parent(n):return data.verified('exp0246',n)
def preflight():
    data.preflight()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=data.SOURCE,text=True).strip()=='codex/exp-0249-integer-attention-attribution'
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
            assert q['attention_probability']['zero_point']==0 and abs(q['attention_probability']['scale']-1/255)<1e-9
            v=np.zeros((64,8,128),np.uint8);v[32:]=255
            fields=list(CONFIG.iter_unpack(build_attention_config(v,q)))
            assert all(f[:1]+f[2:]==fields[0][:1]+fields[0][2:] for f in fields)
            configs[name].append(ia.config(fields[0]))
    p=parent('inputs.json');refs[str(p)]=data.sha(p)
    write('inputs.json',dict(names=NAMES,parameters=parameters,configs=configs,development=json.loads(p.read_text())['development'],prefix=[151645],source_head=r.b.head()))
    write('freeze.json',dict(files={'inputs.json':data.sha(ROOT/'inputs.json'),'dataset.json':data.sha(ROOT/'dataset.json')},references=refs,
        protocol_sha256=data.sha(data.MEMORY/'docs/experiments/EXP-0249.md'),frozen_before_inference=True))
def frozen():
    data.frozen();f=read('freeze.json')
    assert f['protocol_sha256']==data.sha(data.MEMORY/'docs/experiments/EXP-0249.md')
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
        self.configure(None if name in ['F','C64'] else 'mse',r.b.a.FAMILIES);self.rotation_counts={}
    def apply(self,name,family,x,layout='last',skip_quant=False):
        if not self.warm and self.mode!='legacy' and name.endswith('.attn_context'):
            return x  # Output already went through native integer AV conversion.
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
        c=self.frozen['configs'][self.param_name][i]
        scale=self.params[p+'q_rope']['mse']['scale']*self.params[p+'k_cache']['mse']['scale']*scaling
        result=ia.core(q,k,v,valid,c,self.mode,scale)
        output=r.b.p.decode(result['av'],self.params[p+'attn_context']['mse'],query.dtype)
        return output.transpose(1,2).contiguous(),(result['probability'].float()/255).to(query.dtype)

def check_arrays(q,k,v,valid,c,mode,expected=None,scale=.002):
    gpu=ia.core(*[torch.from_numpy(x).cuda() for x in [q,k,v,valid]],c,mode,scale)
    ref=ia.numpy_oracle(q,k,v,valid,c,mode,scale)
    for n in ['acc','raw','score','probability','av']:
        assert np.array_equal(gpu[n].cpu().numpy(),ref[n]),(mode,n)
        if expected and n in expected:assert np.array_equal(ref[n],expected[n]),('capture',mode,n)
    return {n:int(ref[n].size) for n in ['raw','probability','av']}

def numerical():
    rng=np.random.default_rng(249);c=read('inputs.json')['configs']['R3_ALL'][0];proofs=[]
    for nq,nk in [(64,64),(64,65),(1,80),(1,1),(7,13)]:
        q=rng.integers(0,256,(2,2,nq,128),dtype=np.uint8)
        k,v=[rng.integers(0,256,(2,2,nk,128),dtype=np.uint8) for _ in range(2)]
        valid=(np.arange(nk)[None,:]<=np.arange(nk-nq,nk)[:,None])[None,None]
        for mode in MODES[1:]:
            check_arrays(q,k,v,valid,c,mode);proofs.append([nq,nk,mode])
    # Negative ties, intermediate saturation and multiplier ordering.
    for shift in [0,1,8,15]:
        d=1<<shift;x=np.array([-300*d,-129*d,-128*d,-d-d//2,-d//2,0,d//2,127*d,256*d],np.int64)
        got,_=ia.requant(torch.from_numpy(x).cuda(),5,shift,121)
        expect=np.clip((np.clip((x+128*d+d//2)//d,0,255)-128)*5+121,0,255)
        assert np.array_equal(got.cpu().numpy(),expect)
    # Old and current retained device captures, independently archived hashes.
    import audit_exp0248 as au
    p=ROOT.parent/'exp0248';ledger=p/'EVIDENCE_SHA256.json'
    assert data.sha(ledger)=='9df15e39189ee4f9904c765321370b5262abdf46cde70e1f12ae39cd6eb9fc1e'
    hashes=json.loads(ledger.read_text())['files'];captures={};references={str(ledger):data.sha(ledger)}
    for tag in ['chain_r3_01','chain_scalar_01','chain_refined_02']:
        path=p/tag/'step00_r3.bin';assert data.sha(path)==hashes[str(path.relative_to(p))]['sha256'];references[str(path)]=data.sha(path)
        _,_,codes,slots=au.capture(tag,0)
        for g,fields in enumerate(au.configs):
            c=ia.config(fields);q=codes[2*g:2*g+2];k=np.repeat(codes[16+g:17+g],2,0);v=np.repeat(au.feat(slots[10],8)[g:g+1],2,0)
            expected=dict(raw=au.scores(slots[0])[2*g:2*g+2],probability=au.scores(slots[1])[2*g:2*g+2],av=au.feat(slots[2],16)[2*g:2*g+2])
            check_arrays(q,k,v,np.tril(np.ones((64,64),bool)),c,'sole',expected)
        captures[tag]='raw_QK_probability_AV_exact'
    old=ROOT.parent/'exp0042/20260829T110712Z_d5f68b72b799_formal'
    manifest=old/'package_manifest.json';assert data.sha(manifest)=='043f61337b2ad2758bb95791bc4ca768150f457f1f28bc7d20a7ad57eba42359'
    cfg=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0042/block_package_layer14_m64_integer_attention_parallel/attention_config_all_groups.bin')
    assert data.sha(cfg)==json.loads(manifest.read_text())['files'][cfg.name]['sha256']
    report=old/'attention_implementation_audit/implementation_reference.json';hashes=json.loads(report.read_text())['dump_sha256'];arr={}
    for n,h in hashes.items():
        f=report.parent/'attention_dump'/n;assert data.sha(f)==h;references[str(f)]=h
        x=np.fromfile(f,np.uint8);kind=n.split('_')[1];arr[kind]=au.scores(x) if kind in ['score','probability'] else au.feat(x,8 if kind in ['k','v'] else 16)
    for g,fields in enumerate(CONFIG.iter_unpack(cfg.read_bytes())):
        expected={n:arr[n][2*g:2*g+2] for n in ['score','probability','av']}
        check_arrays(arr['q'][2*g:2*g+2],np.repeat(arr['k'][g:g+1],2,0),np.repeat(arr['v'][g:g+1],2,0),np.tril(np.ones((64,64),bool)),ia.config(fields),'sole',expected)
    references.update({str(f):data.sha(f) for f in [manifest,cfg,report,au.P/'attention_config_all_groups.bin',au.P/'manifest.json']})
    write('checks/numerical.json',dict(pass_all=True,random_cases=proofs,captures=captures,EXP0042='scaled_QK_probability_AV_exact',negative_ties_and_saturation=True,
        references=references,integer_dot_abs_sum_bound=128*255*128,TF32=False,scope='retained_device_prefill_exact_plus_independent_random_prefill_and_decode'))
    print('NUMERICAL PASS',flush=True)

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
    rows=[];ins.rotation_counts={}
    for j in range(0,len(samples),4):
        logits,nll,cache=r.forward(model,ins,samples[j:j+4]);assert torch.isfinite(nll).all()
        assert all(k.dtype==v.dtype==(torch.uint8 if ins.policy else torch.float16) for k,v in zip(cache.key_cache,cache.value_cache))
        rows.extend(dict(id=x['id'],cell=x['cell'],nll=loss,top1=top) for x,loss,top in zip(samples[j:j+4],nll.cpu().tolist(),logits.argmax(-1).cpu().tolist()))
        if j%64==0:print('PROGRESS',phase,name,j+4,len(samples),flush=True)
    expected=len(ins.rotation)*len(samples)//4
    assert ins.rotation_counts==({'prefix_K':expected,'body_K':expected*16,'Q':expected*16} if expected else {})
    assert before==r.b.prefix_digest(ins)
    means={c:float(np.mean([x['nll'] for x in rows if x['cell']==c])) for c in data.CELLS};mean=float(np.mean(list(means.values())))
    d=dict(name=name,phase=phase,ppl=math.exp(mean),mean_nll=mean,cell_nll=means,samples=rows,source_head=r.b.head(),freeze_sha256=data.sha(ROOT/'freeze.json'),
        checks=dict(repeat_exact=True,causal_exact=True,CE_max_abs=error,prefix_immutable=True,rotation_counts=ins.rotation_counts),elapsed_s=time.monotonic()-start,
        role='conditional_integer_attention_software_PPL_not_full_DSP_PPL_or_throughput')
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
            if phase=='development' and name in ['F','C64','OFF_legacy','R3_legacy']:
                parent_name={'OFF_legacy':'A8','R3_legacy':'R3_ALL'}.get(name,name)
                p=parent(f'scores/development_{parent_name}_sequential.json');old=json.loads(p.read_text())
                assert d['samples']==old['samples'] and d['ppl']==old['ppl'],('legacy reproduction',name)
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
