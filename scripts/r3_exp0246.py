#!/usr/bin/env python3
"""PC061 explicit dense post-RoPE Q/K R3. No fast-Hadamard kernels."""
import argparse, copy, hashlib, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch
import ablate_exp0244 as b
import data_exp0246 as data

ROOT=data.RESULT
SCOPES={'L0':{0},'ALL':set(range(28))}
FINAL=['F','C64','A8','recal_L0','recal_ALL','R3_L0','R3_ALL']
DEV=FINAL+['C64_R3_L0','C64_R3_ALL']
def read(n):return json.loads((ROOT/n).read_text())
def write(n,x):data.write(n,x)
def sha(p):return data.sha(p)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def base_params():return json.loads(data.verified('exp0243','calibration/eos.json').read_text())['parameters']
def sign_matrix():return np.asarray([[1-2*((i&j).bit_count()%2) for j in range(128)] for i in range(128)],np.int8)
def dense(x,s):return ((x.float()@s)*(1/math.sqrt(128))).to(x.dtype)
def preflight():
    data.preflight()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=data.SOURCE,text=True).strip()=='codex/exp-0246-dense-online-r3'
def bind():b.ROOT=ROOT;b.data=data

class DenseCache(b.SelectiveCache):
    def update(self,k,v,i,cache_kwargs=None):
        k=self.ins.prepare_k(k,i,bool(cache_kwargs and cache_kwargs.get('prefix_seed')))
        return super().update(k,v,i,cache_kwargs)
class DenseFloatCache(b.FloatOracleCache):
    def update(self,k,v,i,cache_kwargs=None):
        k=self.ins.prepare_k(k,i,bool(cache_kwargs and cache_kwargs.get('prefix_seed')))
        return super().update(k,v,i,cache_kwargs)

class Instrument(b.Instrument):
    def __init__(self,model):
        self.rotation=set();self.qk_collect=False;self.chunks={};self.s=torch.from_numpy(sign_matrix()).to('cuda',torch.float32)
        self.rotation_counts={};super().__init__(model);self.cache_type=DenseCache
    def observe(self,name,x):
        if self.qk_collect:
            entry=self.chunks.setdefault(name,{'raw':[],'rotated':[]})
            entry['raw'].append(x.detach().clone());entry['rotated'].append(dense(x,self.s).detach())
    def prepare_k(self,x,i,seed):
        if not seed:self.observe(f'L{i:02d}.k_cache',x)
        if i in self.rotation:
            key='prefix_K' if seed else 'body_K';self.rotation_counts[key]=self.rotation_counts.get(key,0)+1
            x=dense(x,self.s)
        return x
    def apply(self,name,family,x,layout='last',skip_quant=False):
        if name.endswith('.q_rope') and not self.warm:
            self.observe(name,x)
            if int(name[1:3]) in self.rotation:
                self.rotation_counts['Q']=self.rotation_counts.get('Q',0)+1;x=dense(x,self.s)
        return super().apply(name,family,x,layout,skip_quant)
    def select(self,name):
        self.restored=set();self.rotation=SCOPES[name.split('_')[-1]] if 'R3_' in name else set()
        self.params=read('parameters/'+name+'.json')['parameters']
        self.configure(None if name in ['F','C64','C64_R3_L0','C64_R3_ALL'] else 'mse',b.a.FAMILIES)
        self.rotation_counts={}

def freeze():
    old=json.loads(data.verified('exp0245','development.json').read_text())
    cal=json.loads(data.verified('exp0243','inputs.json').read_text())['calibration']
    assert len(cal)==128 and all(len(r['token_ids'])==128 for r in cal)
    write('inputs.json',dict(calibration=cal,development=old['samples'],final_names=FINAL,development_names=DEV,prefix=[151645],
        source_head=b.head(),dataset_sha256=sha(ROOT/'dataset.json'),parent_development_sha256=sha(data.verified('exp0245','development.json'))))
    p=ROOT/'H128_signs_i8.bin';p.write_bytes(sign_matrix().tobytes())
    write('inputs_freeze.json',dict(inputs_sha256=sha(ROOT/'inputs.json'),matrix_sha256=sha(p),protocol_sha256=sha(data.MEMORY/'docs/experiments/EXP-0246.md'),
        base_parameters_sha256=sha(data.verified('exp0243','calibration/eos.json')),final_list_frozen_before_scoring=True))
def frozen():
    f=read('inputs_freeze.json');assert f['inputs_sha256']==sha(ROOT/'inputs.json') and f['matrix_sha256']==sha(ROOT/'H128_signs_i8.bin')
    assert f['protocol_sha256']==sha(data.MEMORY/'docs/experiments/EXP-0246.md') and f['base_parameters_sha256']==sha(data.verified('exp0243','calibration/eos.json'))
    assert read('inputs.json')['final_names']==FINAL

def numerical():
    s=sign_matrix();i=np.arange(128,dtype=np.uint8);bits=np.unpackbits(np.bitwise_and(i[:,None],i[None,:])[...,None],axis=-1).sum(-1)
    independent=np.where(bits%2,-1,1);assert np.array_equal(s,independent)
    h=s.astype(np.float64)/math.sqrt(128);orth=float(np.max(np.abs(h@h.T-np.eye(128))));assert orth<=1e-12
    torch.manual_seed(246);q=torch.randn(2,4,17,128,device='cuda')*torch.logspace(-2,2,128,device='cuda');k=torch.randn_like(q)
    sg=torch.from_numpy(s).to('cuda',torch.float32)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:z=dense(q,sg)
    ops=[x.key for x in prof.key_averages()];assert any(x in ops for x in ['aten::mm','aten::bmm'])
    ref=q.double()@torch.from_numpy(h).cuda();err=((z.double()-ref).square().mean()/ref.square().mean()).sqrt().item();assert err<=2e-6
    orig=q@k.transpose(-1,-2);rot=dense(q,sg)@dense(k,sg).transpose(-1,-2)
    ae=((orig-rot).square().mean()/orig.square().mean()).sqrt().item();assert ae<=2e-5
    write('checks/dense_numerical.json',dict(pass_all=True,independent_parity_exact=True,orthogonality_max_abs=orth,dense_FP32_relative_RMS=err,paired_attention_relative_RMS=ae,
        observed_operations=ops,tf32_enabled=torch.backends.cuda.matmul.allow_tf32,method='explicit_dense_GEMM_signs_then_normalize_no_butterfly',scalar=b.a.scalar_oracle()))

def calibrate():
    model,before,mh=b.model_session('C64');ins=Instrument(model);ins.build_prefix([151645]);ins.qk_collect=True
    stats={mode:{} for mode in ['raw','rotated']};samples=read('inputs.json')['calibration']
    for start in range(0,128,4):
        x=torch.tensor([r['token_ids'] for r in samples[start:start+4]],device='cuda');cache=ins.new_cache(4)
        model.model(input_ids=x[:,:64],past_key_values=cache,use_cache=True)
        for t in range(64,128):model.model(input_ids=x[:,t:t+1],past_key_values=cache,use_cache=True)
        assert cache.get_seq_length()==129 and len(cache.appended)==28*65 and len(ins.chunks)==56
        for name,modes in ins.chunks.items():
            for mode,parts in modes.items():
                value=torch.cat(parts,2);assert value.shape[2]==128
                st=stats[mode].setdefault(name,b.a.Stats('qk_kv'));st.add(value,'heads')
        ins.chunks={};print('CALIBRATION',start+4,128,flush=True)
    ins.qk_collect=False;results={}
    for mode in stats:
        records={};params={};arrays={}
        for name,st in stats[mode].items():
            seed=None
            if name.endswith('k_cache'):
                seed=ins.prefix_raw[int(name[1:3])][0]
                if mode=='rotated':seed=dense(seed,ins.s)
                st.lo=min(st.lo,seed.min().item());st.hi=max(st.hi,seed.max().item())
            records[name]=st.record();params[name]=b.a.choose_params(st,records[name])
            if seed is not None:
                for method in params[name]:
                    p=params[name][method];params[name][method]=b.a.qparams(min(p['requested_lo'],seed.min().item()),max(p['requested_hi'],seed.max().item()))
            arrays[name+'/hist']=st.hist;arrays[name+'/channel_max']=st.channel_max;arrays[name+'/token_absmax']=np.asarray(st.token_absmax,np.float32)
        file=ROOT/f'calibration/{mode}_arrays.npz';file.parent.mkdir(exist_ok=True);np.savez_compressed(file,**arrays)
        write(f'calibration/{mode}.json',dict(parameters=params,sites=records,arrays_sha256=sha(file),prefix_K_extrema_included=True))
        results[mode]=params
    original=base_params()
    for name in DEV:
        params=copy.deepcopy(original);changed=[]
        if name.startswith(('recal_','R3_')):
            mode='rotated' if name.startswith('R3_') else 'raw'
            for i in SCOPES[name.split('_')[-1]]:
                for short in ['q_rope','k_cache']:
                    site=f'L{i:02d}.{short}';params[site]=results[mode][site];changed.append(site)
        assert all(params[n]==original[n] for n in original if n not in changed)
        write(f'parameters/{name}.json',dict(parameters=params,changed_sites=sorted(changed),other_sites_exact=True,parameters_digest=digest(params)))
    assert before==b.a.state_digest(model);write('checks/calibration_weights.json',dict(unchanged=True,digest=before,manifest_sha256=mh,source_head=b.head()))
    ins.close();del model,ins;torch.cuda.empty_cache()
    write('parameters_freeze.json',dict(files={n:sha(ROOT/f'parameters/{n}.json') for n in DEV},frozen_before_scoring=True,inputs_sha256=sha(ROOT/'inputs.json')))

def forward(model,ins,batch):
    x=torch.tensor([r['token_ids'] for r in batch],device='cuda');cache=ins.new_cache(len(batch))
    parts=[model(input_ids=x[:,:64],past_key_values=cache,use_cache=True).logits[:,-1:].float()]
    for j in range(15):parts.append(model(input_ids=x[:,64+j:65+j],past_key_values=cache,use_cache=True).logits[:,-1:].float())
    logits=torch.cat(parts,1);assert cache.get_seq_length()==80 and len(cache.appended)==28*16
    if ins.cache_type is DenseCache:
        for k,v in zip(cache.key_cache,cache.value_cache):assert k.dtype==v.dtype==(torch.uint8 if ins.policy else torch.float16)
    labels=x[:,64:80];nll=torch.logsumexp(logits,-1)-logits.gather(-1,labels[:,:,None]).squeeze(-1)
    return logits,nll,cache

def score(model,ins,name,samples,phase):
    path=f'scores/{phase}_{name}_sequential.json';param_sha=sha(ROOT/f'parameters/{name}.json')
    if (ROOT/path).exists():
        d=read(path);assert d['parameters_sha256']==param_sha and d['dataset_sha256']==sha(ROOT/'dataset.json');return d
    ins.select(name);before=b.prefix_digest(ins);start=time.monotonic()
    logits,nll,_=forward(model,ins,samples[:4]);again,rn,_=forward(model,ins,samples[:4]);assert torch.equal(logits,again) and torch.equal(nll,rn)
    labels=torch.tensor([r['target_ids'] for r in samples[:4]],device='cuda');ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.flatten(),reduction='none').reshape(4,16)
    err=(ce-nll).abs().max().item();assert err<5e-6
    changed=[dict(r,token_ids=r['token_ids'][:70]+[123]*10) for r in samples[:4]];fl,_,_=forward(model,ins,changed);assert torch.equal(logits[:,:7],fl[:,:7])
    if phase=='development':
        ins.cache_type=DenseFloatCache;oracle,_,_=forward(model,ins,samples[:4]);ins.cache_type=DenseCache;assert torch.equal(logits,oracle)
    output=[];ins.rotation_counts={}
    for j in range(0,len(samples),4):
        logits,nll,_=forward(model,ins,samples[j:j+4]);assert torch.isfinite(nll).all()
        output.extend(dict(id=r['id'],cell=r['cell'],nll=v,top1=t) for r,v,t in zip(samples[j:j+4],nll.cpu().tolist(),logits.argmax(-1).cpu().tolist()))
    expected=len(ins.rotation)*(len(samples)//4)
    assert ins.rotation_counts==({'prefix_K':expected,'body_K':expected*16,'Q':expected*16} if expected else {})
    assert before==b.prefix_digest(ins)
    means={c:float(np.mean([r['nll'] for r in output if r['cell']==c])) for c in b.a.CELLS};mean=float(np.mean(list(means.values())))
    d=dict(name=name,phase=phase,mode='sequential',ppl=math.exp(mean),mean_nll=mean,cell_nll=means,samples=output,rotation_layers=sorted(ins.rotation),rotation_counts=ins.rotation_counts,
        checks=dict(repeat_exact=True,causal_exact=True,independent_CE_error=err,float_cache_oracle_exact=True if phase=='development' else None),prefix_immutable=True,prefix_digest=before,
        source_head=b.head(),dataset_sha256=sha(ROOT/'dataset.json'),parameters_sha256=param_sha,elapsed_s=time.monotonic()-start,role='software_dense_R3_not_HMX_or_speed')
    write(path,d);print('SCORE',phase,name,d['ppl'],round(d['elapsed_s'],1),flush=True);return d

def run(phase):
    pf=read('parameters_freeze.json')
    for n,h in pf['files'].items():assert sha(ROOT/f'parameters/{n}.json')==h
    names=DEV if phase=='development' else FINAL;samples=read('inputs.json')['development'] if phase=='development' else read('dataset.json')['samples']
    if phase=='final':assert read('development_complete.json')['pass_all']
    for recipe in ['C64','F']:
        proof=f'checks/{recipe}_{phase}_weights.json'
        if (ROOT/proof).exists():continue
        model,before,mh=b.model_session(recipe);ins=Instrument(model);ins.build_prefix([151645])
        for n in names:
            if (n=='F')!=(recipe=='F'):continue
            result=score(model,ins,n,samples,phase)
            if phase=='development' and n in ['F','C64','A8']:
                p=data.verified('exp0245',f'scores/development_{n}_sequential.json');old=json.loads(p.read_text());assert result['samples']==old['samples'] and result['ppl']==old['ppl']
                write(f'checks/reproduction_{n}.json',dict(exact=True,parent_score_sha256=sha(p)))
        if phase=='development' and recipe=='C64':
            delta={n:read(f'scores/development_{n}_sequential.json')['mean_nll']-read('scores/development_C64_sequential.json')['mean_nll'] for n in ['C64_R3_L0','C64_R3_ALL']}
            checks={};ins.select('C64');ref,_,_=forward(model,ins,samples[:4])
            for n in delta:
                assert abs(delta[n])<=.005,(n,delta[n]);ins.select(n);v,_,_=forward(model,ins,samples[:4]);checks[n]=dict(delta_meanNLL=delta[n],max_logit_abs=(v-ref).abs().max().item())
            write('checks/A16_dense_equivalence.json',dict(pass_all=True,controls=checks))
        assert before==b.a.state_digest(model);write(proof,dict(unchanged=True,digest=before,manifest_sha256=mh,source_head=b.head()))
        ins.close();del model,ins;torch.cuda.empty_cache()
    if phase=='development':write('development_complete.json',dict(pass_all=True,configurations=9,final_not_used=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','numerical','calibrate','development','final']);args=parser.parse_args()
    preflight();data.frozen();b.a.settings();bind()
    with torch.inference_mode():
        if args.phase=='freeze':freeze()
        else:
            frozen()
            if args.phase=='numerical':numerical()
            elif args.phase=='calibrate':calibrate()
            else:run(args.phase)
if __name__=='__main__':main()
