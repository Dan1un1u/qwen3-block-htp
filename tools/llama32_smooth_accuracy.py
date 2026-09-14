#!/usr/bin/env python3
"""L32-0014 precision-only smooth/dense-R4 experiment; no device claims.

All frozen activation boundaries are simulated explicitly. BF16 arithmetic in
this model is a floating quantization control, not an HMX arithmetic oracle.
"""
import argparse, json, math, random, time, gc, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from llama_reference import PROJECTIONS, provenance, sha256, rope
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0014')
MOD=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0014')
ORIG=Path('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin')
OLD=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0013/data')
HADS={}
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def log(*a):print(*a,flush=True)
def had(n,device='cuda'):
    key=(n,str(device))
    if key not in HADS:
        # Construct explicit Sylvester matrix; no activation butterfly operation.
        h=torch.ones(1,1,device=device)
        while len(h)<n:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
        HADS[key]=h/math.sqrt(n)
    return HADS[key]
def dense(x,n):return (x.float().reshape(-1,n)@had(n)).reshape(x.shape).to(x.dtype)
def aq(x,step):return ((x.float()/step).round().clamp(-128,127)*step).to(x.dtype)
def aff(lo,hi):
    lo=min(0.,lo);hi=max(0.,hi);step=float(np.float32(max(hi-lo,1e-8)/255))
    return dict(scale=step,zero_point=int(np.clip(np.floor(-lo/step+.5),0,255)),minimum=lo,maximum=hi)
def uq(x,q):return (((x.float()/q['scale']+q['zero_point']+.5).floor().clamp(0,255)-q['zero_point'])*q['scale']).to(x.dtype)
def norm(x):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)).to(x.dtype)
class Model(torch.nn.Module):
    def __init__(self,m):
        super().__init__();self.m=m;self.steps={};self.qparams={};self.r4=False;self.r3=True;self.mode='float';self.observer=None;self.cache=None
        self.config=json.loads((ORIG/'config.json').read_text());self.config['head_dim']=64
    def event(self,name,x):
        if self.observer is not None:self.observer(name,x)
        return x
    def boundary(self,name,x):
        self.event(name,x)
        enabled=self.mode=='all' or (self.mode=='residual_only' and name.endswith(('block_input','post_attention_residual','block_output'))) or (self.mode=='down_only' and name.endswith('.down'))
        if self.mode=='except_residual':enabled=not name.endswith(('block_input','post_attention_residual','block_output','.down'))
        if enabled:return uq(x,self.qparams[name])
        return x
    def linear(self,i,short,x):
        name=f'{i}.{short}';module=self.m.model.layers[i].get_submodule(PROJECTIONS[short])
        if short=='down' and self.r4:x=dense(x,8192)
        self.event(name+'.input',x)
        if self.mode!='float' and not (self.mode=='down_a16' and short=='down'):x=aq(x,self.steps[name])
        y=module(x);self.event(name+'.linear',y)
        return self.boundary(name,y)
    def hidden(self,ids,cache=None):
        x=self.m.model.embed_tokens(ids);x=self.boundary('0.block_input',x)
        past=0 if cache is None else cache[0][0].shape[2]
        pos=torch.arange(past,past+ids.shape[1],device=ids.device)[None,:]
        c,s=self.m.model.rotary_emb(x,pos);c=c[:,None];s=s[:,None]
        def rot(v):return v*c+torch.cat((-v[...,32:],v[...,:32]),-1)*s
        new=[]
        for i in range(16):
            pre=norm(x);self.event(f'{i}.input_norm',pre)
            q=self.linear(i,'q',pre).reshape(1,-1,32,64).transpose(1,2)
            k=self.linear(i,'k',pre).reshape(1,-1,8,64).transpose(1,2)
            v=self.linear(i,'v',pre).reshape(1,-1,8,64).transpose(1,2)
            q=rot(q);k=rot(k)
            # R3 output is quantized after dense multiplication. In all-A8 mode
            # also preserve the existing pre-R3 post-RoPE quantization boundary.
            q=self.boundary(f'{i}.q_rope',q);k=self.boundary(f'{i}.k_rope',k)
            if self.r3:q=dense(q,64);k=dense(k,64)
            q=self.boundary(f'{i}.q_r3',q);k=self.boundary(f'{i}.k_r3',k)
            if cache is not None:k=torch.cat((cache[i][0],k),2);v=torch.cat((cache[i][1],v),2)
            new.append((k,v));kk=k.repeat_interleave(4,1);vv=v.repeat_interleave(4,1)
            # Float softmax control; quantized carriers can be isolated separately
            # from native log2 attention, which the integer oracle evaluates.
            av=F.scaled_dot_product_attention(q,kk,vv,is_causal=cache is None)
            av=av.transpose(1,2).reshape(1,-1,2048);av=self.boundary(f'{i}.attention_concat',av)
            o=self.linear(i,'o',av);res=self.boundary(f'{i}.post_attention_residual',x+o)
            post=norm(res);self.event(f'{i}.post_norm',post)
            g=self.linear(i,'gate',post);u=self.linear(i,'up',post)
            z=F.silu(g)*u;self.event(f'{i}.middle_pre_r4',z)
            down=self.linear(i,'down',z);x=self.boundary(f'{i}.block_output',res+down)
            if i<15:self.event(f'{i+1}.block_input',x)
        return x,new
    def forward(self,ids,cache=None,select=None):
        x,new=self.hidden(ids,cache);x=norm(x);x=self.boundary('final_norm',x)
        if select is not None:x=x[:,select]
        logits=self.m.lm_head(x);logits=self.boundary('logits',logits)
        return logits,new

def load(stage='smooth'):
    m=AutoModelForCausalLM.from_pretrained(ORIG,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval()
    m.lm_head=torch.nn.Linear(2048,128256,bias=False,dtype=torch.bfloat16)
    state=torch.load(MOD/(stage+'.pt'),map_location='cpu',weights_only=True);m.load_state_dict(state);del state
    model=Model(m.cuda()).eval().requires_grad_(False);model.r4=stage!='folded';return model

def windows():return torch.load(MOD/'data.pt',weights_only=True)
@torch.no_grad()
def prepare():
    prov=provenance(ORIG)
    old=json.loads((OLD/'freeze.json').read_text())
    for n,h in old['files'].items():assert sha256(OLD/n)==h
    # Eight512 calibration windows and eight512 selection windows from distinct
    # train2048 blocks frozen in L32-0013; no validation sample used for fitting.
    a=np.fromfile(OLD/'calibration.bin',dtype='<u4').reshape(32,2048)
    data=dict(calibration=torch.tensor(a[:8,:512].astype('i8')),selection=torch.tensor(a[8:16,:512].astype('i8')),train=torch.tensor(a[16:26,:512].astype('i8')))
    
    if not (MOD/'data.pt').exists():torch.save(data,MOD/'data.pt')
    else:
        old_data=torch.load(MOD/'data.pt',weights_only=True)
        assert all(torch.equal(data[k],old_data[k]) for k in data)
    if not (OUT/'data-freeze.json').exists():save(OUT/'data-freeze.json',dict(parent_freeze=sha256(OLD/'freeze.json'),train_blocks=old['train_windows'],calibration_blocks=list(range(8)),selection_blocks=list(range(8,16)),train_blocks_for_affine=list(range(16,26)),length=512,sha256=sha256(MOD/'data.pt'),validation_sha256=sha256(OLD/'validation.bin'),bridge_sha256=sha256(OLD/'dataset.json')))
    torch.manual_seed(42);m=AutoModelForCausalLM.from_pretrained(ORIG,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().cuda()
    m.requires_grad_(False)
    ids=data['selection'][0:1,:80].cuda();before=m(ids).logits.float().cpu()
    r=had(2048)*torch.randint(0,2,(2048,1),device='cuda').float().mul(2).sub(1)
    matrices={'R1':r.cpu()};head=torch.nn.Linear(2048,128256,bias=False,device='cuda',dtype=torch.bfloat16)
    # Do not center embedding: exact RMSNorm-equivalent original-weight fold.
    for start in range(0,128256,1024):
        w=m.model.embed_tokens.weight[start:start+1024].float()
        head.weight[start:start+1024].copy_(((w*m.model.norm.weight.float())@r).bfloat16())
        m.model.embed_tokens.weight[start:start+1024].copy_((w@r).bfloat16())
    m.lm_head=head;m.config.tie_word_embeddings=False;m.model.norm.weight.fill_(1)
    checks=[]
    for i,l in enumerate(m.model.layers):
        r2=had(64)*torch.randint(0,2,(64,1),device='cuda').float().mul(2).sub(1);matrices[f'R2.{i}']=r2.cpu()
        for short,path in PROJECTIONS.items():
            mod=l.get_submodule(path);w=mod.weight.float()
            if short in ['q','k','v']:w=w*l.input_layernorm.weight.float()
            if short in ['gate','up']:w=w*l.post_attention_layernorm.weight.float()
            w=r.T@w if short in ['o','down'] else w@r
            if short=='v':w=(w.T.reshape(-1,8,64)@r2).reshape(2048,512).T
            if short=='o':w=(w.reshape(2048,32,64)@r2).reshape(2048,2048)
            mod.weight.copy_(w.bfloat16())
        l.input_layernorm.weight.fill_(1);l.post_attention_layernorm.weight.fill_(1)
    model=Model(m).eval().requires_grad_(False);model.r3=False
    debug={}
    def trace(n,x):
        if n.endswith('block_output'):debug[n]=x.detach().float().cpu()
    model.observer=trace
    custom,_=model(ids);hf_output=m(ids,output_hidden_states=True);hf=hf_output.logits
    model.observer=None
    for i in range(15):
        actual=debug[f'{i}.block_output'];expected=hf_output.hidden_states[i+1].float().cpu();log('CUSTOM_LAYER_ERROR',i,float((actual-expected).norm()/expected.norm()))
    custom_error=float((custom.float()-hf.float()).norm()/hf.float().norm());log('CUSTOM_LOGITS_ERROR',custom_error);assert custom_error<.003
    fold_error=float((custom.float().cpu()-before).norm()/before.norm());assert fold_error<.03
    torch.save({n:v.cpu() for n,v in m.state_dict().items()},MOD/'folded.pt');torch.save(matrices,MOD/'rotations.pt')
    del before,custom,hf
    maxima={i:torch.zeros(8192,device='cuda') for i in range(16)}
    def capture(n,x):
        if n.endswith('.middle_pre_r4'):
            i=int(n.split('.')[0]);maxima[i]=torch.maximum(maxima[i],x.float().abs().reshape(-1,8192).amax(0))
    model.observer=capture;model.r3=True
    for i,w in enumerate(data['calibration']):model.hidden(w[None].cuda());log('SMOOTH_MAX_WINDOW',i)
    model.observer=None;smooth={}
    for i,l in enumerate(m.model.layers):
        w=l.mlp.down_proj.weight.float();a=maxima[i].clamp_min(1e-6);b=w.abs().amax(0).clamp_min(1e-6)
        scale=(a/b).sqrt();scale=(scale/scale.median()).clamp(1/32,32)
        # Independent double-precision vector equivalence audit before casting.
        x=torch.randn(4,2048,device='cuda');g=x@l.mlp.gate_proj.weight.float().T;u=x@l.mlp.up_proj.weight.float().T;z=F.silu(g)*u
        wd=w*scale;wr=wd@had(8192)
        lhs=(dense(z/scale,8192).double()@wr.double().T);rhs=z.double()@w.double().T
        err=float((lhs-rhs).norm()/rhs.norm());assert err<2e-5
        l.mlp.up_proj.weight.copy_((l.mlp.up_proj.weight.float()/scale[:,None]).bfloat16());l.mlp.down_proj.weight.copy_(wr.bfloat16())
        smooth[i]=scale.cpu();checks.append(dict(layer=i,equivalence_relative_l2=err,min=float(scale.min()),max=float(scale.max()),a_max=float(a.max())))
        log('SMOOTH_FOLDED',i,checks[-1])
    torch.save(smooth,MOD/'smoothing.pt');torch.save({n:v.cpu() for n,v in m.state_dict().items()},MOD/'smooth.pt')
    save(OUT/'prepare.json',dict(original=prov,fold_logits_relative_l2=fold_error,custom_vs_hf_relative_l2=custom_error,smoothing=checks,embedding_centered=False,rotation='seed42 signed Hadamard R1/R2; R3dense64 and R4dense8192',folding='FP32 TF32 disabled products, BF16 stored transformed weights; independent equivalence audit',artifacts={n:sha256(MOD/n) for n in ['folded.pt','smooth.pt','rotations.pt','smoothing.pt']}))

# Input/gradient bank is collected with transformed BF16 weights and no A8.
def collect():
    model=load();data=windows();gen=torch.Generator().manual_seed(42)
    picks=[torch.cat((torch.arange(32),torch.randperm(480,generator=gen)[:64]+32)) for _ in range(8)]
    bank={};current=[0]
    def observe(n,x):
        if not n.endswith(('.input','.linear')):return
        key=n.rsplit('.',1)[0];row=bank.setdefault(key,dict(raw=[],saliency=[]));idx=picks[current[0]].to(x.device)
        if n.endswith('.input'):row['raw'].append(x.detach().reshape(-1,x.shape[-1])[idx].cpu())
        else:
            def backward(g):row['saliency'].append(g.detach().reshape(-1,g.shape[-1])[idx].float().square().cpu())
            x.register_hook(backward)
    model.observer=observe
    # Only embedding requires gradients to connect the graph; it is never updated.
    model.m.model.embed_tokens.weight.requires_grad_(True);losses=[]
    for i,w in enumerate(data['calibration']):
        current[0]=i;ids=w[None].cuda();logits,_=model(ids)
        loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,128256),ids[:,1:].reshape(-1));loss.backward();losses.append(float(loss.detach()))
        model.m.model.embed_tokens.weight.grad=None;del logits,loss;log('GUIDED_BACKWARD',i,losses[-1])
    bank={n:{k:torch.cat(v) for k,v in row.items()} for n,row in bank.items()}
    assert len(bank)==112 and all(v['raw'].shape[0]==768 and v['saliency'].shape[0]==768 for v in bank.values())
    torch.save(bank,MOD/'guided-bank.pt');save(OUT/'guided-bank.json',dict(losses=losses,rows=768,picks=[p.tolist() for p in picks],prefix_weight=64/480,source='fresh transformed float trajectory',sha256=sha256(MOD/'guided-bank.pt')))

@torch.no_grad()
def fit_grid(x,w,importance):
    bound=max(float(x.abs().max()),1e-8);grid=np.linspace(-16,1,33);trials=[]
    def score(v):
        step=bound*2**float(v)/127;qx=aq(x,step)
        err=F.linear(qx.float()-x.float(),w.float());loss=float((err.square()*importance).mean());trials.append((loss,step));return loss
    vals=[score(v) for v in grid];i=int(np.argmin(vals))
    for v in np.linspace(grid[max(i-1,0)],grid[min(i+1,32)],9):score(v)
    loss,step=min(trials);qx=aq(x,step);target=F.linear(x.float(),w.float());base=w.float().abs().amax(1,keepdim=True).clamp_min(1e-8)/7
    best=torch.full((w.shape[0],),float('inf'),device='cuda');scales=torch.empty_like(base);codes=torch.empty_like(w,dtype=torch.int8)
    for ratio in [.5,.6,.7,.8,.9,1.,1.1]:
        s=base*ratio;c=(w.float()/s).round().clamp(-7,7);qw=(c*s).to(w.dtype)
        error=F.linear(qx.float(),qw.float())-target;err=(error.square()*importance).mean(0);mask=err<best
        scales[mask]=s[mask];codes[mask]=c[mask].to(torch.int8);best=torch.minimum(best,err)
    return step,scales,codes,dict(activation_weighted_mse=loss,weight_joint_mse=float(best.mean()))

@torch.no_grad()
def gptq(w,scale,x,imp):
    # Fixed per-output grid; window uncertainty shrinkage as referenced method.
    a=x.float()*imp.sqrt()[:,None];h=2*(a.T@a)/len(a);diag=h.diagonal().clone();row=h.square().sum(1);blockrow=torch.zeros_like(diag)
    for block in a.reshape(8,96,-1):blockrow+=((block.T@block)*(16/len(a))).square().sum(1)
    variation=(blockrow-8*row).clamp_min(0)/56;off=(row-diag.square()).clamp_min(1e-30);alpha=(variation/off).clamp(0,1);r=(1-alpha).sqrt()
    h.mul_(r[:,None]).mul_(r[None,:]);h.diagonal().copy_(diag)
    dead=diag==0;h[dead,dead]=1;wf=w.float().clone();wf[:,dead]=0
    perm=torch.argsort(h.diagonal(),descending=True);inv=torch.argsort(perm);wf=wf[:,perm];h=h[perm][:,perm]
    h.diagonal().add_(.01*h.diagonal().mean());hinv=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)),upper=True)
    codes=torch.empty_like(wf,dtype=torch.int8)
    for start in range(0,w.shape[1],128):
        end=min(start+128,w.shape[1]);local=wf[:,start:end].clone();errors=torch.zeros_like(local)
        for j in range(end-start):
            q=(local[:,j]/scale[:,0]).round().clamp(-7,7);codes[:,start+j]=q.to(torch.int8)
            e=(local[:,j]-q*scale[:,0])/hinv[start+j,start+j];errors[:,j]=e
            local[:,j:]-=e[:,None]*hinv[start+j,start+j:end][None,:]
        wf[:,end:]-=errors@hinv[start:end,end:]
    return codes[:,inv],dict(shrinkage_mean=float(alpha.mean()),shrinkage_max=float(alpha.max()))

@torch.no_grad()
def fit():
    model=load().cpu();torch.cuda.empty_cache();bank=torch.load(MOD/'guided-bank.pt',weights_only=True);state=model.m.state_dict();steps={};scales={};details={};codes={};rtn={}
    frequency=torch.cat((torch.full((32,),64/480),torch.ones(64))).repeat(8).cuda()[:,None]
    for i in range(16):
        for short,path in PROJECTIONS.items():
            key=f'{i}.{short}';name=f'model.layers.{i}.{path}.weight';w=state[name].cuda();x=bank[key]['raw'].cuda();g2=bank[key]['saliency'].cuda()
            importance=frequency*(g2/g2.mean().clamp_min(1e-30)+.01);importance/=importance.mean()
            step,s,c,detail=fit_grid(x,w,importance);rtn[name]=(c.float()*s).bfloat16().cpu()
            gtoken=g2.mean(1);imp=frequency[:,0]*(gtoken/gtoken.mean().clamp_min(1e-30)+.01);imp/=imp.mean()
            qc,gd=gptq(w,s,aq(x,step),imp);details[key]=dict(detail,**gd,step=step,changed_fraction=float((qc!=c).float().mean()))
            codes[key]=qc.cpu();scales[key]=s.cpu();steps[key]=step;state[name]=(qc.float()*s).bfloat16().cpu()
            log('GUIDED_GPTQ_FIT',key,details[key]);del w,x,g2,importance,s,c,qc;torch.cuda.empty_cache()
        # Each layer is immutable recoverable checkpoint, do not overwrite.
        torch.save(dict(codes={k:v for k,v in codes.items() if k.startswith(str(i)+'.')},scales={k:v for k,v in scales.items() if k.startswith(str(i)+'.')},steps={k:v for k,v in steps.items() if k.startswith(str(i)+'.')},details={k:v for k,v in details.items() if k.startswith(str(i)+'.')}),MOD/f'fit-layer{i}.pt')
    torch.save(state,MOD/'gptq.pt');state.update(rtn);torch.save(state,MOD/'rtn.pt');torch.save(dict(codes=codes,scales=scales),MOD/'quantized.pt');save(MOD/'steps.json',steps);save(OUT/'fit.json',details)

def load_quant(stage='gptq'):
    model=load(stage);model.steps=json.loads((MOD/'steps.json').read_text());model.mode='input';return model
@torch.no_grad()
def carriers():
    model=load_quant();ranges={}
    def observe(n,x):
        if n.endswith(('.input','.linear')):return
        lo=float(x.min());hi=float(x.max());old=ranges.get(n,(lo,hi));ranges[n]=(min(lo,old[0]),max(hi,old[1]))
    model.observer=observe
    for i,w in enumerate(windows()['calibration']):model(w[None].cuda());log('CARRIER_CALIBRATION',i)
    q={n:aff(*v) for n,v in ranges.items()}
    for i in range(1,16):q[f'{i}.block_input']=q[f'{i-1}.block_output']
    save(MOD/'carriers.json',q);save(OUT/'carrier-calibration.json',dict(ranges=ranges,source='same train-only input-A8 GPTQ trajectory; no eval fitting'))

@torch.no_grad()
def evaluate(stage,mode,scope):
    target=OUT/f'{stage}-{mode}-{scope}.json'
    if target.exists():raise FileExistsError(target)
    model=load_quant(stage);model.mode=mode
    if mode!='input':model.qparams=json.loads((MOD/'carriers.json').read_text())
    samples=[]
    if scope=='bridge':
        ds=json.loads((OLD/'dataset.json').read_text())
        for r in ds['samples']:samples.append((r['id'],r['prompt_ids']+r['target_ids'],63,79))
    elif scope=='full':
        ids=np.fromfile(OLD/'validation.bin',dtype='<u4')
        for start in range(0,len(ids),2048):
            chunk=ids[start:start+2048]
            if len(chunk)>1:samples.append((start,chunk.astype('i8').tolist(),0,len(chunk)-1))
    elif scope=='selection':
        samples=[(i,w.tolist(),0,len(w)-1) for i,w in enumerate(windows()['selection'])]
    rows=[];started=time.monotonic()
    for j,(index,tokens,a,b) in enumerate(samples):
        ids=torch.tensor([tokens],device='cuda');hidden,_=model.hidden(ids);h=norm(hidden);h=model.boundary('final_norm',h)
        total=0.;targets=0
        for start in range(a,b,128):
            end=min(start+128,b);logits=model.m.lm_head(h[:,start:end]);logits=model.boundary('logits',logits)
            loss=F.cross_entropy(logits.float().reshape(-1,128256),ids[:,start+1:end+1].reshape(-1),reduction='sum');total+=float(loss);targets+=end-start
        rows.append(dict(id=index,nll_sum=total,targets=targets))
        if j%16==0:log('EVAL',stage,mode,scope,j,total/targets)
    n=sum(r['targets'] for r in rows);nll=sum(r['nll_sum'] for r in rows)/n
    result=dict(stage=stage,mode=mode,scope=scope,targets=n,nll=nll,ppl=math.exp(nll),rows=rows,elapsed_seconds=time.monotonic()-started,arithmetic='BF16 fakequant boundaries, float SDPA; NOT hardware log2/HMX',r3='dense64',r4='dense8192',sp2=False)
    save(target,result);log('PPL_RESULT',stage,mode,scope,result['ppl'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','collect','fit','carriers','eval']);p.add_argument('--weights',default='gptq');p.add_argument('--mode',default='input');p.add_argument('--scope',default='bridge');args=p.parse_args()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.stage=='eval':evaluate(args.weights,args.mode,args.scope)
    else:globals()[args.stage]()
