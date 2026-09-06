#!/usr/bin/env python3
"""AWQ-style diagonal search; original per-channel GPTQ remains the export.

Method reference mit-han-lab/llm-awq d6e797a42b9ef7778de8ee2352116e0f48a78d61.
Official activation-only twenty-ratio search, adapted to symmetric [-7,7].
Copyright (c) 2023 MIT HAN Lab; MIT license retained in docs/licenses/llm-awq-MIT.txt.
"""
import argparse,copy,gc,hashlib,json,os,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from data_exp0232 import RESULT,OUTPUT,SOURCE,MEMORY,MODEL,sha,write,verified,preflight
from gptq_exp0221 import fetch
from run_exp0164_semantic_gate import load_model
from experiment_exp0220 import load_packed,metrics
import rotation_exp0219 as rot

REFERENCE='d6e797a42b9ef7778de8ee2352116e0f48a78d61'
DATA_HASH='2baf928b58551c9aaef5f0d6847ac6497e7db6273a99d8cc5826f1d7f84df68a'
C64=OUTPUT.parent/'exp0230/C64'
C64_HASH='7de4f0758d83f2ba3b58696c695bfbfed72a25dd3bf308475abee0a0f0575a89'
RATIOS=[i/20 for i in range(20)]

def stream_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def references():
    files={}
    for n in ['awq/quantize/auto_scale.py','awq/quantize/quantizer.py','LICENSE']:
        p=RESULT/'references'/n;fetch(f'https://raw.githubusercontent.com/mit-han-lab/llm-awq/{REFERENCE}/{n}',p)
        files[n]=sha(p)
    write('references/provenance.json',dict(repository='https://github.com/mit-han-lab/llm-awq',commit=REFERENCE,files=files))
    print('AWQ_REFERENCES_PINNED',REFERENCE,flush=True)

def frozen():
    assert sha(RESULT/'dataset_freeze.json')==DATA_HASH
    f=json.loads((RESULT/'dataset_freeze.json').read_text())
    for n,h in f['files'].items():assert sha(RESULT/n)==h,n
    assert sha(MEMORY/'docs/experiments/EXP-0232.md')==f['protocol_sha256']
    assert json.loads((RESULT/'independent_data_audit.json').read_text())['pass_all']
    return json.loads((RESULT/'dataset.json').read_text())

def inputs():
    provenance=json.loads((RESULT/'references/provenance.json').read_text());assert provenance['commit']==REFERENCE
    for n,h in provenance['files'].items():assert sha(RESULT/'references'/n)==h,n
    assert sha(C64/'manifest.json')==C64_HASH
    manifest=json.loads((C64/'manifest.json').read_text())
    for n,e in manifest['files'].items():assert sha(C64/n)==e['sha256'],n
    ledger=json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())
    for n,h in ledger.items():assert sha(MODEL/n)==(h['sha256'] if isinstance(h,dict) else h),n
    raw=verified('exp0230','inputs/C64_calibration_u32.bin').read_bytes()
    ids=np.frombuffer(raw,dtype='<u4').reshape(512,128).astype(np.int64)
    small=verified('exp0230','inputs/C8_calibration_u32.bin').read_bytes()
    assert np.array_equal(ids[:64],np.frombuffer(small,dtype='<u4').reshape(64,128))
    return torch.from_numpy(ids),json.loads(verified('exp0230','artifacts_sha256.json').read_text())

def candidate_scale(mean,ratio):
    scale=mean.float().pow(ratio).clamp_min(1e-4)
    scale=scale/(scale.max()*scale.min()).sqrt()
    assert torch.isfinite(scale).all() and (scale>0).all()
    return scale

def rtn(w):
    w=w.float();maximum=w.abs().amax(1)
    scale=torch.where(maximum>0,maximum/7,torch.ones_like(maximum))
    codes=(w/scale[:,None]).round().clamp(-7,7)
    return codes*scale[:,None]

def search_scale(module,linears,x,kwargs=None):
    kwargs={} if kwargs is None else kwargs
    def forward():
        out=module(x,**kwargs);return out[0] if isinstance(out,tuple) else out
    target=forward();original=[fc.weight.detach().clone() for fc in linears]
    mean=x.float().abs().reshape(-1,x.shape[-1]).mean(0)
    history=[];best=float('inf');selected=None;choice=-1
    try:
        for i,ratio in enumerate(RATIOS):
            scale=candidate_scale(mean,ratio)
            for fc,w in zip(linears,original):
                # Quantized transformed weights represented back in input coordinates.
                # Native FP16 folding/GPTQ is independently checked and scored later.
                fc.weight.copy_((rtn(w.float()*scale[None,:]).half().float()/scale[None,:]).half())
            output=forward();loss=float((output.float()-target.float()).square().mean())
            assert np.isfinite(loss);history.append(loss)
            if loss<best:best=loss;selected=scale.clone();choice=i
            for fc,w in zip(linears,original):fc.weight.copy_(w)
    finally:
        for fc,w in zip(linears,original):fc.weight.copy_(w)
    assert choice==int(np.argmin(history))
    return selected,dict(choice=choice,ratio=RATIOS[choice],mse=history,selected_mse=best,
        calibration_positions=x.numel()//x.shape[-1],channels=x.shape[-1],
        input_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
        mean_abs_sha256=hashlib.sha256(mean.cpu().numpy().tobytes()).hexdigest(),
        scale_min=float(selected.min()),scale_max=float(selected.max()),
        proxy='symmetric_per_row_RTN_FP16_effective_weight_original_input_coordinates')

def fold_layer(layer,scales):
    d1,d2,d3=[scales[k].to(device=layer.input_layernorm.weight.device,dtype=torch.float32) for k in ['qkv','mlp','down']]
    assert layer.input_layernorm.weight.dtype==torch.float32
    layer.input_layernorm.weight.div_(d1)
    for n in ['q','k','v']:layer.get_submodule(rot.PROJECTIONS[n]).weight.mul_(d1[None,:])
    layer.post_attention_layernorm.weight.div_(d2)
    for n in ['gate','up']:layer.get_submodule(rot.PROJECTIONS[n]).weight.mul_(d2[None,:])
    layer.mlp.up_proj.weight.div_(d3[:,None]);layer.mlp.down_proj.weight.mul_(d3[None,:])
    assert all(torch.isfinite(p).all() for p in layer.parameters())

def setup():
    torch.set_grad_enabled(False);torch.set_num_threads(8);torch.manual_seed(232)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.use_deterministic_algorithms(True)

def check():
    setup();torch.manual_seed(232)
    # Independent FP64 norm/linear and nonlinear Up/Down algebra.
    x=torch.randn(37,32,dtype=torch.float64);gamma=torch.randn(32,dtype=torch.float64)
    d1=torch.exp(torch.randn(32,dtype=torch.float64));d2=torch.exp(torch.randn(32,dtype=torch.float64));d3=torch.exp(torch.randn(64,dtype=torch.float64))
    norm=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6);w=torch.randn(48,32,dtype=torch.float64)
    a=F.linear(norm*gamma,w);b=F.linear(norm*(gamma/d1),w*d1)
    assert torch.allclose(a,b,rtol=1e-12,atol=1e-12)
    gate=torch.randn(64,32,dtype=torch.float64);up=torch.randn(64,32,dtype=torch.float64);down=torch.randn(32,64,dtype=torch.float64)
    a=F.linear(F.silu(F.linear(norm*gamma,gate))*F.linear(norm*gamma,up),down)
    y=norm*(gamma/d2)
    b=F.linear(F.silu(F.linear(y,gate*d2))*F.linear(y,(up*d2)/d3[:,None]),down*d3)
    assert torch.allclose(a,b,rtol=1e-12,atol=1e-10)
    # Independent NumPy full grid and signed RTN oracle, includes zero rows/channels.
    x=torch.randn(257,32);x[:,0]=0;x[:,1]*=8
    fc=torch.nn.Linear(32,32,bias=False).half();fc.weight[0]=0
    original=fc.weight.detach().clone();scale,st=search_scale(fc,[fc],x.half())
    xn=x.half().numpy().astype(np.float32);wn=original.numpy().astype(np.float32)
    means=np.abs(xn).mean(0,dtype=np.float64).astype(np.float32);losses=[]
    for ratio in RATIOS:
        sc=np.maximum(np.power(means,ratio),1e-4);sc=sc/np.sqrt(sc.max()*sc.min())
        np.testing.assert_allclose(candidate_scale(torch.from_numpy(means),ratio).numpy(),sc,rtol=3e-6)
        transformed=wn*sc;maximum=np.abs(transformed).max(1);row=np.where(maximum>0,maximum/7,1.)
        q=np.clip(np.rint(transformed/row[:,None]),-7,7)*row[:,None]
        np.testing.assert_allclose(rtn(torch.from_numpy(transformed)).numpy(),q,rtol=2e-6,atol=1e-6)
        effective=(q.astype(np.float16).astype(np.float32)/sc).astype(np.float16).astype(np.float64)
        target=(xn.astype(np.float64)@wn.astype(np.float64).T).astype(np.float16).astype(np.float64)
        actual=(xn.astype(np.float64)@effective.T).astype(np.float16).astype(np.float64)
        losses.append(float(np.square(actual-target).mean()))
    np.testing.assert_allclose(st['mse'],losses,rtol=4e-3,atol=2e-5)
    assert st['choice']==int(np.argmin(losses));assert torch.equal(fc.weight,original)
    return dict(pass_all=True,FP64_legal_compensation_algebra=True,independent_numpy_RTN=True,
        independent_numpy_scale_grid_and_choice=True,twenty_ratios=RATIOS,zero_cases=True,
        original_weights_restored=True,oracle_cpu_FP16_GEMM_rounding_tolerance=dict(rtol=.004,atol=2e-5),
        note='Synthetic GEMM tolerance is a new oracle comparison, not a change to model/DSP gates.')

def search():
    preflight();setup();frozen();ids,ledger=inputs()
    assert json.loads((RESULT/'awq_oracle.json').read_text())['pass_all']
    assert not (RESULT/'search.json').exists() and not (OUTPUT/'scales').exists()
    model=load_model(MODEL,torch.float32);model.model.embed_tokens.half().cuda();model.model.rotary_emb.cuda()
    initial=model.model.embed_tokens(ids[:64].cuda());position=torch.arange(128,device='cuda').unsqueeze(0)
    rope=model.model.rotary_emb(initial,position)
    mask=torch.full((128,128),torch.finfo(torch.float16).min,dtype=torch.float16,device='cuda').triu(1)[None,None]
    kwargs=dict(position_embeddings=rope,attention_mask=mask)
    records=[];started=time.monotonic();(OUTPUT/'scales').mkdir(parents=True)
    for i,layer in enumerate(model.model.layers):
        name=f'checkpoints/C64/layer{i}_hidden.npy';checkpoint=Path(ledger['root'])/name
        assert stream_sha(checkpoint)==ledger['files'][name]['sha256']
        saved=np.load(checkpoint,mmap_mode='r');assert saved.shape==(512,128,2048) and saved.dtype==np.float16
        if i==0:incoming=initial
        else:
            previous=Path(ledger['root'])/f'checkpoints/C64/layer{i-1}_hidden.npy'
            incoming=torch.from_numpy(np.array(np.load(previous,mmap_mode='r')[:64])).cuda()
        original_fp32={n:p.detach().clone() for n,p in layer.named_parameters()}
        layer.half().cuda();control=copy.deepcopy(layer)
        for name,long in rot.PROJECTIONS.items():load_packed(control.get_submodule(long).weight,C64/f'layer{i}',name)
        qkv=control.input_layernorm(incoming)
        attention=control.self_attn(qkv,**kwargs)[0];residual=incoming+attention
        mlp=control.post_attention_layernorm(residual)
        down=control.mlp.act_fn(control.mlp.gate_proj(mlp))*control.mlp.up_proj(mlp)
        control_output=residual+control.mlp.down_proj(down)
        reference=torch.from_numpy(np.array(saved[:64])).cuda();concordance=metrics(control_output,reference)
        assert concordance['finite'] and concordance['nrmse']<=.003 and concordance['cosine']>=.99999,concordance
        del control,control_output,reference,residual,attention
        scales={};trials={}
        for family,module,linears,x,kw in [
            ('qkv',layer.self_attn,[layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj],qkv,kwargs),
            ('mlp',layer.mlp,[layer.mlp.gate_proj,layer.mlp.up_proj],mlp,{}),
            ('down',layer.mlp.down_proj,[layer.mlp.down_proj],down,{})]:
            scales[family],trials[family]=search_scale(module,linears,x,kw)
        # Real-layer equivalence uses fresh original FP32 values then final FP16.
        transformed=copy.deepcopy(layer).float()
        for n,p in transformed.named_parameters():p.copy_(original_fp32[n])
        fold_layer(transformed,scales);transformed.half()
        a=layer(incoming[:4],**kwargs)[0];b=transformed(incoming[:4],**kwargs)[0]
        equivalence=metrics(b,a)
        path=OUTPUT/'scales'/f'layer{i}.npz'
        np.savez(path,**{k:v.cpu().numpy() for k,v in scales.items()})
        record=dict(layer=i,trials=trials,unquantized_FP16_equivalence=equivalence,
            C64_GPU_CPU_calibration_concordance=concordance,scales_sha256=sha(path),
            C64_checkpoint_sha256=ledger['files'][f'checkpoints/C64/layer{i}_hidden.npy']['sha256'])
        write(f'search/layer{i}.json',record);records.append(record)
        assert equivalence['finite'] and equivalence['nrmse']<=.003 and equivalence['cosine']>=.99999,equivalence
        layer.cpu();del transformed,original_fp32,scales,qkv,mlp,down,incoming,a,b
        gc.collect();torch.cuda.empty_cache()
        print('AWQ_LAYER',i,'ratios',{k:v['ratio'] for k,v in trials.items()},'elapsed_s',round(time.monotonic()-started,1),flush=True)
    write('search.json',dict(experiment='EXP-0232',pass_all=True,records=records,
        calibration_manifest_sha256=C64_HASH,calibration_tokens_for_scale_search=8192,actual_GPTQ_tokens=65536,
        ratios=RATIOS,reference_commit=REFERENCE,torch_version=torch.__version__,elapsed_s=time.monotonic()-started,
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        final_or_development_data_used=False))
    print('AWQ_SEARCH_COMPLETE',flush=True)

def load_scales(i):
    search=json.loads((RESULT/'search.json').read_text());assert search['pass_all']
    p=OUTPUT/'scales'/f'layer{i}.npz';assert sha(p)==search['records'][i]['scales_sha256']
    with np.load(p) as f:return {k:torch.from_numpy(f[k].copy()) for k in ['qkv','mlp','down']}

def invariance():
    preflight();setup();frozen();ids,_=inputs();assert json.loads((RESULT/'search.json').read_text())['pass_all']
    assert not (RESULT/'invariance.json').exists();outputs={};started=time.monotonic()
    for variant in ['original','transformed']:
        model=load_model(MODEL,torch.float32)
        if variant=='transformed':
            for i,layer in enumerate(model.model.layers):fold_layer(layer,load_scales(i))
        model.half().cuda()
        hidden=model.model(input_ids=ids[:8].cuda(),use_cache=False).last_hidden_state
        logits=model.lm_head(hidden[:,-1])
        assert torch.isfinite(hidden).all() and torch.isfinite(logits).all()
        outputs[variant]=dict(hidden=hidden.cpu(),logits=logits.cpu())
        np.savez(OUTPUT/f'unquantized_{variant}_FP16.npz',hidden=hidden.cpu().numpy(),logits=logits.cpu().numpy())
        del model,hidden,logits;gc.collect();torch.cuda.empty_cache()
    checks={k:metrics(outputs['transformed'][k],outputs['original'][k]) for k in ['hidden','logits']}
    passed=all(r['finite'] and r['nrmse']<=.003 and r['cosine']>=.99999 for r in checks.values())
    write('invariance.json',dict(pass_all=passed,checks=checks,calibration_windows=8,
        search_sha256=sha(RESULT/'search.json'),canonical_FP16_RoPE=True,elapsed_s=time.monotonic()-started,
        unquantized_head_on_both_sides=True,torch_version=torch.__version__))
    assert passed,checks
    print('AWQ_UNQUANTIZED_FP16_INVARIANCE_PASS',json.dumps(checks),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['references','oracle','search','invariance']);a=p.parse_args()
    if a.phase=='references':preflight();references()
    elif a.phase=='oracle':preflight();write('awq_oracle.json',check());print('AWQ_ORACLES_PASS')
    elif a.phase=='search':search()
    else:invariance()
