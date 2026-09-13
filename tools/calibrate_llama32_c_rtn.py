#!/usr/bin/env python3
"""Fresh train-only native boundary calibration and SP2 output-MSE alpha fit."""
import argparse,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from llama_reference import sha256,rms,rope,rotate
from llama32_sp2_contract import integer_levels
ROOT=Path(__file__).resolve().parents[1]
MODELS=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0013')
RESULTS=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0013')
ORIGINAL=Path('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin')

def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def freeze():
    from datasets import load_dataset
    from transformers import LlamaTokenizerFast
    out=RESULTS/'data';out.mkdir(exist_ok=False)
    tok=LlamaTokenizerFast.from_pretrained(ORIGINAL,local_files_only=True,add_bos_token=False,add_eos_token=False)
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1')
    ids=tok('\n\n'.join(ds['train']['text']),return_tensors='np')['input_ids'][0]
    selected=random.Random(42).sample(range(len(ids)//2048),32)
    calibration=np.stack([ids[i*2048:(i+1)*2048] for i in selected]).astype('<u4');calibration.tofile(out/'calibration.bin')
    val=tok('\n\n'.join(ds['validation']['text']),return_tensors='np')['input_ids'][0].astype('<u4');val.tofile(out/'validation.bin')
    assert len(val)==252852
    # Freeze an initial native M64+16 bridge before scoring. It is not WT2-2048.
    starts=np.linspace(0,len(val)-80,128,dtype=np.int64)
    rows=[];samples=[]
    for i,start in enumerate(starts):
        prompt=val[start:start+64].tolist();targets=val[start+64:start+80].tolist()
        rows.append([i,1,16,*prompt,*targets]);samples.append(dict(id=i,start=int(start),prompt_ids=prompt,target_ids=targets))
    with (out/'native-eval.bin').open('xb') as f:
        np.array([0x51424556,1,128,83],dtype='<u4').tofile(f);np.array(rows,dtype='<u4').tofile(f)
    save(out/'dataset.json',dict(name='WT2-validation-native-M64-plus16-bridge-v1',samples=samples,targets=2048,selection='128 deterministic evenly spaced non-overlapping80-token windows; all16 suffix targets; reset KV/position per sample; no BOS/EOS',historical_scope_match=False))
    save(out/'freeze.json',dict(train_split='train',train_windows=selected,calibration_shape=[32,2048],seed=42,validation_input_tokens=len(val),historical_2048_scored_targets=252728,files={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    print('C_DATA_FROZEN',flush=True)

def affine(lo,hi):
    lo=min(0.,lo);hi=max(0.,hi);s=np.float32(max(hi-lo,1e-8)/255)
    return dict(scale=float(s),zero_point=int(np.clip(np.floor(-lo/s+.5),0,255)),minimum=lo,maximum=hi)
def signed(scale):return dict(scale=scale,zero_point=128,minimum=-128*scale,maximum=127*scale)
def native_quant(x,q):return ((x.float()/q['scale']+q['zero_point']+.5).floor().clamp(0,255)-q['zero_point'])*q['scale']
def aq(x,scale):return ((x.float()/scale).round().clamp(-128,127)*scale).to(x.dtype)

def fit_alpha(x,w,full_absmax):
    """Same33 coarse/17 fine log-alpha response-MSE search, fixed native stimuli."""
    levels=torch.tensor(integer_levels(),device=x.device,dtype=torch.float32);levels=levels[levels>=0]/32768
    mid=(levels[:-1]+levels[1:])*.5
    x=x.float();w=w.float();reference=F.linear(x,w)
    peak=max(float(full_absmax),float(torch.finfo(torch.float32).tiny))
    def error(alpha):
        scaled=(x.abs()/alpha).clamp(max=1)
        v=levels[torch.bucketize(scaled.contiguous(),mid,right=False)]*x.sign()*alpha
        return float((F.linear(v,w)-reference).double().square().sum())
    # Match reference exponent search bounds, kept explicit in the artifact.
    logs=np.linspace(-16,1,33);coarse=[dict(log2_relative=float(v),alpha=float(np.float32(peak*2**v))) for v in logs]
    for row in coarse:row['sse']=error(row['alpha'])
    best=min(coarse,key=lambda r:r['sse']);best_index=coarse.index(best)
    lower=logs[max(0,best_index-1)];upper=logs[min(len(logs)-1,best_index+1)]
    fine=[dict(log2_relative=float(v),alpha=float(np.float32(peak*2**v))) for v in np.linspace(lower,upper,17)]
    for row in fine:row['sse']=error(row['alpha'])
    best=min(coarse+fine,key=lambda r:r['sse'])
    return dict(alpha=best['alpha'],sse=best['sse'],peak=peak,samples=x.shape[0],coarse=coarse,fine=fine)

@torch.inference_mode()
def calibrate():
    quant=MODELS/'quant-a01';m=json.loads((quant/'manifest.json').read_text())
    for n,h in m['files'].items():assert sha256(quant/n)==h['sha256']
    data=RESULTS/'data';fm=json.loads((data/'freeze.json').read_text())
    for n,h in fm['files'].items():assert sha256(data/n)==h
    out=MODELS/'calibration-a01';out.mkdir(exist_ok=False)
    cfg=json.loads((ORIGINAL/'config.json').read_text());sa=json.loads((quant/'activation_scales.json').read_text())
    ids=np.fromfile(data/'calibration.bin',dtype='<u4').reshape(32,2048)
    embed=np.memmap(quant/'embedding_weight_f16.bin',dtype='<f2',mode='r',shape=(128256,2048))
    hidden=torch.tensor(np.array(embed[ids.astype(np.int64)]),device='cuda')
    c,s=rope(cfg,torch.arange(2048,device='cuda')[None],torch.float16)
    qparams=[];ranges=[];alphas=[];torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    positions=torch.linspace(0,2047,64,device='cuda').long()
    for index in range(16):
        weights=load_file(quant/f'layer{index}/dequant.safetensors',device='cuda');r={};middles=[];gate_samples=[];up_samples=[];next_hidden=[]
        def update(name,x):
            if not torch.isfinite(x).all():raise ValueError((index,name,'nonfinite'))
            lo,hi=float(x.min()),float(x.max());old=r.get(name,[0,0]);r[name]=[min(old[0],lo),max(old[1],hi)]
        ins=sa[f'model.layers.{index}.self_attn.q_proj.quantizer'];posts=sa[f'model.layers.{index}.mlp.gate_proj.quantizer'];avs=sa[f'model.layers.{index}.self_attn.o_proj.quantizer']
        for sample in range(32):
            x=hidden[sample:sample+1];update('block_input',x)
            n=rms(x,weights['input_layernorm.weight'],cfg['rms_norm_eps']);update('input_norm',n);n=aq(n,ins)
            q=F.linear(n,weights['self_attn.q_proj.weight']);k=F.linear(n,weights['self_attn.k_proj.weight']);v=F.linear(n,weights['self_attn.v_proj.weight'])
            update('q_projection',q);update('k_projection',k);update('v',v)
            q=rotate(q.reshape(1,2048,32,64).transpose(1,2),c,s);k=rotate(k.reshape(1,2048,8,64).transpose(1,2),c,s)
            update('q_rope',q);update('k_rope',k)
            vv=v.reshape(1,2048,8,64).transpose(1,2).repeat_interleave(4,1)
            av=F.scaled_dot_product_attention(q,k.repeat_interleave(4,1),vv,is_causal=True).transpose(1,2).reshape(1,2048,2048)
            update('attention_concat',av);o=F.linear(aq(av,avs),weights['self_attn.o_proj.weight']);update('attention_projection',o)
            residual=x+o;update('post_attention_residual',residual)
            post=rms(residual,weights['post_attention_layernorm.weight'],cfg['rms_norm_eps']);update('post_attention_norm',post);post=aq(post,posts)
            g=F.linear(post,weights['mlp.gate_proj.weight']);u=F.linear(post,weights['mlp.up_proj.weight']);middle=F.silu(g)*u
            update('gate',g);update('up',u);update('middle',middle)
            down=F.linear(middle,weights['mlp.down_proj.weight']);update('down',down);y=residual+down;update('block_output',y);next_hidden.append(y)
            gate_samples.append(g[0,positions].clone());up_samples.append(u[0,positions].clone())
        hidden=torch.cat(next_hidden)
        qp={n:affine(*values) for n,values in r.items()};qp['attention_probability']=affine(0,1)
        qp['input_norm']=signed(ins);qp['post_attention_norm']=signed(posts);qp['attention_concat']=signed(avs)
        if index:qp['block_input']=qparams[-1]['block_output'].copy()
        g=native_quant(torch.cat(gate_samples),qp['gate']);u=native_quant(torch.cat(up_samples),qp['up']);middle=F.silu(g)*u
        alpha=fit_alpha(middle,weights['mlp.down_proj.weight'],max(abs(v) for v in r['middle']))
        qp['middle']['scale']=float(np.float32(alpha['alpha']/32768));qp['middle']['zero_point']=0
        qparams.append(qp);ranges.append(r);alphas.append(alpha)
        save(out/f'layer{index}.json',dict(layer=index,ranges=r,qparams=qp,sp2=alpha));print('C_CALIBRATED',index,alpha['alpha'],flush=True)
    hr={};head=np.memmap(quant/'head/dequant_f16.bin',dtype='<f2',mode='r',shape=(128256,2048));head=torch.tensor(np.array(head),device='cuda');gamma=torch.ones(2048,dtype=torch.float16,device='cuda')
    for sample in range(32):
        x=rms(hidden[sample,positions],gamma,cfg['rms_norm_eps']);y=F.linear(x,head)
        for name,value in [('generation_final_norm_output',x),('generation_lm_head_output',y)]:
            lo,hi=float(value.min()),float(value.max());old=hr.get(name,[0,0]);hr[name]=[min(lo,old[0]),max(hi,old[1])]
    save(out/'calibration.json',dict(experiment='L32-0013',qparams=qparams,ranges=ranges,sp2=alphas,generation_qparams={n:affine(*v) for n,v in hr.items()},weight_manifest_sha256=sha256(quant/'manifest.json'),data_freeze_sha256=sha256(data/'freeze.json'),method='fresh train32x2048, frozen learned shared SA; native-only outputs affine minmax; SP2 alpha response-MSE on2048 native-quantized Gate/Up sample rows per layer; no validation fitting',contract_differences=['native U8 intermediate/residual/KV and head', 'native input ties floor(x/scale+zp+.5)', '48 shared SA for96 sites','SP2 reconstruction remains integer, not BF16-rounded fakequant']))
    print('C_CALIBRATION_COMPLETE',flush=True)

if __name__=='__main__':
    import subprocess
    subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','calibrate']);globals()[p.parse_args().stage]()
