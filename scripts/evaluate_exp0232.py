#!/usr/bin/env python3
"""Packed FP16 software development evaluation; final data stays sealed until selection."""
import argparse,json,os,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from data_exp0232 import RESULT,OUTPUT,MODEL,SOURCE,CELLS,write,sha,preflight,verified
from awq_exp0232 import frozen as new_frozen
from experiment_exp0220 import load_packed
import evaluate_exp0230 as prior
import rotation_exp0219 as rot

def frozen():
    d=new_frozen()
    old=json.loads(verified('exp0230','dataset.json').read_text())
    d['samples'] += [r for r in old['samples'] if r['split']=='development']
    return d

def load(v,device):
    if v!='E64':
        model,h=prior.load(v,device)
        if v in ['C8','C64']:
            assert h==json.loads(verified('exp0230',v+'/package.json').read_text())['manifest_sha256']
        return model,h
    root=OUTPUT/'E64';expected=json.loads((RESULT/'E64/package.json').read_text())['manifest_sha256']
    assert sha(root/'manifest.json')==expected
    m=json.loads((root/'manifest.json').read_text());assert m['experiment']=='EXP-0232'
    assert m['coordinates']=='fresh_AWQ_input_diagonals_with_legal_norm_and_Up_compensation'
    assert json.loads((RESULT/'invariance.json').read_text())['pass_all']
    for n,e in m['files'].items():assert sha(root/n)==e['sha256'],n
    model,_=prior.load('C64','cpu')
    changed={f'model.layers.{i}.{long}.weight' for i in range(28) for long in rot.PROJECTIONS.values()}
    changed.update(f'model.layers.{i}.{norm}.weight' for i in range(28) for norm in ['input_layernorm','post_attention_layernorm'])
    import hashlib
    def digest(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
    other={n:digest(t) for n,t in list(model.named_parameters())+list(model.named_buffers()) if n not in changed}
    for i,layer in enumerate(model.model.layers):
        for name,long in rot.PROJECTIONS.items():load_packed(layer.get_submodule(long).weight,root/f'layer{i}',name)
        for name,long in [('input','input_layernorm'),('post','post_attention_layernorm')]:
            layer.get_submodule(long).weight.copy_(torch.from_numpy(np.fromfile(root/f'layer{i}/{name}_norm_weight_f16.bin',dtype='<f2')))
    assert all(digest(t)==other[n] for n,t in list(model.named_parameters())+list(model.named_buffers()) if n not in changed)
    return model.to(device).eval(),expected

def evaluate(phase,v,cpu=False,output_prefix=''):
    preflight();data=frozen()
    if phase in ['primary','reserve']:
        for control in ['F','C8','C64','E64']:
            dev=json.loads((RESULT/f'software/development_{control}.json').read_text())
            assert dev['repeat_exact'] and dev['causal_mask_exact']
            if control!='E64':assert dev['exact_prior_development_regression']
        if phase=='reserve':assert json.loads((RESULT/'summary_primary.json').read_text())['reserve_trigger']
    rows=[r for r in data['samples'] if r['split']==phase]
    if cpu:assert phase=='development';rows=rows[:8]
    name=output_prefix+f'software/{phase}_{v}'+('_cpu_check' if cpu else '')+'.json'
    assert not (RESULT/name).exists()
    torch.set_num_threads(8);torch.manual_seed(232);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.use_deterministic_algorithms(True)
    started=time.monotonic();device='cpu' if cpu else 'cuda';model,manifest_hash=load(v,device)
    def forward(batch):
        x=torch.tensor([r['prompt_ids']+r['target_ids'] for r in batch],device=device)
        logits=model(input_ids=x[:,:79],use_cache=False).logits[:,63:79,:].float()
        labels=x[:,64:80]
        scoring=logits.double() if cpu else logits
        nll=torch.logsumexp(scoring,-1)-scoring.gather(-1,labels[:,:,None]).squeeze(-1)
        return x,logits,nll
    with torch.inference_mode():
        x,logits,nll=forward(rows[:4]);_,again,repeated=forward(rows[:4])
        assert torch.equal(logits,again) and torch.equal(nll,repeated)
        scoring=logits.double() if cpu else logits
        independent=torch.nn.functional.cross_entropy(scoring.reshape(-1,logits.shape[-1]),x[:,64:80].reshape(-1),reduction='none').reshape(4,16)
        ce_error=float((independent-nll).abs().max())
        if cpu:
            legacy_nll=torch.logsumexp(logits,-1)-logits.gather(-1,x[:,64:80,None]).squeeze(-1)
            legacy_ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,64:80].reshape(-1),reduction='none').reshape(4,16)
            write(f'recovery/cpu_reduction_{v}_{time.time_ns()}.json',dict(
                original_fp32_method_difference=float((legacy_nll-legacy_ce).abs().max()),
                fp32_logsumexp_error_vs_fp64=float((legacy_nll-nll).abs().max()),
                fp32_cross_entropy_error_vs_fp64=float((legacy_ce-independent).abs().max()),
                fp64_method_difference=ce_error,unchanged_model_logits=True,
                repair='FP64 CPU reference scoring accumulation; GPU outputs and 5e-6 check unchanged'))
            print('CPU_REDUCTION_DIAGNOSTIC',float((legacy_nll-legacy_ce).abs().max()),ce_error,flush=True)
        assert ce_error<5e-6,ce_error
        changed=x[:,:79].clone();changed[:,70:]=123
        other=model(input_ids=changed,use_cache=False).logits[:,63:70,:].float()
        assert torch.equal(logits[:,:7],other)
        output=[]
        for start in range(0,len(rows),4):
            batch=rows[start:start+4];_,logits,nll=forward(batch)
            assert torch.isfinite(nll).all()
            for row,loss,top in zip(batch,nll.cpu().tolist(),logits.argmax(-1).cpu().tolist()):
                output.append(dict(id=row['id'],cell=row['cell'],nll=loss,top1=top))
            if start%64==0:print('SOFTWARE_PROGRESS',phase,v,start+4,len(rows),flush=True)
    regression=None
    if phase=='development' and v!='E64' and not cpu:
        old=json.loads(verified('exp0230',f'software/development_{v}.json').read_text())
        assert output==old['samples'],('unchanged control regression failed',v)
        regression=True
    means={c:float(np.mean([s['nll'] for s in output if s['cell']==c])) for c in CELLS}
    write(name,dict(variant=v,phase=phase,device=device,role='packed_FP16_software_not_exact_DSP',
        samples=output,mean_nll=float(np.mean(list(means.values()))),cell_nll=means,
        ppl=float(np.exp(np.mean(list(means.values())))),manifest_sha256=manifest_hash,
        dataset_sha256=sha(RESULT/'dataset.json'),repeat_exact=True,causal_mask_exact=True,independent_CE_max_abs=ce_error,exact_prior_development_regression=regression,
        torch_version=torch.__version__,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        scoring_accumulation='float64' if cpu else 'float32',elapsed_s=time.monotonic()-started))
    print('SOFTWARE_COMPLETE',phase,v,'PPL',float(np.exp(np.mean(list(means.values())))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','primary','reserve'])
    p.add_argument('variant',choices=['F','C8','C64','E64']);p.add_argument('--cpu-check',action='store_true');a=p.parse_args()
    evaluate(a.phase,a.variant,a.cpu_check)
