#!/usr/bin/env python3
"""Fixed C64 restorations on one resident canonical FP16 model, with full state audits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc,hashlib,json,math,subprocess,time
import numpy as np
import torch
from data_exp0235 import RESULT,OUTPUT,SOURCE,MODEL,BASE,CELLS,FAMILIES,specs,write,sha,verified,preflight,frozen
import evaluate_exp0230 as prior

def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def tensors(model):return dict(list(model.named_parameters())+list(model.named_buffers()))

def main():
    preflight();data=frozen();variants=specs();assert not (RESULT/'software/development_F.json').exists(), 'preserve previous attempt'
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    torch.set_num_threads(8);torch.set_grad_enabled(False);torch.manual_seed(235)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.use_deterministic_algorithms(True)
    origin_ledger=json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())
    for n,h in origin_ledger.items():assert sha(MODEL/n)==(h['sha256'] if isinstance(h,dict) else h),n
    original_model,fh=prior.load('F','cpu');original_state=tensors(original_model)
    original_hashes={n:digest(t) for n,t in original_state.items()}
    originals={n:original_state[n].detach().clone() for n in variants['F']}
    shapes={n:list(t.shape) for n,t in originals.items()};counts={n:t.numel() for n,t in originals.items()}
    del original_state,original_model;gc.collect()
    model,ch=prior.load('C64','cpu');assert ch==json.loads(verified('exp0230','C64/package.json').read_text())['manifest_sha256']
    baseline_state=tensors(model);baseline_hashes={n:digest(t) for n,t in baseline_state.items()}
    assert set(baseline_hashes)==set(original_hashes)
    for n,h in baseline_hashes.items():
        if n not in originals:assert h==original_hashes[n],('nonrestored floating control differs',n)
    baseline={n:baseline_state[n].detach().clone() for n in variants['F']}
    del baseline_state
    assert model.lm_head.weight.data_ptr()!=model.model.embed_tokens.weight.data_ptr()
    manifest=dict(original_shard_ledger_sha256=sha(verified('exp0218','original_checkpoint_sha256.json')),floating_manifest_sha256=fh,C64_manifest_sha256=ch,
        original_state_hashes=original_hashes,C64_state_hashes=baseline_hashes,shapes=shapes,weight_counts=counts,
        variants={v:dict(restored_names=names,restored_weight_count=sum(counts[n] for n in names)) for v,names in variants.items()},
        quantizable_weight_count=sum(counts.values()),source_head=source,other_weights_identical_between_controls=True)
    write('restoration_manifest.json',manifest)
    model=model.to('cuda').eval();state=tensors(model);previous=set();audits=[]
    def restore(v):
        nonlocal previous
        chosen=set(variants[v])
        for n in previous:state[n].copy_(baseline[n])
        for n in chosen:state[n].copy_(originals[n])
        actual={n:digest(t) for n,t in state.items()}
        expected={n:(original_hashes[n] if n in chosen else h) for n,h in baseline_hashes.items()}
        assert actual==expected,('restoration/frozen-state identity',v)
        if v=='F':assert actual==original_hashes
        if v=='C64':assert actual==baseline_hashes
        previous=chosen
        audit=dict(variant=v,restored_names=variants[v],restored_weight_count=sum(counts[n] for n in chosen),all_parameter_and_buffer_hashes_match=True,state_hashes=actual)
        audits.append(audit)
        return audit
    def score(phase,v,sentinel=False):
        rows=[r for r in data['samples'] if r['split']==phase];assert len(rows)==(128 if phase=='development' else 512)
        name=f'software/{phase}_{v}.json' if not sentinel else 'C64_final_sentinel.json';assert not (RESULT/name).exists()
        def forward(batch):
            x=torch.tensor([r['prompt_ids']+r['target_ids'] for r in batch],device='cuda')
            logits=model(input_ids=x[:,:79],use_cache=False).logits[:,63:79,:].float()
            assert torch.isfinite(logits).all()
            nll=torch.logsumexp(logits,-1)-logits.gather(-1,x[:,64:80,None]).squeeze(-1)
            return x,logits,nll
        started=time.monotonic()
        with torch.inference_mode():
            x,z,nll=forward(rows[:4]);_,z2,nll2=forward(rows[:4]);assert torch.equal(z,z2) and torch.equal(nll,nll2)
            ce=torch.nn.functional.cross_entropy(z.reshape(-1,z.shape[-1]),x[:,64:80].reshape(-1),reduction='none').reshape(4,16)
            error=float((ce-nll).abs().max());assert error<5e-6,error
            changed=x[:,:79].clone();changed[:,70:]=123
            other=model(input_ids=changed,use_cache=False).logits[:,63:70,:].float();assert torch.equal(z[:,:7],other)
            output=[]
            for start in range(0,len(rows),4):
                batch=rows[start:start+4];_,z,nll=forward(batch);assert torch.isfinite(nll).all()
                for row,loss,top in zip(batch,nll.cpu().tolist(),z.argmax(-1).cpu().tolist()):output.append(dict(id=row['id'],cell=row['cell'],nll=loss,top1=top))
        regression=None
        if phase=='development' and v in ['F','C64']:
            old=json.loads(verified('exp0230',f'software/development_{v}.json').read_text())
            assert output==old['samples'],('control regression',v);regression=True
        if sentinel:assert output==json.loads((RESULT/'software/development_C64.json').read_text())['samples']
        means={c:math.fsum(x for row in output if row['cell']==c for x in row['nll'])/(16*sum(row['cell']==c for row in output)) for c in CELLS}
        nll=math.fsum(x for row in output for x in row['nll'])/(16*len(rows));ppl=math.exp(nll)
        write(name,dict(variant=v,phase=phase,role='FP16_restoration_software_diagnostic_not_deployable_recipe',samples=output,cell_nll=means,mean_nll=nll,ppl=ppl,
            repeat_exact=True,causal_mask_exact=True,independent_CE_max_abs=error,all_logits_finite=True,exact_prior_development_regression=regression,
            restoration_manifest_sha256=sha(RESULT/'restoration_manifest.json'),dataset_sha256=sha(RESULT/'dataset.json'),
            development_dataset_sha256=sha(BASE/'exp0230/dataset.json'),torch_version=torch.__version__,source_head=source,elapsed_s=time.monotonic()-started))
        print('SENSITIVITY_SCORE',phase,v,'PPL',ppl,'sentinel',sentinel,flush=True)
    for v in variants:
        restore(v);score('development',v)
    for v in ['ATT_ALL','MLP_ALL','HEAD']:
        restore(v)
        for phase in ['primary','reserve']:score(phase,v)
    restore('C64');score('development','C64',sentinel=True)
    for n,t in originals.items():assert digest(t)==original_hashes[n],('original snapshot modified',n)
    for n,t in baseline.items():assert digest(t)==baseline_hashes[n],('C64 snapshot modified',n)
    write('restoration_audits.json',dict(pass_all=True,audits=audits,immutable_snapshots_exact=True,final_C64_sentinel_exact=True,all_restoration_sets_fixed_before_scoring=True))
    write('environment.json',dict(torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),source_head=source,
        TF32=False,FP16_reduced_precision_reduction=False,deterministic=True,CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],batch_size=4,use_cache=False))
    print('EXP235_ALL_FIXED_RESTORATIONS_COMPLETE',flush=True)
if __name__=='__main__':main()
