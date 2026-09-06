#!/usr/bin/env python3
"""Additional original BF16 teacher on exactly the frozen M64+16 windows."""
import json,subprocess,time,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from data_exp0229 import RESULT,MODEL,SOURCE,write,preflight,sha,verified
from measure_exp0229 import frozen

def main():
    preflight();frozen();assert not (RESULT/'teacher_bf16.json').exists()
    index=json.loads(verified('exp0218','original_checkpoint_sha256.json').read_text())
    # Retained checkpoint ledger schema is a filename->digest mapping.
    for name,entry in index.items():
        digest=entry['sha256'] if isinstance(entry,dict) else entry
        assert sha(MODEL/name)==digest,(name,'original checkpoint')
    torch.set_num_threads(8);torch.manual_seed(229)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=True).eval().to('cuda')
    samples=json.loads((RESULT/'dataset.json').read_text())['samples']
    start=time.monotonic();output=[]
    def forward(rows):
        x=torch.tensor([r['prompt_ids']+r['target_ids'] for r in rows],device='cuda')
        logits=model(input_ids=x[:,:79],use_cache=False).logits[:,63:79,:].float()
        labels=x[:,64:80]
        nll=torch.logsumexp(logits,dim=-1)-logits.gather(-1,labels[:,:,None]).squeeze(-1)
        return x,logits,nll
    with torch.inference_mode():
        x,logits,nll=forward(samples[:4]);_,again,repeat=forward(samples[:4])
        assert torch.equal(logits,again) and torch.equal(nll,repeat)
        independent=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,64:80].reshape(-1),reduction='none').reshape(4,16)
        error=(nll-independent).abs().max().item();assert error<5e-6
        changed=x[:,:79].clone();changed[:,70:]=123
        changed_logits=model(input_ids=changed,use_cache=False).logits[:,63:70,:].float()
        assert torch.equal(logits[:,:7],changed_logits),'future token leaked through causal mask'
        cached=model(input_ids=x[:,:64],use_cache=True);cache=cached.past_key_values
        values=[]
        for i in range(16):
            lg=cached.logits[:,-1,:].float();target=x[:,64+i]
            values.append(torch.logsumexp(lg,-1)-lg.gather(1,target[:,None]).squeeze(1))
            if i<15:
                cached=model(input_ids=x[:,64+i:65+i],past_key_values=cache,use_cache=True)
                cache=cached.past_key_values
        cache_nll=torch.stack(values,1)
        assert torch.isfinite(cache_nll).all()
        write('teacher_checks.json',dict(same_batch_repeat_exact=True,independent_CE_max_abs_error=error,
            future_token_causal_mask_exact=True,cache_vs_dense_nll_max_abs=float((cache_nll-nll).abs().max()),
            cache_vs_dense_nll_mean_abs=float((cache_nll-nll).abs().mean()),
            cache_comparison_role='BF16 shape-dependent rounding diagnostic; primary teacher uses fixed dense batch4',
            checkpoint_ledger_sha256=sha(verified('exp0218','original_checkpoint_sha256.json'))))
        for offset in range(0,len(samples),4):
            rows=samples[offset:offset+4];_,logits,loss=forward(rows)
            assert torch.isfinite(loss).all()
            top=logits.argmax(-1).cpu().tolist();loss=loss.cpu().tolist()
            for row,values,ids in zip(rows,loss,top):
                output.append(dict(id=row['id'],nll=values,top1=ids))
            if offset%64==0:print('TEACHER_PROGRESS',offset+4,len(samples),flush=True)
    write('teacher_bf16.json',dict(role='original_BF16_GPU_additional_reference_not_DSP_FP16',
        dataset_sha256=sha(RESULT/'dataset.json'),samples=output,torch_version=torch.__version__,
        GPU=torch.cuda.get_device_name(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        total_s=time.monotonic()-start,context_tokens=64,targets_per_window=16,batch=4))
    print('TEACHER_COMPLETE',len(output),time.monotonic()-start,flush=True)

if __name__=='__main__':main()
