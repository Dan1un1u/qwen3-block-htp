#!/usr/bin/env python3
"""Three predeclared development-only A0 FP16 family restores, never candidates."""
import argparse,hashlib,json
from contextlib import ExitStack
import torch
from safetensors import safe_open
import evaluate_exp0230 as ev
from data_exp0230 import RESULT,MODEL,write
from experiment_exp0220 import verify_origin

def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def main(family):
    ev.preflight();ev.frozen();origin=verify_origin()
    original_load=ev.load;records={};prefix='diagnostics/'+family+'/'
    def hybrid_load(v,device):
        model,manifest=original_load(v,device)
        def selected(name):
            if family=='head':return name=='lm_head.weight'
            prefix='.self_attn.' if family=='attention' else '.mlp.'
            suffixes=('q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight') if family=='attention' else ('gate_proj.weight','up_proj.weight','down_proj.weight')
            return prefix in name and name.endswith(suffixes)
        targets={n:p for n,p in model.named_parameters() if selected(n)}
        assert len(targets)==dict(attention=112,mlp=84,head=1)[family]
        before={n:digest(p) for n,p in model.named_parameters() if n not in targets}
        before_buffers={n:digest(p) for n,p in model.named_buffers()}
        index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
        with ExitStack() as stack:
            handles={n:stack.enter_context(safe_open(str(MODEL/n),framework='pt',device='cpu')) for n in set(index.values())}
            for n,p in targets.items():
                key=n if n in index else 'model.embed_tokens.weight'
                assert key==n or n=='lm_head.weight'
                reference=handles[index[key]].get_tensor(key).float().half()
                assert reference.shape==p.shape
                p.copy_(reference.to(device));assert torch.equal(p.cpu(),reference)
                records[n]=dict(original_tensor=key,fp16_sha256=digest(reference))
        assert before=={n:digest(p) for n,p in model.named_parameters() if n not in targets}
        assert before_buffers=={n:digest(p) for n,p in model.named_buffers()}
        return model,manifest
    def diagnostic_write(name,value):
        assert name==prefix+'software/development_A0.json'
        value.update(variant='A0_FP16_'+family,role='development_only_software_hybrid_not_candidate',
            restored_tensors=records,original_shards=origin,all_other_parameters_and_buffers_exact=True,
            affects_selection=False,not_deployed=True,interactions_nonadditive=True)
        write(name,value)
    ev.load=hybrid_load;ev.write=diagnostic_write
    ev.evaluate('development','A0',output_prefix=prefix)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('family',choices=['attention','mlp','head'])
    main(p.parse_args().family)
