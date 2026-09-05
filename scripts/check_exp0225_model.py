#!/usr/bin/env python3
"""Independent HF full-model folding and chunked training-graph checks."""
import copy,json
import torch
import torch.nn.functional as F
from learned_rotation_exp0225 import LearnedModel,fold,RESULT
from experiment_exp0220 import metrics,verify_origin
from run_exp0164_semantic_gate import load_model
from eval_exp0218 import MODEL
from rotation_exp0219 import write_json

def main():
    torch.set_num_threads(8);verify_origin()
    data=json.loads((RESULT/'learning_data.json').read_text());ids=torch.tensor([data['train'][0]['token_ids'][:16]])
    with torch.no_grad():
        original=load_model(MODEL,torch.float32)
        ref=original(ids,use_cache=False).logits[:, -2:].clone()
        # Build training model separately, retaining exact original BF16 values in FP16.
        learned=LearnedModel(copy.deepcopy(original),device='cpu');learned.recompute=False;learned.set_quantized(False)
        rotations=learned.rotations();fold(original,rotations)
        fp32=original(ids,use_cache=False).logits[:,-2:];invariance=metrics(fp32,ref)
        assert invariance['finite'] and invariance['nrmse']<=.003 and invariance['cosine']>=.99999,invariance
        original.half();hidden=learned.hidden(ids)
        chunks=[learned.head_chunk(hidden[:,-2:],i,min(i+2048,len(learned.head.weight))) for i in range(0,len(learned.head.weight),2048)]
        dynamic=torch.cat(chunks,-1);folded=original(ids,use_cache=False).logits[:,-2:].float()
        agreement=metrics(dynamic,folded);assert agreement['finite'] and agreement['nrmse']<=.003 and agreement['cosine']>=.99999,agreement
        learned.set_quantized(True)
        full_h=learned.hidden(ids[:,:-1]);full_z=torch.cat([learned.head_chunk(full_h,i,min(i+2048,len(learned.head.weight))) for i in range(0,len(learned.head.weight),2048)],-1)
        expected=F.cross_entropy(full_z.reshape(-1,full_z.shape[-1]),ids[:,1:].reshape(-1))
        actual=learned.loss(ids)
        print('CE_DIAGNOSTIC',float(expected),float(actual),float(expected-actual),flush=True)
        write_json(RESULT/'chunked_ce_diagnostic_fp64_merge.json',dict(expected=float(expected),actual=float(actual),difference=float(expected-actual),invariance=invariance,agreement=agreement))
        assert abs(float(expected-actual))<1e-5
    write_json(RESULT/'full_model_oracle.json',dict(fp32_original_vs_folded=invariance,FP16_dynamic_vs_folded=agreement,chunked_CE=float(actual),full_CE=float(expected),data_role='training_text_only'))
    print('FULL_MODEL_ORACLE_PASS',invariance,agreement,flush=True)
if __name__=='__main__':main()
