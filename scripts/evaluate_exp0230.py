#!/usr/bin/env python3
"""Packed FP16 software development evaluation; final data stays sealed until selection."""
import argparse,json,os,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from data_exp0230 import RESULT,OUTPUT,MODEL,SOURCE,CELLS,write,sha,preflight,verified
from export_exp0230 import frozen,BASE_PACKAGE,BASE_HASH
from experiment_exp0220 import load_packed
from run_exp0164_semantic_gate import load_model
import rotation_exp0219 as rot

def package(v):
    if v=='F':return OUTPUT.parent/'exp0217/f16f16_greedy16'
    return BASE_PACKAGE if v=='A0' else OUTPUT/v

def load(v,device):
    root=package(v);manifest=json.loads((root/'manifest.json').read_text())
    expected='0f8a359b559f252cb13329f57d4cabd56f17ca9bfb64bc5643f4d6014795070f' if v=='F' else BASE_HASH if v=='A0' else json.loads((RESULT/v/'package.json').read_text())['manifest_sha256']
    assert sha(root/'manifest.json')==expected
    old=manifest['files'];files={n.replace(chr(92),'/'):old.get(n.replace(chr(92),'/'),x) for n,x in old.items()}
    assert all(sha(root/n)==entry['sha256'] for n,entry in files.items())
    model=load_model(MODEL,torch.float32)
    if model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings=False
    if v!='F':
        for i,layer in enumerate(model.model.layers):
            for name,long in rot.PROJECTIONS.items():load_packed(layer.get_submodule(long).weight,root/f'layer{i}',name)
        load_packed(model.lm_head.weight,root,'generation_lm_head')
    # Reload non-transformer and norm tensors from the actual package too.
    for n,param in [('generation_embedding_weight_f16.bin',model.model.embed_tokens.weight),
                    ('generation_final_norm_weight_f16.bin',model.model.norm.weight)]:
        a=np.fromfile(root/n,dtype='<f2').reshape(param.shape);param.copy_(torch.from_numpy(a).float())
    for i,layer in enumerate(model.model.layers):
        for n,param in [('input',layer.input_layernorm.weight),('post',layer.post_attention_layernorm.weight)]:
            param.copy_(torch.from_numpy(np.fromfile(root/f'layer{i}/{n}_norm_weight_f16.bin',dtype='<f2')).float())
    # Match the prior canonical CPU FP16 model construction, including RoPE buffers.
    return model.half().to(device).eval(),expected

def evaluate(phase,v,cpu=False):
    preflight();data=frozen()
    if phase in ['primary','reserve']:
        selection=json.loads((RESULT/'selection.json').read_text())
        assert v in ['F','A0',selection['selected']]
        assert sha(package(selection['selected'])/'manifest.json')==selection['selected_manifest_sha256']
    rows=[r for r in data['samples'] if r['split']==phase]
    if cpu:assert phase=='development';rows=rows[:8]
    name=f'software/{phase}_{v}'+('_cpu_check' if cpu else '')+'.json'
    assert not (RESULT/name).exists()
    torch.set_num_threads(8);torch.manual_seed(230);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.use_deterministic_algorithms(True)
    started=time.monotonic();device='cpu' if cpu else 'cuda';model,manifest_hash=load(v,device)
    def forward(batch):
        x=torch.tensor([r['prompt_ids']+r['target_ids'] for r in batch],device=device)
        logits=model(input_ids=x[:,:79],use_cache=False).logits[:,63:79,:].float()
        labels=x[:,64:80];nll=torch.logsumexp(logits,-1)-logits.gather(-1,labels[:,:,None]).squeeze(-1)
        return x,logits,nll
    with torch.inference_mode():
        x,logits,nll=forward(rows[:4]);_,again,repeated=forward(rows[:4])
        assert torch.equal(logits,again) and torch.equal(nll,repeated)
        independent=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,64:80].reshape(-1),reduction='none').reshape(4,16)
        ce_error=float((independent-nll).abs().max());assert ce_error<5e-6
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
    means={c:float(np.mean([s['nll'] for s in output if s['cell']==c])) for c in CELLS}
    write(name,dict(variant=v,phase=phase,device=device,role='packed_FP16_software_not_exact_DSP',
        samples=output,mean_nll=float(np.mean(list(means.values()))),cell_nll=means,
        ppl=float(np.exp(np.mean(list(means.values())))),manifest_sha256=manifest_hash,
        dataset_sha256=sha(RESULT/'dataset.json'),repeat_exact=True,causal_mask_exact=True,independent_CE_max_abs=ce_error,
        torch_version=torch.__version__,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        elapsed_s=time.monotonic()-started))
    print('SOFTWARE_COMPLETE',phase,v,'PPL',float(np.exp(np.mean(list(means.values())))),flush=True)

def select():
    preflight();frozen()
    runs={v:json.loads((RESULT/f'software/development_{v}.json').read_text()) for v in ['F','A0','C8','C64']}
    expected=[r['id'] for r in json.loads((RESULT/'dataset.json').read_text())['samples'] if r['split']=='development']
    for v,d in runs.items():assert [r['id'] for r in d['samples']]==expected and d['repeat_exact'] and d['causal_mask_exact']
    selected='C64' if runs['C64']['mean_nll']<runs['C8']['mean_nll']-1e-6 else 'C8'
    write('selection.json',dict(selected=selected,rule='equal_cell_development_mean_NLL_tie1e-6_smaller_budget',
        development={v:dict(nll=d['mean_nll'],ppl=d['ppl'],cell_nll=d['cell_nll']) for v,d in runs.items()},
        inputs={v:sha(RESULT/f'software/development_{v}.json') for v in runs},
        selected_manifest_sha256=sha(package(selected)/'manifest.json'),final_scoring_used=False,
        selected_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),baseline_promoted=False))
    print('SELECTED_ON_DEVELOPMENT_ONLY',selected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','primary','reserve','select'])
    p.add_argument('variant',nargs='?',choices=['F','A0','C8','C64']);p.add_argument('--cpu-check',action='store_true');a=p.parse_args()
    select() if a.phase=='select' else evaluate(a.phase,a.variant,a.cpu_check)
