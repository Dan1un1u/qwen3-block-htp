#!/usr/bin/env python3
"""Final-format row-scale reconstruction with frozen GPTQ codes and rotations.

Algorithm reference: OmniQuant block-output reconstruction, arXiv:2308.13137.
Project adaptation: optimize scales of existing GPTQ codes, not RTN weights.
No source code is copied from OmniQuant and no equivalent transform is trained.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import eval_exp0218 as ev
import rotation_exp0219 as rot
import prepare_exp0164_generation_package as pack
from experiment_exp0220 import verify_origin, load_packed, metrics
from learned_rotation_exp0225 import fold, orthogonality
from run_exp0164_semantic_gate import load_model

RESULT = Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0227')
OUTPUT = Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0227')
ROOT = Path(__file__).resolve().parents[1]
CONTROLS = {'A': OUTPUT.parent/'exp0224/A', 'R': OUTPUT.parent/'exp0225/step100'}
MANIFESTS = {'A': 'a5de4e6c4e02ac913e69fbddb0d4b0b9e12b5cfe88ff606cf1ea18842dc0c179',
             'R': '44ac941297f1aab82ec4daff2ca8191379def73f6796a1c9c5d7d700820c5673'}
ROTATION = OUTPUT.parent/'exp0225/training_exact/step100.pt'
ROTATION_SHA = 'e48b1dbb3441e418a182e2c066acaef69607bbd118ca3f2727a8b157bd3499fd'
CALIBRATION = RESULT.parent/'exp0221/calibration.json'
VALIDATION = RESULT.parent/'exp0226/learning_data.json'
PLAN = dict(steps=100, checkpoints=[0, 50, 100], batch_size=8, seed=227,
            lr=.003, schedule='cosine', optimizer='Adam', weight_decay=0.,
            grad_norm_clip=1., log_scale_bound=math.log(2), layers=28,
            train_samples=64, validation_samples=64, sequence_length=128,
            trainable='transformer_row_scales_only', codes='frozen_GPTQ',
            objective='mean_per_sample_relative_full_block_output_MSE_teacher_own_stream_student_own_stream',
            selection='lowest_independent_block_validation_relative_MSE_at_0_50_100_tie_earliest',
            eval_used_for_training_or_selection=False)


def preflight():
    subprocess.run(['python3', '/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py',
                    'preflight', '--source-worktree', str(ROOT)], check=True)


def digest(path):
    return pack.sha256_file(path)


def data():
    assert digest(CALIBRATION) == 'e65edb14cb774956df92b27b5dc728b976f7ba00e9435145004180c6538a1669'
    assert digest(VALIDATION) == '7126e60d0dd74508982082fdab4cd2e2ecb8cb85375511b0f2fe04a5d0d573c6'
    train = json.loads(CALIBRATION.read_text())['samples']
    valid = json.loads(VALIDATION.read_text())['validation']
    assert len(train) == len(valid) == 64
    assert [s['language'] for s in train] == ['en']*32 + ['zh']*32
    assert all(len(s['token_ids']) == 128 for s in train+valid)
    for domain in ['wiki', 'news']:
        for lang in ['en', 'zh']:
            assert sum(s['domain'] == domain and s['language'] == lang for s in valid) == 16
    grams = lambda ids: {tuple(ids[i:i+32]) for i in range(len(ids)-31)}
    a = set().union(*(grams(s['token_ids']) for s in train))
    b = set().union(*(grams(s['token_ids']) for s in valid))
    assert not a & b
    qbh = ev.dataset()
    q = set().union(*(grams(s['prompt_ids']+s.get('target_ids', [])) for s in qbh['samples']))
    assert not a & q and not b & q
    return train, valid


def verify_package(v):
    root = CONTROLS[v]
    assert digest(root/'manifest.json') == MANIFESTS[v]
    manifest = json.loads((root/'manifest.json').read_text())
    for n, record in manifest['files'].items():
        assert digest(root/n.replace(chr(92), '/')) == record['sha256'], n
    return manifest


def read_codes(root, prefix, shape):
    n, k = shape
    raw = np.fromfile(root/(prefix+'_weight_w4_hmx.bin'), dtype=np.uint8).reshape(n//32,k//32,512)
    q = rot.unpack(raw,n,k).to(torch.int8)
    assert int(q.min()) >= -7 and int(q.max()) <= 7
    scale = torch.from_numpy(np.fromfile(root/(prefix+'_weight_w4_scale_f32.bin'),dtype='<f4'))
    assert scale.shape == (n,) and torch.isfinite(scale).all() and (scale > 0).all()
    return q, scale


class ScaleLinear(nn.Module):
    def __init__(self, codes, scales):
        super().__init__()
        self.register_buffer('codes', codes)
        self.register_buffer('base_scale', scales.float())
        self.theta = nn.Parameter(torch.zeros_like(scales, dtype=torch.float32))

    def scale(self):
        return self.base_scale * self.theta.exp()

    def weight(self):
        return (self.codes.float()*self.scale()[:,None]).half()

    def forward(self,x):
        return F.linear(x,self.weight())


def rel_mse(pred, target):
    num = (pred.float()-target.float()).square().mean(dim=(-2,-1))
    den = target.float().square().mean(dim=(-2,-1)).clamp_min(1e-12)
    return (num/den).mean()


def layer_forward(layer, x, rope, mask):
    return layer(x, attention_mask=mask, position_embeddings=rope, use_cache=False)[0]


@torch.no_grad()
def run_batches(layer, hidden, rope, mask):
    return torch.cat([layer_forward(layer, hidden[i:i+8], rope, mask).detach()
                      for i in range(0,len(hidden),8)])


@torch.no_grad()
def staged_forward(layer, x, rope, mask):
    residual = x
    x = layer.input_layernorm(x)
    x = layer.self_attn(x, attention_mask=mask, position_embeddings=rope, use_cache=False)[0]
    x = residual+x
    return x+layer.mlp(layer.post_attention_layernorm(x))


def batch_schedule():
    gen = torch.Generator().manual_seed(PLAN['seed'])
    out = []
    while len(out) < PLAN['steps']:
        en, zh = torch.randperm(32,generator=gen), torch.randperm(32,generator=gen)+32
        out.extend([torch.cat((en[i:i+4],zh[i:i+4])).tolist() for i in range(0,32,4)])
    return out[:PLAN['steps']]


def initialize():
    RESULT.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    assert not (RESULT/'protocol.json').exists()
    train,valid = data()
    manifests = {v:verify_package(v) for v in CONTROLS}
    assert digest(ROTATION) == ROTATION_SHA
    origin = verify_origin()
    rot.write_json(RESULT/'protocol.json',dict(plan=PLAN,train=train,validation=valid,
        calibration_sha256=digest(CALIBRATION),validation_source_sha256=digest(VALIDATION),
        control_manifests=MANIFESTS,rotation_checkpoint_sha256=ROTATION_SHA,original_shards=origin,
        batch_schedule=batch_schedule(),roles_32gram_disjoint=True,
        reference=dict(paper='https://arxiv.org/abs/2308.13137',
            algorithm='OmniQuant block output reconstruction; project-owned fixed-GPTQ-code scale-only adaptation',
            source_copied=False),control_file_counts={v:len(m['files']) for v,m in manifests.items()}))
    rot.write_json(RESULT/'learning_data.json',dict(validation=valid,plan=PLAN,source_sha256=digest(VALIDATION)))
    print('PROTOCOL_FROZEN',digest(RESULT/'protocol.json'),flush=True)


def unit_check():
    # Independent scalar-address carrier oracle, not the unpack reshape algorithm.
    from gptq_exp0221 import pack_codes
    gen=torch.Generator().manual_seed(227)
    q=torch.randint(-7,8,(64,96),generator=gen,dtype=torch.int8)
    packed=np.frombuffer(pack_codes(q),dtype=np.uint8).reshape(2,3,512)
    scalar=np.empty(q.shape,dtype=np.int8)
    for row in range(64):
        for col in range(96):
            inner=(col%32//4)*128+(row%32)*4+col%4
            byte=int(packed[row//32,col//32,inner//2])
            value=(byte>>(4*(inner%2)))&15
            scalar[row,col]=value-16 if value>=8 else value
    assert np.array_equal(q.numpy(),scalar)
    scale=torch.linspace(.001,.08,64)
    m=ScaleLinear(q,scale)
    expected=(scalar.astype(np.float32)*scale.numpy()[:,None]).astype(np.float16)
    assert np.array_equal(expected,m.weight().detach().numpy())
    # Exact analytic core derivative; FP16 forward itself is a staircase.
    qc=q.double();sc=scale.double();x=torch.randn(3,96,generator=gen,dtype=torch.float64)
    theta=torch.zeros(64,dtype=torch.float64,requires_grad=True)
    f=lambda t:F.linear(x,qc*(sc*t.exp())[:,None]).square().mean()
    grad=torch.autograd.grad(f(theta),theta)[0]
    errors=[]
    for i in [0,19,63]:
        a=theta.detach().clone();b=a.clone();a[i]+=1e-6;b[i]-=1e-6
        diff=(f(a)-f(b))/2e-6
        errors.append(float(abs(diff-grad[i])))
    assert max(errors)<1e-8
    pred=m(x.half());loss=pred.float().square().mean();loss.backward()
    assert torch.isfinite(m.theta.grad).all() and float(m.theta.grad.abs().sum())>0
    rot.write_json(RESULT/'unit_oracle.json',dict(passed=True,scalar_carrier_exact=True,
        numpy_FP16_dequant_exact=True,analytic_scale_gradient_max_abs=max(errors),
        FP16_cast_backward='straight-through cast derivative; actual forward weights are exactly exported'))
    print('UNIT_ORACLE_PASS',flush=True)


@torch.no_grad()
def fresh_teacher(v, probes, result_dir):
    model=load_model(ev.MODEL,torch.float32)
    for p in model.parameters():p.requires_grad_(False)
    if v=='R':
        assert digest(ROTATION)==ROTATION_SHA
        rotations=torch.load(ROTATION,map_location='cpu',weights_only=False)['rotations']
        assert max(orthogonality(rotations).values())<1e-4
        refs=[rot.logits(model,s) for s in probes]
        fold(model,rotations)
        checks=[dict(sample=s['id'],**rot.difference(rot.logits(model,s),ref)) for s,ref in zip(probes,refs)]
        assert all(c['passed'] for c in checks)
        rot.write_json(result_dir/'fresh_fold_invariance.json',dict(checks=checks,rotation_sha256=ROTATION_SHA))
    return model.half()


def train(v, smoke=False):
    protocol=json.loads((RESULT/'protocol.json').read_text());assert protocol['plan']==PLAN
    assert json.loads((RESULT/'unit_oracle.json').read_text())['passed']
    train_data,valid_data=data();verify_package(v)
    torch.manual_seed(227);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    work=OUTPUT/('smoke' if smoke else 'training')/v
    result=RESULT/('smoke' if smoke else 'training')/v
    work.mkdir(parents=True,exist_ok=True);result.mkdir(parents=True,exist_ok=True)
    assert not (result/'complete.json').exists()
    attempt=result/f'attempt_{time.time_ns()}';attempt.mkdir()
    probes=[dict(id=i,prompt_ids=train_data[i]['token_ids'][:64],target_ids=train_data[i]['token_ids'][64:80]) for i in [0,32]]
    started=time.monotonic();model=fresh_teacher(v,probes,attempt)
    ids=torch.tensor([s['token_ids'] for s in train_data+valid_data])
    with torch.no_grad():teacher_hidden=model.model.embed_tokens(ids).cuda()
    embedding=np.memmap(CONTROLS[v]/'generation_embedding_weight_f16.bin',dtype='<f2',mode='r',shape=model.model.embed_tokens.weight.shape)
    student_hidden=torch.from_numpy(np.asarray(embedding[ids.numpy()]).copy()).cuda()
    del embedding
    rotary=model.model.rotary_emb.cuda()
    with torch.no_grad():rope=rotary(teacher_hidden[:1],torch.arange(128,device='cuda')[None,:])
    mask=torch.full((128,128),torch.finfo(torch.float16).min,device='cuda',dtype=torch.float16).triu(1)[None,None]
    # Only one teacher/student block lives on GPU. CPU model is the original teacher.
    selected=[];total_params=0
    for i,teacher_layer in enumerate(model.model.layers):
        if smoke and i>=1:break
        teacher_layer=teacher_layer.cuda()
        with torch.no_grad():target=run_batches(teacher_layer,teacher_hidden,rope,mask)
        student=copy.deepcopy(teacher_layer)
        modules={}
        for short,long in rot.PROJECTIONS.items():
            old=student.get_submodule(long)
            codes,scales=read_codes(CONTROLS[v]/f'layer{i}',short,old.weight.shape)
            m=ScaleLinear(codes,scales).cuda();parent,name=long.rsplit('.',1)
            setattr(student.get_submodule(parent),name,m);modules[short]=m
        # Frozen original norms must match each package exactly.
        for short,long in [('input','input_layernorm'),('post','post_attention_layernorm')]:
            expected=np.fromfile(CONTROLS[v]/f'layer{i}/{short}_norm_weight_f16.bin',dtype='<f2')
            assert np.array_equal(student.get_submodule(long).weight.detach().cpu().numpy(),expected)
        params=[m.theta for m in modules.values()];total_params+=sum(p.numel() for p in params)
        with torch.no_grad():
            # Independently loaded dense baseline vs exact zero-step scale wrapper.
            dense=copy.deepcopy(teacher_layer)
            for short,long in rot.PROJECTIONS.items():
                load_packed(dense.get_submodule(long).weight,CONTROLS[v]/f'layer{i}',short)
            checks=[]
            for sample in [0,32]:
                x=student_hidden[sample:sample+1]
                a=layer_forward(student,x,rope,mask);b=layer_forward(dense,x,rope,mask)
                assert torch.equal(a,b),(v,i,'zero_step_dense_parity')
                c=staged_forward(dense,x,rope,mask)
                check=metrics(a,c)
                assert check['finite'] and check['nrmse']<=.003 and check['cosine']>=.99999
                checks.append(dict(sample=sample,zero_step_exact=True,**check))
            del dense
        layer_start=time.monotonic();score_rows=[];saved={}
        selection_path=result/f'layer{i:02d}_selection.json'
        if selection_path.exists():
            previous=json.loads(selection_path.read_text())
            chosen=previous['selected_step'];record=Path(previous['selected_checkpoint'])
            assert digest(record)==previous['selected_sha256']
            state=torch.load(record,map_location='cuda',weights_only=True)
            with torch.no_grad():
                for n,m in modules.items():m.theta.copy_(state[n])
            selected.append(previous)
        else:
            opt=torch.optim.Adam(params,lr=PLAN['lr'],weight_decay=0.)
            log=(attempt/f'layer{i:02d}.jsonl').open('x')
            steps=2 if smoke else PLAN['steps']
            snapshots={0,2} if smoke else set(PLAN['checkpoints'])
            for step in range(steps+1):
                if step in snapshots:
                    with torch.no_grad():
                        pred=run_batches(student,student_hidden[64:],rope,mask)
                        per_sample=((pred.float()-target[64:].float()).square().mean((-2,-1))/target[64:].float().square().mean((-2,-1)).clamp_min(1e-12)).cpu()
                        score=float(per_sample.mean());assert math.isfinite(score)
                        state={n:m.theta.detach().cpu().clone() for n,m in modules.items()}
                    path=work/f'layer{i:02d}_step{step:03d}.pt'
                    if path.exists():path=work/f'layer{i:02d}_step{step:03d}_{attempt.name}.pt'
                    torch.save(state,path);saved[step]=(state,path)
                    score_rows.append(dict(step=step,validation_relative_mse=score,per_sample=per_sample.tolist(),checkpoint=str(path),sha256=digest(path)))
                if step==steps:break
                indices=protocol['batch_schedule'][step]
                lr=PLAN['lr']*.5*(1+math.cos(math.pi*step/steps))
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True)
                pred=layer_forward(student,student_hidden[indices],rope,mask)
                loss=rel_mse(pred,target[indices]);assert torch.isfinite(loss)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,PLAN['grad_norm_clip'],error_if_nonfinite=True)
                opt.step()
                with torch.no_grad():
                    for p in params:p.clamp_(-PLAN['log_scale_bound'],PLAN['log_scale_bound'])
                row=dict(step=step+1,loss=float(loss),grad_norm=float(norm),lr=lr)
                log.write(json.dumps(row)+'\n');log.flush()
                if (step+1)%25==0:print(json.dumps(dict(variant=v,layer=i,**row)),flush=True)
            log.close()
            best=min(score_rows,key=lambda r:(r['validation_relative_mse'],r['step']))
            chosen=best['step'];state,path=saved[chosen]
            with torch.no_grad():
                for n,m in modules.items():m.theta.copy_(state[n].cuda())
            selection=dict(layer=i,selected_step=chosen,selected_checkpoint=str(path),selected_sha256=digest(path),
                candidates=score_rows,checks=checks,elapsed_s=time.monotonic()-layer_start,plan_sha256=digest(RESULT/'protocol.json'),qbh_used=False)
            rot.write_json(selection_path,selection);selected.append(selection)
            del opt,saved,state
        # Exact same FP32 scales are materialized once and used for propagation/export.
        final_scales={n:m.scale().detach().cpu() for n,m in modules.items()}
        with torch.no_grad():
            for n,m in modules.items():
                assert torch.isfinite(final_scales[n]).all() and (final_scales[n]>0).all()
                assert torch.isfinite(m.weight()).all()
            student_hidden=run_batches(student,student_hidden,rope,mask)
            teacher_hidden=target
        scales_path=work/f'layer{i:02d}_scales.pt'
        if not scales_path.exists():torch.save(final_scales,scales_path)
        else:
            old=torch.load(scales_path,weights_only=True)
            assert all(torch.equal(old[n],s) for n,s in final_scales.items())
        teacher_layer.cpu();del student,modules,params,target,final_scales
        gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(variant=v,completed_layer=i,selected_step=chosen,elapsed_s=time.monotonic()-started)),flush=True)
    torch.save(dict(teacher=teacher_hidden.cpu(),student=student_hidden.cpu()),work/'final_hidden.pt')
    rot.write_json(result/'complete.json',dict(variant=v,smoke=smoke,layers=len(selected),trainable_parameters=total_params,
        selected_steps=[s['selected_step'] for s in selected],protocol_sha256=digest(RESULT/'protocol.json'),
        elapsed_s=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        final_train_relative_mse=float(rel_mse(student_hidden[:64],teacher_hidden[:64])),
        final_validation_relative_mse=float(rel_mse(student_hidden[64:],teacher_hidden[64:]))))
    print('TRAIN_COMPLETE',v,smoke,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','unit','smoke','train']);p.add_argument('--variant',choices=['A','R']);a=p.parse_args()
    preflight();torch.set_num_threads(16)
    if a.phase=='init':initialize()
    elif a.phase=='unit':unit_check()
    else:
        assert a.variant
        train(a.variant,a.phase=='smoke')
