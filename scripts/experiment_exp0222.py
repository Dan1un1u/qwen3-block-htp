#!/usr/bin/env python3
"""Host-only paired LM-head ablation; never regenerates transformer weights."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from transformers import AutoTokenizer
import eval_exp0218 as ev
import rotation_exp0219 as rot
from experiment_exp0220 import load_packed, verify_origin, metrics
from run_exp0164_semantic_gate import load_model
from summarize_exp0221 import score

SOURCE = Path(__file__).resolve().parents[1]
RESULT = ev.ROOT.parent / 'exp0222'
OUTPUT = Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0222')
PREVIOUS = ev.ROOT.parent / 'exp0221'


def tensor_hash(x):
    return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()


def non_head_hashes(model):
    return {n: tensor_hash(p) for n, p in model.named_parameters() if n != 'lm_head.weight'}


def compare(actual, expected, tolerance):
    assert len(actual) == len(expected)
    worst = 0.0
    for a, b in zip(actual, expected):
        assert a.keys() == b.keys(), (a['id'], a.keys(), b.keys())
        for key in a:
            if key == 'nll':
                error = max(abs(x-y) for x, y in zip(a[key], b[key]))
                worst = max(worst, error)
                assert error <= tolerance, (a['id'], error)
            else:
                assert a[key] == b[key], (a['id'], key)
    return dict(samples=len(actual),max_nll_difference=worst,other_fields_exact=True)


def run(v):
    started = time.monotonic()
    directory = RESULT/v
    assert not directory.exists(), 'Retained experiment outputs must not be overwritten'
    directory.mkdir(parents=True)
    closure_path = PREVIOUS/'closure.json'
    assert ev.digest(closure_path) == '0b0b183eeca9b96c38a2e78454e2c68a7e1442579c8ff8d6f420918dc2ca489a'
    closure = json.loads(closure_path.read_text())
    package = closure['packages'][v]
    root = Path(package['path'])
    assert ev.digest(root/'manifest.json') == package['manifest_sha256']
    manifest = json.loads((root/'manifest.json').read_text())
    ledger = json.loads((PREVIOUS/'evidence_sha256.json').read_text())
    assert ev.digest(PREVIOUS/'evidence_sha256.json') == '8242201af9041c25352689c3e93a5112800a1b8186620477e3fc31072d18ab8d'
    old_quality = PREVIOUS/v/'software_quality.json'
    assert ev.digest(old_quality) == ledger[f'{v}/software_quality.json']
    original_shards = verify_origin()
    consumed = {}

    def verify(name):
        actual = ev.digest(root/name)
        assert actual == manifest['files'][name]['sha256'], name
        consumed[name] = actual

    def raw_copy(param, name):
        verify(name)
        value = np.fromfile(root/name, dtype='<f2').reshape(param.shape)
        param.copy_(torch.from_numpy(value))

    def packed_copy(param, prefix):
        for suffix in ['_weight_w4_hmx.bin', '_weight_w4_scale_f32.bin']:
            verify(prefix+suffix)
        load_packed(param, root, prefix)

    model = load_model(ev.MODEL, torch.float32)
    # Fresh head, before any package loading. Keep gamma in A's final norm;
    # B/C have identity final norm and therefore require gamma in their head.
    original_head = model.lm_head.weight.detach()
    gamma = model.model.norm.weight.detach().clone()
    fp16_head = torch.empty(original_head.shape, dtype=torch.float16)
    for first in range(0, len(original_head), 1024):
        w = original_head[first:first+1024]
        value = w if v == 'A' else w*gamma[None,:]
        if v == 'C':
            value = rot.fwht(value, 1)
        fp16_head[first:first+1024].copy_(value.half())
    # Independent dense matrix verifies selected fresh transformed head rows.
    h = torch.ones(1, 1)
    while len(h) < model.config.hidden_size:
        h = torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    h /= math.sqrt(len(h))
    w = original_head[:32]
    expected = w if v == 'A' else w*gamma[None,:]
    if v == 'C':
        expected = expected@h
    head_check = metrics(fp16_head[:32], expected)
    assert head_check['finite'] and head_check['nrmse'] <= .003 and head_check['cosine'] >= .99999
    model.lm_head.weight = torch.nn.Parameter(original_head.clone())
    model.config.tie_word_embeddings = False
    del original_head, expected, w, h
    model.half()
    for i, layer in enumerate(model.model.layers):
        for short, long in rot.PROJECTIONS.items():
            packed_copy(layer.get_submodule(long).weight, f'layer{i}/{short}')
        for short, long in [('input','input_layernorm'),('post','post_attention_layernorm'),
                            ('q','self_attn.q_norm'),('k','self_attn.k_norm')]:
            raw_copy(layer.get_submodule(long).weight, f'layer{i}/{short}_norm_weight_f16.bin')
        print(json.dumps(dict(variant=v,loaded_layer=i)), flush=True)
    raw_copy(model.model.embed_tokens.weight, 'generation_embedding_weight_f16.bin')
    raw_copy(model.model.norm.weight, 'generation_final_norm_weight_f16.bin')
    packed_copy(model.lm_head.weight, 'generation_lm_head')
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    before = non_head_hashes(model)
    data = ev.dataset()
    tokenizer = AutoTokenizer.from_pretrained(ev.MODEL, local_files_only=True)
    teacher = {s['id']:s for s in json.loads((ev.ROOT/'teacher_bf16.json').read_text())['samples']}
    # Four fixed repeat cases: first full NLL/task in each language.
    repeat_ids = [next(s['id'] for s in data['samples'] if s['split']=='full' and s['kind']==kind and s['language']==lang)
                  for kind in ['nll','task'] for lang in ['zh','en']]
    subset = dict(data, samples=[s for s in data['samples'] if s['id'] in repeat_ids])
    scores = {}
    for head in ['W4','FP16']:
        if head == 'FP16':
            model.lm_head.weight = torch.nn.Parameter(fp16_head)
        gc.collect()
        print(json.dumps(dict(variant=v,phase=head)), flush=True)
        rows = rot.quality(model, tokenizer, data)
        rot.write_json(directory/(head+'_quality.json'),dict(samples=rows,role='host_FP16_software_not_DSP'))
        if head == 'W4':
            control = compare(rows, json.loads(old_quality.read_text())['samples'], 1e-5)
            rot.write_json(directory/'control_reproduction.json', control)
        repeat = rot.quality(model, tokenizer, subset)
        expected_rows = [s for s in rows if s['id'] in repeat_ids]
        check = compare(repeat, expected_rows, 0.0)
        rot.write_json(directory/(head+'_repeat.json'),dict(samples=repeat,check=check))
        assert non_head_hashes(model) == before, 'Non-head tensor changed during ablation'
        scores[head] = score(rows, teacher)
        print(json.dumps(dict(variant=v,head=head,score=scores[head])), flush=True)
    artifact = OUTPUT/v
    artifact.mkdir(parents=True, exist_ok=False)
    path = artifact/'lm_head_weight_f16.bin'
    with path.open('xb') as f:
        f.write(fp16_head.numpy().tobytes())
    rot.write_json(directory/'provenance.json',dict(variant=v,original_shards=original_shards,
        package_manifest_sha256=package['manifest_sha256'],consumed_files=consumed,
        non_head_tensor_sha256=before,non_head_unchanged=True,head_construction=head_check,
        head_formula={'A':'W','B':'W*final_gamma','C':'(W*final_gamma)*H2048'}[v],
        fp16_head_path=str(path),fp16_head_sha256=ev.digest(path),
        dataset_sha256=ev.digest(ev.ROOT/'dataset_v1.json'),holdout_scored=False,
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        elapsed_s=time.monotonic()-started,device_work=False,profiling='N/A_host_only'))
    rot.write_json(directory/'summary.json',scores)
    print(json.dumps(dict(variant=v,completed=True,elapsed_s=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(16)
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['A','B','C'])
    run(parser.parse_args().variant)
