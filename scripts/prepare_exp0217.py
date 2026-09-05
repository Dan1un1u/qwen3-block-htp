#!/usr/bin/env python3
"""Independent FP16 greedy reference and immutable full generation package."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

import prepare_exp0164_generation_package as base
from export_exp0022_block import pack_fp16_hmx_weight
from run_exp0164_semantic_gate import load_model, generate, render_prompt, DEFAULT_PROMPT


def main():
    torch.set_num_threads(16)
    model = Path('/mnt/d/llm_exp/models/Qwen3-origin')
    output = Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0217/f16f16_greedy16')
    result = Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0217/semantic_reference.json')
    if output.exists() or result.exists():
        raise FileExistsError('Refusing to replace retained EXP-0217 evidence')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rendered, ids, mask = render_prompt(tokenizer, DEFAULT_PROMPT)
    assert ids.shape[1] == 64
    teacher = load_model(model, torch.float16)
    generated = generate(teacher, tokenizer, ids, mask, 16)
    del teacher
    reference = dict(experiment='EXP-0217', recipe='F16F16', prompt=DEFAULT_PROMPT,
                     rendered_prompt=rendered, prompt_token_ids=ids[0].tolist(),
                     tokenizer_sha256=base.sha256_file(model/'qwen3-tokenizer.json'),
                     f16f16=generated, reference_kind='independent_transformers_FP16_eager')
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(reference, ensure_ascii=False, indent=2)+'\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree('/mnt/d/llm_exp/models/qwen3-block-htp/exp0158/f16f16',
                    output, copy_function=base.copy_link)
    manifest = json.loads((output/'manifest.json').read_text())
    files = manifest['files']
    for name, values in [('generation_prompt_token_ids_u32.bin', ids[0].tolist()),
                         ('generation_expected_token_ids_u32.bin', generated['token_ids'])]:
        p = output/name
        base.atomic_write_bytes(p, np.asarray(values, dtype='<u4').tobytes())
        files[name] = base.file_record(p)
    files['generation_final_norm_weight_f16.bin'] = base.export_final_norm(model, output)
    # Reuse the byte-identical FP16 embedding retained for W4F16.
    emb = output/'generation_embedding_weight_f16.bin'
    base.copy_link('/mnt/d/llm_exp/models/qwen3-block-htp/exp0164/w4f16_greedy16/'+emb.name, str(emb))
    files[emb.name] = base.file_record(emb)
    head = output/'generation_lm_head_weight_f16_hmx.bin'
    with safe_open(base.model_tensor_location(model, 'lm_head.weight'), framework='pt', device='cpu') as src:
        weight = src.get_slice('lm_head.weight')
        assert tuple(weight.get_shape()) == (base.VOCAB, base.HIDDEN)
        with head.open('wb') as stream:
            for first in range(0, base.VOCAB, 1024):
                stream.write(pack_fp16_hmx_weight(weight[first:first+1024, :]).tobytes())
    assert head.stat().st_size == base.VOCAB*base.HIDDEN*2
    files[head.name] = base.file_record(head)
    for p in base.export_rope(output)+base.export_zero_caches(output):
        files[str(p.relative_to(output))] = base.file_record(p)
    manifest.update(experiment='EXP-0217', recipe='f16f16',
                    execution_unit='token_ids_embedding_28_layers_final_norm_FP16_head_greedy',
                    generation=dict(prompt_tokens=64, generated_tokens=16, cache_capacity=80,
                                    weight_format='FP16_HMX_N32_K32',
                                    semantic_reference_sha256=base.sha256_file(result),
                                    timed_full_logits_ddr=False))
    manifest['contract']['cache_capacity_per_layer'] = 80
    # Atomic replacement is required: the copied manifest may be a hard link.
    base.atomic_write_bytes(output/'manifest.json', (json.dumps(manifest, indent=2)+'\n').encode())
    print(json.dumps(reference, ensure_ascii=False, indent=2))
    print('PACKAGE='+str(output), flush=True)


if __name__ == '__main__':
    main()
