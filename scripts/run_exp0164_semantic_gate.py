#!/usr/bin/env python3
"""Run the EXP-0164 tokenizer/model semantic gate on the original weights.

This is deliberately independent of the DSP runtime.  It first runs the
original BF16 checkpoint, then reloads the checkpoint as FP16 and replaces
every Linear weight (including lm_head) with the project's signed symmetric
per-output-channel W4 reconstruction.  Both paths use the same real Qwen3
chat template and deterministic greedy decoding.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = (
    "请用一段简洁、完整的中文回答：低比特量化为什么能够减少大模型推理的"
    "内存带宽？为什么更低的位宽又不一定直接带来更低的端到端延迟？请同时"
    "提到权重搬运、计算核选择和流水调度。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/d/llm_exp/models/Qwen3-origin"),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_prompt(tokenizer: AutoTokenizer, prompt: str) -> tuple[str, torch.Tensor, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    batch = tokenizer(rendered, return_tensors="pt")
    return rendered, batch.input_ids, batch.attention_mask


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    new_tokens: int,
) -> dict[str, object]:
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    generated = output[0, input_ids.shape[1] :]
    return {
        "token_ids": generated.tolist(),
        "text": tokenizer.decode(generated, skip_special_tokens=False),
        "elapsed_s": time.monotonic() - started,
    }


def load_model(model_root: Path, dtype: torch.dtype) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()


def quantize_linear_weights_w4(model: AutoModelForCausalLM) -> dict[str, object]:
    tensor_count = 0
    element_count = 0
    names: list[str] = []
    started = time.monotonic()
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight.data.float()
            maximum = weight.abs().amax(dim=1)
            scale = torch.where(
                maximum > 0.0, maximum / 7.0, torch.ones_like(maximum)
            )
            quantized = torch.round(weight / scale[:, None]).clamp(-7, 7)
            module.weight.data.copy_(
                (quantized * scale[:, None]).to(module.weight.dtype)
            )
            tensor_count += 1
            element_count += module.weight.numel()
            names.append(name)
            del weight, maximum, scale, quantized
    return {
        "scheme": "signed_symmetric_per_output_channel_w4_qmin_-7_qmax_7",
        "tensor_count": tensor_count,
        "element_count": element_count,
        "includes_lm_head": "lm_head" in names,
        "elapsed_s": time.monotonic() - started,
    }


def main() -> None:
    args = parse_args()
    if args.new_tokens <= 0:
        raise ValueError("--new-tokens must be positive")
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, args.threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_root = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, use_fast=True
    )
    rendered, input_ids, attention_mask = render_prompt(
        tokenizer, args.prompt
    )
    if input_ids.shape[1] > 64:
        raise ValueError(
            f"EXP-0164 prompt has {input_ids.shape[1]} tokens, maximum is 64"
        )

    record: dict[str, object] = {
        "experiment": "EXP-0164",
        "model_root": str(model_root),
        "model_index_sha256": sha256_file(
            model_root / "model.safetensors.index.json"
        ),
        "tokenizer_sha256": sha256_file(model_root / "qwen3-tokenizer.json"),
        "prompt": args.prompt,
        "rendered_prompt": rendered,
        "prompt_token_count": int(input_ids.shape[1]),
        "prompt_token_ids": input_ids[0].tolist(),
        "new_tokens": args.new_tokens,
        "decode": "greedy",
    }

    load_started = time.monotonic()
    teacher = load_model(model_root, torch.bfloat16)
    record["teacher_load_s"] = time.monotonic() - load_started
    record["teacher_bf16"] = generate(
        teacher, tokenizer, input_ids, attention_mask, args.new_tokens
    )
    del teacher
    gc.collect()

    load_started = time.monotonic()
    w4f16 = load_model(model_root, torch.float16)
    record["w4f16_load_s"] = time.monotonic() - load_started
    record["w4f16_quantization"] = quantize_linear_weights_w4(w4f16)
    record["w4f16"] = generate(
        w4f16, tokenizer, input_ids, attention_mask, args.new_tokens
    )
    del w4f16
    gc.collect()

    record["semantic_gate"] = {
        "teacher_nonempty": bool(record["teacher_bf16"]["text"].strip()),
        "w4f16_nonempty": bool(record["w4f16"]["text"].strip()),
        "w4f16_readable_manual_review_required": True,
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        target = args.output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    print(payload, end="")


if __name__ == "__main__":
    main()
