#!/usr/bin/env python3
"""Freeze a tied-head Llama generation package and lightweight heldout NLL set."""
import argparse,ast,json,os,struct
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from llama_quantized import load_quantized,link_projection
from llama_reference import provenance,pack_weight,rope,layer,sha256
from export_llama32 import write


def native_cache(cache,kind):
    # M64 immutable HMX base plus 16 row-major decode slots, per KV head.
    carriers=[]
    for head in range(8):
        rows=cache[head,:64]
        logical=rows if kind=="k" else rows.T
        base=pack_weight(logical).reshape(-1)
        carriers.append(np.concatenate((base,np.zeros(16*64,dtype="<f2"))))
    return np.concatenate(carriers)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--evidence",type=Path,required=True)
    ap.add_argument("--model",type=Path,default=Path("/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin"))
    ap.add_argument("--quantized",type=Path)
    ap.add_argument("--dataset",type=Path)
    args=ap.parse_args()
    if args.output.exists() or args.evidence.exists():raise FileExistsError("Refuse to replace frontend evidence")
    prov=provenance(args.model)
    stack=json.loads((args.stack/"manifest.json").read_text());assert stack["layers"]==16
    for n,r in stack["files"].items():assert sha256(args.stack/n)==r["sha256"],n
    args.output.mkdir(parents=True);args.evidence.mkdir(parents=True)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    cfg=json.loads((args.model/"config.json").read_text())
    # Fixed natural-language M64 instruction, selected solely by token length.
    system="You are a helpful assistant. Give an accurate answer in one short sentence. Avoid unrelated details and unnecessary explanations."
    messages=[{"role":"system","content":system},{"role":"user","content":"What is the capital of France?"}]
    ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,date_string="10 Sep 2026")
    if len(ids)!=64:raise ValueError(f"Expected M64 prompt, got {len(ids)}")
    # Re-tokenize retained raw CC0 holdout passages; never reuse Qwen token IDs.
    source=Path(__file__).resolve().parents[1]/"scripts/prepare_exp0218_dataset.py"
    lists={}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ["EN","ZH"]:
            lists[node.targets[0].id]=ast.literal_eval(node.value)
    samples=[]
    for lang,key in [("en","EN"),("zh","ZH")]:
        for index,text in enumerate(lists[key][-4:]):
            encoded=tok.encode(text,add_special_tokens=True)
            if len(encoded)<80:raise ValueError("Holdout text too short")
            samples.append({"id":len(samples),"language":lang,"domain":"narrative","text":text,"prompt_ids":encoded[:64],"target_ids":encoded[64:80]})
    dataset={"id":"llama32-port-nll-v1","purpose":"heldout numerical integration diagnostic; not a general model-quality benchmark","license":"CC0-1.0","source_text_script_sha256":sha256(source),"tokenizer_sha256":sha256(args.model/"tokenizer.json"),"context_tokens":64,"scored_tokens_per_sample":16,"samples":samples,"thresholds":{"overall_ppl_ratio_max":1.05,"each_language_domain_ppl_ratio_max":1.10},"calibration_used":False,"selection_used":False}
    if args.dataset:
        frozen=json.loads((args.dataset/"freeze.json").read_text())
        for n,h in frozen.items():assert sha256(args.dataset/n)==h,n
        dataset=json.loads((args.dataset/"dataset.json").read_text())
        samples=[s for s in dataset["samples"] if s["split"]=="heldout"]
        dataset["samples"]=samples
    (args.evidence/"dataset.json").write_text(json.dumps(dataset,ensure_ascii=False,indent=2)+"\n")
    words=[0x51424556,1,len(samples),83]
    for s in samples:words += [s["id"],1,16]+s["prompt_ids"]+s["target_ids"]
    (args.evidence/"heldout.bin").write_bytes(struct.pack("<"+"I"*len(words),*words))
    (args.evidence/"freeze.json").write_text(json.dumps({n:sha256(args.evidence/n) for n in ["dataset.json","heldout.bin"]},indent=2))
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,attn_implementation="eager",local_files_only=True).cuda().eval()
    if args.quantized:load_quantized(model,args.quantized)
    w=model.state_dict();input_ids=torch.tensor([ids],device="cuda")
    prompt_x=torch.nn.functional.embedding(input_ids,w["model.embed_tokens.weight"])
    x=prompt_x
    for index in range(16):
        out=args.output/f"layer{index}";out.mkdir()
        for src in (args.stack/f"layer{index}").iterdir():
            if "cache" not in src.name:os.link(src,out/src.name)
        x,cache=layer(x,w,index,cfg,torch.arange(64,device="cuda")[None])
        for j,name in enumerate(["k","v"]):
            write(out/f"kv_cache_{name}_hmx_f16.bin",np.zeros(8*80*64,dtype="<f2"))
            write(out/f"reference_kv_cache_{name}_hmx_f16_step00.bin",native_cache(cache[j][0],name))
    write(args.output/"block_input_f16.bin",prompt_x)
    write(args.output/("reference_w4f16_block_output_f16.bin" if args.quantized else "reference_f16f16_block_output_f16.bin"),x)
    for start in range(16):
        pos=0 if start==0 else 63+start
        cos,sin=rope(cfg,torch.arange(pos,pos+64,device="cuda")[None],torch.float16)
        for name,value in [("cos",cos),("sin",sin)]:
            file=f"rope_{name}_f16.bin" if start==0 else f"generation_decode_rope_{name}_{start-1:02d}_f16.bin"
            write(args.output/file,value)
    write(args.output/"generation_embedding_weight_f16.bin",w["model.embed_tokens.weight"])
    if args.quantized:
        for src in (args.quantized/"head").glob("generation_lm_head_weight_w4_*.bin"):os.link(src,args.output/src.name)
    else:write(args.output/"generation_lm_head_weight_f16_hmx.bin",pack_weight(w["model.embed_tokens.weight"]))
    write(args.output/"generation_final_norm_weight_f16.bin",w["model.norm.weight"])
    write(args.output/"generation_prompt_token_ids_u32.bin",np.asarray(ids,dtype="<u4"))
    # Fixed 16-step replay, including tokens after EOS for arithmetic reproducibility.
    generated=[];past=None;current=input_ids
    for _ in range(16):
        r=model(current,past_key_values=past,use_cache=True)
        token=int(r.logits[0,-1].argmax());generated.append(token)
        past=r.past_key_values;current=torch.tensor([[token]],device="cuda")
    write(args.output/"generation_expected_token_ids_u32.bin",np.asarray(generated,dtype="<u4"))
    shown=generated[:next((i for i,t in enumerate(generated) if t in cfg["eos_token_id"]),len(generated))]
    teacher={"prompt_messages":messages,"prompt_ids":ids,"fp16_generated_ids":generated,"fp16_text":tok.decode(shown,skip_special_tokens=True),"nll":{}}
    for dtype in [torch.float16,torch.bfloat16]:
        # Reload original BF16 tensors; do not call a BF16<-FP16 roundtrip the original teacher.
        if dtype == torch.bfloat16 or args.quantized:
            model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=dtype,attn_implementation="eager",local_files_only=True).cuda().eval()
        else:
            model.to(dtype=dtype)
        rows=[]
        for s in samples:
            inp=torch.tensor([s["prompt_ids"]+s["target_ids"][:-1]],device="cuda")
            logits=model(inp,use_cache=False).logits[0,63:79].float()
            target=torch.tensor(s["target_ids"],device="cuda")
            rows.append({"id":s["id"],"language":s["language"],"nll":(logits.logsumexp(-1)-logits.gather(-1,target[:,None]).squeeze(-1)).tolist()})
        teacher["nll"][str(dtype)]=rows
    (args.evidence/"teacher.json").write_text(json.dumps(teacher,ensure_ascii=False,indent=2)+"\n")
    manifest={"experiment":("L32-0002" if args.quantized else "L32-0001"),"recipe":("W4A16" if args.quantized else "W16A16"),"layers":16,"generation_tokens":16,"prompt_tokens":64,"cache_capacity":80,"original":prov,"frontend_teacher_sha256":sha256(args.evidence/"teacher.json"),"dataset_freeze_sha256":sha256(args.evidence/"freeze.json"),"tied_head":("same original tensor; FP16 lookup and fresh W4 GPTQ head" if args.quantized else "same original embedding tensor, separate row-major and HMX carriers"),"files":{str(f.relative_to(args.output)):{"bytes":f.stat().st_size,"sha256":sha256(f)} for f in sorted(args.output.rglob("*")) if f.is_file()}}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"teacher_text":teacher["fp16_text"],"heldout_samples":len(samples),"heldout_tokens":len(samples)*16}),flush=True)

if __name__=="__main__":main()
