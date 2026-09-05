#!/usr/bin/env python3
"""Frozen qbh-lite-v1 scoring and original BF16 reference, independent of DSP."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

ROOT=Path("/mnt/d/llm_exp/results/qwen3-block-htp/exp0218")
MODEL=Path("/mnt/d/llm_exp/models/Qwen3-origin")

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def grade(sample,text):
    text=text.strip()
    if sample["scorer"]=="text": return text in sample["answer"]
    if sample["scorer"]=="numeric":
        return bool(re.fullmatch(r"[+-]?\d+(?:\.0+)?",text)) and float(text)==sample["answer"]
    if sample["scorer"]=="json":
        try:
            value=json.loads(text)
            return value==sample["answer"] and type(value)==dict and all(
                type(value[k])==type(v) for k,v in sample["answer"].items())
        except (ValueError,KeyError,TypeError): return False
    raise ValueError(sample["scorer"])

def decode(tok,ids):
    if tok.eos_token_id in ids: ids=ids[:ids.index(tok.eos_token_id)]
    return tok.decode(ids,skip_special_tokens=False)

def dataset():
    freeze=json.loads((ROOT/"dataset_freeze.json").read_text())
    for name,h in freeze.items(): assert digest(ROOT/name)==h,name
    data=json.loads((ROOT/"dataset_v1.json").read_text())
    assert digest(MODEL/"qwen3-tokenizer.json")==data["tokenizer_sha256"]
    return data

def teacher():
    import torch
    import transformers
    from run_exp0164_semantic_gate import load_model
    data=dataset()
    out=ROOT/"teacher_bf16.json"
    assert not out.exists(),"Refusing to replace frozen teacher"
    torch.set_num_threads(16)
    tok=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    start=time.monotonic();model=load_model(MODEL,torch.bfloat16)
    record=dict(dataset_sha256=digest(ROOT/"dataset_v1.json"),
        checkpoint_index_sha256=digest(MODEL/"model.safetensors.index.json"),
        tokenizer_sha256=data["tokenizer_sha256"],torch_version=torch.__version__,
        transformers_version=transformers.__version__,load_s=time.monotonic()-start,samples=[])
    for s in data["samples"]:
        if s["split"]!="full": continue
        started=time.monotonic()
        ids=torch.tensor([s["prompt_ids"]],dtype=torch.long)
        r=dict(id=s["id"],kind=s["kind"],language=s["language"])
        with torch.inference_mode():
            if s["kind"]=="nll":
                inputs=torch.tensor([s["prompt_ids"]+s["target_ids"][:-1]])
                logits=model(inputs,use_cache=False).logits[0,63:79].float()
                targets=torch.tensor(s["target_ids"])
                v=logits[torch.arange(16),targets]
                r.update(nll=(torch.logsumexp(logits,dim=-1)-v).tolist(),
                    top1=logits.argmax(-1).tolist(),rank=(1+(logits>v[:,None]).sum(-1)).tolist())
            else:
                gen=model.generate(ids,attention_mask=torch.ones_like(ids),
                    max_new_tokens=16,do_sample=False,use_cache=True,
                    pad_token_id=tok.eos_token_id,temperature=None,top_p=None,top_k=None)
                generated=gen[0,64:].tolist();text=decode(tok,generated)
                r.update(token_ids=generated,text=text)
                if s["kind"]=="task":r["correct"]=grade(s,text)
        r["elapsed_s"]=time.monotonic()-started;record["samples"].append(r)
        print(json.dumps(dict(id=s["id"],kind=s["kind"],elapsed_s=r["elapsed_s"]),ensure_ascii=False),flush=True)
    record["total_s"]=time.monotonic()-start
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")

def parse_device(path,expected_name):
    data=dataset()
    binary=(ROOT/(expected_name+".bin")).read_bytes()
    import struct
    words=struct.unpack("<"+"I"*(len(binary)//4),binary)
    rows=[words[4+i*83:4+(i+1)*83] for i in range(words[2])]
    records=[]
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("{"):records.append(json.loads(line))
    begins=[r for r in records if r.get("record")=="eval_sample_begin"]
    steps=[r for r in records if r.get("record")=="eval_step"]
    complete=[r for r in records if r.get("record")=="eval_suite_complete"]
    assert len(complete)==1 and complete[0]["samples"]==len(rows)
    assert len(begins)==len(rows)
    assert len(steps)==sum(r[2] for r in rows)
    pos=0;result=[]
    for row in rows:
        sample_steps=steps[pos:pos+row[2]];pos+=row[2]
        for step,r in enumerate(sample_steps):
            assert r["sample_id"]==row[0] and r["step"]==step
            assert r["pass"] and r["vtcm_bytes"]==8388608
            assert r["intermediate_read"]==r["intermediate_write"]==r["spill"]==0
            assert r["cache_valid"]==64+step
            if row[1]==1:
                assert r["teacher_forcing"] and r["target_token"]==row[67+step]
                assert r["vocab_count"]==151936 and r["nonfinite"]==0
                assert math.isfinite(r["nll"]) and r["nll"]>=0
                assert 1<=r["rank"]<=151936 and r["target_ties"]>=1 and r["max_ties"]>=1
                if r["token_id"]==r["target_token"]: assert r["rank"]==1
        result.append(dict(id=row[0],steps=sample_steps))
    return dict(raw_sha256=digest(path),suite=expected_name,
        model_ready_ns=next(r["model_ready_ns"] for r in records if r.get("record")=="eval_model_ready"),
        suite_wall_ns=complete[0]["suite_wall_ns"],samples=result)

def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=["teacher","validate"])
    p.add_argument("--raw",type=Path);p.add_argument("--suite",default="full")
    a=p.parse_args()
    if a.command=="teacher":teacher()
    else:
        result=parse_device(a.raw,a.suite)
        a.raw.with_suffix(".validated.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(dict(valid=True,samples=len(result["samples"]),suite_wall_s=result["suite_wall_ns"]/1e9)))
if __name__=="__main__":main()
