#!/usr/bin/env python3
"""Run one quality suite or a rotated speed gate, retaining every raw result."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import time
from eval_exp0218 import ROOT,MODEL,parse_device
from summarize_exp0217 import normalized

SOURCE=Path(__file__).resolve().parents[1]
RECIPES=["f16f16","w4f16","w4u8"]
TOKENIZER=None
# Reuse the previously audited additive ledger, not prior measurements.
tree=ast.parse((SOURCE/"scripts/summarize_exp0217.py").read_text())
LEDGER=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=="ledger" for t in n.targets))
OVERVIEW=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=="overview" for t in n.targets))

def speed_validate(path,recipe):
    records=[json.loads(l) for l in path.read_text().splitlines() if l.startswith("{")]
    profiles=[r for r in records if r.get("record")=="generation_profile"]
    steps=[r for r in records if "generation_step" in r and r.get("record")!="generation_profile"]
    final=[r for r in records if r.get("generation_sequence_complete")]
    expected=json.loads((ROOT/"regression_expected.json").read_text())[recipe]
    assert len(profiles)==len(steps)==16 and len(final)==1
    assert final[0]["all_steps_pass"] and final[0]["token_ids"]==expected
    layer_fields=["metadata_stage_ticks","input_stage_ticks","input_norm_ticks","qkv_projection_ticks",
        "qk_norm_rope_ticks","attention_ticks","o_projection_ticks","post_attention_residual_ticks",
        "post_attention_norm_ticks","gate_up_ticks","activation_ticks","down_ticks","final_residual_ticks",
        "cache_append_pack_ticks","cache_append_dma_ticks","block_orchestration_ticks","layer_bookkeeping_ticks",
        "layer_unattributed_ticks"]
    for step,(p,s) in enumerate(zip(profiles,steps)):
        assert s["pass"] and s["selected_token_id"]==expected[step]
        assert p["variant"].lower()==recipe and p["generation_step"]==step
        assert p["backend"]=="standalone_fastrpc_dsp" and p["qnn"]=="none"
        assert p["vtcm_requested_bytes"]==p["vtcm_acquired_bytes"]==8388608
        assert p["repeat_count"]==1 and p["block_invocation_count"]==28
        assert p["boundary_ddr_write_bytes"]==p["intermediate_ddr_read_bytes"]==p["intermediate_ddr_write_bytes"]==p["intermediate_spill_fill_count"]==0
        n=normalized([p])
        assert sum(n[k] for _,k in LEDGER)==p["invocation_ticks"],(path,step,"ledger")
        for i in range(28):
            lp=p[f"slice_layer_{i}"]
            assert lp["status"]==3 and lp["layer_index"]==i
            assert sum(lp[k] for k in layer_fields)==lp["layer_ticks"]
            assert lp["layer_unattributed_ticks"]==lp["hidden_ddr_read_bytes"]==lp["hidden_ddr_write_bytes"]==0
            assert lp["cache_valid_before"]==(0 if step==0 else 63+step)
            assert lp["cache_valid_after"]==64+step
    return profiles,final[0]

def run(recipe,suite,path):
    assert not path.exists(),f"Refusing to overwrite {path}"
    path.parent.mkdir(parents=True,exist_ok=True)
    frontend={}
    if not suite:
        global TOKENIZER
        if TOKENIZER is None:
            from transformers import AutoTokenizer
            t=time.monotonic_ns()
            TOKENIZER=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
            frontend["tokenizer_initial_load_ns"]=time.monotonic_ns()-t
        reference=json.loads((ROOT.parent/"exp0217/semantic_reference.json").read_text())
        t=time.monotonic_ns()
        rendered=TOKENIZER.apply_chat_template([dict(role="user",content=reference["prompt"])],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=TOKENIZER.encode(rendered,add_special_tokens=False)
        frontend["tokenize_template_ns"]=time.monotonic_ns()-t
        assert ids==reference["prompt_token_ids"]
    started=time.monotonic()
    cmd=["bash",str(SOURCE/"scripts/run_exp0218.sh"),recipe]+([suite] if suite else [])
    selected=[];detokenize_ns=0
    with path.open("w") as f:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in p.stdout:
            f.write(line)
            if not suite and line.startswith("{"):
                record=json.loads(line)
                if "generation_step" in record and record.get("record")!="generation_profile":
                    selected.append(record["selected_token_id"])
                    t=time.monotonic_ns()
                    text=TOKENIZER.decode(selected,skip_special_tokens=False)
                    detokenize_ns+=time.monotonic_ns()-t
                    if len(selected)==1:
                        frontend["launch_to_first_text_s"]=time.monotonic()-started
                        frontend["first_text"]=text
        p.wait()
    elapsed=time.monotonic()-started
    if not suite:
        frontend["incremental_detokenize_ns"]=detokenize_ns
        frontend["text"]=TOKENIZER.decode(selected,skip_special_tokens=False)
        frontend["timing_scope"]="Tokenizer and incremental detokenizer on WSL CPU; launch-to-text includes ADB transport and device cold staging. Device startup phases in raw generation_startup."
    metadata=dict(command=cmd,returncode=p.returncode,elapsed_s=elapsed,frontend=frontend)
    path.with_suffix(".execution.json").write_text(json.dumps(metadata,indent=2)+"\n")
    assert p.returncode==0,(path,p.returncode)
    return metadata

def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["quick","full","repeat","warmup","short","formal"])
    p.add_argument("--recipe",choices=RECIPES)
    a=p.parse_args();recipes=[a.recipe] if a.recipe else RECIPES
    if a.phase in ["quick","full","repeat"]:
        for recipe in recipes:
            path=ROOT/a.phase/(recipe+".jsonl")
            meta=run(recipe,a.phase,path)
            d=parse_device(path,a.phase)
            if a.phase=="repeat":
                fields=["token_id","target_token","target_code","nll","rank","target_ties","max_ties","saturated"]
                repeat_equal=all(all(all(x[k]==y[k] for k in fields) for x,y in zip(
                    d["samples"][i]["steps"],d["samples"][i+2]["steps"])) for i in [0,1])
                d["repeat_equal"]=repeat_equal
                # A failed diagnostic is retained, never retried to obtain a passing score.
            path.with_suffix(".validated.json").write_text(json.dumps(d,indent=2)+"\n")
            print(json.dumps(dict(recipe=recipe,phase=a.phase,**meta,
                repeat_equal=d.get("repeat_equal"))),flush=True)
    else:
        count=dict(warmup=1,short=5,formal=10)[a.phase]
        for i in range(count):
            order=recipes[i%len(recipes):]+recipes[:i%len(recipes)]
            for recipe in order:
                path=ROOT/a.phase/f"round_{i+1:02d}_{recipe}.jsonl"
                meta=run(recipe,None,path)
                speed_validate(path,recipe)
                print(json.dumps(dict(recipe=recipe,phase=a.phase,round=i+1,**meta)),flush=True)

if __name__=="__main__":main()
