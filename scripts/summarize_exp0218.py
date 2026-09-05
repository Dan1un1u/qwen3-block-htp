#!/usr/bin/env python3
"""Archive full diagnostics and produce the user's compact comparison."""
import json
import math
from pathlib import Path
from statistics import mean,median
from transformers import AutoTokenizer
from eval_exp0218 import ROOT,MODEL,dataset,decode,grade,digest
from measure_exp0218 import RECIPES,OVERVIEW,speed_validate
from summarize_exp0217 import normalized

def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+
        ["| "+" | ".join(map(str,row))+" |" for row in rows])+"\n"

def quality():
    data=dataset();tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    samples={s["id"]:s for s in data["samples"]}
    teacher=json.loads((ROOT/"teacher_bf16.json").read_text())
    tref={s["id"]:s for s in teacher["samples"]}
    all_scores={"BF16 teacher":teacher["samples"]}
    diagnostics={}
    for recipe in RECIPES:
        full=json.loads((ROOT/"full"/(recipe+".validated.json")).read_text())
        result=[]
        for r in full["samples"]:
            s=samples[r["id"]];steps=r["steps"]
            entry=dict(id=s["id"],kind=s["kind"],language=s["language"])
            if s["kind"]=="nll":
                entry.update(nll=[x["nll"] for x in steps],top1=[x["token_id"] for x in steps],
                    rank=[x["rank"] for x in steps],max_ties=[x["max_ties"] for x in steps],
                    saturated=[x["saturated"] for x in steps])
            else:
                ids=[x["token_id"] for x in steps];text=decode(tok,ids)
                entry.update(token_ids=ids,text=text)
                if s["kind"]=="task":entry["correct"]=grade(s,text)
            result.append(entry)
        all_scores[recipe]=result
        quick=json.loads((ROOT/"quick"/(recipe+".validated.json")).read_text())
        repeat=json.loads((ROOT/"repeat"/(recipe+".validated.json")).read_text())
        byid={r["id"]:r for r in full["samples"]}
        fields=["token_id","target_code","nll","rank","max_ties"]
        mismatches=[]
        for r in quick["samples"]+repeat["samples"]:
            if samples[r["id"]]["kind"]!="nll":continue
            for i,(a,b) in enumerate(zip(r["steps"],byid[r["id"]]["steps"])):
                different=[k for k in fields if a[k]!=b[k]]
                if different:mismatches.append(dict(id=r["id"],step=i,fields=different))
        diagnostics[recipe]=dict(repeat_equal=repeat["repeat_equal"],
            quick_repeat_vs_full_mismatches=mismatches,
            full_suite_wall_s=full["suite_wall_ns"]/1e9,
            full_model_ready_s=full["model_ready_ns"]/1e9)
    summary={}
    for recipe,entries in all_scores.items():
        nll=[v for e in entries if e["kind"]=="nll" for v in e["nll"]]
        lang={l:mean(v for e in entries if e["kind"]=="nll" and e["language"]==l for v in e["nll"]) for l in ["zh","en"]}
        tasks=[e for e in entries if e["kind"]=="task"]
        top1=[v==w for e in entries if e["kind"]=="nll" for v,w in zip(e["top1"],tref[e["id"]]["top1"])]
        summary[recipe]=dict(nll=mean(nll),ppl=math.exp(mean(nll)),language_nll=lang,
            tokens=len(nll),teacher_top1_agreement=mean(top1),
            tasks_correct=sum(e["correct"] for e in tasks),tasks_total=len(tasks),
            failed_tasks=[dict(id=e["id"],text=e["text"],expected=samples[e["id"]]["answer"]) for e in tasks if not e["correct"]],
            open_outputs=[e for e in entries if e["kind"]=="open"])
        summary[recipe]["delta_nll_teacher"]=summary[recipe]["nll"]-summary["BF16 teacher"]["nll"]
        summary[recipe]["delta_nll_recipe_baseline"]=None
        if recipe in RECIPES:
            summary[recipe]["diagnostics"]=diagnostics[recipe]
            summary[recipe]["mean_target_rank"]=mean(v for e in entries if e["kind"]=="nll" for v in e["rank"])
            summary[recipe]["maximum_tie_fraction"]=mean(v>1 for e in entries if e["kind"]=="nll" for v in e["max_ties"])
            summary[recipe]["saturation_fraction"]=sum(v for e in entries if e["kind"]=="nll" for v in e["saturated"])/(512*151936)
    out=dict(dataset_version=data["version"],dataset_sha256=digest(ROOT/"dataset_v1.json"),
        scope="512 conditional targets, 24 fixed short tasks; no general benchmark claim; 8 holdout windows not executed.",
        initial_snapshot=True,selected_baseline_promoted=False,summary=summary,samples=all_scores)
    (ROOT/"quality_summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    rows=[[r,f'{v["nll"]:.4f}',f'{v["ppl"]:.2f}',f'{v["delta_nll_teacher"]:+.4f}',
        f'{v["language_nll"]["zh"]:.4f} / {v["language_nll"]["en"]:.4f}',
        f'{v["teacher_top1_agreement"]*100:.2f}%',f'{v["tasks_correct"]}/24'] for r,v in summary.items()]
    (ROOT/"quality_table.md").write_text(table(["实现","NLL ↓","条件 PPL ↓","ΔNLL / BF16","中文 / 英文 NLL","Teacher top-1 一致率","短题"],rows))
    return out

def speed():
    data={};timings={}
    for recipe in RECIPES:
        paths=sorted((ROOT/"formal").glob("round_*_"+recipe+".jsonl"));assert len(paths)==10
        runs=[];loops=[]
        for path in paths:
            profiles,final=speed_validate(path,recipe)
            runs.append(profiles);loops.append(final["generation_loop_wall_ns"])
        data[recipe]={m:[normalized([p for p in r if p["mode"]==m]) for r in runs] for m in ["prefill","decode"]}
        pwall=median(d["host_us"] for d in data[recipe]["prefill"])
        dwall=median(d["host_us"]*15 for d in data[recipe]["decode"])
        timings[recipe]=dict(prefill_tokens=64,prefill_host_us=pwall,prefill_tokens_per_second=64e6/pwall,
            decode_tokens=15,decode_host_us=dwall,decode_tokens_per_second=15e6/dwall,
            generation_loop_median_us=median(loops)/1000)
    labels=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE','QK–Softmax–AV','O projection',
        'Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down','Final residual',
        'KV carrier conversion','KV append DMA','Block orchestration','Layer bookkeeping',
        'Stage-boundary bookkeeping','DSP unattributed','Runtime setup/teardown',
        'Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall']
    rows=[]
    for label,(_,fields) in zip(labels,OVERVIEW):
        vals=[median(sum(p[f] if f.endswith("_us") else p[f]/19.2 for f in fields)
            for p in data[r]["prefill"]) for r in RECIPES]
        cells=[f'{v:.1f} ({100*v/timings[r]["prefill_host_us"]:.2f}%)' for r,v in zip(RECIPES,vals)]
        rows.append([label,*cells,f'{100*(vals[1]/vals[2]-1):+.2f}%' if vals[2] else "N/A"])
    (ROOT/"module_table.md").write_text(table(["模块","F16A16","W4A16","W4A8","W4A8 相对 W4A16 增速"],rows))
    out=dict(experiment="EXP-0218",sessions_per_recipe=10,invocation_ledgers=480,
        layer_ledgers=13440,mode="same M64 plus 15 feedback decodes, rotated order",
        times=timings,data=data)
    (ROOT/"speed_summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    report=["# EXP-0218 full profiling archive",
        "Ten rotated sessions per recipe. 480 invocation ledgers and 13440 layer ledgers close exactly. Each call has one complete model RPC, exact 8 MiB VTCM, no hidden/logits DDR or intermediate spill. Final norm is removed from inclusive head time for additive module reporting. Independent medians do not necessarily sum.",
        (ROOT/"module_table.md").read_text()]
    for recipe in RECIPES:
        for mode,series in data[recipe].items():
            report += ["## "+recipe+" "+mode,
                "All numeric exported fields retained; ticks use 19.2 ticks/us. No audit or quality score collection in these sessions.",
                table(["Field","R1","R10 median"],[[k,f"{series[0][k]:.6f}",f"{median(p[k] for p in series):.6f}"] for k in sorted(series[0])])]
    (ROOT/"full_profiling_report.md").write_text("\n\n".join(report)+"\n")
    return out

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--quality-only",action="store_true");a=p.parse_args()
    q=quality();print((ROOT/"quality_table.md").read_text())
    if not a.quality_only:
        s=speed();print(json.dumps(s["times"],indent=2))
