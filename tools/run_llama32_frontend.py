#!/usr/bin/env python3
"""Validate real Llama token-to-token generation, then frozen heldout device NLL."""
import argparse,json,math,re,shlex,subprocess
from pathlib import Path
from transformers import AutoTokenizer
from run_llama32_layer import ROOT,adb,windows
from llama_reference import sha256

def records(text):
    out=[]
    for line in text.splitlines():
        try:out.append(json.loads(re.sub(r":-?(?:nan|inf)([,}])",r":null\1",line)))
        except json.JSONDecodeError:pass
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package",type=Path,required=True)
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--reuse-package-from")
    args=ap.parse_args()
    subprocess.run(["python3","/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py","preflight","--source-worktree",str(ROOT)],check=True)
    m=json.loads((args.package/"manifest.json").read_text());assert m["layers"]==16
    for f,r in m["files"].items():assert sha256(args.package/f)==r["sha256"],f
    for f,h in json.loads((args.reference/"freeze.json").read_text()).items():assert sha256(args.reference/f)==h,f
    assert sha256(args.reference/"teacher.json")==m["frontend_teacher_sha256"]
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    remote="/data/local/tmp/llama32-htp/l32-0001/"+args.output.name
    assert adb("shell",f"test ! -e {shlex.quote(remote)}",check=False).returncode==0
    adb("shell",f"mkdir -p {shlex.quote(remote)}")
    builds={}
    for name,build in [("qwen3_block_cli","android_ReleaseG_aarch64"),("libqwen3_probe.so","android_ReleaseG_aarch64"),("libqwen3_probe_skel.so","hexagon_ReleaseG_toolv19_v79")]:
        cache=(ROOT/build/"CMakeCache.txt").read_text();assert "QBH_LLAMA_LAYER_COUNT:STRING=16" in cache and "QBH_MODEL_LLAMA32:BOOL=ON" in cache
        src=ROOT/build/"ship"/name;builds[name]=sha256(src);adb("push",windows(src),remote+"/"+name)
    print("DEPLOYING_FRONTEND",flush=True)
    if args.reuse_package_from:
        source=args.reuse_package_from
        if not re.fullmatch(r"/data/local/tmp/llama32-htp/l32-0001/[a-zA-Z0-9_-]+/package",source):
            raise ValueError("Unexpected reuse path")
        names=list(m["files"])+["manifest.json"]
        checked={}
        for first in range(0,len(names),32):
            selected=names[first:first+32]
            output=adb("shell","sha256sum "+" ".join(shlex.quote(source+"/"+n) for n in selected)).stdout
            for line in output.splitlines():
                digest,name=line.split(maxsplit=1);checked[name.removeprefix(source+"/")]=digest
        for name in names:
            expected=sha256(args.package/"manifest.json") if name=="manifest.json" else m["files"][name]["sha256"]
            assert checked.get(name)==expected,name
        adb("shell","ln -s "+shlex.quote(source)+" "+shlex.quote(remote+"/package"))
    else:
        adb("push",windows(args.package),remote+"/package")
    adb("push",windows(args.reference/"heldout.bin"),remote+"/heldout.bin")
    adb("shell",f"chmod 755 {remote}/qwen3_block_cli")
    env={"LD_LIBRARY_PATH":remote,"DSP_LIBRARY_PATH":remote,"ADSP_LIBRARY_PATH":remote,
         "QBH_VERTICAL_SLICE":"1","QBH_REPLAY_SEQUENCE":"1","QBH_GENERATION_SEQUENCE":"10","QBH_GENERATION_STEPS":"16",
         "QBH_SCAN_MODE":"prefill","QBH_LOGICAL_M":"64","QBH_KV_CACHE_LENGTH":"0","QBH_KV_CACHE_CAPACITY":"80","QBH_KV_CACHE_LAYOUT":"hmx_native_f16"}
    argv=["./qwen3_block_cli",remote+"/package","F16F16","1","2","32","hvx","on","off","fused","gate8_interleaved","control","hvx","crouton_native_batch8","4","64","parallel_qk_norm_rope","4","norms","serial","scalar","input_norm_pool_post_norm_pool","4","3","1","0"]
    command="cd "+shlex.quote(remote)+" && "+" ".join(k+"="+shlex.quote(v) for k,v in env.items())+" "+shlex.join(argv)
    protocol={"experiment":"L32-0001","source_head":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"builds":builds,"package_manifest_sha256":sha256(args.package/"manifest.json"),"dataset_sha256":sha256(args.reference/"dataset.json"),"reused_package":args.reuse_package_from,"command":command,"timing_scope":"single functional run, not formal profiling; includes embedding/16 layers/norm/head/greedy/FastRPC, excludes loading and external tokenizer"}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2))
    run=adb("shell",command,check=False)
    (args.output/"generation.stdout.txt").write_text(run.stdout);(args.output/"generation.stderr.txt").write_text(run.stderr)
    rec=records(run.stdout);steps=[r for r in rec if isinstance(r,dict) and "selected_token_id" in r and "generation_step" in r]
    tokens=[r["selected_token_id"] for r in steps]
    stop=next((i for i,t in enumerate(tokens) if t in [128001,128008,128009]),len(tokens))
    tok=AutoTokenizer.from_pretrained(m["original"]["original_root"],local_files_only=True)
    result={"generation_process_exit_code":run.returncode,"generation_pass":run.returncode==0 and len(steps)==16 and all(s["pass"] for s in steps),"generated_ids":tokens,"text":tok.decode(tokens[:stop],skip_special_tokens=True),"generation_steps":steps}
    (args.output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="generation_steps"},ensure_ascii=False),flush=True)
    if not result["generation_pass"]:
        print(run.stderr,flush=True);print(run.stdout[-2000:],flush=True);raise SystemExit(1)
    pre=steps[0]["host_wall_ns"];dec=sum(s["host_wall_ns"] for s in steps[1:])
    result["functional_run_speed"]={"prefill_tokens":64,"prefill_host_wall_ns":pre,"prefill_tokens_per_second":64e9/pre,"decode_tokens":15,"decode_host_wall_ns":dec,"decode_tokens_per_second":15e9/dec}
    evalcmd=command.replace(" && "," && QBH_EVAL_QUIET=1 QBH_EVAL_FILE="+shlex.quote(remote+"/heldout.bin")+" ",1)
    protocol["evaluation_command"]=evalcmd;(args.output/"protocol.json").write_text(json.dumps(protocol,indent=2))
    print("EVALUATING_HELDOUT",flush=True)
    run=adb("shell",evalcmd,check=False)
    (args.output/"evaluation.stdout.txt").write_text(run.stdout);(args.output/"evaluation.stderr.txt").write_text(run.stderr)
    er=records(run.stdout);evalsteps=[r for r in er if isinstance(r,dict) and r.get("record")=="eval_step"]
    ds=json.loads((args.reference/"dataset.json").read_text());teacher=json.loads((args.reference/"teacher.json").read_text())
    result["eval_process_exit_code"]=run.returncode
    complete=run.returncode==0 and len(evalsteps)==len(ds["samples"])*16 and all(s["pass"] and s["nll"] is not None and math.isfinite(s["nll"]) for s in evalsteps)
    result["evaluation_complete"]=complete;result["ppl"]={}
    if complete:
        for lang in ["all","en","zh"]:
            chosen={s["id"] for s in ds["samples"] if lang=="all" or s["language"]==lang}
            nll=[s["nll"] for s in evalsteps if s["sample_id"] in chosen]
            row={"tokens":len(nll),"device":math.exp(sum(nll)/len(nll))}
            for dtype,key in [("torch.float16","fp16_teacher"),("torch.bfloat16","bf16_teacher")]:
                vals=[v for s in teacher["nll"][dtype] if s["id"] in chosen for v in s["nll"]]
                row[key]=math.exp(sum(vals)/len(vals))
                row["ratio_to_"+key]=row["device"]/row[key]
            row["pass"]=row["ratio_to_bf16_teacher"] <= (1.05 if lang=="all" else 1.10)
            result["ppl"][lang]=row
    result["pass"]=complete and all(r["pass"] for r in result["ppl"].values())
    (args.output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"pass":result["pass"],"ppl":result["ppl"],"functional_run_speed":result["functional_run_speed"]}),flush=True)
    if not result["pass"]:
        print(run.stderr,flush=True);print(run.stdout[-2000:],flush=True);raise SystemExit(1)

if __name__=="__main__":main()
