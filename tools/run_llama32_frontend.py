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
    ap.add_argument("--generation-steps",type=int,default=16,choices=range(1,17))
    args=ap.parse_args()
    subprocess.run(["python3","/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py","preflight","--source-worktree",str(ROOT)],check=True)
    m=json.loads((args.package/"manifest.json").read_text());assert m["layers"]==16
    for f,r in m["files"].items():assert sha256(args.package/f)==r["sha256"],f
    for f,h in json.loads((args.reference/"freeze.json").read_text()).items():assert sha256(args.reference/f)==h,f
    assert sha256(args.reference/"teacher.json")==m["frontend_teacher_sha256"]
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    remote="/data/local/tmp/llama32-htp/"+m["experiment"].lower()+"/"+args.output.name
    assert adb("shell",f"test ! -e {shlex.quote(remote)}",check=False).returncode==0
    adb("shell",f"mkdir -p {shlex.quote(remote)}")
    builds={}
    for name,build in [("qwen3_block_cli","android_ReleaseG_aarch64"),("libqwen3_probe.so","android_ReleaseG_aarch64"),("libqwen3_probe_skel.so","hexagon_ReleaseG_toolv19_v79")]:
        cache=(ROOT/build/"CMakeCache.txt").read_text();assert "QBH_LLAMA_LAYER_COUNT:STRING=16" in cache and "QBH_MODEL_LLAMA32:BOOL=ON" in cache
        src=ROOT/build/"ship"/name;builds[name]=sha256(src);adb("push",windows(src),remote+"/"+name)
    print("DEPLOYING_FRONTEND",flush=True)
    if args.reuse_package_from:
        source=args.reuse_package_from
        if not re.fullmatch(r"/data/local/tmp/llama32-htp/"+re.escape(m["experiment"].lower())+r"/[a-zA-Z0-9_-]+/package",source):
            raise ValueError("Unexpected reuse path")
        parent_text=adb("shell","cat "+shlex.quote(source+"/manifest.json")).stdout
        parent=json.loads(parent_text)
        parent_digest=adb("shell","sha256sum "+shlex.quote(source+"/manifest.json")).stdout.split()[0]
        expected_parent=m.get("parent_package_manifest_sha256",sha256(args.package/"manifest.json"))
        assert parent_digest==expected_parent,"Remote parent manifest mismatch"
        names=[n for n,r in m["files"].items() if n in parent["files"] and parent["files"][n]["sha256"]==r["sha256"]]
        checked={}
        for first in range(0,len(names),32):
            selected=names[first:first+32]
            output=adb("shell","sha256sum "+" ".join(shlex.quote(source+"/"+n) for n in selected)).stdout
            for line in output.splitlines():
                digest,name=line.split(maxsplit=1);checked[name.removeprefix(source+"/")]=digest
        for name in names:assert checked.get(name)==m["files"][name]["sha256"],name
        parents={str(Path(n).parent) for n in m["files"]}
        assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in m["files"])
        adb("shell","mkdir -p "+" ".join(shlex.quote(remote+"/package/"+n) for n in sorted(parents)))
        for first in range(0,len(names),32):
            adb("shell"," && ".join("ln -s "+shlex.quote(source+"/"+n)+" "+shlex.quote(remote+"/package/"+n) for n in names[first:first+32]))
        for name in list(set(m["files"])-set(names))+["manifest.json"]:
            adb("push",windows(args.package/name),remote+"/package/"+name)
    else:
        adb("push",windows(args.package),remote+"/package")
    adb("push",windows(args.reference/"heldout.bin"),remote+"/heldout.bin")
    adb("shell",f"chmod 755 {remote}/qwen3_block_cli")
    env={"LD_LIBRARY_PATH":remote,"DSP_LIBRARY_PATH":remote,"ADSP_LIBRARY_PATH":remote,
         "QBH_VERTICAL_SLICE":"1","QBH_REPLAY_SEQUENCE":"1","QBH_GENERATION_SEQUENCE":("7" if m["recipe"]=="W4A16" else "10"),"QBH_GENERATION_STEPS":str(args.generation_steps),
         "QBH_SCAN_MODE":"prefill","QBH_LOGICAL_M":"64","QBH_KV_CACHE_LENGTH":"0","QBH_KV_CACHE_CAPACITY":"80","QBH_KV_CACHE_LAYOUT":"hmx_native_f16"}
    argv=["./qwen3_block_cli",remote+"/package",("W4F16" if m["recipe"]=="W4A16" else "F16F16"),"1","2","32","hvx","on","off","fused","gate8_interleaved","control","hvx","crouton_native_batch8","4","64","parallel_qk_norm_rope","4","norms","serial","scalar","input_norm_pool_post_norm_pool","4","3","1","0"]
    if m["recipe"]=="W4A16":
        argv[4]="4";argv[10]="serial";argv[11]="adaptive_down96_gate4_dma8_cross"
        env.update(QBH_W4F16_DECODE_OPT="2",QBH_W4F16_GROUP_FENCE="join_only_down",QBH_W4F16_EXPAND_CLAIM_REGIONS="1",QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER="1",QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER="1",QBH_W4F16_GATE_UP_STREAM_GROUP_TILES="4")
    if m['recipe']=='W4A8':
        env.pop('QBH_KV_CACHE_LAYOUT');env.update(QBH_GENERATION_SEQUENCE="9",QBH_W4U8_DECODE_PROJECTION_MODE="direct_n",QBH_W4U8_DECODE_DIRECT_N_MASK="63")
        argv=["./qwen3_block_cli",remote+"/package","W4U8","1","2","32","rms_rope_softmax","on","off","fused","serial","control","hvx","w4u8_streaming_persistent_mlp_hvx","3","64","u8_log2_gqa","4","w4u8_mlp_io_qkv_o","serial","scalar","control","4","3","1","0"]
    command="cd "+shlex.quote(remote)+" && "+" ".join(k+"="+shlex.quote(v) for k,v in env.items())+" "+shlex.join(argv)
    protocol={"experiment":m["experiment"],"source_head":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"builds":builds,"package_manifest_sha256":sha256(args.package/"manifest.json"),"dataset_sha256":sha256(args.reference/"dataset.json"),"reused_package":args.reuse_package_from,"requested_generation_steps":args.generation_steps,"command":command,"timing_scope":"single functional run, not formal profiling; includes embedding/16 layers/norm/head/greedy/FastRPC, excludes loading and external tokenizer"}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2))
    run=adb("shell",command,check=False)
    (args.output/"generation.stdout.txt").write_text(run.stdout);(args.output/"generation.stderr.txt").write_text(run.stderr)
    rec=records(run.stdout);steps=[r for r in rec if isinstance(r,dict) and "selected_token_id" in r and "generation_step" in r]
    tokens=[r["selected_token_id"] for r in steps]
    stop=next((i for i,t in enumerate(tokens) if t in [128001,128008,128009]),len(tokens))
    tok=AutoTokenizer.from_pretrained(m["original"]["original_root"],local_files_only=True)
    result={"generation_process_exit_code":run.returncode,"generation_pass":run.returncode==0 and len(steps)==args.generation_steps and all(s["pass"] for s in steps),"generated_ids":tokens,"text":tok.decode(tokens[:stop],skip_special_tokens=True),"generation_steps":steps}
    if m['recipe']=='W4A8':
        oracle=json.loads((args.reference/'teacher.json').read_text())
        result['arithmetic_token_match']=tokens==oracle['u8_generated_ids'][:args.generation_steps]
        result['arithmetic_code_match']=[s['selected_logit_half_bits'] for s in steps]==oracle['u8_selected_codes'][:args.generation_steps] and all(s['selected_logit_encoding']=='u8_code' for s in steps)
        result['generation_pass']=result['generation_pass'] and result['arithmetic_token_match'] and result['arithmetic_code_match']
    (args.output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="generation_steps"},ensure_ascii=False),flush=True)
    if not result["generation_pass"]:
        print(run.stderr,flush=True);print(run.stdout[-2000:],flush=True);raise SystemExit(1)
    pre=steps[0]["host_wall_ns"];dec=sum(s["host_wall_ns"] for s in steps[1:])
    result["functional_run_speed"]={"prefill_tokens":64,"prefill_host_wall_ns":pre,"prefill_tokens_per_second":64e9/pre,"decode_tokens":len(steps)-1,"decode_host_wall_ns":dec,"decode_tokens_per_second":((len(steps)-1)*1e9/dec if dec else None)}
    evalcmd=command.replace(" && "," && QBH_EVAL_QUIET=1 QBH_EVAL_FILE="+shlex.quote(remote+"/heldout.bin")+" ",1)
    protocol["evaluation_command"]=evalcmd;(args.output/"protocol.json").write_text(json.dumps(protocol,indent=2))
    print("EVALUATING_HELDOUT",flush=True)
    run=adb("shell",evalcmd,check=False)
    (args.output/"evaluation.stdout.txt").write_text(run.stdout);(args.output/"evaluation.stderr.txt").write_text(run.stderr)
    er=records(run.stdout);evalsteps=[r for r in er if isinstance(r,dict) and r.get("record")=="eval_step"]
    ds=json.loads((args.reference/"dataset.json").read_text());ds["samples"]=[s for s in ds["samples"] if s.get("split","heldout")=="heldout"];teacher=json.loads((args.reference/"teacher.json").read_text())
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
            row["pass"]=(None if m["recipe"]=="W4A8" else row["ratio_to_bf16_teacher"] <= (1.05 if lang=="all" else 1.10))
            result["ppl"][lang]=row
    result["quality_gate_applied"]=m["recipe"]!="W4A8"
    result["pass"]=complete and (m["recipe"]=="W4A8" or all(r["pass"] for r in result["ppl"].values()))
    (args.output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"pass":result["pass"],"ppl":result["ppl"],"functional_run_speed":result["functional_run_speed"]}),flush=True)
    if not result["pass"]:
        print(run.stderr,flush=True);print(run.stdout[-2000:],flush=True);raise SystemExit(1)

if __name__=="__main__":main()
