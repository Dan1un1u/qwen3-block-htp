#!/usr/bin/env python3
"""Llama-only single-layer device gate; immutable attempts, explicit provenance."""
import argparse
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import numpy as np
from llama_reference import sha256

ROOT=Path(__file__).resolve().parents[1]
ADB="/mnt/c/adb/adb.exe"
SERIAL="3B15C8007Z300000"

def adb(*args, check=True):
    r=subprocess.run([ADB,"-s",SERIAL,*args],capture_output=True,text=True)
    if check and r.returncode:
        raise RuntimeError(r.stdout+r.stderr)
    return r

def windows(path):
    return subprocess.check_output(["wslpath","-w",str(path)],text=True).strip()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--scan",action="store_true")
    args=ap.parse_args()
    subprocess.run(["python3","/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py","preflight","--source-worktree",str(ROOT)],check=True)
    manifest=json.loads((args.package/"manifest.json").read_text())
    if manifest["recipe"]=="W4A8":raise ValueError("Use run_llama32_stack.py with a one-layer replay package for native W4A8")
    assert (manifest["experiment"],manifest["recipe"]) in [("L32-0001","W16A16"),("L32-0002","W4A16")]
    for name,record in manifest["files"].items():
        assert sha256(args.package/name)==record["sha256"],name
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    active=re.search(r"^  active_experiment: (L32-[0-9]+)$",Path("/home/daniuniu/work/llama32-htp-project-memory/PROJECT_STATUS.yaml").read_text(),re.M).group(1)
    remote="/data/local/tmp/llama32-htp/"+active.lower()+"/"+args.output.name
    assert adb("shell",f"test ! -e {shlex.quote(remote)}",check=False).returncode==0,"Remote attempt exists"
    adb("shell",f"mkdir -p {shlex.quote(remote)}")
    artifacts={}
    for name,build in [("qwen3_block_cli","android_ReleaseG_aarch64"),("libqwen3_probe.so","android_ReleaseG_aarch64"),("libqwen3_probe_skel.so","hexagon_ReleaseG_toolv19_v79")]:
        src=ROOT/build/"ship"/name
        artifacts[name]={"sha256":sha256(src),"path":str(src)}
        adb("push",windows(src),remote+"/"+name)
    adb("push",windows(args.package),remote+"/package")
    adb("shell",f"chmod 755 {remote}/qwen3_block_cli")
    env={"LD_LIBRARY_PATH":remote,"DSP_LIBRARY_PATH":remote,"ADSP_LIBRARY_PATH":remote,
         "QBH_DUMP_OUTPUT_PATH":remote+"/actual.bin"}
    if args.scan:
        env.update(QBH_SCAN_MODE=manifest["phase"],QBH_LOGICAL_M=str(manifest["logical_rows"]),QBH_KV_CACHE_LENGTH=str(manifest["past_tokens"]),QBH_KV_CACHE_CAPACITY="80",QBH_DUMP_CACHE_DIR=remote,QBH_DUMP_ATTENTION_DIR=remote)
    elif manifest["phase"]!="prefill":raise ValueError("Decode requires --scan")
    argv=["./qwen3_block_cli",remote+"/package",("W4F16" if manifest["recipe"]=="W4A16" else "F16F16"),"1","2","32","hvx","on","on","fused","gate8_interleaved","control","hvx","crouton_native_batch8","4","64","parallel_qk_norm_rope","4","norms","serial","scalar","input_norm_pool_post_norm_pool","4","3","1","0"]
    if manifest["recipe"]=="W4A16":
        argv[4]="4";argv[10]="serial";argv[11]="adaptive_down96_gate4_dma8_cross"
        env.update(QBH_W4F16_GROUP_FENCE="join_only_down",QBH_W4F16_EXPAND_CLAIM_REGIONS="1",QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER="1",QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER="1",QBH_W4F16_GATE_UP_STREAM_GROUP_TILES="4")
    command="cd "+shlex.quote(remote)+" && "+" ".join(k+"="+shlex.quote(v) for k,v in env.items())+" "+shlex.join(argv)
    protocol={"source_head":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"builds":artifacts,"package_manifest_sha256":sha256(args.package/"manifest.json"),"command":command,"thresholds":{"cosine_min":0.99999,"nrmse_max":0.003,"nonfinite_max":0,"vtcm_bytes":8388608,"intermediate_ddr_bytes":0},"scope":"single layer correctness, no throughput claim"}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    run=adb("shell",command,check=False)
    (args.output/"stdout.txt").write_text(run.stdout);(args.output/"stderr.txt").write_text(run.stderr)
    adb("pull",remote+"/actual.bin",windows(args.output/"actual.bin"),check=False)
    if args.scan:
        for kind in ["k", "v"]:
            name=f"actual_kv_cache_{kind}_f16.bin"
            adb("pull",remote+"/"+name,windows(args.output/name),check=False)
    if args.scan:
        for name in ["actual_scan_q_f16.bin","actual_scan_attention_f16.bin","actual_scan_o_projection_f16.bin","actual_post_residual_f16.bin","actual_post_norm_carrier_f16.bin","actual_down_f16.bin"]+[f"actual_middle_carrier_{i}_f16.bin" for i in range(4)]:
            adb("pull",remote+"/"+name,windows(args.output/name),check=False)
    result={"process_exit_code":run.returncode,"records":[]}
    for line in run.stdout.splitlines():
        try:result["records"].append(json.loads(re.sub(r":-?(?:nan|inf)([,}])", r":null\1", line)))
        except json.JSONDecodeError:pass
    if (args.output/"actual.bin").exists():
        n=manifest["logical_rows"]*2048
        actual=np.fromfile(args.output/"actual.bin",dtype="<f2")[:n].astype(np.float64)
        reference=np.fromfile(args.package/("reference_w4f16_block_output_f16.bin" if manifest["recipe"]=="W4A16" else "reference_f16f16_block_output_f16.bin"),dtype="<f2")[:n].astype(np.float64)
        result["output"]={"finite":bool(np.isfinite(actual).all()),"max_abs":float(np.max(np.abs(actual-reference))),"nrmse":float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),"cosine":float(np.dot(actual,reference)/np.linalg.norm(actual)/np.linalg.norm(reference))}
        result["output_gate"]=result["output"]["finite"] and result["output"]["nrmse"]<=0.003 and result["output"]["cosine"]>=0.99999
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"}),flush=True)
    print(run.stdout[-1200:],flush=True)
    if run.returncode or not result.get("output_gate"):raise SystemExit(1)

if __name__=="__main__":main()
