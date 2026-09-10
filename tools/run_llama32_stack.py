#!/usr/bin/env python3
"""Run one continuous Llama FP16 slice with retained provenance and replay gates."""
import argparse,json,re,shlex,subprocess
from pathlib import Path
from run_llama32_layer import ROOT,adb,windows
from llama_reference import sha256

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--a8-audit",action="store_true",help="Audit scalar softmax equivalence and poison unused decode rows")
    args=ap.parse_args()
    subprocess.run(["python3","/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py","preflight","--source-worktree",str(ROOT)],check=True)
    m=json.loads((args.package/"manifest.json").read_text())
    assert (m["experiment"],m["recipe"]) in [("L32-0001","W16A16"),("L32-0002","W4A16"),("L32-0003","W4A8")]
    for f,r in m["files"].items():assert sha256(args.package/f)==r["sha256"],f
    for build in ["android_ReleaseG_aarch64","hexagon_ReleaseG_toolv19_v79"]:
        cache=(ROOT/build/"CMakeCache.txt").read_text()
        assert re.search(r"QBH_LLAMA_LAYER_COUNT:[^=]+="+str(m["layers"])+r"\n",cache)
        assert "QBH_MODEL_LLAMA32:BOOL=ON" in cache
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    active=re.search(r"^  active_experiment: (L32-[0-9]+)$",Path("/home/daniuniu/work/llama32-htp-project-memory/PROJECT_STATUS.yaml").read_text(),re.M).group(1)
    remote="/data/local/tmp/llama32-htp/"+active.lower()+"/"+args.output.name
    assert adb("shell",f"test ! -e {shlex.quote(remote)}",check=False).returncode==0
    adb("shell",f"mkdir -p {shlex.quote(remote)}")
    builds={}
    for name,build in [("qwen3_block_cli","android_ReleaseG_aarch64"),("libqwen3_probe.so","android_ReleaseG_aarch64"),("libqwen3_probe_skel.so","hexagon_ReleaseG_toolv19_v79")]:
        src=ROOT/build/"ship"/name;builds[name]=sha256(src)
        adb("push",windows(src),remote+"/"+name)
    adb("push",windows(args.package),remote+"/package")
    adb("shell",f"chmod 755 {remote}/qwen3_block_cli")
    env={"LD_LIBRARY_PATH":remote,"DSP_LIBRARY_PATH":remote,"ADSP_LIBRARY_PATH":remote,
         "QBH_VERTICAL_SLICE":"1","QBH_REPLAY_SEQUENCE":"1","QBH_REPLAY_DECODE_STEPS":"1",
         "QBH_SCAN_MODE":"prefill","QBH_LOGICAL_M":"64","QBH_KV_CACHE_LENGTH":"0","QBH_KV_CACHE_CAPACITY":"80","QBH_REPLAY_DUMP_DIR":remote}
    argv=["./qwen3_block_cli",remote+"/package",("W4F16" if m["recipe"]=="W4A16" else "F16F16"),"1","2","32","hvx","on","off","fused","gate8_interleaved","control","hvx","crouton_native_batch8","4","64","parallel_qk_norm_rope","4","norms","serial","scalar","input_norm_pool_post_norm_pool","4","3","1","0"]
    if m["recipe"]=="W4A16":
        argv[4]="4";argv[10]="serial";argv[11]="adaptive_down96_gate4_dma8_cross"
        env.update(QBH_W4F16_DECODE_OPT="2",QBH_W4F16_GROUP_FENCE="join_only_down",QBH_W4F16_EXPAND_CLAIM_REGIONS="1",QBH_W4F16_GATE_UP_EXTRA_EXPAND_WORKER="1",QBH_W4F16_GATE_UP_EXTRA_STREAM_WORKER="1",QBH_W4F16_GATE_UP_STREAM_GROUP_TILES="4")
    if m['recipe']=='W4A8':
        argv=["./qwen3_block_cli",remote+"/package","W4U8","1","2","32","rms_rope_softmax","on","off","fused","serial","control","hvx","w4u8_streaming_persistent_mlp_hvx","3","64","u8_log2_gqa","4","w4u8_mlp_io_qkv_o","serial","scalar","control","4","3","1","0"]
        env.update(QBH_W4U8_DECODE_COMMON_OP_ROWS="4",QBH_W4U8_DECODE_SWIGLU_ROWS="4",QBH_W4U8_DECODE_SOFTMAX="hvx_tile4",QBH_W4U8_DECODE_PROJECTION_MODE="direct_n",QBH_W4U8_DECODE_DIRECT_N_MASK="63")
    if m["recipe"]=="W4A8":
        argv[20]="hvx_tree";argv[9]="hvx_fused_post_norm_pool4"
    if args.a8_audit:
        assert m["recipe"]=="W4A8"
        argv[8]="on"
        env.update(QBH_W4U8_DECODE_COMMON_PADDING_POISON="1",QBH_W4U8_DECODE_SWIGLU_PADDING_POISON="1")
    command="cd "+shlex.quote(remote)+" && "+" ".join(k+"="+shlex.quote(v) for k,v in env.items())+" "+shlex.join(argv)
    (args.output/"protocol.json").write_text(json.dumps({"experiment":m["experiment"],"layers":m["layers"],"source_head":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"builds":builds,"package_manifest_sha256":sha256(args.package/"manifest.json"),"command":command,"gate":("exact integer output and KV replay" if m["recipe"]=="W4A8" else "existing composition_v2 FP16 replay; no relaxation")},indent=2))
    r=adb("shell",command,check=False)
    (args.output/"stdout.txt").write_text(r.stdout);(args.output/"stderr.txt").write_text(r.stderr)
    records=[]
    for line in r.stdout.splitlines():
        try:records.append(json.loads(re.sub(r":-?(?:nan|inf)([,}])",r":null\1",line)))
        except json.JSONDecodeError:pass
    steps=[x for x in records if isinstance(x,dict) and "output_nrmse" in x and "pass" in x]
    passed=r.returncode==0 and len(steps)==2 and all(x["pass"] for x in steps)
    if args.a8_audit:
        profiles=[x for x in records if x.get("record")=="replay_profile" and x.get("mode")=="decode"]
        passed=passed and len(profiles)==1 and all(x["w4u8_decode_softmax_hvx_tile4_call_count"]>0 and x["w4u8_decode_softmax_hvx_tile4_mismatch_count"]==0 and x["w4u8_common_padding_poison_count"]>0 and x["w4u8_decode_swiglu_padding_poison_count"]>0 for x in profiles)
    for step in range(2):
        name=f"actual_replay_output_{step:02d}_{'u8' if m['recipe']=='W4A8' else 'f16'}.bin"
        adb("pull",remote+"/"+name,windows(args.output/name),check=False)
    (args.output/"result.json").write_text(json.dumps({"process_exit_code":r.returncode,"pass":passed,"steps":steps,"records":records},indent=2)+"\n")
    print(json.dumps({"process_exit_code":r.returncode,"pass":passed,"steps":steps}),flush=True)
    if not passed:
        print(r.stderr,flush=True);print(r.stdout[-1500:],flush=True);raise SystemExit(1)

if __name__=="__main__":main()
