#!/usr/bin/env python3
"""L32-0017 fixed ten paired generation profiles; reuse verified functional deployments."""
import argparse,json,shlex,subprocess
from pathlib import Path
import numpy as np
import run_llama32_sp2_pipeline_e2e as common
from prototype_llama32_sp2 import preflight
from llama_reference import sha256
from run_llama32_layer import ROOT,adb
from report_llama32_pipeline_profile import MODULES
R=Path("/mnt/d/llm_exp/results/llama32-htp/l32-0017")
def save(p,v):
 assert not p.exists(),p
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
def main():
 preflight()
 ap=argparse.ArgumentParser();ap.add_argument("--base",required=True);ap.add_argument("--fp32",required=True);ap.add_argument("--attempt",required=True);a=ap.parse_args()
 out=R/a.attempt;out.mkdir(exist_ok=False);common.OUT=out
 seal=json.loads((ROOT/"build/llama-build-seal.json").read_text());head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip();assert seal["source_head"]==head
 cfg={}
 for arm,name in [("base",a.base),("fp32",a.fp32)]:
  root=R/name;d=json.loads((root/"result.json").read_text());assert d["generation_pass"]
  p=json.loads((root/"protocol.json").read_text());assert p["source_head"]==head
  remote=p["command"].split(" && ")[0].removeprefix("cd ")
  for n,h in p["builds"].items():assert adb("shell","sha256sum "+shlex.quote(remote+"/"+n)).stdout.split()[0]==h
  package=Path("/mnt/d/llm_exp/models/llama32-htp/l32-0016")/("frontend-base-a01" if arm=="base" else "frontend-a02")
  assert sha256(package/"manifest.json")==p["package_manifest_sha256"]
  common.remote_verify(remote+"/package",json.loads((package/"manifest.json").read_text()))
  teacher=R.parent/"l32-0016"/("frontend-base-a01-reference" if arm=="base" else "frontend-a02-reference")/"teacher.json"
  cfg[arm]=dict(command=p["command"],oracle=str(teacher),oracle_sha256=sha256(teacher),manifest_sha256=p["package_manifest_sha256"],builds=p["builds"])
 assert cfg["base"]["builds"]==cfg["fp32"]["builds"]
 save(out/"protocol.json",dict(experiment="L32-0017",source_head=head,build_seal=seal,arms=cfg,pairs=10,order="AB/BA",prompt_tokens=64,decode_tokens=15,bootstrap_seed=17017,bootstrap_samples=20000,timing_scope="complete warm Host wall, embedding/16layers/norm/head/greedy/FastRPC; excludes loading/tokenizer; functional runs excluded"))
 pairs=[]
 for i in range(10):
  pair={}
  for arm in (["base","fp32"] if i%2==0 else ["fp32","base"]):
   pair[arm]=common.execute(arm,f"formal-{i:02d}-{arm}")
   for q in pair[arm]["profiles"]:
    assert q["dense_r3_total_calls"]==q["dense_r4_calls"]==0
    assert q["llama_fp32_residual"]==int(arm=="fp32")
    ticks=[sum(q[k] for k in fields) for _,fields in MODULES];ticks[-1]-=q["generation_final_norm_ticks"]
    assert sum(ticks)==q["invocation_ticks"]
    assert q["host_wall_ns"]/1000>=q["invocation_ticks"]/19.2
  pairs.append(pair);print("PAIR_COMPLETE",i+1,flush=True)
 rng=np.random.default_rng(17017);idx=rng.integers(0,10,size=(20000,10));speed={};modules={}
 for mode,tokens in [("prefill",64),("decode",15)]:
  x=np.array([p["base"][mode+"_ns"] for p in pairs],float);y=np.array([p["fp32"][mode+"_ns"] for p in pairs],float)
  ci=np.quantile(y[idx].mean(1)/x[idx].mean(1),[.025,.975]);ratio=float(y.mean()/x.mean())
  speed[mode]=dict(base_mean_ns=float(x.mean()),fp32_mean_ns=float(y.mean()),base_tokens_per_second=tokens*1e9/float(x.mean()),fp32_tokens_per_second=tokens*1e9/float(y.mean()),latency_ratio=ratio,latency_increase_pct=(ratio-1)*100,ci95=ci.tolist(),gate_pass=bool(ci[1]<=1.10))
  modules[mode]={}
  for arm in ["base","fp32"]:
   groups=[[p[arm]["profiles"][0]] if mode=="prefill" else p[arm]["profiles"][1:] for p in pairs]
   host=float(np.mean([np.mean([q["host_wall_ns"]/1000 for q in g]) for g in groups]));rows=[]
   for name,fields in MODULES:
    us=float(np.mean([np.mean([sum(q[k] for k in fields)-(q["generation_final_norm_ticks"] if fields==["generation_lm_head_ticks"] else 0) for q in g])/19.2 for g in groups]))
    rows.append(dict(module=name,us=us,host_share_percent=100*us/host))
   boundary=host-sum(r["us"] for r in rows)
   rows.extend([dict(module="Host–DSP 边界",us=boundary,host_share_percent=100*boundary/host),dict(module="完整 Host wall",us=host,host_share_percent=100.)]);modules[mode][arm]=rows
 save(out/"summary.json",dict(speed=speed,modules=modules,verified_token_boundaries=320,formal_pairs=10))
 print(json.dumps(speed),flush=True)
if __name__=="__main__":main()
