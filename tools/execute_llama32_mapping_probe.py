#!/usr/bin/env python3
import argparse,hashlib,json,subprocess
from pathlib import Path
a=argparse.ArgumentParser()
a.add_argument("result_root",type=Path);a.add_argument("tag");a.add_argument("count",type=int);a.add_argument("mib",type=int);a.add_argument("window",type=int);a.add_argument("cycles",type=int);a.add_argument("pin_kib",type=int,nargs="?",default=0)
x=a.parse_args();seal=json.loads((x.result_root/"runtime-a02/seal.json").read_text())
adb=["/mnt/c/adb/adb.exe","-s","3B15C8007Z300000"];remote=seal["remote"]
for name,expected in seal["files"].items():
 got=subprocess.check_output(adb+["shell","sha256sum "+remote+"/"+name],text=True).split()[0]
 assert got==expected,(name,got,expected)
out=x.result_root/x.tag;out.mkdir(exist_ok=False)
args=[x.count,x.mib,x.window,x.cycles,x.pin_kib]
cmd="cd "+remote+" && LD_LIBRARY_PATH="+remote+" DSP_LIBRARY_PATH="+remote+" ADSP_LIBRARY_PATH="+remote+" ./llama_mapping_cli "+" ".join(map(str,args))
(out/"command.json").write_text(json.dumps(dict(command=cmd,seal=seal),indent=2))
with (out/"stdout.jsonl").open("w") as f,(out/"stderr.txt").open("w") as e:
 z=subprocess.run(adb+["shell",cmd],stdout=f,stderr=e,timeout=240)
(out/"exit.json").write_text(json.dumps(dict(returncode=z.returncode)))
records=[]
for l in (out/"stdout.jsonl").read_text().splitlines():
 try:records.append(json.loads(l))
 except ValueError:pass
summary=[a for a in records if a.get("record")=="summary"]
print(json.dumps(dict(tag=x.tag,returncode=z.returncode,summary=summary),indent=2))
raise SystemExit(z.returncode)
