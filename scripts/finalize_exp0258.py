#!/usr/bin/env python3
"""Seal EXP0258 completed performance-only diagnosis without promotion."""
import sys,subprocess,json
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from device_exp0258 import *
preflight();s=read(R/'summary.json');assert s['execution_state']=='completed' and not s['numerical_eligible'] and read(R/'independent_integrity_checks.json')['pass_all']
rt=read(read(R/'runtime_l28.json')['manifest']);assert not subprocess.check_output(['git','diff',rt['source_head'],'HEAD','--','src','include','CMakeLists.txt'],cwd=S,text=True)
remote=adb('shell',f'cd {rt["remote"]} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
for n,h in rt['binaries'].items():assert sha(Path(read(R/'runtime_l28.json')['manifest']).parent/n)==h and h in remote
assert adb('shell','cat /proc/sys/kernel/random/boot_id').strip()==rt['boot']
a=read(R/'export_audit.json');m=read(O/'r3/manifest.json');assert sha(O/'r3/manifest.json')==a['manifest_sha256']
for n,v in m['files'].items():assert sha(O/'r3'/n)==v['sha256']
for root in [PARENT_REMOTE,REMOTE+'-package']:
 out=adb('shell',f'cd {root} && sha256sum -c files.sha256',timeout=600);assert out.count(': OK')==906 and 'FAILED' not in out
assert sha(O/'prefix/prefix_kv_u8.bin')==a['seed_sha256'] and a['seed_sha256'] in adb('shell','sha256sum '+REMOTE+'-package/prefix_kv_u8.bin')
old=read(R.parent/'exp0257/prefix_device.json');assert old['sha256'] in adb('shell','sha256sum '+CONTROL_SEED)
files=[S/'include/block_protocol.h',S/'src/host/block_main.c',S/'src/dsp/block_imp.c',S/'src/dsp/dense_r3.inc',S/'src/dsp/hvx_u8_ops.c',S/'CMakeLists.txt',*sorted((S/'scripts').glob('*exp0258*'))]
write(R/'ARTIFACT_PROVENANCE.json',dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),source_files={str(p.relative_to(S)):sha(p) for p in files},runtime=rt,native_source_unchanged_since_stage=True,ABI=120,toolchain='SDK6.6 Tools19.0.07 V79 NDKr26c',package_manifest_sha256=sha(O/'r3/manifest.json'),prefix_sha256=a['seed_sha256'],prompts_sha256=sha(R/'prompts.json'),protocol_sha256=sha(M/'docs/experiments/EXP-0258.md'),binaries_and_all_package_files_reverified=True,boot_unchanged=True,numerical_exception='PC072 only understood R3 ideal-reference gate; failed status retained'))
ledger={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(R.rglob('*')) if p.is_file() and p.name!='EVIDENCE_SHA256.json' and '__pycache__' not in p.parts}
write(R/'EVIDENCE_SHA256.json',dict(experiment='EXP-0258',files=ledger))
for n,v in ledger.items():assert sha(R/n)==v['sha256']
print(json.dumps(dict(sealed=True,files=len(ledger),ledger_sha256=sha(R/'EVIDENCE_SHA256.json'))))
