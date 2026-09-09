#!/usr/bin/env python3
"""Reverify and seal EXP0259 exact dense R3 pipeline optimization evidence."""
import sys,subprocess,json
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from device_full_exp0259 import *
adb=prev.adb
preflight()
s=read(R/'summary.json');assert s['execution_state']=='completed' and not s['numerical_eligible']
for n in ['independent_integrity_checks','selected_code_determinism','numerical_gate','slice_gate','confirmation_integrity','layer_independent_integrity','full_packages_verified']:
 assert read(R/(n+'.json'))['pass_all'],n
assert read(R/'confirmation_gate.json')['fullmodel_escalation_allowed']
assert read(R/'selected_code_determinism.json')['steps']==5280
runtimes={}
for count in [1,3,28]:
 path=Path(read(R/f'runtime_l{count}.json')['manifest']);rt=read(path)
 assert not subprocess.check_output(['git','diff',rt['source_head'],'HEAD','--','src','include','CMakeLists.txt'],cwd=S,text=True)
 remote=adb('shell',f'cd {rt["remote"]} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
 for n,h in rt['binaries'].items():assert sha(path.parent/n)==h and h in remote
 assert adb('shell','cat /proc/sys/kernel/random/boot_id').strip()==rt['boot']
 runtimes[str(count)]=rt
assert sha(P/'EVIDENCE_SHA256.json')=='875397fc4185340c7ada2f35aa09c2e3d6f1799e4f0566187fe02d9ca1c3cd79'
seal=read(P/'EVIDENCE_SHA256.json')
def parent_file(n):
 assert sha(P/n)==seal['files'][n]['sha256'],n
 return P/n
for n in ['export_audit.json','prompts.json','smoke_a0/stdout.jsonl','smoke_a1/stdout.jsonl','smoke_a0/validated.json','smoke_a1/validated.json']:parent_file(n)
assert sha(R/'prompts.json')==sha(parent_file('prompts.json'))
for arm,label in [(0,'slice_off'),(1,'slice_stream'),(1,'slice_stream_repeat')]:
 for step in range(9):
  for suffix in ['output','r3']:
   assert (R/f'{label}/step{step:02}_{suffix}.bin').read_bytes()==parent_file(f'slice_a{arm}/step{step:02}_{suffix}.bin').read_bytes()
a=read(P/'export_audit.json');o=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0258')
m=read(o/'r3/manifest.json');assert sha(o/'r3/manifest.json')==a['manifest_sha256']
for n,v in m['files'].items():assert sha(o/'r3'/n)==v['sha256']
for root in [prev.PARENT_REMOTE,prev.REMOTE+'-package']:
 out=adb('shell',f'cd {root} && sha256sum -c files.sha256',timeout=600);assert out.count(': OK')==906 and 'FAILED' not in out
assert sha(o/'prefix/prefix_kv_u8.bin')==a['seed_sha256'] and a['seed_sha256'] in adb('shell','sha256sum '+prev.REMOTE+'-package/prefix_kv_u8.bin')
old=read(R.parent/'exp0257/prefix_device.json');assert old['sha256'] in adb('shell','sha256sum '+prev.CONTROL_SEED)
files=[S/'include/block_protocol.h',S/'include/hvx_u8_ops.h',S/'src/host/block_main.c',S/'src/dsp/block_imp.c',S/'src/dsp/dense_r3.inc',S/'src/dsp/hvx_u8_ops.c',S/'src/dsp/r3_sign_matrix.inc',S/'CMakeLists.txt',*sorted((S/'scripts').glob('*exp0259*'))]
write(R/'ARTIFACT_PROVENANCE.json',dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),source_files={str(p.relative_to(S)):sha(p) for p in files},runtimes=runtimes,native_source_unchanged_since_stage=True,ABI=121,toolchain='SDK6.6 Tools19.0.07 V79 NDKr26c',package_manifest_sha256=sha(o/'r3/manifest.json'),prefix_sha256=a['seed_sha256'],prompts_sha256=sha(R/'prompts.json'),protocol_sha256=sha(M/'docs/experiments/EXP-0259.md'),binaries_and_all_package_files_reverified=True,boot_unchanged=True,layer_timed_RPCs=7920,fullmodel_timed_RPCs=5280,numerical_status='PC073 exact implementation optimization; ideal R3 whole-layer failure unchanged; no PPL acceptance'))
ledger={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(R.rglob('*')) if p.is_file() and p.name!='EVIDENCE_SHA256.json' and '__pycache__' not in p.parts}
write(R/'EVIDENCE_SHA256.json',dict(experiment='EXP-0259',files=ledger))
for n,v in ledger.items():assert sha(R/n)==v['sha256']
print(json.dumps(dict(sealed=True,files=len(ledger),ledger_sha256=sha(R/'EVIDENCE_SHA256.json'))))
