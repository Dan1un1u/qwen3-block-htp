#!/usr/bin/env python3
"""Seal EXP0257 after read-only independent audit and clean source closure."""
import sys,json,subprocess,hashlib
from pathlib import Path
S=Path('/home/daniuniu/work/qwen3-block-htp');sys.path.insert(0,str(S/'scripts'))
import device_exp0257 as d
R=d.R;M=d.M

def read(p):return json.loads(Path(p).read_text())
def write(n,z):
 with (R/n).open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False);f.write('\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=S,text=True).strip()
d.preflight();assert read(R/'summary.json')['speed_gate']=='incomplete_formal_control_nondeterminism';assert read(R/'independent_integrity_checks.json')['pass_all']
rt=read(read(R/'runtime_l28.json')['manifest']);assert not git('diff',rt['source_head'],'HEAD','--','src','include','CMakeLists.txt')
remote=d.adb('shell',f'cd {rt["remote"]} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
for name,h in rt['binaries'].items():assert d.sha(Path(read(R/'runtime_l28.json')['manifest']).parent/name)==h and h in remote
assert d.adb('shell','cat /proc/sys/kernel/random/boot_id').strip()==rt['boot']
pack=read(R/'device_package.json');manifest=d.O/'package/manifest.json';assert d.sha(manifest)==pack['manifest_sha256']==read(R/'export_audit.json')['manifest_sha256'];pm=read(manifest)
for n,v in pm['files'].items():assert d.sha(d.O/'package'/n)==v['sha256'],n
out=d.adb('shell',f'cd {pack["remote"]} && sha256sum -c files.sha256',timeout=600);assert out.count(': OK')==len(pm['files']) and 'FAILED' not in out
seed=read(R/'prefix_device.json');assert d.sha(d.O/'prefix/prefix_kv_u8.bin')==seed['sha256']==read(R/'prefix_export.json')['sha256'];assert seed['sha256'] in d.adb('shell','sha256sum '+seed['remote'])
cmanifest=d.C/'manifest.json';assert d.sha(cmanifest)==read(R/'prefix_export.json')['manifest_sha256']==read(R/'W4A16_prompt_control.json')['manifest_sha256'];cm=read(cmanifest)
for n,v in cm['files'].items():assert d.sha(d.C/n)==v['sha256'],n
files={str(p.relative_to(S)):d.sha(p) for p in [S/'CMakeLists.txt',S/'include/block_protocol.h',S/'src/host/block_main.c',S/'src/dsp/block_imp.c',*sorted((S/'scripts').glob('*exp0257*'))] if p.is_file()}
write('ARTIFACT_PROVENANCE.json',dict(source_head=git('rev-parse','HEAD'),capture_source_head='65915ede3ec216214047a3d78b40ab6552d3a639',binary_manifest=rt,runtime_source_unchanged_since_stage=True,source_files=files,package=pack,package_files_reverified=len(pm['files']),C64_manifest_sha256=d.sha(cmanifest),C64_files_reverified=len(cm['files']),prefix=seed,protocol_sha256=d.sha(M/'docs/experiments/EXP-0257.md'),ABI=118,toolchain='SDK6.6 Tools19.0.07 V79 NDKr26c',models_verified_after_execution=True,remote_binaries_package_seed_verified=True,boot_unchanged=True))
ledger={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=d.sha(p)) for p in sorted(R.rglob('*')) if p.is_file() and p.name!='EVIDENCE_SHA256.json' and '__pycache__' not in p.parts}
write('EVIDENCE_SHA256.json',dict(experiment='EXP-0257',files=ledger))
for n,v in ledger.items():assert d.sha(R/n)==v['sha256']
print(json.dumps(dict(files=len(ledger),ledger_sha256=d.sha(R/'EVIDENCE_SHA256.json'),source_head=git('rev-parse','HEAD'),sealed=True)))
