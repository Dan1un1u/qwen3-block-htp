#!/usr/bin/env python3
from device_exp0260 import *
preflight();s=read(R/'summary.json');assert s['implementation_eligible'] and read(R/'independent_integrity_checks.json')['pass_all']
for n in ['layer_numerical_gate','slice_gate']:assert read(R/(n+'.json'))['pass_all']
for n in ['layer_short_gate','layer_formal_gate','full_short_gate','full_formal_gate']:assert read(R/(n+'.json'))['integrity_pass']
runtimes={}
for count in [1,3,28]:
 path=Path(read(R/f'runtime_l{count}.json')['manifest']);rt=read(path);assert not subprocess.check_output(['git','diff',rt['source_head'],'HEAD','--','src','include','CMakeLists.txt'],cwd=S,text=True)
 remote=adb('shell',f'cd {rt["remote"]} && sha256sum qwen3_block_cli libqwen3_probe.so libqwen3_probe_skel.so')
 for n,h in rt['binaries'].items():assert sha(path.parent/n)==h and h in remote
 assert adb('shell','cat /proc/sys/kernel/random/boot_id').strip()==rt['boot'];runtimes[str(count)]=rt
m=read(C/'manifest.json');assert sha(C/'manifest.json')=='7de4f0758d83f2ba3b58696c695bfbfed72a25dd3bf308475abee0a0f0575a89'
for n,v in m['files'].items():assert sha(C/n)==v['sha256']
raw=adb('shell',f'cd {PKG} && sha256sum -c {PARENT}/files.sha256');assert raw.count(': OK')==len(m['files']) and 'FAILED' not in raw
for tag,control in [('layer_control_final','layer_control'),('layer_batch_audit','layer_control'),('layer_vector_exact','layer_control'),('slice_candidate','slice_control'),('slice_repeat','slice_control')]:
 for p in (R/control).glob('*.bin'):assert p.read_bytes()==(R/tag/p.name).read_bytes()
files=[S/'include/block_protocol.h',S/'src/host/block_main.c',S/'src/dsp/block_imp.c',S/'CMakeLists.txt',*sorted((S/'scripts').glob('*exp0260*'))]
write(R/'ARTIFACT_PROVENANCE.json',dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),source_files={str(p.relative_to(S)):sha(p) for p in files},runtimes=runtimes,ABI=122,native_source_unchanged_since_stage=True,toolchain='SDK6.6 Tools19.0.07 V79 NDKr26c',package_manifest_sha256=sha(C/'manifest.json'),package_files=len(m['files']),protocol_sha256=sha(M/'docs/experiments/EXP-0260.md'),binaries_and_packages_reverified=True,boot_unchanged=True,layer_timed_RPCs=2970,fullmodel_timed_RPCs=5280,no_baseline_promotion=True))
ledger={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(R.rglob('*')) if p.is_file() and p.name!='EVIDENCE_SHA256.json' and '__pycache__' not in p.parts};write(R/'EVIDENCE_SHA256.json',dict(experiment='EXP-0260',files=ledger))
for n,v in ledger.items():assert sha(R/n)==v['sha256']
print(json.dumps(dict(sealed=True,files=len(ledger),ledger_sha256=sha(R/'EVIDENCE_SHA256.json'))))
