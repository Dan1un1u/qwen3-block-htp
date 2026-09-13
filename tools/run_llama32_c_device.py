#!/usr/bin/env python3
"""Sealed C candidate deployment, exact greedy gate and matched device PPL."""
import argparse,json,math,shlex,shutil,subprocess
from pathlib import Path
from run_llama32_layer import ROOT,adb,windows
from run_llama32_frontend import records
from llama_reference import sha256
from llama32_sp2_contract import contract
MODELS=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0013');RESULTS=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0013');FRONT=MODELS/'frontend-a01'
OUT=RESULTS/'device-e2e-a01';REMOTE='/data/local/tmp/llama32-htp/l32-0013/device-e2e-a01'
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def deploy():
    m=json.loads((FRONT/'manifest.json').read_text());assert m['layers']==16 and m['sp2_encoding']==contract()
    for n,h in m['files'].items():assert sha256(FRONT/n)==h['sha256']
    for name in ['device-c-layer0-a01','device-c-layer7-a01','device-c-layer15-a01','device-c-stack3-a01']:
        r=json.loads((RESULTS/name/'result.json').read_text());assert r['pass'],name
        p=json.loads((RESULTS/name/'protocol.json').read_text())
        assert p['experiment']=='L32-0013'
    seal=json.loads((ROOT/'build/llama-build-seal.json').read_text());head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();assert head==seal['source_head']
    for n,h in seal['files'].items():assert sha256(Path(n))==h
    for build in ['android_ReleaseG_aarch64','hexagon_ReleaseG_toolv19_v79']:
        cache=(ROOT/build/'CMakeCache.txt').read_text();assert 'QBH_LLAMA_LAYER_COUNT:STRING=16' in cache and 'QBH_MODEL_LLAMA32:BOOL=ON' in cache
    OUT.mkdir(exist_ok=False);(OUT/'binaries').mkdir();shutil.copy2(ROOT/'build/llama-build-seal.json',OUT/'binaries/build-seal.json')
    assert adb('shell','test ! -e '+REMOTE,check=False).returncode==0;adb('shell','mkdir -p '+REMOTE)
    for name,build in [('qwen3_block_cli','android_ReleaseG_aarch64'),('libqwen3_probe.so','android_ReleaseG_aarch64'),('libqwen3_probe_skel.so','hexagon_ReleaseG_toolv19_v79')]:
        p=ROOT/build/'ship'/name;shutil.copy2(p,OUT/'binaries'/name);adb('push',windows(p),REMOTE+'/'+name)
        assert adb('shell','sha256sum '+REMOTE+'/'+name).stdout.split()[0]==sha256(p)
    adb('push',windows(FRONT),REMOTE+'/package');adb('push',windows(RESULTS/'data/native-eval.bin'),REMOTE+'/eval.bin');adb('shell','chmod 755 '+REMOTE+'/qwen3_block_cli')
    names=list(m['files'])+['manifest.json']
    for start in range(0,len(names),32):
        got=adb('shell','sha256sum '+' '.join(shlex.quote(REMOTE+'/package/'+n) for n in names[start:start+32])).stdout
        for line in got.splitlines():
            h,n=line.split(maxsplit=1);rel=n.removeprefix(REMOTE+'/package/');assert h==sha256(FRONT/rel),rel
    assert adb('shell','sha256sum '+REMOTE+'/eval.bin').stdout.split()[0]==sha256(RESULTS/'data/native-eval.bin')
    env=dict(LD_LIBRARY_PATH=REMOTE,DSP_LIBRARY_PATH=REMOTE,ADSP_LIBRARY_PATH=REMOTE,QBH_VERTICAL_SLICE='1',QBH_REPLAY_SEQUENCE='1',QBH_GENERATION_SEQUENCE='9',QBH_GENERATION_STEPS='16',QBH_SCAN_MODE='prefill',QBH_LOGICAL_M='64',QBH_KV_CACHE_LENGTH='0',QBH_KV_CACHE_CAPACITY='80',QBH_LLAMA_SP2='9',QBH_W4U8_DECODE_COMMON_OP_ROWS='4',QBH_W4U8_DECODE_SWIGLU_ROWS='4',QBH_W4U8_DECODE_SOFTMAX='hvx_tile4',QBH_W4U8_DECODE_PROJECTION_MODE='direct_n',QBH_W4U8_DECODE_DIRECT_N_MASK='63',QBH_W4U8_DECODE_DIRECT_N_GATE_UP_BATCH_N_TILES='32',QBH_W4U8_DECODE_DIRECT_N_GATE_UP_CONTINUOUS='1',QBH_W4U8_DECODE_DIRECT_N_O_GATE_PREFETCH='1',QBH_W4U8_DECODE_DIRECT_N_GATE_UP_SWIGLU_STREAM='1',QBH_W4U8_DECODE_DIRECT_N_QKV_BATCH_N_TILES='16',QBH_W4U8_DECODE_DIRECT_N_DOWN_BATCH_N_TILES='8',QBH_W4U8_DECODE_DIRECT_N_DOWN_SINGLE_DMA='1',QBH_W4U8_DECODE_O_BATCH_N_TILES='16',QBH_W4U8_DECODE_DIRECT_N_O_SINGLE_DMA='1')
    args=['./qwen3_block_cli',REMOTE+'/package','W4U8','1','2','32','rms_rope_softmax','on','off','hvx_fused_post_norm_pool4','serial','control','hvx','w4u8_streaming_persistent_mlp_hvx','3','64','u8_log2_gqa','4','w4u8_mlp_io_qkv_o','serial','hvx_tree','control','4','3','1','0']
    command='cd '+REMOTE+' && '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' '+shlex.join(args)
    save(OUT/'protocol.json',dict(experiment='L32-0013',source_head=head,build_seal=seal,package_manifest_sha256=sha256(FRONT/'manifest.json'),dataset_sha256=sha256(RESULTS/'data/dataset.json'),command=command,evaluation_command=command.replace(' && ',' && QBH_EVAL_QUIET=1 QBH_EVAL_FILE='+REMOTE+'/eval.bin ',1),scope='M64+16 bridge exact integer alignment; not full WT2-2048 historical PPL',nll_atol=5e-5))
    print('C_DEPLOYED',flush=True)
def generate():
    p=json.loads((OUT/'protocol.json').read_text());r=adb('shell',p['command'],check=False)
    (OUT/'generation.stdout.txt').open('x').write(r.stdout);(OUT/'generation.stderr.txt').open('x').write(r.stderr)
    rr=records(r.stdout);steps=[v for v in rr if 'selected_token_id' in v and 'generation_step' in v];profiles=[v for v in rr if v.get('record')=='generation_profile']
    oracle=json.loads((RESULTS/'oracle-a01/teacher.json').read_text());tokens=[v['selected_token_id'] for v in steps]
    exact=r.returncode==0 and len(steps)==16 and tokens==oracle['u8_generated_ids'] and [v['selected_logit_half_bits'] for v in steps]==oracle['u8_selected_codes'] and all(v['pass'] and v['selected_logit_encoding']=='u8_code' for v in steps)
    physical=len(profiles)==16 and all(v['dsp_status']==3 and v['block_invocation_count']==16 and v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608 and all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','hmx_fp16_tile_pair_count','generation_lm_head_expand_ticks','w4u8_qkvo_weight_expand_ticks']) for v in profiles)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin',local_files_only=True)
    speed=None
    if len(steps)==16:speed=dict(prefill_tokens_per_second=64e9/steps[0]['host_wall_ns'],decode_tokens_per_second=15e9/sum(v['host_wall_ns'] for v in steps[1:]),scope='one functional run; no formal profiling/speed gate')
    save(OUT/'generation-result.json',dict(pass_gate=exact and physical,arithmetic=exact,physical=physical,process_exit=r.returncode,steps=steps,profiles=profiles,text=tok.decode(tokens,skip_special_tokens=True),speed=speed))
    assert exact and physical,(r.stdout[-1500:],r.stderr[-1500:]);print('C_E2E_GATE_PASS',flush=True)
def ppl():
    assert json.loads((OUT/'generation-result.json').read_text())['pass_gate']
    p=json.loads((OUT/'protocol.json').read_text());r=adb('shell',p['evaluation_command'],check=False)
    (OUT/'evaluation.stdout.txt').open('x').write(r.stdout);(OUT/'evaluation.stderr.txt').open('x').write(r.stderr)
    rr=records(r.stdout);rows=[v for v in rr if v.get('record')=='eval_step']
    complete=r.returncode==0 and len(rows)==2048 and all(v['pass'] and v['nll'] is not None and math.isfinite(v['nll']) for v in rows)
    save(OUT/'ppl-device.json',dict(complete=complete,process_exit=r.returncode,rows=rows,targets=len(rows),nll=sum(v['nll'] for v in rows)/len(rows) if complete else None,ppl=math.exp(sum(v['nll'] for v in rows)/len(rows)) if complete else None))
    assert complete,(r.stdout[-1500:],r.stderr[-1500:]);print('C_DEVICE_PPL_COMPLETE',flush=True)
def compare():
    software=json.loads((RESULTS/'integer-ppl-a01/result.json').read_text());hardware=json.loads((OUT/'ppl-device.json').read_text());teacher=json.loads((RESULTS/'teacher-a01/bridge.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    assert hardware['complete'] and software['package_manifest_sha256']==p['package_manifest_sha256'] and software['dataset_sha256']==teacher['dataset_sha256']==p['dataset_sha256']
    ref={(v['sample_id'],v['step']):v for v in software['rows']};pairs=[]
    for row in hardware['rows']:
        key=(row['sample_id'],row['step']);assert key in ref;expected=ref.pop(key)
        pairs.append(dict(sample_id=key[0],step=key[1],nll_abs=abs(row['nll']-expected['nll'])))
    assert not ref
    worst=max(v['nll_abs'] for v in pairs);passed=worst<=p['nll_atol']
    save(OUT/'alignment.json',dict(pass_gate=passed,max_token_nll_abs=worst,nll_atol=p['nll_atol'],targets=2048,bf16_teacher_ppl=teacher['ppl'],integer_software_ppl=software['ppl'],device_ppl=hardware['ppl'],quality_ratio_to_teacher=hardware['ppl']/teacher['ppl'],quality_gate_applied=False,historical_17_6424_comparable=False,scope=p['scope'],per_token=pairs))
    assert passed;print('C_DEVICE_SOFTWARE_PPL_ALIGNED',hardware['ppl'],flush=True)
if __name__=='__main__':
    subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['deploy','generate','ppl','compare']);globals()[p.parse_args().stage]()
