#!/usr/bin/env python3
"""Exact C64 layer0 weights, frozen Q/K parameters, fresh dense-R3 replay inputs."""
from pathlib import Path
import json,hashlib,subprocess,shutil,struct,math
import numpy as np
import torch
from export_exp0149_vertical_slice import write_qparams,build_silu_lut,build_attention_config
S=Path('/home/daniuniu/work/qwen3-block-htp');M=S.parent/'qwen3-block-htp-project-memory'
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0247');P=R.parent/'exp0246'
C=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0230/C64');O=C.parent.parent/'exp0247'
OLD=C.parent.parent/'exp0148/w4u8'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    with (R/n).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
    subprocess.run(['python3',str(M/'scripts/project_memory.py'),'preflight','--source-worktree',str(S)],check=True)
    assert sha(P/'EVIDENCE_SHA256.json')=='de6b2a5bcd10369a9c0f188ae11af115e04f951fb3f970a1b0a0d5914f9e25dc'
    ledger=json.loads((P/'EVIDENCE_SHA256.json').read_text())
    for n,d in ledger['files'].items():assert sha(P/n)==d['sha256'],n
    assert sha(C/'manifest.json')=='7de4f0758d83f2ba3b58696c695bfbfed72a25dd3bf308475abee0a0f0575a89'
    manifest=json.loads((C/'manifest.json').read_text())
    for n,d in manifest['files'].items():assert sha(C/n.replace('\\','/'))==d['sha256'],n
    old=json.loads((OLD/'manifest.json').read_text())
    # Only the legacy standalone file-size catalog is used for unused allocations.
    assert all((OLD/n).stat().st_size==d['bytes'] and sha(OLD/n)==d['sha256'] for n,d in old['files'].items())
    ids=[151645]+json.loads((P/'inputs.json').read_text())['calibration'][0]['token_ids'][:71]
    assert len(ids)==72
    embedding=np.memmap(C/'generation_embedding_weight_f16.bin',dtype='<f2',mode='r',shape=(151936,2048))
    inputs=np.asarray(embedding[ids]).copy();inv=1.0/(1000000.0**(torch.arange(0,128,2,dtype=torch.float32)/128.0))
    phase=torch.arange(72,dtype=torch.float32)[:,None]*inv[None,:];phase=torch.cat([phase,phase],-1)
    cos=phase.cos().half().numpy();sin=phase.sin().half().numpy()
    # Independent double formula verifies every emitted half coefficient.
    oracle=np.arange(72)[:,None]*(1000000.0**(-np.arange(0,128,2)/128.0))[None,:]
    oc=np.tile(np.cos(oracle),(1,2)).astype('<f2');os=np.tile(np.sin(oracle),(1,2)).astype('<f2')
    assert np.max(np.abs(cos.astype(float)-oc.astype(float)))<=2**-10
    assert np.max(np.abs(sin.astype(float)-os.astype(float)))<=2**-10
    mapping=dict(block_input='embedding_out',input_norm='norm_qkv',q_projection='q_out',k_projection='k_out',v='v_out',q_rope='q_rope',k_rope='k_cache',
        attention_probability='attention_prob',attention_concat='attn_context',attention_projection='o_out',post_attention_residual='residual_mid',post_attention_norm='norm_mlp',
        gate='gate_out',up='up_out',middle='swiglu',down='down_out',block_output='residual_out')
    proof={};O.mkdir(exist_ok=False)
    for cell,variant in [('control','A8'),('r3','R3_L0')]:
        d=O/cell;d.mkdir();layer=d/'layer0';layer.mkdir()
        parameters=json.loads((P/f'parameters/{variant}.json').read_text())['parameters']
        assert parameters['L00.v_out']['mse']==parameters['L00.v_cache']['mse']
        q={name:dict(scale=parameters['L00.'+site]['mse']['scale'],zero_point=parameters['L00.'+site]['mse']['zero'],
            minimum=parameters['L00.'+site]['mse']['lo'],maximum=parameters['L00.'+site]['mse']['hi']) for name,site in mapping.items()}
        for name,info in old['files'].items():
            if 'weight' not in name:
                (d/name).write_bytes(bytes(info['bytes']))
        for name in ['q','k','v','o','gate','up','down']:
            for suffix in ['w4_hmx','w4_scale_f32']:
                n=f'{name}_weight_{suffix}.bin';shutil.copy2(C/'layer0'/n,d/n)
        for n in ['input_norm_weight_f16.bin','post_norm_weight_f16.bin','q_norm_weight_f16.bin','k_norm_weight_f16.bin']:
            shutil.copy2(C/'layer0'/n,d/n)
        write_qparams(d/'qparams_u8.bin',q)
        records=struct.Struct('<32sfi2f');serialized={}
        for rec in records.iter_unpack((d/'qparams_u8.bin').read_bytes()):serialized[rec[0].split(b'\0')[0].decode()]=rec[1:]
        assert len(serialized)==17 and all(serialized[n][0]==v['scale'] and serialized[n][1]==v['zero_point'] for n,v in q.items())
        qp=q['block_input'];encoded=np.clip(np.floor(inputs.astype(np.float32)*np.float32(1.0/qp['scale'])+np.float32(qp['zero_point'])+np.float32(.5)),0,255).astype(np.uint8)
        inputs[:64].tofile(d/'block_input_f16.bin');encoded[:64].tofile(d/'reference_w4u8_block_input_u8.bin')
        cos[:64].tofile(d/'rope_cos_f16.bin');sin[:64].tofile(d/'rope_sin_f16.bin')
        for i in range(8):
            z=np.full((64,2048),qp['zero_point'],np.uint8);z[0]=encoded[64+i];z.tofile(d/f'replay_decode_input_{i:02d}_u8.bin')
            for name,array in [('cos',cos),('sin',sin)]:
                z=np.zeros((64,128),'<f2');z[0]=array[64+i];z.tofile(d/f'replay_decode_rope_{name}_{i:02d}_f16.bin')
        # Fixed full-code-range V carrier; identical in both arms, no fitting to device data.
        v=np.zeros((64,8,128),np.uint8);v[32:]=255
        (d/'attention_config_all_groups.bin').write_bytes(build_attention_config(v,q))
        build_silu_lut(q).tofile(d/'silu_up_lut_u16.bin')
        for p in d.iterdir():
            if p.is_file():shutil.copy2(p,layer/p.name)
        for kind,size in [('k',102400),('v',106496)]:
            for prefix,suffix in [('', ''),('reference_','_step00')]:
                (layer/f'{prefix}kv_cache_{kind}_hmx_u8_segmented{suffix}.bin').write_bytes(bytes(size))
        dm=dict(experiment='EXP-0247',actual_layer=0,variant=cell,C64_manifest_sha256=sha(C/'manifest.json'),parameters_sha256=sha(P/f'parameters/{variant}.json'),
            token_ids=ids,cache='empty initial; EOS key computed online exactly once; real eight sequential appends',unused_references='zero allocation placeholders, not compared by experiment harness',
            qparams=q,files={str(p.relative_to(d)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(d.rglob('*')) if p.is_file()})
        (d/'manifest.json').write_text(json.dumps(dm,indent=2)+'\n');proof[cell]=dict(manifest_sha256=sha(d/'manifest.json'),files=len(dm['files']))
    a=json.loads((O/'control/manifest.json').read_text());b=json.loads((O/'r3/manifest.json').read_text());changed=[n for n in a['files'] if a['files'][n]!=b['files'][n]]
    assert set(changed)=={'qparams_u8.bin','layer0/qparams_u8.bin','attention_config_all_groups.bin','layer0/attention_config_all_groups.bin'},changed
    save('export_audit.json',dict(pass_all=True,parent_files_verified=len(ledger['files']),all_C64_files_verified=True,weights_unchanged=True,other_parameters_unchanged=True,changed_files=changed,
        manifests=proof,token_ids=ids,first_calibration_document=json.loads((P/'inputs.json').read_text())['calibration'][0]['id'],protocol_sha256=sha(M/'docs/experiments/EXP-0247.md'),
        dense_matrix='S128 parity signs plus half normalization',normalization_half=float(np.float16(1/math.sqrt(128))),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()))
    print('EXPORT_VERIFIED',proof,flush=True)
if __name__=='__main__':main()
