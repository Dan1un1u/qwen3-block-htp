#!/usr/bin/env python3
"""Independent SDK/HMX-arithmetic all-A8 oracle with dense R3/R4.

Host/GPU simulation only. No real-device result is implied. All activation
carriers are U8; dense rotation has FP32 temporary accumulation prior to U8.
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from llama32_smooth_accuracy import ROOT,OUT,MOD,OLD,ORIG,had,aff,save,log
from llama_reference import PROJECTIONS,sha256,rope
from quantize_llama32 import pack
from llama_u8_reference import HmxU8Converter,exact_rms_norm_u8,exact_qk_norm_rope_u8,exact_residual_add_u8,projection_bias_words,_cached_w4_projection
from llama32_c_integer_oracle import NativeOracle
from export_llama32_u8 import configs,quantize
FRONT=MOD/'native-a01'
def signed(step):return dict(scale=step,zero_point=128,minimum=-128*step,maximum=127*step)
@torch.no_grad()
def prepare():
    FRONT.mkdir(exist_ok=False);state=torch.load(MOD/'gptq.pt',weights_only=True);quant=torch.load(MOD/'quantized.pt',weights_only=True);carriers=json.loads((MOD/'carriers.json').read_text());steps=json.loads((MOD/'steps.json').read_text());qs=[]
    for i in range(16):
        path=FRONT/f'layer{i}';path.mkdir();q={n:carriers[f'{i}.{n}'] for n in ['block_input','block_output','q','k','v','q_rope','k_rope','q_r3','k_r3','attention_concat','o','post_attention_residual','gate','up','down']}
        for short in PROJECTIONS:
            c=quant['codes'][f'{i}.{short}'];s=quant['scales'][f'{i}.{short}'].reshape(-1)
            pack(c).tofile(path/(short+'_weight_w4_hmx.bin'));s.numpy().astype('<f4').tofile(path/(short+'_weight_w4_scale_f32.bin'))
            q[short+'_input']=signed(steps[f'{i}.{short}'])
        q['attention_probability']=dict(scale=1/255,zero_point=0)
        qs.append(q)
    embed=state['model.embed_tokens.weight'].float().numpy();quantize(embed,qs[0]['block_input']).tofile(FRONT/'embedding_u8.bin')
    w=state['lm_head.weight'].float();s=w.abs().amax(1).clamp_min(1e-8)/7;c=(w/s[:,None]).round().clamp(-7,7).to(torch.int8)
    pack(c).tofile(FRONT/'head_weight_w4_hmx.bin');s.numpy().astype('<f4').tofile(FRONT/'head_weight_w4_scale_f32.bin')
    save(FRONT/'qparams.json',dict(layers=qs,final_norm=carriers['final_norm'],logits=carriers['logits']))
    save(FRONT/'manifest.json',dict(experiment='L32-0014',original_model=str(ORIG),source_weights_sha256=sha256(MOD/'quantized.pt'),source_steps_sha256=sha256(MOD/'steps.json'),source_carriers_sha256=sha256(MOD/'carriers.json'),files={str(p.relative_to(FRONT)):sha256(p) for p in FRONT.rglob('*') if p.is_file()}))
    log('NATIVE_PACKAGE_PREPARED')
def dense_fixed(x,n):
    rows=x.reshape(-1,n);out=torch.empty_like(rows)
    for start in range(0,len(rows),64):
        tile=rows[start:start+64].contiguous();count=len(tile)
        if count<64:tile=F.pad(tile,(0,0,0,64-count))
        out[start:start+count]=(tile@had(n))[:count]
    return out.reshape(x.shape)
class Runner:
    def __init__(self):
        m=json.loads((FRONT/'manifest.json').read_text())
        for n,h in m['files'].items():assert sha256(FRONT/n)==h
        self.q=json.loads((FRONT/'qparams.json').read_text());self.embed=np.memmap(FRONT/'embedding_u8.bin',mode='r',dtype='u1',shape=(128256,2048))
        self.conv=HmxU8Converter(ROOT/'build/l32-0003/qbh_hmx_u8_reference.so');self.helper=NativeOracle();self.config=json.loads((ORIG/'config.json').read_text());self.config['head_dim']=64
        self.gamma=np.ones(2048,dtype='f2');self.audit=[]
    def dense_codes(self,c,inq,outq,n):
        x=torch.from_numpy((c.astype('f4')-inq['zero_point'])*np.float32(inq['scale'])).cuda()
        y=dense_fixed(x,n).cpu().numpy()
        return quantize(y,outq)
    def layer(self,x,i,cos,sin,past):
        q=self.q['layers'][i];p=FRONT/f'layer{i}'
        def proj(v,s):return self.helper.project(v,p,s,dict(q=2048,k=512,v=512,o=2048,gate=8192,up=8192,down=2048)[s],v.shape[-1],q[s+'_input'],q[s],self.conv)
        qr=proj(exact_rms_norm_u8(x,q['block_input'],self.gamma,q['q_input']),'q')
        kr=proj(exact_rms_norm_u8(x,q['block_input'],self.gamma,q['k_input']),'k')
        v=proj(exact_rms_norm_u8(x,q['block_input'],self.gamma,q['v_input']),'v')
        qr=exact_qk_norm_rope_u8(qr,32,q['q'],q['q_rope'],None,cos,sin)
        kr=exact_qk_norm_rope_u8(kr,8,q['k'],q['k_rope'],None,cos,sin)
        qr=self.dense_codes(qr,q['q_rope'],q['q_r3'],64).reshape(-1,32,64)
        kr=self.dense_codes(kr,q['k_rope'],q['k_r3'],64).reshape(-1,8,64)
        k=kr.transpose(1,0,2);v=v.reshape(-1,8,64).transpose(1,0,2);count=0
        if past is not None:count=past[0].shape[1];k=np.concatenate((past[0],k),1);v=np.concatenate((past[1],v),1)
        cfg=dict(q,q_rope=q['q_r3'],k_rope=q['k_r3'])
        from export_llama32_u8 import divide
        av,score,prob=self.helper.attention(qr,k,v,count,configs(cfg),self.conv,divide);av=av.reshape(len(x),2048)
        oin=quantize((av.astype('f4')-q['attention_concat']['zero_point'])*q['attention_concat']['scale'],q['o_input'])
        o=proj(oin,'o');res=exact_residual_add_u8(x,q['block_input'],o,q['o'],q['post_attention_residual'])
        g=proj(exact_rms_norm_u8(res,q['post_attention_residual'],self.gamma,q['gate_input']),'gate')
        u=proj(exact_rms_norm_u8(res,q['post_attention_residual'],self.gamma,q['up_input']),'up')
        gf=torch.from_numpy((g.astype('f4')-q['gate']['zero_point'])*np.float32(q['gate']['scale'])).cuda()
        uf=torch.from_numpy((u.astype('f4')-q['up']['zero_point'])*np.float32(q['up']['scale'])).cuda()
        # Fused high precision SwiGLU-to-dense-R4 temporary, then ONE A8 output.
        z=F.silu(gf)*uf;zr=dense_fixed(z,8192);zin=quantize(zr.cpu().numpy(),q['down_input'])
        down=proj(zin,'down');out=exact_residual_add_u8(res,q['post_attention_residual'],down,q['down'],q['block_output'])
        return out,(k,v)
    def forward(self,ids,past=None,only_last=True):
        x=np.array(self.embed[ids],copy=True);offset=0 if past is None else past[0][0].shape[1]
        cos,sin=rope(self.config,torch.arange(offset,offset+len(ids))[None],torch.float16);cos=cos.numpy().reshape(-1,64);sin=sin.numpy().reshape(-1,64);cache=[]
        for i in range(16):x,kv=self.layer(x,i,cos,sin,None if past is None else past[i]);cache.append(kv)
        if only_last:x=x[-1:]
        n=exact_rms_norm_u8(x,self.q['layers'][-1]['block_output'],self.gamma,self.q['final_norm'])
        y=self.helper.project(n,FRONT,'head',128256,2048,self.q['final_norm'],self.q['logits'],self.conv)
        return y,cache
@torch.no_grad()
def evaluate():
    attempt=OUT/'native-a02';attempt.mkdir(exist_ok=False)
    runner=Runner();ds=json.loads((OLD/'dataset.json').read_text());rows=[];started=time.monotonic()
    for sample in ds['samples']:
        # One80-token causal prefill is equivalent to teacher-forced prefix/decode
        # for static arithmetic. Separate cache equivalence is audited below.
        ids=sample['prompt_ids']+sample['target_ids'];codes,_=runner.forward(ids,only_last=False);codes=codes[63:79]
        logits=(torch.from_numpy(codes.astype('f4'))-runner.q['logits']['zero_point'])*runner.q['logits']['scale']
        loss=F.cross_entropy(logits,torch.tensor(sample['target_ids']),reduction='none').tolist()
        rows.append(dict(id=sample['id'],nll=loss,target_codes=[int(codes[j,t]) for j,t in enumerate(sample['target_ids'])]))
        save(attempt/f'native-sample-{sample["id"]:03d}.json',rows[-1])
        if sample['id']==0:
            pc,cache=runner.forward(ids[:64]);errors=[int(np.max(np.abs(pc[0].astype('i4')-codes[0].astype('i4'))))]
            for j in range(15):
                dc,cache=runner.forward(ids[64+j:65+j],cache);errors.append(int(np.max(np.abs(dc[0].astype('i4')-codes[1+j].astype('i4')))))
            save(attempt/'cache-equivalence.json',dict(max_code_errors=errors,scope='first heldout window, all128256 vocabulary logits codes; full80 vs actual M64+15',dense='FP32 fixed64-row dense GEMM; no shape-dependent GEMV',pass_gate=max(errors)==0))
            log('CACHE_EQUIVALENCE',errors);assert max(errors)==0
        if len(rows)%8==0:log('NATIVE_BRIDGE',len(rows),sum(loss)/16)
    vals=[v for row in rows for v in row['nll']];mean=sum(vals)/len(vals)
    save(OUT/'native-bridge.json',dict(ppl=math.exp(mean),nll=mean,targets=len(vals),rows=rows,elapsed_seconds=time.monotonic()-started,scope='native HMX SDK conversion and existing integer softmax/RMS/RoPE/residual oracle; dense FP32 R3/R4 host simulation, NOT device',integer_dot_matrices_audited=len(runner.helper.audited),sp2=False))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ORIG,local_files_only=True);prompt=json.loads((Path('/mnt/d/llm_exp/models/llama32-htp/l32-0013/frontend-a01')/'manifest.json').read_text()) if False else None
    ids=tok.apply_chat_template([dict(role='user',content='What is the capital of France? Answer briefly.')],tokenize=True,add_generation_prompt=True,date_string='14 Sep 2026')
    generated=[];cache=None
    for j in range(24):
        logits,cache=runner.forward(ids if j==0 else [generated[-1]],cache);generated.append(int(logits[0].argmax()))
        if generated[-1] in [128001,128008,128009]:break
    save(OUT/'native-generation.json',dict(ids=generated,text=tok.decode(generated,skip_special_tokens=True),scope='integer oracle, not device'))
    log('NATIVE_PPL',math.exp(mean),'TEXT',tok.decode(generated))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate']);args=p.parse_args();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    globals()[args.stage]()
