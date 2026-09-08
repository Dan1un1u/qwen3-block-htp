#!/usr/bin/env python3
"""PC057: immutable C64 weights, static U8 QDQ diagnostics; never a DSP oracle."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import transformers.models.qwen3.modeling_qwen3 as qwen
import evaluate_exp0230 as canonical

SOURCE = Path(__file__).resolve().parents[1]
MEMORY = SOURCE.parent / 'qwen3-block-htp-project-memory'
RESULT = Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0242')
DATA = RESULT.parent / 'exp0230/dataset.json'
DATA_HASH = '1d908de405d4bcf4c46f0acdf4de0ab57cb45b6da298a6dbe28fd4cdb0dff8b0'
C64_HASH = '7de4f0758d83f2ba3b58696c695bfbfed72a25dd3bf308475abee0a0f0575a89'
CELLS = ['en_wiki', 'zh_wiki', 'en_news', 'zh_news']
FAMILIES = ['norm_input', 'qkv_output', 'projection_output', 'swiglu',
            'qk_kv', 'attention', 'residual', 'head_input']
NBINS = 8192
LOG_MIN, LOG_MAX = -24., 16.
MAG_CENTERS = np.exp2(LOG_MIN + (np.arange(NBINS) + .5) * (LOG_MAX - LOG_MIN) / NBINS)
CENTERS = np.concatenate([-MAG_CENTERS, [0.], MAG_CENTERS])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, obj):
    path = RESULT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def preflight():
    subprocess.run(['python3', str(MEMORY / 'scripts/project_memory.py'), 'preflight',
                    '--source-worktree', str(SOURCE)], check=True)
    assert json.loads(subprocess.check_output(['python3', '-c',
        'import yaml,json; print(json.dumps(yaml.safe_load(open("' + str(MEMORY / 'PROJECT_STATUS.yaml') + '"))["governance"]))'], text=True))['active_experiment'] == 'EXP-0242'


def settings():
    torch.set_num_threads(8)
    torch.manual_seed(242)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True)


def dataset():
    assert sha(DATA) == DATA_HASH
    data = json.loads(DATA.read_text())
    assert data['document_disjoint'] and data['all_roles_32gram_disjoint']
    return data['samples']


def rows_for(split):
    return [x for x in dataset() if x['split'] == split]


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def qparams(lo, hi):
    lo, hi = min(float(lo), 0.), max(float(hi), 0.)
    scale = float(np.float32(max((hi - lo) / 255., 1e-12)))
    zero = int(np.clip(np.floor(-lo / scale + .5), 0, 255))
    return dict(scale=scale, inv_scale=float(np.float32(1. / scale)), zero=zero, requested_lo=lo, requested_hi=hi,
                lo=-zero * scale, hi=(255 - zero) * scale)


def qdq_np(x, p):
    x = np.asarray(x, np.float32)
    return (np.clip(np.floor(x * np.float32(p['inv_scale']) + np.float32(p['zero']) + np.float32(.5)), 0, 255) - np.float32(p['zero'])) * np.float32(p['scale'])


def qdq(x, p):
    return ((torch.floor(x.float() * p['inv_scale'] + p['zero'] + .5).clamp(0, 255)
             - p['zero']) * p['scale']).to(x.dtype)


def scalar_oracle():
    for lo, hi in [(-3., 5.), (0., 1.), (-100., 0.), (0., 0.)]:
        p = qparams(lo, hi)
        values = [-1000., lo, 0., hi, 1000.] + [(i - p['zero'] + .5) * p['scale'] for i in range(256)]
        # Oracle uses the same FP32 operations, including native zero preservation.
        x = np.asarray(values, np.float32)
        out = []
        for v in x:
            z = np.float32(np.float32(v * np.float32(p['inv_scale'])) + np.float32(p['zero']))
            code = max(0, min(255, math.floor(float(np.float32(z + np.float32(.5))))))
            out.append(np.float32(np.float32(code - p['zero']) * np.float32(p['scale'])))
        got = qdq(torch.tensor(x, device='cuda'), p).cpu().numpy()
        assert np.array_equal(got, np.array(out)), (lo, hi, np.max(np.abs(got - out)))
        assert qdq(torch.tensor([0.], device='cuda'), p).item() == 0.
    return dict(scalar_u8_oracle='exact_FP32', zero_preserved=True)


class Stats:
    """Full-tensor histograms/moments; bounded first-document heatmaps, no tail subsampling."""
    def __init__(self, family):
        self.family = family
        self.count = 0
        self.hist = np.zeros(2 * NBINS + 1, np.int64)
        self.lo, self.hi = 0., 0.
        self.energy = 0.
        self.total = 0.
        self.token_absmax = []
        self.token_rms = []
        self.channel_max = None
        self.channel_energy = None
        self.heatmap = None

    def add(self, x, layout='last'):
        z = x.detach().float()
        assert torch.isfinite(z).all(), 'nonfinite reference activation'
        self.lo = min(self.lo, z.min().item())
        self.hi = max(self.hi, z.max().item())
        self.count += z.numel()
        self.energy += z.square().sum(dtype=torch.float64).item()
        self.total += z.sum(dtype=torch.float64).item()
        # Full log histogram, split by sign, including an explicit exact-zero bin.
        logs = torch.log2(z.abs().clamp_min(2. ** LOG_MIN)).clamp(LOG_MIN, LOG_MAX - 1e-5)
        neg = torch.histc(logs[z < 0], bins=NBINS, min=LOG_MIN, max=LOG_MAX).long().cpu().numpy()
        pos = torch.histc(logs[z > 0], bins=NBINS, min=LOG_MIN, max=LOG_MAX).long().cpu().numpy()
        self.hist += np.concatenate([neg, [(z == 0).sum().item()], pos])
        # Flatten heads into channels for Q/K/V and probabilities into key positions.
        if layout == 'heads':
            z = z.transpose(1, 2).reshape(z.shape[0], z.shape[2], -1)
        elif layout == 'prob':
            z = z.transpose(1, 2).reshape(z.shape[0], z.shape[2], -1)
        z = z.reshape(-1, z.shape[-1])
        cm = z.abs().amax(0).cpu().numpy()
        ce = z.square().sum(0, dtype=torch.float64).cpu().numpy()
        self.channel_max = cm if self.channel_max is None else np.maximum(cm, self.channel_max)
        self.channel_energy = ce if self.channel_energy is None else ce + self.channel_energy
        self.token_absmax.extend(z.abs().amax(-1).cpu().tolist())
        self.token_rms.extend(z.square().mean(-1).sqrt().cpu().tolist())
        if self.heatmap is None:
            # Max pool channel chunks to show sparse outlier positions without hiding extremes.
            view = z[:128]
            width = view.shape[-1]
            chunks = min(128, width)
            edges = np.linspace(0, width, chunks + 1, dtype=int)
            self.heatmap = torch.stack([view[:, a:b].abs().amax(-1) for a, b in zip(edges[:-1], edges[1:])], -1).cpu().numpy()

    def record(self):
        assert int(self.hist.sum()) == self.count
        order = np.argsort(np.abs(CENTERS))
        cdf = np.cumsum(self.hist[order]) / self.count
        quantiles = {str(p): float(abs(CENTERS[order[min(np.searchsorted(cdf, p), len(order)-1)]]))
                     for p in [.5, .9, .99, .999, .9999, .99999]}
        return dict(family=self.family, count=self.count, minimum=self.lo, maximum=self.hi,
                    mean=self.total/self.count, rms=math.sqrt(self.energy/self.count),
                    energy=self.energy, abs_quantiles=quantiles,
                    maximum_over_p999=max(-self.lo,self.hi)/max(quantiles['0.999'],1e-12),
                    token_max_median=float(np.median(self.token_absmax)),
                    token_max_p99=float(np.quantile(self.token_absmax,.99)),
                    channel_max_median=float(np.median(self.channel_max)),
                    channel_max_max=float(self.channel_max.max()))


class Instrument:
    def __init__(self, model):
        self.model = model
        self.policy = None
        self.params = {}
        self.families = set()
        self.depth = 28
        self.collect = False
        self.stats = {}
        self.local_errors = None
        self.capture = None
        self.handles = []
        self.original_attention = qwen.eager_attention_forward
        qwen.eager_attention_forward = self.attention
        self.install()

    def apply(self, name, family, x, layout='last', skip_quant=False):
        if self.collect:
            if name not in self.stats:
                self.stats[name] = Stats(family)
            self.stats[name].add(x, layout)
        if self.local_errors is not None and name in self.params:
            # Independent development tensors, unperturbed C64 trajectory.
            e = self.local_errors.setdefault(name, {})
            for method in ['minmax', 'mse', 'percentile']:
                p = self.params[name][method]
                delta = (qdq(x, p).float() - x.float())
                v = e.setdefault(method, dict(sse=0., energy=0., count=0, clipped=0))
                v['sse'] += delta.square().sum(dtype=torch.float64).item()
                v['energy'] += x.float().square().sum(dtype=torch.float64).item()
                v['count'] += x.numel()
                v['clipped'] += ((x < p['lo']) | (x > p['hi'])).sum().item()
        layer = int(name.split('.')[0][1:]) if name.startswith('L') else 28
        if self.policy and family in self.families and not skip_quant and (layer < self.depth or name == 'head_input' and self.depth == 28):
            x = qdq(x, self.params[name][self.policy])
        return x

    def install(self):
        def post(name, family):
            return lambda module, args, output: self.apply(name, family, output)
        def pre(name, family):
            return lambda module, args: (self.apply(name, family, args[0]),) + args[1:]
        for i, layer in enumerate(self.model.model.layers):
            prefix = f'L{i:02d}.'
            self.handles.append(layer.input_layernorm.register_forward_hook(post(prefix+'norm_qkv', 'norm_input')))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(post(prefix+'norm_mlp', 'norm_input')))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(pre(prefix+'residual_mid', 'residual')))
            for short, long in canonical.rot.PROJECTIONS.items():
                family = 'qkv_output' if short in ['q','k','v'] else 'projection_output'
                self.handles.append(layer.get_submodule(long).register_forward_hook(post(prefix+short+'_out', family)))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre(prefix+'swiglu', 'swiglu')))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(pre(prefix+'attn_context', 'attention')))
            def block_post(module, args, output, p=prefix):
                y = self.apply(p+'residual_out', 'residual', output[0])
                if self.capture is not None:
                    self.capture[p+'residual_out'] = y.detach().float().cpu().numpy()
                return (y,) + output[1:]
            self.handles.append(layer.register_forward_hook(block_post))
        self.handles.append(self.model.model.embed_tokens.register_forward_hook(post('L00.embedding_out','residual')))
        self.handles.append(self.model.lm_head.register_forward_pre_hook(pre('head_input','head_input')))

    def attention(self, module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        p = f'L{module.layer_idx:02d}.'
        query = self.apply(p+'q_rope', 'qk_kv', query, 'heads')
        key = self.apply(p+'k_cache', 'qk_kv', key, 'heads')
        # V projection output and cache value are the same numerical boundary.
        skip = self.policy and 'qkv_output' in self.families
        value = self.apply(p+'v_cache', 'qk_kv', value, 'heads', skip_quant=skip)
        keys = qwen.repeat_kv(key, module.num_key_value_groups)
        values = qwen.repeat_kv(value, module.num_key_value_groups)
        weights = torch.matmul(query, keys.transpose(2,3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, :keys.shape[-2]]
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = self.apply(p+'attention_prob', 'attention', weights, 'prob')
        weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
        output = torch.matmul(weights, values)
        return output.transpose(1,2).contiguous(), weights

    def configure(self, policy=None, families=(), depth=28):
        self.policy, self.families, self.depth = policy, set(families), depth

    def close(self):
        for h in self.handles:
            h.remove()
        qwen.eager_attention_forward = self.original_attention


def forward(model, batch, calibration=False):
    ids = [r['token_ids'] if calibration else r['prompt_ids']+r['target_ids'] for r in batch]
    x = torch.tensor(ids, device='cuda')
    if calibration:
        return model(input_ids=x, use_cache=False).logits
    logits = model(input_ids=x[:, :79], use_cache=False).logits[:,63:79].float()
    nll = torch.logsumexp(logits,-1) - logits.gather(-1,x[:,64:80,None]).squeeze(-1)
    return x, logits, nll


def choose_params(stat, rec):
    candidates = []
    for ratio in [1., .9, .8, .7, .6, .5, .4, .3, .2, .1, .05]:
        candidates.append((f'ratio_{ratio}', qparams(stat.lo*ratio, stat.hi*ratio)))
    for quant in ['0.999','0.9999','0.99999']:
        limit = rec['abs_quantiles'][quant]
        candidates.append((f'p{quant}', qparams(max(stat.lo,-limit),min(stat.hi,limit))))
    for name, p in candidates:
        p['histogram_mse'] = float(np.dot(stat.hist, (qdq_np(CENTERS,p)-CENTERS)**2) / stat.count)
        p['candidate'] = name
    best = min(range(len(candidates)),key=lambda i:(candidates[i][1]['histogram_mse'],i))
    return dict(minmax=candidates[0][1], mse=candidates[best][1], percentile=candidates[-2][1])


def calibrate():
    model, mh = canonical.load('C64','cuda'); assert mh == C64_HASH
    before = state_digest(model)
    dev = rows_for('development')[:4]
    _, uninstrumented, _ = forward(model,dev)
    ins = Instrument(model)
    _, instrumented, _ = forward(model,dev)
    assert torch.equal(uninstrumented,instrumented), 'instrumentation changes logits'
    checks = scalar_oracle()
    checks['disabled_instrumentation_logits_exact'] = True
    ins.collect = True
    rows = sum(([r for r in rows_for('calibration') if r['cell']==c][:32] for c in CELLS), [])
    for start in range(0,len(rows),4):
        forward(model,rows[start:start+4],calibration=True)
        print('CALIBRATION',start+4,len(rows),flush=True)
    ins.collect = False
    records = {n:s.record() for n,s in ins.stats.items()}
    params = {n:choose_params(s,records[n]) for n,s in ins.stats.items()}
    arrays = {}
    for n,s in ins.stats.items():
        arrays[n+'/hist'] = s.hist
        arrays[n+'/channel_max'] = s.channel_max
        arrays[n+'/channel_energy'] = s.channel_energy
        arrays[n+'/token_absmax'] = np.array(s.token_absmax,np.float32)
        arrays[n+'/token_rms'] = np.array(s.token_rms,np.float32)
        arrays[n+'/heatmap'] = s.heatmap
    arrays['centers'] = CENTERS
    assert not (RESULT/'activation_arrays.npz').exists()
    np.savez_compressed(RESULT/'activation_arrays.npz',**arrays)
    after = state_digest(model); assert before == after
    write('calibration.json',dict(manifest_sha256=mh,dataset_sha256=DATA_HASH,
          calibration_ids=[r['id'] for r in rows],calibration_tokens=16384,
          sites=records,parameters=params,checks=checks,state_digest=before,
          no_parameter_mutation=True,histogram_bins_per_sign=NBINS,
          histogram_log2_range=[LOG_MIN,LOG_MAX],
          source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
          arrays_sha256=sha(RESULT/'activation_arrays.npz')))
    ins.close()
    print('CALIBRATION_COMPLETE',len(records),flush=True)


def score(model, ins, name, rows, policy=None, families=(), depth=28, phase='development'):
    outpath = RESULT/f'scores/{phase}_{name}.json'
    if outpath.exists():
        old = json.loads(outpath.read_text())
        assert old['parameter_sha256'] == sha(RESULT/'calibration.json')
        assert old['ids'] == [r['id'] for r in rows]
        print('RETAINED',phase,name,old['ppl'],flush=True)
        return old
    ins.configure(policy,families,depth)
    started = time.monotonic()
    x, logits, nll = forward(model,rows[:4])
    _, again, repeated = forward(model,rows[:4])
    assert torch.equal(logits,again) and torch.equal(nll,repeated)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,64:80].reshape(-1),reduction='none').reshape(4,16)
    ce_error = (ce-nll).abs().max().item(); assert ce_error<5e-6
    changed = x[:,:79].clone();changed[:,70:] = 123
    causal = model(input_ids=changed,use_cache=False).logits[:,63:70].float()
    assert torch.equal(logits[:,:7],causal)
    output = []
    for start in range(0,len(rows),4):
        _, logits, nll = forward(model,rows[start:start+4])
        assert torch.isfinite(nll).all(), (phase,name,'nonfinite NLL')
        for r,loss,top in zip(rows[start:start+4],nll.cpu().tolist(),logits.argmax(-1).cpu().tolist()):
            output.append(dict(id=r['id'],cell=r['cell'],nll=loss,top1=top))
    cell_nll = {c:float(np.mean([r['nll'] for r in output if r['cell']==c])) for c in CELLS}
    mean = float(np.mean(list(cell_nll.values())))
    result = dict(name=name,phase=phase,policy=policy,families=list(families),depth=depth,
                  ids=[r['id'] for r in rows],samples=output,cell_nll=cell_nll,mean_nll=mean,ppl=math.exp(mean),
                  parameter_sha256=sha(RESULT/'calibration.json'),dataset_sha256=DATA_HASH,
                  repeat_exact=True,causal_mask_exact=True,independent_CE_max_abs=ce_error,
                  elapsed_s=time.monotonic()-started,role='software_QDQ_not_DSP_or_throughput')
    write(f'scores/{phase}_{name}.json',result)
    print('SCORE',phase,name,'PPL',result['ppl'],'seconds',result['elapsed_s'],flush=True)
    return result


def evaluation(phase):
    cal = json.loads((RESULT/'calibration.json').read_text())
    model,mh = canonical.load('C64','cuda'); assert mh==C64_HASH
    before=state_digest(model); assert before==cal['state_digest']
    ins=Instrument(model);ins.params=cal['parameters']
    rows=rows_for(phase)
    if phase=='development':
        score(model,ins,'C64',rows)
        for family in FAMILIES:
            for policy in ['minmax','mse']:
                score(model,ins,f'{family}_{policy}',rows,policy,[family])
        combined={p:score(model,ins,'all_'+p,rows,p,FAMILIES) for p in ['minmax','mse','percentile']}
        for depth in [1,2,4,8,14,21,28]:
            if depth!=28:
                score(model,ins,f'prefix{depth}_minmax',rows,'minmax',FAMILIES,depth)
        worst=sorted(FAMILIES,key=lambda f:json.loads((RESULT/f'scores/development_{f}_minmax.json').read_text())['mean_nll'],reverse=True)[:2]
        for family in worst:
            score(model,ins,'without_'+family,rows,'minmax',[f for f in FAMILIES if f!=family])
        selected=min(combined,key=lambda p:(round(combined[p]['mean_nll'],6),['minmax','mse','percentile'].index(p)))
        if not (RESULT/'selection.json').exists():
            write('selection.json',dict(selected=selected,rule='lowest_development_equal_cell_NLL',
                  combined={p:dict(ppl=r['ppl'],mean_nll=r['mean_nll']) for p,r in combined.items()},
                  parameter_sha256=sha(RESULT/'calibration.json'),primary_not_used=True,
                  selected_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    else:
        selection=json.loads((RESULT/'selection.json').read_text())
        assert selection['parameter_sha256']==sha(RESULT/'calibration.json')
        score(model,ins,'C64',rows,phase=phase)
        for policy in dict.fromkeys(['minmax',selection['selected']]):
            score(model,ins,'all_'+policy,rows,policy,FAMILIES,phase=phase)
    after=state_digest(model); assert before==after
    write(f'checks/{phase}_weight_integrity.json',dict(before=before,after=after,unchanged=True))
    ins.close();del ins,model;torch.cuda.empty_cache()
    model,mh=canonical.load('F','cuda');ins=Instrument(model)
    score(model,ins,'F',rows,phase=phase)
    ins.close()


def traces():
    cal=json.loads((RESULT/'calibration.json').read_text())
    model,mh=canonical.load('C64','cuda');assert mh==C64_HASH
    ins=Instrument(model);ins.params=cal['parameters']
    rows=sum(([r for r in rows_for('development') if r['cell']==c][:4] for c in CELLS),[])
    ins.local_errors={}
    refs=[]
    for start in range(0,len(rows),4):
        ins.capture={};forward(model,rows[start:start+4]);refs.append(ins.capture)
    write('local_tensor_error.json',dict(ids=[r['id'] for r in rows],sites=ins.local_errors,
          role='independent_development_on_unperturbed_C64_tensors'))
    ins.local_errors=None
    drift={}
    for policy in ['minmax','mse','percentile']:
        ins.configure(policy,FAMILIES);metrics={}
        for batch_index,start in enumerate(range(0,len(rows),4)):
            ins.capture={};forward(model,rows[start:start+4])
            for name,value in ins.capture.items():
                ref=refs[batch_index][name].astype(np.float64);value=value.astype(np.float64)
                v=metrics.setdefault(name,dict(sse=0.,energy=0.,dot=0.,candidate_energy=0.,count=0))
                v['sse']+=float(np.sum((value-ref)**2));v['energy']+=float(np.sum(ref**2))
                v['dot']+=float(np.sum(ref*value));v['candidate_energy']+=float(np.sum(value**2));v['count']+=ref.size
        for v in metrics.values():
            v['relative_rmse']=math.sqrt(v['sse']/max(v['energy'],1e-30))
            v['cosine']=v['dot']/math.sqrt(max(v['energy']*v['candidate_energy'],1e-30))
        drift[policy]=metrics
    write('residual_drift.json',dict(ids=[r['id'] for r in rows],policies=drift))
    assert state_digest(model)==cal['state_digest']
    ins.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['calibrate','development','primary','traces'])
    args=parser.parse_args();preflight();settings();RESULT.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        if args.phase=='calibrate':calibrate()
        elif args.phase=='traces':traces()
        else:evaluation(args.phase)


if __name__=='__main__':
    main()
