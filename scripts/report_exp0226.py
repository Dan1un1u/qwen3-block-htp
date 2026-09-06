#!/usr/bin/env python3
"""Paired fixed-step sampling attribution and separately frozen final evaluation."""
import json,math
import numpy as np
from learned_rotation_exp0226 import RESULT,PLAN
from rotation_exp0219 import write_json
from prepare_exp0164_generation_package import sha256_file as sha
from summarize_exp0218 import table

def report():
    data=json.loads((RESULT/'learning_data.json').read_text());actual={};surrogate={}
    for v in ['control_A','step000','old100','step050','step100']:
        actual[v]=json.loads((RESULT/v/'validation.json').read_text());assert actual[v]['dataset_sha256']==sha(RESULT/'learning_data.json')
        if v!='control_A':
            surrogate[v]=json.loads((RESULT/f'surrogate_{v}.json').read_text());assert surrogate[v]['dataset_sha256']==sha(RESULT/'learning_data.json')
    a,b=actual['old100']['samples'],actual['step100']['samples'];assert len(a)==len(b)==64
    assert all((x['language'],x['domain'],x['row_index'])==(y['language'],y['domain'],y['row_index']) for x,y in zip(a,b))
    delta={};rng=np.random.default_rng(226)
    for group in ['all','en','zh','wiki_en','wiki_zh','news_en','news_zh']:
        values=np.array([y['nll']-x['nll'] for x,y in zip(a,b) if group=='all' or group==x['language'] or group==x['domain']+'_'+x['language']])
        ci=np.quantile(values[rng.integers(0,len(values),(10000,len(values)))].mean(1),[.025,.975])
        delta[group]=dict(new_minus_old_nll=float(values.mean()),ppl_change_percent=100*math.expm1(float(values.mean())),samples=len(values),improved_samples=int((values<0).sum()),paired_sample_bootstrap_95pct=[float(x) for x in ci])
    matched={v:dict(actual_nll=actual[v]['nll'],STE_nll=surrogate[v]['nll'],GPTQ_minus_STE=actual[v]['nll']-surrogate[v]['nll']) for v in surrogate}
    out=dict(primary_fixed_step100=delta,quantizer_validation=matched,actual=actual,surrogate=surrogate,plan=PLAN,data_sha256=sha(RESULT/'learning_data.json'),bootstrap_scope='small fixed validation windows, descriptive uncertainty only; no repeated-training uncertainty')
    write_json(RESULT/'sampling_summary.json',out)
    columns=['模型','全部 NLL','英文 NLL','中文 NLL','英文百科','中文百科','英文新闻','中文新闻']
    rows=[[v,f"{r['nll']:.6f}",*[f"{r['language_nll'][l]:.6f}" for l in ['en','zh']],*[f"{r['domain_language_nll'][d+'_'+l]:.6f}" for d in ['wiki','news'] for l in ['en','zh']]] for v,r in actual.items()]
    text=['# EXP0226 sampling ablation results','Training uses the same Wikipedia corpora,100steps,batch8,400x128 tokens/language, seed225,lr1.5cosine, exact Cayley, original BF16 checkpoint, per-channel W4 and frozen GPTQ8192. Only document coverage/body sampling changes. English30->200documents,max46->2windows; Chinese same400documents, random body windows. Samplingseed226 and all64validationwindows frozen before training. Wiki32 reused; independent XL-Sum BBC news32 added. News is expository prose, not a full general-domain or fiction benchmark. Validation never includes qbh scoring/holdout. Other recipes and runtime frozen.',table(columns,rows),
          '## Primary attribution: fixed old100 versus new100',table(['子集','新−旧 NLL','PPL变化','改善片段','描述性 paired bootstrap95% NLL'],[[k,f"{v['new_minus_old_nll']:+.6f}",f"{v['ppl_change_percent']:+.3f}%",f"{v['improved_samples']}/{v['samples']}",str(v['paired_sample_bootstrap_95pct'])] for k,v in delta.items()]),
          '## Same-rotation training surrogate versus actual export',table(['旋转','STE RTN NLL','实际 GPTQ NLL','GPTQ−STE'],[[v,f"{r['STE_nll']:.6f}",f"{r['actual_nll']:.6f}",f"{r['GPTQ_minus_STE']:+.6f}"] for v,r in matched.items()]),
          'STE uses unchanged GPU training FP16 path; actual packages use CPU software FP16 scoring with identical64x127targets. This diagnoses objective mismatch, not bitwise CPU/GPU equivalence. The small validation and one training seed do not establish broad statistical generalization.']
    for title,name in [('Selection before qbh','selection.json'),('Final DSP quality and immutable historical controls','quality_table.md'),('Three-recipe profile, historical other-recipe columns','module_table.md')]:
        text.append('## '+title);p=RESULT/name;text.append(p.read_text() if p.suffix=='.md' else '```json\n'+p.read_text()+'```')
    speed=json.loads((RESULT/'speed_summary.json').read_text());text.append('## Direct E2E\n\n```json\n'+json.dumps(speed['times'],indent=2)+'\n```')
    (RESULT/'REPORT.md').write_text('\n\n'.join(text)+'\n')
if __name__=='__main__':report()
