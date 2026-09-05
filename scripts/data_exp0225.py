#!/usr/bin/env python3
"""Freeze bilingual train/validation by source document, before learning."""
import json,re,hashlib
from urllib.parse import urlencode
from pathlib import Path
from transformers import AutoTokenizer
import eval_exp0218 as ev
from gptq_exp0221 import fetch,sha
from rotation_exp0219 import write_json
from learned_rotation_exp0225 import RESULT,PLAN

def grams(ids):return {tuple(ids[i:i+32]) for i in range(len(ids)-31)}
def prepare():
    assert not (RESULT/'learning_data.json').exists()
    tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    calibration=json.loads((RESULT.parent/'exp0221/calibration.json').read_text())
    forbidden=set()
    for s in ev.dataset()['samples']:forbidden.update(grams(s['prompt_ids']+s.get('target_ids',[])))
    for s in calibration['samples']:forbidden.update(grams(s['token_ids']))
    train=[];validation=[];sources=[];rejected=0
    for lang,ds,config in [('en','Salesforce/wikitext','wikitext-2-raw-v1'),('zh','wikimedia/wikipedia','20231101.zh')]:
        cal_rows={r['row_index'] for r in calibration['samples'] if r['language']==lang}
        rows=[];banned_docs=set();doc='preamble';pages=[];enough=False
        for offset in range(0,12000,100):
            url='https://datasets-server.huggingface.co/rows?'+urlencode(dict(dataset=ds,config=config,split='train',offset=offset,length=100))
            raw=fetch(url,RESULT/'learning_raw'/f'{lang}_{offset}.json');payload=json.loads(raw)
            pages.append(dict(url=url,sha256=sha(raw)))
            for item in payload['rows']:
                text=item['row']['text'].strip()
                if lang=='en' and re.match(r'^= [^=].*[^=] =$' ,text):doc=text
                key=doc if lang=='en' else str(item['row'].get('id',item['row_idx']))
                if item['row_idx'] in cal_rows:banned_docs.add(key)
                if 'text' in item.get('truncated_cells',[]):continue
                ids=tok.encode(text,add_special_tokens=False)
                if len(ids)<128:continue
                window=ids[:128]
                if grams(window)&forbidden:
                    banned_docs.add(key);rejected+=1;continue
                rows.append(dict(language=lang,row_index=item['row_idx'],document=key,text_sha256=sha(text.encode()),token_ids=window))
            eligible=[r for r in rows if r['document'] not in banned_docs]
            chosen=eligible[:400];train_docs={r['document'] for r in chosen}
            valid=[r for r in eligible[400:] if r['document'] not in train_docs][:16]
            if len(chosen)==400 and len(valid)==16:enough=True;break
            if not payload['rows']:break
        assert enough,(lang,len(eligible),len(valid))
        # A document is assigned one role. Explicitly reject token duplicates across all roles.
        train.extend(chosen);validation.extend(valid)
        sources.append(dict(language=lang,dataset=ds,config=config,split='train',pages=pages,banned_documents=len(banned_docs)))
        print('DATA_LANGUAGE_READY',lang,len(chosen),len(valid),flush=True)
    train_grams=set().union(*(grams(s['token_ids']) for s in train))
    val_grams=set().union(*(grams(s['token_ids']) for s in validation))
    assert not train_grams&val_grams and not (train_grams|val_grams)&forbidden
    assert not {(s['language'],s['document']) for s in train}&{(s['language'],s['document']) for s in validation}
    write_json(RESULT/'learning_data.json',dict(plan=PLAN,train=train,validation=validation,sources=sources,
        overlap_rejections=rejected,calibration_ids_sha256=calibration['ids_sha256'],
        tokenizer_sha256=ev.digest(ev.MODEL/'qwen3-tokenizer.json'),
        role='training and validation text only; no evaluation labels or scores',
        train_validation_document_disjoint=True,all_roles_32gram_disjoint=True))
    print('DATA_FROZEN',ev.digest(RESULT/'learning_data.json'),flush=True)
if __name__=='__main__':prepare()
