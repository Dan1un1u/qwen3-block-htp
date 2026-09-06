#!/usr/bin/env python3
"""Independent standard-library audit of complete EXP234 ledgers and raw-token PPL."""
import hashlib,json,math
from pathlib import Path
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0234')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(16*1024*1024):h.update(b)
    return h.hexdigest()
def main():
    ledger=json.loads((R/'evidence_sha256.json').read_text())
    actual={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p.name!='evidence_sha256.json'}
    assert actual==set(ledger),('result coverage',actual^set(ledger))
    for n,h in ledger.items():assert sha(R/n)==h,n
    artifacts=json.loads((R/'artifacts_sha256.json').read_text());root=Path(artifacts['root']);files=artifacts['files']
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert actual==set(files),('artifact coverage',actual^set(files))
    for n,e in files.items():assert (root/n).stat().st_size==e['bytes'] and sha(root/n)==e['sha256'],n
    closure=json.loads((R/'closure.json').read_text())
    assert closure['artifact_ledger_sha256']==sha(R/'artifacts_sha256.json')
    assert closure['data_freeze_sha256']==sha(R/'dataset_freeze.json')=='a69984650c65d3a1b5d0ffa632be4b9e1c4ae6c5651978281aa694f9be01de80'
    summary=closure['quality'];phases=['primary','reserve'] if summary['phase']=='combined' else ['primary']
    rows=[r for r in json.loads((R/'dataset.json').read_text())['samples'] if r['split'] in phases]
    assert len(rows)==summary['documents'] and len(rows)*16==summary['targets']
    def chosen(name,row):
        if name=='overall':return True
        if name in ['en','zh']:return row['cell'].startswith(name+'_')
        if name in ['wiki','news']:return row['cell'].endswith('_'+name)
        return row['cell']==name
    count=0
    for v in ['F','C64','G8','G64']:
        samples={}
        for phase in phases:
            run=json.loads((R/f'software/{phase}_{v}.json').read_text())
            for row in run['samples']:
                assert row['id'] not in samples and len(row['nll'])==16
                assert all(math.isfinite(x) for x in row['nll']);samples[row['id']]=row
        assert set(samples)=={r['id'] for r in rows}
        for name,g in summary['statistics'].items():
            ids=[r['id'] for r in rows if chosen(name,r)]
            nll=math.fsum(x for i in ids for x in samples[i]['nll'])/(16*len(ids))
            assert abs(nll-g['nll'][v])<1e-12 and abs(math.exp(nll)-g['ppl'][v])<1e-10,(name,v)
            count+=1
    assert count==36
    print(json.dumps(dict(pass_all=True,evidence_files=len(ledger),artifact_files=len(files),independent_PPL_reductions=count,evidence_ledger_sha256=sha(R/'evidence_sha256.json'),artifact_ledger_sha256=sha(R/'artifacts_sha256.json'),closure_sha256=sha(R/'closure.json')),indent=2),flush=True)
if __name__=='__main__':main()
