#!/usr/bin/env python3
"""Independent standard-library ledger, restoration map, and594 raw-token reductions."""
import hashlib,json,math
from pathlib import Path
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0235')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(16*1024*1024):h.update(b)
    return h.hexdigest()
def selected(name,row):
    if name=='overall':return True
    if name in ['en','zh']:return row['cell'].startswith(name+'_')
    if name in ['wiki','news']:return row['cell'].endswith('_'+name)
    return row['cell']==name

def main():
    ledger=json.loads((R/'evidence_sha256.json').read_text());actual={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p.name!='evidence_sha256.json'}
    assert actual==set(ledger),actual^set(ledger)
    for n,h in ledger.items():assert sha(R/n)==h,n
    art=json.loads((R/'artifacts_sha256.json').read_text());root=Path(art['root']);files=art['files']
    assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==set(files)
    for n,e in files.items():assert sha(root/n)==e['sha256'] and (root/n).stat().st_size==e['bytes'],n
    closure=json.loads((R/'closure.json').read_text());assert closure['dataset_freeze_sha256']==sha(R/'dataset_freeze.json')=='2decbae51004dccd6f06d66c103d1664fc3f8de6af079b2575a76e1b18002d58'
    assert closure['artifact_ledger_sha256']==sha(R/'artifacts_sha256.json') and closure['summary_sha256']==sha(R/'summary.json')
    summary=json.loads((R/'summary.json').read_text());m=json.loads((R/'restoration_manifest.json').read_text());spec=json.loads((R/'variants.json').read_text())['development'];a=json.loads((R/'restoration_audits.json').read_text())
    assert len(spec)==61 and len(a['audits'])==65
    for n,shape in m['shapes'].items():assert math.prod(shape)==m['weight_counts'][n]
    for audit in a['audits']:
        names=spec[audit['variant']];assert audit['restored_names']==names
        expected={n:(m['original_state_hashes'][n] if n in names else h) for n,h in m['C64_state_hashes'].items()}
        assert audit['state_hashes']==expected
        assert audit['restored_weight_count']==sum(m['weight_counts'][n] for n in names)
    count=0
    for group,phases in [('development',['development']),('confirmation',['primary','reserve'])]:
        samples={}
        for v in summary[group]['variants']:
            records=[r for phase in phases for r in json.loads((R/f'software/{phase}_{v}.json').read_text())['samples']]
            assert len({r['id'] for r in records})==summary[group]['documents'] and len(records)*16==summary[group]['targets']
            samples[v]=records
        for v,out in summary[group]['variants'].items():
            assert out['restored_weight_count']==sum(m['weight_counts'][n] for n in spec[v])
            for name,g in out['statistics'].items():
                def mean(w):
                    rows=[r for r in samples[w] if selected(name,r)]
                    return math.fsum(x for r in rows for x in r['nll'])/(16*len(rows))
                n=mean(v);base=mean('C64');f=mean('F');assert abs(n-g['nll'])<1e-12 and abs(math.exp(n)-g['ppl'])<1e-10
                assert abs(math.exp(n-f)-g['vs_F16']['ppl_ratio'])<1e-12
                assert abs(math.exp(n-base)-g['vs_C64']['ppl_ratio'])<1e-12
                if base>f:assert abs((base-n)/(base-f)-g['fraction_C64_excess_NLL_removed'])<1e-10
                count+=1
    assert count==594
    sentinel=json.loads((R/'C64_final_sentinel.json').read_text());assert sentinel['samples']==json.loads((R/'software/development_C64.json').read_text())['samples']
    print(json.dumps(dict(pass_all=True,evidence_files=len(ledger),artifact_files=len(files),independent_PPL_reductions=count,all65_state_maps=True,evidence_ledger_sha256=sha(R/'evidence_sha256.json'),artifact_ledger_sha256=sha(R/'artifacts_sha256.json'),closure_sha256=sha(R/'closure.json')),indent=2),flush=True)
if __name__=='__main__':main()
