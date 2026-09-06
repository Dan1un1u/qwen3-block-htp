#!/usr/bin/env python3
"""Reproduce the earlier implementation-equivalence inspection in an archived stage."""
import ast,json
from data_exp0234 import RESULT,SOURCE,sha,write,preflight,verified

def main():
    preflight();s=SOURCE/'scripts';record=json.loads((RESULT/'implementation_equivalence.json').read_text())
    a=(s/'group_exp0231.py').read_text();b=(s/'group_exp0234.py').read_text()
    assert a==b.replace('from data_exp0234 import','from data_exp0231 import')
    def loop(n):
        t=ast.parse((s/n).read_text());f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='prepare')
        node=next(x for x in f.body if isinstance(x,ast.For) and isinstance(x.target,ast.Tuple) and [a.id for a in x.target.elts]==['i','layer'])
        return ast.dump(node,include_attributes=False)
    a=loop('export_exp0231.py');b=loop('export_exp0234.py');assert a==b
    import hashlib
    assert hashlib.sha256(a.encode()).hexdigest()==record['layer_loop_AST_sha256']
    old=json.loads(verified('exp0231','G128/package.json').read_text())
    for n,h in record['core_source_identical_to_G8_package'].items():assert sha(s/n)==h==old['quantizer_source'][n]
    for n,h in record['source_files'].items():assert sha(s/n)==h
    assert record['group_source_identical_except_result_import'] and record['CPU_calibration_and_quantization_layer_loop_AST_exact']
    write('implementation_equivalence_recheck.json',dict(pass_all=True,original_inspection_sha256=sha(RESULT/'implementation_equivalence.json'),purpose='Reproduce prior inspection using retained source and stage command; original evidence unchanged'))
    print('IMPLEMENTATION_EQUIVALENCE_ARCHIVED_RECHECK_PASS',flush=True)
if __name__=='__main__':main()
