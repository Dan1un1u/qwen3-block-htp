#!/usr/bin/env python3
"""Audit the fixed L32-0048 campaign; component costs are not inference timings."""
import json, hashlib, statistics, sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/d/llm_exp/results/llama32-htp/l32-0048')
read=lambda p:json.loads(p.read_text())
seal=read(root/'runtime-a02/seal.json')
for name,digest in seal['files'].items():
    assert hashlib.sha256((root/'runtime-a02'/name).read_bytes()).hexdigest()==digest
plan=read(root/'fixed-final-plan.json')
runs=[]
for name,count,mib,window,cycles,pin_kib in plan:
    d=root/name
    assert read(d/'exit.json')['returncode']==0
    command=read(d/'command.json')
    assert command['seal']==seal
    assert command['command'].endswith(f'./llama_mapping_cli {count} {mib} {window} {cycles} {pin_kib}')
    records=[json.loads(line) for line in (d/'stdout.jsonl').read_text().splitlines() if line.startswith('{')]
    events=[x for x in records if x.get('record')=='mapping']
    summaries=[x for x in records if x.get('record')=='summary']
    retained=[x for x in records if x.get('record')=='retained_checks']
    assert len(summaries)==len(retained)==1
    summary=summaries[0]
    assert summary['pass'] and summary['completed']==len(events)==count*cycles
    for key in ['run_result','dsp_status','dsp_mismatches','host_mismatches','cleanup_errors','dsp_cleanup_errors']:
        assert summary[key]==0,(name,key,summary[key])
    assert summary['count']==count and summary['window']==window and summary['cycles']==cycles
    assert summary['buffer_bytes']==mib*2**20 and summary['pinned_bytes']==pin_kib*1024
    assert summary['resident_bytes']==count*mib*2**20+pin_kib*1024
    assert summary['vtcm_bytes']==8*2**20
    registrations=[x for x in records if 'register_buffer' in x]
    assert len(registrations)==count and {x['register_buffer'] for x in registrations}==set(range(count))
    assert all(x['result']==0 for x in registrations)
    alloc=[x for x in records if 'allocated_buffer' in x]
    assert len(alloc)==count and all(x['bytes']==mib*2**20 for x in alloc)
    assert len({x['fd'] for x in alloc})==count
    for index,event in enumerate(events):
        cycle=index//count
        expected=index%count if cycle%2==0 else count-1-index%count
        assert event['cycle']==cycle and event['buffer']==expected
        assert event['map_result']==event['unmap_result']==event['mismatches']==0
        assert event['bytes']==mib*2**20
        assert event['va']>0 and event['va']+event['bytes']<=2**32
        assert event['checks']==33*512*(1 if cycle==0 else 2)
        assert event['map_ticks']>0 and event['unmap_ticks']>0 and event['dma_ticks']>0
    expected_retained=(len(events)-1)*32 if window==2 else 0
    if pin_kib: expected_retained+=len(events)*96
    assert retained[0]['count']==expected_retained
    # SDK6.6 incs/HAP_perf.h: HAP_perf_get_qtimer_count is 19.2 MHz.
    map_ticks=sum(x['map_ticks'] for x in events)
    unmap_ticks=sum(x['unmap_ticks'] for x in events)
    assert map_ticks+unmap_ticks<summary['dsp_ticks']
    runs.append(dict(name=name,summary=summary,
        map_us=map_ticks/len(events)/19.2,
        unmap_us=unmap_ticks/len(events)/19.2,
        map_unmap_per_sweep_ms=(map_ticks+unmap_ticks)/cycles/19200,
        unique_va_count=len({x['va'] for x in events}),
        dsp_sample_word_checks=sum(x['checks'] for x in events),
        retained_word_checks=retained[0]['count'],
        host_sample_word_checks=count*33*1024,
        host_pin_byte_checks=pin_kib*1024//4096))
groups={}
for prefix in ['single-final','double-pinned-final','fullsize-final']:
    selected=[x for x in runs if x['name'].startswith(prefix)]
    assert len(selected)==5
    costs=[x['map_unmap_per_sweep_ms'] for x in selected]
    groups[prefix]=dict(processes=5,mappings=sum(x['summary']['completed'] for x in selected),
        resident_gib=selected[0]['summary']['resident_bytes']/2**30,
        active_windows=selected[0]['summary']['window'],
        map_us=statistics.mean(x['map_us'] for x in selected),
        unmap_us=statistics.mean(x['unmap_us'] for x in selected),
        map_unmap_per_sweep_ms=statistics.mean(costs),
        per_process_cost_min_ms=min(costs),per_process_cost_max_ms=max(costs),
        mean_allocation_ms=statistics.mean(x['summary']['allocation_ns']/1e6 for x in selected),
        mean_registration_ms=statistics.mean(x['summary']['registration_ns']/1e6 for x in selected),
        mean_probe_host_ms=statistics.mean(x['summary']['host_run_ns']/1e6 for x in selected),
        unique_va_count_per_process=[x['unique_va_count'] for x in selected])
out=dict(experiment='L32-0048',measured_source=seal['source_head'],all_pass=True,
    qtimer_ticks_per_us=19.2,large_mapping_count=sum(x['mappings'] for x in groups.values()),
    control_mapping_count=12,groups=groups,runs=runs,
    inference_measured=False,quality_measured=False,baseline_promoted=False,
    scope='Synthetic fully touched resident buffers; sampled DMA read/write checks, not complete FP16 model execution.')
(root/'SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(groups,indent=2))
