#!/usr/bin/env python3
"""Independent CPU package expansion versus selected GPU training parameters."""
import argparse
from block_reconstruction_exp0227 import *


@torch.no_grad()
def check(v):
    preflight();rows=[]
    for i in range(28):
        selection=json.loads((RESULT/'training'/v/f'layer{i:02d}_selection.json').read_text())
        state=torch.load(selection['selected_checkpoint'],weights_only=True)
        for n,theta in state.items():
            scales=np.fromfile(OUTPUT/v/f'layer{i}/{n}_weight_w4_scale_f32.bin',dtype='<f4')
            k=(OUTPUT/v/f'layer{i}/{n}_weight_w4_hmx.bin').stat().st_size*2//len(scales)
            q,s0=read_codes(CONTROLS[v]/f'layer{i}',n,(len(scales),k))
            m=ScaleLinear(q,s0).cuda();m.theta.copy_(theta.cuda())
            assert np.array_equal(m.scale().cpu().numpy(),scales)
            actual=(q.numpy().astype(np.float32)*scales[:,None]).astype(np.float16)
            assert np.array_equal(m.weight().cpu().numpy(),actual)
            rows.append(dict(layer=i,projection=n,GPU_scale_exact=True,GPU_training_weight_CPU_packed_weight_exact=True))
    rot.write_json(RESULT/v/'GPU_export_oracle.json',dict(passed=True,checks=rows,manifest_sha256=digest(OUTPUT/v/'manifest.json')))
    print('GPU_EXPORT_ORACLE_PASS',v,len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['A','R']);a=p.parse_args();check(a.variant)
