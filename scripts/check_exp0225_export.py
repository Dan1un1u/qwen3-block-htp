#!/usr/bin/env python3
"""Run unchanged EXP0224 quantization oracles into EXP0225 evidence."""
import torch
from output_scale_exp0224 import check
from rotation_exp0219 import write_json
from learned_rotation_exp0225 import RESULT
if __name__=='__main__':
    torch.set_grad_enabled(False);torch.set_num_threads(8)
    write_json(RESULT/'output_scale_oracle.json',check())
