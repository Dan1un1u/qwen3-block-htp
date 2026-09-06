#!/usr/bin/env python3
"""Sampling-only ablation; reuse EXP225 learned model and exact optimizer unchanged."""
import argparse
from pathlib import Path
import learned_rotation_exp0225 as base
RESULT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0226')
OUTPUT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0226')
PLAN=dict(base.PLAN,validation_samples=64,checkpoint_selection='minimum_equal_language_equal_domain_actual_export_validation_NLL_tie_earliest',sampling_seed=226,validation_domains=['wiki','news'],sampling_ablation='en200docs_two_random_nonoverlapping_body_windows_zh_same400docs_one_random_body_window',primary_attribution='fixed_step100_old225_vs_new226_same_validation')
base.RESULT=RESULT;base.OUTPUT=OUTPUT;base.PLAN=PLAN
LearnedModel=base.LearnedModel
fold=base.fold
orthogonality=base.orthogonality
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','smoke']);p.add_argument('--resume',action='store_true');a=p.parse_args();base.train(a.phase=='smoke',a.resume)
