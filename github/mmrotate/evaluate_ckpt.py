#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
  python evaluate_ckpt_fix.py /path/to/config.py /path/to/epoch_24.pth --out results.pkl

This script:
 - loads the config
 - forces cfg.data.test = cfg.data.val (so you evaluate on the same val set used in training logs)
 - builds dataset + dataloader
 - builds model (init_detector) and loads checkpoint
 - runs single_gpu_test (no show_progress arg)
 - calls dataset.evaluate(...) and prints results with 4 decimal places
"""
import os
import argparse
import mmcv
from mmcv import Config
import torch

# mmdet/mmrotate apis
from mmdet.apis import init_detector, single_gpu_test
from mmrotate.datasets import build_dataloader, build_dataset
from mmcv.runner import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('checkpoint', help='path to checkpoint (epoch_24.pth)')
    parser.add_argument('--out', help='output file to save raw results (pkl)', default=None)
    parser.add_argument('--device', help='cuda device, e.g. cuda:0', default='cuda:0')
    parser.add_argument('--eval-metric', help='metric to pass to evaluate (default mAP)', default='mAP')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='iou threshold (if needed)')
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # --- Important: make test the same as val used in training logs ---
    # If your training log used cfg.data.val (trainval.txt), we set cfg.data.test = cfg.data.val
    if 'val' in cfg.data:
        cfg.data.test = cfg.data.val
        mmcv.print_log('Set cfg.data.test = cfg.data.val to match validation used during training', 'mmcv')
    else:
        mmcv.print_log('WARNING: cfg has no data.val. Please ensure cfg.data.test points to the correct dataset', 'mmcv')

    # adjust some cfg options to ensure deterministic test
    cfg.device = args.device
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        dist=False,
        shuffle=False)

    # build model and load checkpoint
    model = init_detector(cfg, args.checkpoint, device=args.device)

    # run inference (single GPU)
    mmcv.print_log('Start inference with single_gpu_test (no show_progress arg).', 'mmcv')
    outputs = single_gpu_test(model, data_loader)  # <- no show_progress param

    # optionally save raw outputs
    if args.out:
        mmcv.dump(outputs, args.out)
        mmcv.print_log(f'Raw outputs saved to {args.out}', 'mmcv')

    # Now evaluate using dataset.evaluate (this calls your HRSCDataset.evaluate)
    mmcv.print_log('Calling dataset.evaluate(...)', 'mmcv')
    # If your dataset.evaluate returns (eval_results) or (eval_results, other),
    # handle both cases. For HRSCDataset.evaluate in your code it returns (eval_results).
    eval_ret = dataset.evaluate(outputs, metric=args.eval_metric, iou_thr=args.iou_thr)

    # dataset.evaluate in your custom HRSCDataset returns eval_results (dict) or (dict, ...)
    # Normalize to a dict
    if isinstance(eval_ret, tuple):
        eval_results = eval_ret[0]
    else:
        eval_results = eval_ret

    # Print formatted to 4 decimal places
    print('\n=== Evaluation (formatted to 4 decimals) ===')
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f'{k}: {v:.4f}')
        else:
            print(f'{k}: {v}')
    print('===========================================')

if __name__ == '__main__':
    main()
