#!/usr/bin/env python

import argparse
import os

import torch
import torch.distributed as dist


def setup_arg_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--checkpoint_dir', type=str,
                        help='checkpoint output path')
    parser.add_argument('--auto_resume', action='store_true',
                        help='auto resume from the last checkpoint from checkpoint_dir')
    parser.add_argument('--resume_path', type=str,
                        help='resume from manually specified checkpoint file, overriding auto_resume')
    parser.add_argument('--pretrain', type=str,
                        help='path to pretrained weights. will NOT override auto_resume of resume_path, '
                             'load optimizer state or enforce strict matching of checkpoint and model weights.')


def _find_autoresume_path(args: argparse.Namespace):
    print('Trying to auto resume from path:', args.checkpoint_dir)

    if not os.path.exists(args.checkpoint_dir):
        print(f'Checkpoint directory {args.checkpoint_dir} does not exist')
        return
    if not os.path.isdir(args.checkpoint_dir):
        print(f'Checkpoint path {args.checkpoint_dir} is not a directory')
        return
        checkpoint_files = [x for x in os.listdir(args.checkpoint_dir) if x.startswith('checkpoint-') and x.endswith('.pth')]
        checkpoint_iters = []
        for x in checkpoint_files:
            try:
                x = x[len('checkpoint-'): -len('.pth')]
                x = int(x)
            except ValueError:
                continue
            checkpoint_iters.append(x)
    else:
        checkpoint_iters = []

    if len(checkpoint_iters) == 0:
        print('Did not find a valid checkpoint file.')
    else:
        checkpoint_iters.sort()
        args.resume_path = os.path.join(args.checkpoint_dir, 'checkpoint-%d.pth' % checkpoint_iters[-1])
        print(f'Found {len(checkpoint_iters)} checkpoint file(s).')


def resume_from_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler._LRScheduler,
    loss_scaler: torch.cuda.amp.grad_scaler.GradScaler,
    args: argparse.Namespace,
) -> int:
    if args.pretrain is not None:
        print(f'Loading pretrain model: {args.pretrain}')
        try:
            if not os.path.exists(args.pretrain):
                raise FileNotFoundError(f'Pretrain file {args.pretrain} not found')
            ckpt = torch.load(args.pretrain, map_location='cpu')
            if 'model' not in ckpt:
                raise ValueError('Invalid checkpoint format: missing model state')
            missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
            if missing_keys:
                print(f'Warning: Missing keys in pretrain model: {missing_keys}')
            if unexpected_keys:
                print(f'Warning: Unexpected keys in pretrain model: {unexpected_keys}')
        except Exception as e:
            print(f'Error loading pretrain model: {str(e)}')
            return 0

    # returns resume_step on successful resume, or 0 otherwise.
    if args.auto_resume and args.resume_path is None:
        _find_autoresume_path(args)
    
    if args.resume_path is None:
        print('Not resuming from a checkpoint.')
        return 0
    else:
        print(f'Resuming from checkpoint file {args.resume_path}')
        try:
            if not os.path.exists(args.resume_path):
                raise FileNotFoundError(f'Resume file {args.resume_path} not found')
            ckpt = torch.load(args.resume_path, map_location='cpu')
            if 'model' not in ckpt:
                raise ValueError('Invalid checkpoint format: missing model state')
            model.load_state_dict(ckpt['model'], strict=True)
        except Exception as e:
            print(f'Error loading checkpoint: {str(e)}')
            return 0
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_sched.load_state_dict(ckpt['lr_sched'])
            loss_scaler.load_state_dict(ckpt['loss_scaler'])
            return ckpt['next_step']
        else:
            print('Optimizer state is NOT found in checkpoint.')
            return 0


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler._LRScheduler,
    loss_scaler: torch.cuda.amp.grad_scaler.GradScaler,
    next_step: int,
    args: argparse.Namespace,
):
    if args.checkpoint_dir is None:
        return

    try:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if not os.access(args.checkpoint_dir, os.W_OK):
            raise PermissionError(f'No write permission for checkpoint directory {args.checkpoint_dir}')
    except Exception as e:
        print(f'Error creating checkpoint directory: {str(e)}')
        return
    
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_sched': lr_sched.state_dict(),
        'loss_scaler': loss_scaler.state_dict(),
        'next_step': next_step,
    }
    torch.save(to_save, os.path.join(args.checkpoint_dir, f'checkpoint-{next_step}.pth'))
