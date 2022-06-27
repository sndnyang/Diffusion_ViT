import os
import sys
import time
import copy
import shutil
import random
import logging as log
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision import utils as tv_utils
import numpy as np

from ExpUtils import *
from diffusion import GaussianDiffusion
from utils.sampler import RASampler, cycle
from utils.losses import LabelSmoothingCrossEntropy
from utils.eval_acc import validate
from utils.training_functions import accuracy
import argparse
from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload
from utils.utils import ema, num_to_groups, sample_ema
from utils.losses import ce_loss
from models.create_model import create_model
import warnings
warnings.filterwarnings("ignore", category=Warning)

MODELS = ['difvit', 'vit', 'swin', 'pit', 'cait', 't2t']


def init_parser():
    arg_parser = argparse.ArgumentParser(description='quick training script')

    # Data args
    arg_parser.add_argument('--data_path', default='../../data', type=str, help='dataset path')
    arg_parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'T-IMNET', 'svhn', 'stl10', 'img12810', 'img128', 'imgnet', 'celeba', 'img10'],
                            type=str, help='Image Net dataset path')
    arg_parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    arg_parser.add_argument('--print-freq', default=500, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    arg_parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    arg_parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
    arg_parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

    arg_parser.add_argument('--wd', default=0.1, type=float, help='weight decay (default: 0.1)')

    arg_parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2'])
    arg_parser.add_argument('--model', type=str, default='difvit', choices=MODELS)
    arg_parser.add_argument('--disable_cos', action='store_true', help='disable cosine lr schedule')
    arg_parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')

    arg_parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    
    arg_parser.add_argument('--resume', default=False, help='Version')

    arg_parser.add_argument('--pyx', default=1.0, type=float, help='classifier')
    arg_parser.add_argument('--px', default=1.0, type=float, help='diffusion')

    arg_parser.add_argument('--channel', type=int, help='disable cuda')
    arg_parser.add_argument('--heads', type=int, help='disable cuda')
    arg_parser.add_argument('--depth', type=int, help='disable cuda')
    arg_parser.add_argument('--dim', default=384, choices=[192, 384, 512, 786], type=int, help='Image Net dataset path')

    arg_parser.add_argument('--ls', action='store_false', help='label smoothing')
    arg_parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    arg_parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')
    arg_parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),
    
    arg_parser.add_argument('--cm', action='store_false', help='Use Cutmix')
    
    arg_parser.add_argument('--beta', default=1.0, type=float, help='hyperparameter beta (default: 1)')
    arg_parser.add_argument('--mu', action='store_false', help='Use Mixup')
    arg_parser.add_argument('--alpha', default=1.0, type=float, help='mixup interpolation coefficient (default: 1)')
    arg_parser.add_argument('--mix_prob', default=0.5, type=float, help='mixup probability')
    
    arg_parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    arg_parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')
    arg_parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    arg_parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    
    arg_parser.add_argument('--LSA', action='store_true', help='Locality Self-Attention')
    arg_parser.add_argument('--SPT', action='store_true', help='Shifted Patch Tokenization')
    arg_parser.add_argument('--ffnt', action='store_true', help='If add time embedding before FFN layer')

    arg_parser.add_argument("--log_dir", type=str, default='./runs')
    arg_parser.add_argument("--log_arg", type=str, default='model-pyx-px-wd-lr')
    arg_parser.add_argument("--novis", action="store_true", help="")
    arg_parser.add_argument("--debug", action="store_true", help="")
    arg_parser.add_argument("--exp_name", type=str, default="GeViT", help="exp name, for description")
    arg_parser.add_argument("--seed", type=int, default=1)
    arg_parser.add_argument("--gpu-id", type=str, default="0")
    arg_parser.add_argument("--note", type=str, default="")

    arg_parser.add_argument("--wandb", action="store_true", help="If set, use wandb")

    return arg_parser


def main(arg):

    data_info = datainfo(logger, arg)
    
    model = create_model(data_info['img_size'], data_info['n_classes'], arg)
    model.to(args.device)
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]
    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(data_info['img_size'], padding=4),
        transforms.ToTensor(),
        *normalize
    ]

    augmentations = transforms.Compose(augmentations)

    if args.dataset == 'cifar10':
        args.img_size = 32
    elif args.dataset == 'cifar100':
        args.img_size = 32

    elif args.dataset == 'svhn':
        args.img_size = 28

    elif args.dataset == 'celeba':
        args.img_size = 128
        args.pyx = 0

    elif args.dataset == 'tinyimagenet':
        args.img_size = 64

    elif args.dataset == 'imgnet':
        args.img_size = 224

    elif 'img128' in args.dataset:
        args.img_size = 128

    elif args.dataset == 'img10':
        args.img_size = 224
    elif args.dataset == 'stl10':
        args.img_size = 96

    '''
        Training
    '''
    diffusion_model = GaussianDiffusion(
        model,
        image_size=arg.img_size,
        timesteps=1000,   # number of steps
        loss_type=arg.loss    # L1 or L2
    ).to(args.device)

    n_ch = 3
    im_sz = arg.img_size
    buffer = torch.FloatTensor(10000, n_ch, im_sz, im_sz).uniform_(-1, 1)
    init_env(arg, logger)
    arg.save_path = arg.dir_path
    print(arg.save_path)

    if arg.resume:
        print(f"load model: {arg.model}")
        print(f'Number of params: {format(n_parameters, ",")}')
        checkpoint = torch.load(arg.resume)
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            final_epoch = arg.epochs
            epoch = checkpoint['epoch'] + 1
            arg.epochs = final_epoch - (checkpoint['epoch'] + 1)
            print(f'trained {epoch} epochs')

    epoch_start = time.time()
    if arg.dataset in ['celeba']:
        print('%s is not a classification dataset' % arg.dataset)
    elif arg.data_path == "none":
        print("Don't load the data and evaluate the accuracy ")
    else:
        train_dataset, val_dataset, px_dataset = dataload(arg, augmentations, normalize, data_info, px=True)
        #
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,  num_workers=arg.workers, pin_memory=True,
        #     batch_sampler=RASampler(len(train_dataset), arg.batch_size, 1, arg.ra, shuffle=True, drop_last=True))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=arg.workers)
        # px_loader = torch.utils.data.DataLoader(px_dataset, batch_size=arg.batch_size, shuffle=False, pin_memory=True, num_workers=arg.workers)
        # px_loader = cycle(px_loader)
        test_acc, test_loss = validate(val_loader, diffusion_model.denoise_fn, criterion, arg)
        print(f'Test Accuracy {test_acc}, loss {test_loss}')

    inc, fid = sample_ema(diffusion_model, buffer, 0, arg)
    print('Sampled images are stored %s' % os.path.join(arg.save_path, 'samples'))

    if fid != 0:
        print(f'Inception Score {inc}   fid {fid}')

    end = time.time()
    total_time = args.epochs * (end - epoch_start)
    print(f'PID {arg.pid} Total ~ {str(timedelta(seconds=total_time))}')

    print(f"load model: {arg.model}")
    print(f'Number of params: {format(n_parameters, ",")}')
    # summary(model, ((3, data_info['img_size'], data_info['img_size']), (1,)))


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    if args.SPT:
        args.log_arg += "-SPT"
    if args.LSA:
        args.log_arg += "-LSA"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']

    print = wlog
    print(' '.join(sys.argv))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    main(args)

    print(args.dir_path)
    print(' '.join(sys.argv))
