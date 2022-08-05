import os
import sys
import time
import copy
import argparse
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from ExpUtils import *
from diffusion import GaussianDiffusion
from utils.sampler import RASampler, cycle
from utils.losses import LabelSmoothingCrossEntropy
from utils.eval_acc import validate
from utils.training_functions import accuracy
from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload
from utils.utils import ema, num_to_groups, sample_ema
from utils.losses import ce_loss
from models.create_model import create_model
import warnings
warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0
MODELS = ['difvit', 'vit', 'swin', 'pit', 'cait', 't2t', 'gevit']


def init_parser():
    arg_parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    # Data args
    arg_parser.add_argument('--data_path', default='../../data', type=str, help='dataset path')
    arg_parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimg', 'svhn', 'mnist', 'stl10'], type=str, help='Image Net dataset path')
    arg_parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    arg_parser.add_argument('--print-freq', default=500, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    arg_parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    arg_parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
    arg_parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

    arg_parser.add_argument('--wd', default=5e-2, type=float, help='weight decay (default: 1e-4)')

    arg_parser.add_argument('--t_step', default=1000, type=int, metavar='N', help='T')
    arg_parser.add_argument('--beta_schd', type=str, default='cosine', choices=['cosine', 'linear'])
    arg_parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2'])
    arg_parser.add_argument('--model', type=str, default='difvit', choices=MODELS)
    arg_parser.add_argument('--disable_cos', action='store_true', help='disable cosine lr schedule')

    arg_parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    
    arg_parser.add_argument('--resume', default=False, help='Version')

    arg_parser.add_argument('--pyx', default=1.0, type=float, help='classifier, cross entropy loss')
    arg_parser.add_argument('--px', default=100.0, type=float, help='diffusion loss')

    arg_parser.add_argument('--ffnt', default=1, type=int, choices=[1, 0], help='If add time embedding before FFN layer')
    arg_parser.add_argument('--channel', type=int, help='disable cuda')
    arg_parser.add_argument('--heads', type=int, help='disable cuda')
    arg_parser.add_argument('--depth', type=int, default=9, help='disable cuda')
    arg_parser.add_argument('--dim', default=384, choices=[192, 384, 512, 576, 768, 1024], type=int, help='Image Net dataset path')
    arg_parser.add_argument('--ps', default=4, type=int, help='patch size')

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

    arg_parser.add_argument("--log_dir", type=str, default='./runs')
    arg_parser.add_argument("--log_arg", type=str, default='model-pyx-px-wd-dim-ps')
    arg_parser.add_argument("--novis", action="store_true", help="")
    arg_parser.add_argument("--no_fid", action="store_true", help="")
    arg_parser.add_argument("--debug", action="store_true", help="")
    arg_parser.add_argument("--exp_name", type=str, default="GeViT", help="exp name, for description")
    arg_parser.add_argument("--seed", type=int, default=1)
    arg_parser.add_argument("--gpu-id", type=str, default="0")
    arg_parser.add_argument("--note", type=str, default="")

    arg_parser.add_argument("--wandb", action="store_true", help="If set, use wandb")

    return arg_parser


def main(arg):
    global best_acc1

    # torch.cuda.set_device(arg.gpu)
    data_info = datainfo(logger, arg)
    
    model = create_model(data_info['img_size'], data_info['n_classes'], arg)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Creating model: {arg.model}")
    print(f'Number of params: {format(n_parameters, ",")}')

    if ',' in args.gpu_id:
        model = nn.DataParallel(model, device_ids=range(len(arg.gpu_id.split(','))))
    else:
        model.to(args.device)
    print(f'Initial learning rate: {arg.lr:.6f}')
    print(f"Start training for {arg.epochs} epochs")

    if arg.ls and arg.pyx:
        print('label smoothing used')
        criterion = LabelSmoothingCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()

    if arg.sd > 0. and arg.pyx:
        print(f'Stochastic depth({arg.sd}) used ')

    criterion = criterion.to(args.device)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]

    if arg.cm and arg.pyx:
        print('Cutmix used')
    if arg.mu and arg.pyx:
        print('Mixup used')
    if arg.ra > 1 and arg.pyx:
        print(f'Repeated Aug({arg.ra}) used')

    '''
        Data Augmentation
    '''
    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(data_info['img_size'], padding=4)
        # other datasets use this one?
        # transforms.RandomResizedCrop(data_info['img_size'])
    ]

    if arg.aa and arg.pyx:
        print('Auto augmentation used')

        if 'cifar' in arg.dataset:
            print("CIFAR Policy")
            from utils.autoaug import CIFAR10Policy
            augmentations += [
                CIFAR10Policy()
            ]

        elif 'svhn' in arg.dataset:
            print("SVHN Policy")
            from utils.autoaug import SVHNPolicy
            augmentations += [
                SVHNPolicy()
            ]

        else:
            print("imagenet Policy")
            from utils.autoaug import ImageNetPolicy
            augmentations += [
                ImageNetPolicy()
            ]

    augmentations += [
        transforms.ToTensor(),
        *normalize
    ]

    if arg.re > 0 and arg.pyx:
        from utils.random_erasing import RandomErasing
        print(f'Random erasing({arg.re}) used ')

        augmentations += [
            RandomErasing(probability=arg.re, sh=arg.re_sh, r1=arg.re_r1, mean=data_info['stat'][0])
        ]

    augmentations = transforms.Compose(augmentations)

    train_dataset, val_dataset, px_dataset = dataload(arg, augmentations, normalize, data_info, px=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,  num_workers=arg.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), arg.batch_size, 1, arg.ra, shuffle=True, drop_last=True))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=arg.workers)
    # you can reduce the batch size for p(x), reduce the training time a bit
    px_loader = torch.utils.data.DataLoader(px_dataset, batch_size=arg.batch_size, shuffle=False, pin_memory=True, num_workers=arg.workers)
    px_loader = cycle(px_loader)

    '''
        Training
    '''
    diffusion_model = GaussianDiffusion(
        model,
        image_size=arg.img_size,
        channels=3 if arg.dataset != 'mnist' else 1,
        timesteps=arg.t_step,   # number of steps
        loss_type=arg.loss    # L1 or L2
    ).to(args.device)
    ema_model = copy.deepcopy(diffusion_model)

    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=arg.lr, weight_decay=arg.wd)
    scheduler = build_scheduler(arg, optimizer, len(train_loader))

    n_ch = 3
    im_sz = arg.img_size
    buffer = torch.FloatTensor(10000 if arg.dataset != 'stl10' else 5000, n_ch, im_sz, im_sz).uniform_(-1, 1)
    # summary(model, (3, data_info['img_size'], data_info['img_size']))

    if arg.resume:
        checkpoint = torch.load(arg.resume)
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            final_epoch = arg.epochs
            arg.epochs = final_epoch - (checkpoint['epoch'] + 1)

    print("Beginning training")
    test_acc = 0
    sample_time = 0
    for epoch in range(arg.epochs):
        epoch_start = time.time()
        lr, avg_loss, avg_acc, avg_ce, avg_dif = train(train_loader, px_loader, diffusion_model, ema_model, criterion, optimizer, epoch, scheduler, arg)
        metrics = {'lr': lr}
        tf_metrics = {"lr": lr, "Train/Loss": avg_loss, "Train/Acc": avg_acc, "Train/CELoss": avg_ce, "Train/DifLoss": avg_dif}
        end = time.time()

        torch.save({'model_state_dict': diffusion_model.state_dict(), 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                   os.path.join(arg.save_path, 'checkpoint.pth'))

        if arg.pyx > 0:
            test_acc, test_loss = validate(val_loader, model, criterion, arg, epoch=epoch)
            acc2, test_loss2 = validate(val_loader, ema_model.denoise_fn, criterion, arg, epoch=epoch)
            if test_acc > best_acc1:
                print('* Best model update *')
                best_acc1 = test_acc
                torch.save({'model_state_dict': diffusion_model.state_dict(), 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                           os.path.join(arg.save_path, 'best.pth'))
            print(f'Best acc1 {best_acc1:.2f}')
            metrics = {'lr': lr, 'Loss': test_loss, 'Acc': test_acc, 'EMALoss': test_loss2, 'EMAAccuracy': acc2}
            tf_metrics = {"lr": lr, "Test/Loss": test_loss, "Test/Accuracy": test_acc, "Test/EMALoss": test_loss2, "Test/EMAAccuracy": acc2, "Train/Loss": avg_loss,
                          "Train/Acc": avg_acc, "Train/CELoss": avg_ce, "Train/DifLoss": avg_dif}

        if args.px > 0 and arg.dataset in ['cifar10', 'cifar100'] and not arg.no_fid:
            sample_start = time.time()
            # sample_ema(diffusion_model, model_buffer, epoch, arg, title=None)
            # model.train()
            inc, fid = sample_ema(ema_model, buffer, epoch, arg)
            sample_end = time.time()
            print(f'sample takes {sample_end - sample_start}')
            sample_time += sample_end - sample_start
            if fid != 0:
                metrics['IS'] = inc
                metrics['fid'] = fid

        for k in tf_metrics:
            v = tf_metrics[k]
            arg.writer.add_scalar(k, v, epoch)

        if arg.wandb:
            import wandb
            wandb.log(metrics)

        remain_time = (args.epochs - epoch) * (end - epoch_start)
        total_time = args.epochs * (end - epoch_start)
        print(f'PID {arg.pid} Total ~ {str(timedelta(seconds=total_time))}, '
              f'epoch {str(timedelta(seconds=end-epoch_start))},'
              f'remain {str(timedelta(seconds=remain_time))}')

    print(f'total sample time {str(timedelta(seconds=sample_time))}')
    print(f"Creating model: {arg.model}")
    print(f'Number of params: {format(n_parameters, ",")}')
    print(f'Initial learning rate: {arg.lr:.6f}')
    print(f'best top-1: {best_acc1:.2f}, final top-1: {test_acc:.2f}')
    torch.save({'model_state_dict': diffusion_model.state_dict(), 'epoch': args.epochs - 1, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
               os.path.join(arg.save_path, 'checkpoint.pth'))
    torch.save({'model_state_dict': ema_model.state_dict()}, os.path.join(arg.save_path, 'ema_checkpoint.pth'))


def train(train_loader, px_loader, model, ema_model, criterion, optimizer, epoch, scheduler, arg):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    avg_loss, avg_acc1 = 0, 0
    avg_ce_loss, avg_dif_loss = 0, 0
    lr = arg.lr
    avg_ce, avg_dif = 0, 0
    for i, data in enumerate(train_loader):
        images, target = data[:2]
        n += images.size(0)
        loss = 0
        if arg.px > 0:
            x_p, _ = px_loader.__next__()
            x_p = x_p.to(arg.device)
            dif_loss = model(x_p)
            loss += arg.px * dif_loss
            avg_dif_loss += float(dif_loss.item() * images.size(0))

        if arg.pyx > 0:
            images = images.to(arg.device)
            target = target.to(arg.device)

            # Classifier
            ce, output = ce_loss(model.denoise_fn, images, target, criterion, arg)
            loss += arg.pyx * ce

            acc = accuracy(output, target, (1,))
            acc1 = acc[0]
            avg_ce_loss += float(ce.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

        loss_val += float(loss.item() * images.size(0))
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 0.999? or 0.995?
        ema(model, ema_model, 0.999)

        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if arg.print_freq >= 0 and i % arg.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            avg_ce, avg_dif = avg_ce_loss / n, avg_dif_loss / n
            size = len(train_loader)
            print(f'[Epoch {epoch+1}/{arg.epochs}][{i:4d}:{size}]  Loss: {avg_loss:.4f} CE: {avg_ce:.4f} Dif: {avg_dif:.4f} Top-1: {avg_acc1:4.3f}  LR: {lr:.6f}')

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    avg_ce, avg_dif = avg_ce_loss / n, avg_dif_loss / n
    return lr, avg_loss, avg_acc1, avg_ce, avg_dif


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

    init_env(args, logger)
    print(args.dir_path)

    main(args)

    print(args.dir_path)
    print(' '.join(sys.argv))
