import os
import numpy as np
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from torchvision import utils as tv_utils
import torch
import torchvision as tv
from .eval_quality import eval_is_fid
from tqdm import tqdm
from ExpUtils import wlog


def sqrt(x):
    return int(torch.sqrt(torch.Tensor([x])))


def plot(p, x):
    return tv.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


def cycle(loader):
    while True:
        for data in loader:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def save_sample_q(model, epoch, arg, num=100, save=True, i=0):
    milestone = str(epoch)
    if i > 0:
        milestone += '-' + str(i)
    batches = num_to_groups(num, num)
    all_images_list = list(map(lambda n: model.sample(batch_size=n), batches))
    all_images = torch.cat(all_images_list, dim=0)
    if save:
        tv_utils.save_image(all_images, os.path.join(arg.save_path, 'samples', f'sample-{milestone}.png'), nrow=10)
    return all_images


def sample_ema(ema_model, buffer, epoch, arg):
    ema_model.eval()
    n, num = 100, 100
    if arg.dataset != 'cifar10':
        n = 1
        num = 25
    for i in tqdm(range(n)):
        if (i + 1) % 10 == 0:
            print(f'Sampling {i}-th batch with 100 images')
        q = save_sample_q(ema_model, i, arg, num=num)
        idx_start = i * 100
        buffer[idx_start:idx_start + 100] = q

    inc_score, fid = 0, 0
    if arg.dataset == 'cifar10':
        metrics = eval_is_fid((buffer + 1) * 127.5)
        inc_score = metrics['inception_score_mean']
        fid = metrics['frechet_inception_distance']
        wlog('Epoch %d  IS, FID: %.3f, %.3f' % (epoch, inc_score, fid))
    return inc_score, fid


# def sample_ema_tf(ema_model, buffer, epoch, arg, num=10):
#     from Task.eval_buffer import eval_is_fid as eval_is_fid_tf
#     ema_model.eval()
#     q = save_sample_q(ema_model, epoch, arg, num=10)
#     # idx_start = 0
#     # if epoch > 20:
#     #     idx_start = (epoch - 20) % 100 * 100
#     # buffer[idx_start:idx_start + 100] = q
#     inc_score, fid = 0, 0
#     torch.save({'model_state_dict': ema_model.state_dict()}, os.path.join(arg.save_path, 'ema_checkpoint.pth'))
#
#     if (epoch * 10) % arg.epochs == 0:
#         torch.save({'model_state_dict': ema_model.state_dict()}, os.path.join(arg.save_path, f'ema_{epoch}_checkpoint.pth'))
#
#     # if epoch % 5 == 0 and epoch >= 100:
#     #     end = (idx_start + 100) if epoch < 120 else 10000
#     #     _, _, fid = eval_is_fid_tf(buffer[:end], arg, eval='fid')
#     #     arg.writer.add_scalar('GEN/FID', fid, epoch)
#     #     wlog('Epoch %d  FID: %.3f' % (epoch, fid))
#     return fid
