import os
import sys
import glob
import json
import time
import socket
import shutil
import signal
import logging
from functools import partial

import torch
import numpy as np
import tensorboardX as tbX
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(filename)s[%(lineno)d]: %(message)s", datefmt="%m-%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
wlog = logger.info


def init_env(args, exp_logger):
    # 1. debug -> num_workers
    init_debug(args)
    args.vis = not args.novis
    args.hostname = socket.gethostname().split('.')[0]

    # 2. select gpu
    auto_select_gpu(args)
    args.dir_path = form_dir_path(args.exp_name, args)
    args.save_path = args.dir_path
    os.makedirs('{}/samples'.format(args.dir_path))
    set_file_logger(exp_logger, args)
    init_logger_board(args)
    wlog(args.dir_path)
    args.n_classes = 10
    if args.dataset == "cifar100":
        args.n_classes = 100
    if args.dataset == "tinyimagenet":
        args.n_classes = 200

    if not args.debug and args.wandb:
        import wandb
        wandb.init(project='biggest')
        name = args.note
        if name:
            wandb.run.name = args.note + str(os.getpid())
        else:
            wandb.run.name = args.exp_name + str(os.getpid())
        wandb.run.save()
        args.pid = os.getpid()
        args.node = os.uname().nodename.split('.')[0]
        wandb.config.update(args)


def init_debug(args):
    # verify the debug mode
    # pytorch loader has a parameter num_workers
    # in debug mode, it should be 0
    # so set args.debug
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        print('No sys.gettrace')
        args.debug = False
    elif gettrace():
        print('Hmm, Big Debugger is watching me')
        args.debug = True
    else:
        args.debug = False


def auto_select_gpu(args):
    if args.gpu_id:
        return
    try:
        import GPUtil
    except ImportError:
        wlog("please install GPUtil for automatically selecting GPU")
        args.gpu_id = '1'
        return

    if len(GPUtil.getGPUs()) == 0:
        return
    id_list = GPUtil.getAvailable(order="load", maxLoad=0.7, maxMemory=0.9, limit=8)
    if len(id_list) == 0:
        print("GPU memory is not enough for predicted usage")
        raise NotImplementedError
    args.gpu_id = str(id_list[0])


def init_logger_board(args):
    if 'vis' in vars(args) and args.vis:
        args.writer = tbX.SummaryWriter(log_dir=args.dir_path)


def vlog(writer, cur_iter, set_name, wlog=None, verbose=True, **kwargs):
    for k in kwargs:
        v = kwargs[k]
        writer.add_scalar('%s/%s' % (set_name, k.capitalize()), v, cur_iter)
    if wlog:
        my_print = wlog
    else:
        my_print = print
    if not verbose:
        prompt = "%d " % cur_iter
        prompt += ','.join("%s: %.4f" % (k, kwargs[k]) for k in ['loss', 'acc', 'acc1', 'acc5'] if k in kwargs)
        my_print(prompt)


def set_file_logger(exp_logger, args):
    # Just use "logger" above
    # use tensorboard + this function to substitute ExpSaver
    device = args.device
    args_dict = vars(args)
    dir_path = args.dir_path
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    args_dict['device'] = ''
    with open(os.path.join(dir_path, "para.json"), "w") as fp:
        json.dump(args_dict, fp, indent=4, sort_keys=True)
    args.device = device
    logfile = os.path.join(dir_path, "exp.log")
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s")
    fh.setFormatter(formatter)
    exp_logger.addHandler(fh)
    # copy_script_to_folder(sys.argv[0], args.dir_path)
    if os.name != 'nt':
        signal.signal(signal.SIGQUIT, partial(rename_quit_handler, args))
        signal.signal(signal.SIGTERM, partial(delete_quit_handler, args))


def list_args(args):
    for e in sorted(vars(args).items()):
        print("args.%s = %s" % (e[0], e[1] if not isinstance(e[1], str) else '"%s"' % e[1]))


def form_dir_path(task, args):
    """
    Params:
        task: the name of your experiment/research
        args: the namespace of argparse
            requires:
                --dataset: always need a dataset.
                --log-arg: the details shown in the name of your directory where logs are.
                --log-dir: the directory to save logs, default is ~/projecct/runs.
    """
    args.pid = os.getpid()
    args_dict = vars(args)
    if "log_dir" not in args_dict:
        args.log_dir = ""
    if "log_arg" not in args_dict:
        args.log_arg = ""

    run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    log_arg_list = []
    if args.debug:
        task += '-debug'
    for e in args.log_arg.split("-"):
        v = args_dict.get(e, None)
        if v is None:
            log_arg_list.append(str(e))
        elif isinstance(v, str):
            log_arg_list.append(str(v))
        else:
            log_arg_list.append("%s=%s" % (e, str(v)))
    args.exp_marker = exp_marker = "-".join(log_arg_list)
    exp_marker = "%s/%s/%s@%s@%d" % (args.dataset, task, run_time, exp_marker, os.getpid())
    base_dir = os.path.join(os.environ['HOME'], 'project/runs') if not args.log_dir else args.log_dir
    dir_path = os.path.join(base_dir, exp_marker)
    return dir_path


def summary(data):
    assert isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)
    wlog("shape: %s, num of points: %d, pixels: %d" % (str(data.shape), data.shape[0], np.prod(data.shape[1:])))
    wlog("max: %g, min %g" % (data.max(), data.min()))
    wlog("mean: %g" % data.mean())
    wlog("mean of abs: %g" % np.abs(data).mean())
    wlog("mean of square sum: %g" % (data ** 2).mean())


def remove_outliers(x, outlier_constant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    iqr = (upper_quartile - lower_quartile) * outlier_constant
    quartile_set = (lower_quartile - iqr, upper_quartile + iqr)

    result = a[np.where((a >= quartile_set[0]) & (a <= quartile_set[1]))]

    return result


def vis_step(writer, step, dicts):
    """
    Add several curves.
    """
    for k in dicts:
        writer.add_scalar(k, dicts[k], step)


def copy_script_to_folder(caller_path, folder):
    '''copy script'''
    script_filename = caller_path.split('/')[-1]
    script_relative_path = os.path.join(folder, script_filename)
    shutil.copy(caller_path, script_relative_path)
    for file in ['diffusion.py', 'models/DifViT.py']:
        shutil.copy(file, folder)
    shutil.copytree('utils', os.path.join(folder, 'utils'))
    shutil.copytree('models', os.path.join(folder, 'models'))


def time_string():
    '''convert time format'''
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch   = total_epoch
        self.current_epoch = 0
        self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
        self.epoch_losses  = self.epoch_losses - 1

        self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
        self.epoch_accuracy= self.epoch_accuracy

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
        self.epoch_losses  [idx, 0] = train_loss
        self.epoch_losses  [idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1
        return self.max_accuracy(False) == val_acc

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
        else:       return self.epoch_accuracy[:self.current_epoch, 1].max()

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)


        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print ('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def plotting(exp_dir):
    # Load the training log dictionary:
    train_dict = pickle.load(open(os.path.join(exp_dir, 'log.pkl'), 'rb'))
    ###########################################################
    #   Make the vanilla train and test loss per epoch plot   #
    ###########################################################

    plt.plot(np.asarray(train_dict['train_loss']), label='train_loss')
    plt.plot(np.asarray(train_dict['test_loss']), label='test_loss')

    # plt.ylim(0,2000)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'loss.png' ))
    plt.clf()

    # accuracy
    plt.plot(np.asarray(train_dict['train_acc']), label='train_acc')
    plt.plot(np.asarray(train_dict['test_acc']), label='test_acc')

    # plt.ylim(0,2000)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'acc.png'))
    plt.clf()


def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax


def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, filename=None):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)
            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0:
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def delete_quit_handler(g_var, signal, frame):
    shutil.rmtree(g_var.dir_path)
    sys.exit(0)


def rename_quit_handler(g_var, signal, frame):
    os.rename(g_var.dir_path, g_var.dir_path + "_stop")
    sys.exit(0)
