import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from ExpUtils import *
from diffusion import GaussianDiffusion
import argparse
from utils.dataloader import datainfo, dataload
from models.create_model import create_model
from Task.eval_tasks import *
from utils.eval_acc import validate
from utils.eval_quality import eval_is_fid
import warnings
warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0
MODELS = ['difvit', 'vit', 'swin', 'pit', 'cait', 't2t']


def init_parser():
    arg_parser = argparse.ArgumentParser(description='evaluate script')
    arg_parser.add_argument("--eval", default="OOD", type=str, choices=["logp_hist", "OOD", "test_clf", "fid", "cali", "gen", 'quality', 'nll'])

    # Data args
    arg_parser.add_argument('--data_path', default='../../data', type=str, help='dataset path')
    arg_parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimg', 'svhn', 'mnist', 'stl10', 'celeba', 'img32', 'img128', 'img12810', 'imgnet', 'img10'],
                            type=str, help='Image Net dataset path')
    arg_parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    arg_parser.add_argument('--print-freq', default=500, type=int, metavar='N', help='log frequency (by iteration)')
    arg_parser.add_argument("--score_fn", default="px", type=str, choices=["px", "py", "pxgrad"], help="For OODAUC, chooses what score function we use.")
    arg_parser.add_argument("--ood_dataset", default="svhn", type=str, choices=["svhn", "cifar_interp", "cifar_100", "celeba"], help="Chooses which dataset to compare against for OOD")
    arg_parser.add_argument("--datasets", nargs="+", type=str, default=[], help="The datasets you wanna use to generate a log p(x) histogram")

    arg_parser.add_argument('--resume', default=False, help='Version')
    arg_parser.add_argument('--buffer_path', default='', type=str, help='path for saved buffer')

    # Optimization hyperparams
    arg_parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    arg_parser.add_argument('--t_step', default=1000, type=int, metavar='N', help='T')
    arg_parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2'])
    arg_parser.add_argument('--model', type=str, default='difvit', choices=MODELS)

    arg_parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    
    arg_parser.add_argument('--pyx', default=1.0, type=float, help='classifier')
    arg_parser.add_argument('--px', default=1.0, type=float, help='diffusion')

    arg_parser.add_argument('--ffnt', default=1, choices=[0, 1], type=int, help='If add time embedding before FFN layer')
    arg_parser.add_argument('--channel', type=int, help='disable cuda')
    arg_parser.add_argument('--heads', type=int, help='disable cuda')
    arg_parser.add_argument('--depth', type=int, help='disable cuda')
    arg_parser.add_argument('--dim', default=384, choices=[192, 384, 512, 576, 768], type=int, help='Image Net dataset path')
    arg_parser.add_argument('--ps', default=4, type=int, help='patch size')

    arg_parser.add_argument('--ls', action='store_false', help='label smoothing')
    arg_parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    arg_parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    arg_parser.add_argument("--multi", action="store_true", help="maybe the model is trained with DataParallel")

    arg_parser.add_argument("--log_dir", type=str, default='eval')
    arg_parser.add_argument("--log_arg", type=str, default='model-pyx-px')
    arg_parser.add_argument("--novis", action="store_true", help="")
    arg_parser.add_argument("--no_fid", action="store_true", help="")
    arg_parser.add_argument("--debug", action="store_true", help="")
    arg_parser.add_argument("--exp_name", type=str, default="GeViT", help="exp name, for description")
    arg_parser.add_argument("--seed", type=int, default=1)
    arg_parser.add_argument("--gpu-id", type=str, default="0")
    arg_parser.add_argument("--note", type=str, default="")

    arg_parser.add_argument("--wandb", action="store_true", help="If set, use wandb")

    return arg_parser


def run_bpd_on_dataset(nll_model, f, loader, arg):
    all_bpd = []
    c = 0
    start = time.time()
    for images, _ in loader:
        images = images.to(arg.device)
        c += images.shape[0]
        minibatch_metrics = nll_model.calc_bpd_loop(f, images, clip_denoised=True)
        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean()
        all_bpd.append(total_bpd.item())
        if c % 1000 == 0:
            print(f'{c} bpd: {total_bpd.item()}')
    bpd = np.mean(all_bpd)
    end = time.time()
    print(f"done {c} samples: bpd={bpd}, takes {end - start}")
    return bpd


def main(arg):
    global best_acc1

    # torch.cuda.set_device(arg.gpu)
    data_info = datainfo(logger, arg)

    if arg.eval == "fid":
        eval_start = time.time()
        print('eval is, fid')
        checkpoint = torch.load(arg.resume)
        buffer = checkpoint['buffer']
        metrics = eval_is_fid((buffer + 1) * 127.5, dataset=arg.dataset)
        inc_score = metrics['inception_score_mean']
        fid = metrics['frechet_inception_distance']
        print(f"sample with {len(buffer)}")
        print("Inception score of {}".format(inc_score))
        print("FID of score {}".format(fid))
        eval_time = time.time() - eval_start
        print(f'takes {eval_time}')
        return

    criterion = nn.CrossEntropyLoss().to(args.device)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]
    augmentations = transforms.Compose([
        transforms.ToTensor(),
        *normalize
    ])

    '''
        model
    '''
    backbone = create_model(data_info['img_size'], data_info['n_classes'], arg)
    if arg.multi:
        backbone = nn.DataParallel(backbone, device_ids=range(len(arg.gpu_id.split(','))))
    else:
        backbone.to(args.device)

    n_parameters = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Creating model: {arg.model}")
    print(f'Number of params: {format(n_parameters, ",")}')

    model = GaussianDiffusion(
        backbone,
        image_size=arg.image_size,
        channels=3 if arg.dataset != 'mnist' else 1,
        timesteps=arg.t_step,   # number of steps
        loss_type=arg.loss    # L1 or L2
    ).to(args.device)

    checkpoint = torch.load(arg.resume)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    except RuntimeError:
        # in train script, it may save model, diffusion model, ema diffusion model,  so~~~
        backbone.load_state_dict(checkpoint['model_state_dict'])
        model.denoise_fn = backbone.to(device)
    f = model.denoise_fn
    f.eval()

    if arg.eval == "OOD":
        OODAUC(f, arg, device)

    if arg.eval == "test_clf":
        _, val_dataset, _ = dataload(arg, augmentations, normalize, data_info, px=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=arg.workers)
        acc, test_loss = validate(val_loader, f, criterion, 0, arg, epoch=-1)
        print(f'Acc {acc}, loss {test_loss}')

    if arg.eval == "cali":
        calibration(f, arg, device)

    if arg.eval == 'gen':
        new_samples(model, arg)

    if arg.eval == 'quality':
        from Task.quality_analysis import qualitative_analysis
        buffer = torch.load(arg.buffer_path)
        qualitative_analysis(f, buffer, buffer_path=arg.buffer_path, dataset=args.dataset)

    if arg.eval == "logp_hist":
        logp_hist(f, arg, device)

    if arg.eval == 'nll':
        assert arg.dataset == 'cifar10', "It's simple to load other datasets"
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py#L709
        from gdiffusion import GaussianDiffusion as NllDiffusion, LossType, ModelVarType, ModelMeanType
        nll_model = NllDiffusion(
            betas=model.betas.cpu(),
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )
        train_set, val_dataset = dataload(arg, augmentations, normalize, data_info)
        # make sure that batch size doesn't matter, since the model use laynorm, not batch norm. small batch size is very slow.
        train_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=arg.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=arg.workers)

        val_bpd = run_bpd_on_dataset(nll_model, f, val_loader, arg)
        train_bpd = run_bpd_on_dataset(nll_model, f, train_loader, arg)
        print(f'train bpd {train_bpd}, test bpd {val_bpd}')


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.SPT = None
    args.LSA = None

    print = wlog
    print(' '.join(sys.argv))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    if args.log_dir == 'eval':
        # by default to eval the model
        args.dir_path = args.resume + "_eval_%s_%s" % (args.eval, run_time)
    set_file_logger(logger, args)
    print(args.dir_path)

    main(args)

    print(args.dir_path)
    print(' '.join(sys.argv))
