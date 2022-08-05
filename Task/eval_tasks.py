import time
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import timedelta

from ExpUtils import *
from utils.utils import plot
from utils.eval_quality import eval_is_fid
from Task.calibration import reliability_diagrams
from Task.calibration import ECELoss

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
seed = 1
n_ch = 3
n_classes = 10
print = wlog


def new_samples(f, arg):
    im_sz = arg.image_size
    start = time.time()
    batches = 100 if arg.dataset != 'stl10' else 50
    replay_buffer = torch.FloatTensor(batches * 100, n_ch, im_sz, im_sz).uniform_(-1, 1)
    eval_time = 0
    bs = 100
    for i in range(batches):

        samples = f.sample(batch_size=bs)
        replay_buffer[i * bs: (i + 1) * bs] = samples

        if (i + 1) % 10 == 0:
            now = time.time()
            print(f'batch {i + 1} {str(timedelta(seconds=now - start - eval_time))}')

        if (i + 1) in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
            eval_start = time.time()
            plot('{}/samples_{}.png'.format(arg.dir_path, i+1), samples)
            metrics = eval_is_fid(replay_buffer[:(i + 1) * bs] * 255, dataset=arg.dataset, args=arg)
            inc_score = metrics['inception_score_mean']
            fid = metrics['frechet_inception_distance']
            print("sample with %d" % (i * bs + bs))
            print("Inception score of {}".format(inc_score))
            print("FID of score {}".format(fid))
            eval_time += time.time() - eval_start
    if "0_check" in arg.resume:
        torch.save(replay_buffer, '{}/buffer.pt'.format(arg.dir_path))


def logp_hist(f, arg, device):
    import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set()
    plt.switch_backend('agg')
    def score_fn(x):
        if arg.score_fn == "px":
            return f(x).logsumexp(1).detach().cpu()
        elif arg.score_fn == "py":
            logits = f(x)
            return nn.Softmax()(logits).max(1)[0].detach().cpu()
        else:
            return f(x).max(1)[0].detach().cpu()
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
        ]
    )
    datasets = {
        "cifar10": tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False),
        "svhn": tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test"),
        "cifar100": tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=False),
        "celeba": tv.datasets.CelebA(root="../../data", download=True, split="test",
                                     transform=tr.Compose([tr.Resize(size=(32, 32)),
                                                           tr.ToTensor(),
                                                           tr.Normalize((.5, .5, .5), (.5, .5, .5)),
                                                           ]))
    }

    score_dict = {}
    num_workers = 0 if arg.debug else 4
    for dataset_name in arg.datasets:
        print(dataset_name)
        dataset = datasets[dataset_name]
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=num_workers, drop_last=False)
        this_scores = []
        for x, _ in dataloader:
            x = x.to(device)
            scores = score_fn(x)
            this_scores.extend(scores.numpy())
        score_dict[dataset_name] = this_scores

    colors = ['green', 'red']
    for i, (name, scores) in enumerate(score_dict.items()):
        plt.hist(scores, label=name, bins=100, alpha=.5, color=colors[i])
    plt.legend(loc='upper left')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(arg.resume + "_%s_logp.pdf" % arg.datasets[1], bbox_inches='tight', pad_inches=0.0)


def OODAUC(f, arg, device):
    print("OOD Evaluation")

    def grad_norm(x):
        x_k = torch.autograd.Variable(x, requires_grad=True)
        f_prime = torch.autograd.grad(f(x_k)[0].sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5))
         ]
    )

    num_workers = 0 if arg.debug else 4
    dset_real = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)
    dload_real = DataLoader(dset_real, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)

    if arg.ood_dataset == "svhn":
        dset_fake = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test")
    elif arg.ood_dataset == "cifar_100":
        dset_fake = tv.datasets.CIFAR100(root="../../data", transform=transform_test, download=True, train=False)
    elif arg.ood_dataset == "celeba":
        dset_fake = tv.datasets.CelebA(root="../../data", download=True, split="test",
                                       transform=tr.Compose([tr.Resize(size=(32, 32)),
                                                             tr.ToTensor(),
                                                             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
                                                             ]))
    else:
        dset_fake = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)

    dload_fake = DataLoader(dset_fake, batch_size=100, shuffle=True, num_workers=num_workers, drop_last=False)
    real_scores = []
    # print("Real scores...")

    def score_fn(x):
        if arg.score_fn == "px":
            return f(x).logsumexp(1).detach().cpu()
        elif arg.score_fn == "py":
            logits = f(x)
            return nn.Softmax()(logits).max(1)[0].detach().cpu()
        else:
            return -grad_norm(x).detach().cpu()

    for x, _ in dload_real:
        x = x.to(device)
        scores = score_fn(x)
        real_scores.append(scores.numpy())
    fake_scores = []
    # print("Fake scores...")
    if arg.ood_dataset == "cifar_interp":
        last_batch = None
        for i, (x, _) in enumerate(dload_fake):
            x = x.to(device)
            if i > 0:
                x_mix = (x + last_batch) / 2 + 0.0 * torch.randn_like(x)
                scores = score_fn(x_mix)
                fake_scores.append(scores.numpy())
            last_batch = x
    else:
        for i, (x, _) in enumerate(dload_fake):
            x = x.to(device)
            scores = score_fn(x)
            fake_scores.append(scores.numpy())
    real_scores = np.concatenate(real_scores)
    fake_scores = np.concatenate(fake_scores)
    real_labels = np.ones_like(real_scores)
    fake_labels = np.zeros_like(fake_scores)
    import sklearn.metrics
    scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([real_labels, fake_labels])
    score = sklearn.metrics.roc_auc_score(labels, scores)
    print('OOD scores %f of %s between cifar10 and %s using %s' % (score, arg.score_fn, arg.ood_dataset, arg.resume))


def test_clf(f, arg, device):
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
        ]
    )

    def sample(x, n_steps=arg.n_steps):
        x_k = torch.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = torch.autograd.grad(f(x_k)[0].sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * torch.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    if arg.dataset == "cifar_train":
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=True)
    elif arg.dataset == "cifar_test":
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)
    elif arg.dataset == "cifar100_train":
        dset = tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=True)
    elif arg.dataset == "cifar100_test":
        dset = tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=False)
    elif arg.dataset == "svhn_train":
        dset = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="train")
    elif arg.dataset == "svhn_test":
        dset = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test")
    elif arg.dataset == 'stl10':
        dset = tv.datasets.STL10(root="../data", transform=transform_test, download=True, split="test")
    else:
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)

    num_workers = 0 if arg.debug else 4
    dload = DataLoader(dset, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)

    corrects, losses, pys, preds = [], [], [], []
    for x_p_d, y_p_d in tqdm(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        if arg.n_steps > 0:
            x_p_d = sample(x_p_d)
        with torch.no_grad():
            logits = f(x_p_d)
        py = nn.Softmax(dim=1)(logits).max(1)[0].detach().cpu().numpy()
        loss = nn.CrossEntropyLoss(reduction='none')(logits, y_p_d).cpu().detach().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
        pys.extend(py)
        preds.extend(logits.max(1)[1].cpu().numpy())

    loss = np.mean(losses)
    correct = np.mean(corrects)
    print('loss %.5g,  accuracy: %g%%' % (loss, correct * 100))
    return correct


def calibration(f, arg, device):
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5))
         ]
    )

    num_workers = 0 if arg.debug else 4
    dset_real = tv.datasets.CIFAR10(root="../../data", transform=transform_test, download=True, train=False)
    dload_real = DataLoader(dset_real, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    f.eval()
    real_scores = []
    labels = []
    pred = []
    ece_com = ECELoss(20)
    ece = 0
    c = 0
    logits_l = []
    correct_num = 0
    for x, y in dload_real:
        x = x.to(device)
        labels.append(y.numpy())
        logits = f(x)
        logits_l.append(logits.detach())
        scores = nn.Softmax(dim=1)(logits).max(dim=1)[0].detach().cpu()
        preds = nn.Softmax(dim=1)(logits).argmax(dim=1).detach().cpu()
        correct_num += (preds == y).sum()
        real_scores.append(scores.numpy())
        pred.append(preds.numpy())
    logits_l = torch.cat(logits_l)
    temps = torch.LongTensor(np.concatenate(labels))
    print(logits_l.shape)
    print(temps.shape)
    ece = ece_com(logits_l, temps.to(device)).item()
    print("On Calibration of Modern Neural Networks code result: %f" % ece)
    real_scores = np.concatenate(real_scores)
    labels = np.concatenate(np.array(labels))
    pred = np.concatenate(pred)
    correct = correct_num / len(pred)
    print(len(real_scores))
    # print(pred.shape)

    reliability_diagrams(list(pred), list(labels), list(real_scores), bin_size=0.05, title="Accuracy: %.2f%%" % (100.0 * correct), args=arg)
