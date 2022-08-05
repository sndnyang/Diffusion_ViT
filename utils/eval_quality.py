import os
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as tr
import torch_fidelity


class TensorDataset(Dataset):

    def __init__(self, x):
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)


def dataset_without_label(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data.to(dtype=torch.uint8)

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def load_dataset(args):
    transform_px = tr.Compose(
        [
            tr.ToTensor(),
            lambda x: x * 255
        ]
    )
    if args.dataset == 'cifar100':
        cls = dataset_without_label(torchvision.datasets.CIFAR100)
        test_dataset = cls(root=args.data_path, transform=transform_px)
    elif args.dataset in ['celeba', 'img32', 'tinyimg']:
        cls = dataset_without_label(torchvision.datasets.ImageFolder)
        # no test set for celeba128, I save all images in args.data_root/train/subdir
        set_name = 'train' if args.dataset in ['celeba'] else 'val'
        test_dataset = cls(root=os.path.join(args.data_path, set_name), transform=transform_px)
    else:
        assert False, 'dataset %s' % args.dataset
    return test_dataset


def eval_is_fid(images, dataset='cifar10', args=None):
    print('eval images num', images.shape, images.min(), images.max())
    px_dataset = TensorDataset(images.to(dtype=torch.uint8))
    target = f'{dataset}-train'
    if dataset not in ['cifar10', 'stl10']:
        target = load_dataset(args)

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=px_dataset,
        input2=target,
        cuda=True,
        isc=True,
        fid=True,
        # kid=True,
        verbose=False,
    )

    # {'inception_score_mean': 1.2051318455224629, 'inception_score_std': 0.004178657147150005, 'frechet_inception_distance': 427.7598114126157}
    # 64.98886704444885 seconds for 10k 3x32x32 images
    print(metrics_dict)
    return metrics_dict
