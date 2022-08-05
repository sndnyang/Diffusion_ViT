import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ExpUtils import wlog


def datainfo(logger, args):
    img_mean, img_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if args.dataset == 'cifar10':
        # print(Fore.YELLOW+'*'*80)
        wlog('CIFAR10')
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_size = 32
        
    elif args.dataset == 'cifar100':
        # print(Fore.YELLOW+'*'*80)
        wlog('CIFAR100')
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_size = 32

    elif args.dataset == 'celeba':
        wlog('CelebA')
        n_classes = 10
        img_size = 128

    elif args.dataset == 'svhn':
        wlog('SVHN')
        n_classes = 10
        img_size = 32
        
    elif args.dataset == 'tinyimg':
        wlog('T-IMNET')
        n_classes = 200
        img_size = 64

    elif args.dataset == 'imgnet':
        wlog('imagenet')
        n_classes = 1000
        img_size = 224

    elif args.dataset == 'mnist':
        wlog('MNIST, 32')
        img_size = 32
        n_classes = 10

    elif args.dataset == 'stl10':
        wlog('STL10')
        n_classes = 10
        img_size = 96

    elif 'img32' in args.dataset:
        wlog('imagenet 32')
        n_classes = 1000
        img_size = 32
        if '10' in args.dataset:
            wlog('32 but 10')
            args.n_classes = 10

    elif 'img128' in args.dataset:
        wlog('imagenet 128')
        n_classes = 1000
        img_size = 128
        if '10' in args.dataset:
            wlog('128 but 10')
            args.n_classes = 10

    elif args.dataset == 'img10':
        wlog('imagenet')
        n_classes = 10
        img_size = 224

    data_info = dict()
    data_info['n_classes'] = n_classes
    data_info['stat'] = (img_mean, img_std)
    data_info['img_size'] = img_size
    args.n_classes = n_classes
    args.image_size = img_size
    
    return data_info


def dataload(args, augmentations, normalize, data_info, px=False):
    base_transform = transforms.Compose([
        transforms.Resize(data_info['img_size']),
        transforms.ToTensor(),
        lambda x: x * 2 - 1
        # *normalize
    ])
    if args.dataset == 'cifar10':
        args.img_size = 32
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=augmentations)
        px_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=False, transform=base_transform)
        val_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=False, transform=base_transform)

    elif args.dataset == 'mnist':
        args.img_size = 32
        augmentations = base_transform
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=augmentations)
        px_dataset = datasets.MNIST(root=args.data_path, train=True, download=False, transform=base_transform)
        val_dataset = datasets.MNIST(root=args.data_path, train=False, download=False, transform=base_transform)

    elif args.dataset == 'cifar100':
        args.img_size = 32
        train_dataset = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=augmentations)
        px_dataset = datasets.CIFAR100(root=args.data_path, train=True, download=False, transform=base_transform)
        val_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=False, transform=base_transform)
        
    elif args.dataset == 'svhn':
        args.img_size = 32
        train_dataset = datasets.SVHN(root=args.data_path, split='train', download=True, transform=augmentations)
        px_dataset = datasets.SVHN(root=args.data_path, split='train', download=True, transform=base_transform)
        val_dataset = datasets.SVHN(root=args.data_path, split='test', download=True, transform=base_transform)

    elif args.dataset == 'celeba':
        args.img_size = 128
        args.pyx = 0
        args.print_freq = 200
        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=augmentations)
        px_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=base_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=base_transform)

    elif args.dataset == 'tinyimg':
        args.img_size = 64
        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=augmentations)
        px_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=base_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'val', 'images'), transform=base_transform)

    elif args.dataset == 'imgnet':
        args.img_size = 224
        args.print_freq = 2000
        base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            lambda x: x * 2 - 1
            # *normalize,
        ])
        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=augmentations)
        px_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=base_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'val'), transform=base_transform)

    elif 'img128' in args.dataset:
        args.img_size = 128
        args.print_freq = 3000
        if '10' in args.dataset:
            args.print_freq = 800

        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=augmentations)
        px_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=base_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'val'), transform=base_transform)

    elif 'img32' in args.dataset:
        args.img_size = 32
        args.print_freq = 3000
        if '10' in args.dataset:
            args.print_freq = 800

        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=augmentations)
        px_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=base_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'val'), transform=base_transform)

    elif args.dataset == 'img10':
        args.img_size = 224
        # args.print_freq = 200
        base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            *normalize,
        ])
        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=augmentations)
        px_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=base_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'val'), transform=base_transform)

    elif args.dataset == 'stl10':
        args.img_size = 96
        # args.print_freq = 100
        train_dataset = datasets.STL10(root=args.data_path, split='train', download=True, transform=augmentations)
        px_dataset = datasets.STL10(root=args.data_path, split='train', download=True, transform=base_transform)
        val_dataset = datasets.STL10(root=args.data_path, split='test', download=True, transform=base_transform)

    if px:
        return train_dataset, val_dataset, px_dataset
    else:
        return train_dataset, val_dataset
