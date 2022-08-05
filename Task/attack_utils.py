import os
import torch as torch
import torchvision as tv
import torchvision.transforms as tr


def setup_exp(exp_dir, seed, folder_list, code_file_list=[]):
    # make directory for saving results
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    for folder in ['code'] + folder_list:
        if not os.path.exists(exp_dir + folder):
            os.mkdir(exp_dir + folder)

    # save copy of code in the experiment folder
    def save_code():
        def save_file(file_name):
            file_in = open('./' + file_name, 'r')
            file_out = open(exp_dir + 'code/' + os.path.basename(file_name), 'w')
            for line in file_in:
                file_out.write(line)
        for file in code_file_list:
            save_file(file)
    save_code()

    # set seed for cpu and CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_data(data_type, use_train=True, use_random_transform=False):
    # transformations for importing data. NOTE: all images scaled to have pixel range [-1, 1]
    if use_random_transform and data_type == 'svhn':
        transform = tr.Compose([
            tr.RandomCrop(32, padding=4),
            tr.ToTensor(),
            tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif use_random_transform:
        transform = tr.Compose([
            tr.RandomCrop(32, padding=4),
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = tr.Compose([tr.ToTensor(), tr.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5))])

    # import either train or test set
    if data_type == 'cifar10':
        data = tv.datasets.CIFAR10(root='../../data', transform=transform, train=use_train, download=True)
        num_classes = 10
    elif data_type == 'cifar100':
        data = tv.datasets.CIFAR100(root='../../data', transform=transform, train=use_train, download=True)
        num_classes = 100
    elif data_type == 'svhn':
        if use_train:
            use_train = 'train'
        else:
            use_train = 'test'
        data = tv.datasets.SVHN(root='../../data', split=use_train, transform=transform, download=True)
        num_classes = 10
    else:
        raise RuntimeError('Invalid method for data_type ("cifar10", "svhn", "cifar100")')

    return data, num_classes


