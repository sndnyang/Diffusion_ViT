import torch
from torch.utils.data import Dataset
import torch_fidelity


class TensorDataset(Dataset):

    def __init__(self, x):
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)


def eval_is_fid(images):
    print('eval images num', images.shape)
    px_dataset = TensorDataset(images.to(dtype=torch.uint8))
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=px_dataset,
        input2='cifar10-train',
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
