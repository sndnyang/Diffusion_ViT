# Thanks to rwightman's timm package
# github.com:rwightman/pytorch-image-models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.mix import cutmix_data, mixup_data, mixup_criterion


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        return self._compute_losses(x, target).mean()


def ce_loss(model, images, target, criterion, arg):

    # Cutmix only
    if arg.cm and not arg.mu:
        r = np.random.rand(1)
        if r < arg.mix_prob:
            slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, arg)
            images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
            output = model(images)

            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        else:
            output = model(images)

            loss = criterion(output, target)

    # Mixup only
    elif not arg.cm and arg.mu:
        r = np.random.rand(1)
        if r < arg.mix_prob:
            images, y_a, y_b, lam = mixup_data(images, target, arg)
            output = model(images)

            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        else:
            output = model(images)

            loss = criterion(output, target)

    # Both Cutmix and Mixup
    elif arg.cm and arg.mu:
        r = np.random.rand(1)
        if r < arg.mix_prob:
            switching_prob = np.random.rand(1)

            # Cutmix
            if switching_prob < 0.5:
                slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, arg)
                images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                output = model(images)

                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            # Mixup
            else:
                images, y_a, y_b, lam = mixup_data(images, target, arg)
                output = model(images)

                loss = mixup_criterion(criterion, output, y_a, y_b, lam)

        else:
            output = model(images)
            loss = criterion(output, target)
    # No Mix
    else:
        output = model(images)
        loss = criterion(output, target)
    return loss, output
