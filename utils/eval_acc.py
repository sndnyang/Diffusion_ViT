import torch
from .training_functions import accuracy
from ExpUtils import wlog


def validate(val_loader, model, criterion, arg, epoch=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(arg.device)
            target = target.to(arg.device)

            output = model(images)
            loss = criterion(output, target)

            acc = accuracy(output, target, (1, 5))
            acc1 = acc[0]
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    wlog(f'[Epoch {epoch + 1}][Test]   Loss: {avg_loss:.4f}   Top-1: {avg_acc1:6.2f}')

    return avg_acc1, avg_loss
