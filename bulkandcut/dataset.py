import torch
import torchvision
import PIL
import numpy as np

from bulkandcut import rng

def load_dataset(data_dir, img_resize_dim):
    data_augmentations = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            size=[img_resize_dim, img_resize_dim],
            interpolation=PIL.Image.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        ])

    data = torchvision.datasets.ImageFolder(root=data_dir, transform=data_augmentations)
    return data


def mixup(data, targets, n_classes, alpha=.25):
    """
    This function was adapted from:
        https://github.com/hysts/pytorch_mixup/blob/master/utils.py.
    To the author my gratitude. :-)
    """
    batch_size = data.size(0)
    indices = torch.randperm(n=batch_size)
    data2 = data[indices]
    targets2 = targets[indices]

    targets = _onehot(targets, n_classes)
    targets2 = _onehot(targets2, n_classes)

    # Original code:
    # lambda_ = torch.FloatTensor([rng.beta(a=alpha, b=alpha)])
    # data = data * lambda_ + data2 * (1 - lambda_)
    # targets = targets * lambda_ + targets2 * (1 - lambda_)

    # My modification:
    lambda_ = torch.FloatTensor(rng.beta(a=alpha, b=alpha, size=batch_size))
    lamb_data = lambda_.reshape((-1, 1, 1, 1))
    lamb_targ = lambda_.reshape((-1, 1))
    data = data * lamb_data + data2 * (1 - lamb_data)
    targets = targets * lamb_targ + targets2 * (1 - lamb_targ)

    return data, targets

def _onehot(label, n_classes):
    template = torch.zeros(label.size(0), n_classes)
    ohe = template.scatter_(1, label.view(-1, 1), 1)
    return ohe
