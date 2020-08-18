import torch
import torchvision
import PIL
import numpy as np

def load_dataset(data_dir, img_resize_dim):
    data_augmentations = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            size=[img_resize_dim, img_resize_dim],
            interpolation=PIL.Image.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),  # TODO: does it mirror or flip upside-down?
        torchvision.transforms.ToTensor(),
        ])

    data = torchvision.datasets.ImageFolder(root=data_dir, transform=data_augmentations)
    return data


def mixup(data, targets, n_classes, rng, alpha=1.):
    """
    This function was adapted from:
        https://github.com/hysts/pytorch_mixup/blob/master/utils.py.
    To the author my gratitude. :-)
    """

    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    targets = _onehot(targets, n_classes)
    targets2 = _onehot(targets2, n_classes)

    #lam = torch.FloatTensor(rng.beta(a=alpha, b=alpha, size=data.size(0)))  #TODO: make this work
    lambda_ = torch.FloatTensor([rng.beta(a=alpha, b=alpha)])
    data = data * lambda_ + data2 * (1 - lambda_)
    targets = targets * lambda_ + targets2 * (1 - lambda_)

    return data, targets

def _onehot(label, n_classes):
    """
    This function was copied from:
        https://github.com/hysts/pytorch_mixup/blob/master/utils.py
    To the author my gratitude. :-)
    """
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)