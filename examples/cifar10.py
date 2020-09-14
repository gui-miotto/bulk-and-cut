import os
from datetime import datetime

import torch
import torchvision

import bulkandcut as bnc

here = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(here, "..", "datasets", "cifar10"))
work_dir = os.path.join("/tmp", "bulkandcut_output", str(datetime.now()))

img_shape = (3, 32, 32)
train_ds = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
    )
valid_ds = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
    )
evolution = bnc.Evolution(
    input_shape=img_shape,
    n_classes=len(train_ds.classes),
    work_directory=work_dir,
    train_dataset=train_ds,
    valid_dataset=valid_ds,
    debugging=True,
    )
evolution.run(time_budget=24. * 60. * 60.)
