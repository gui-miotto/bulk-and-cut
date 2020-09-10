import os
from datetime import datetime

import torch
import torchvision

import bulkandcut as bnc

here = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(here, "..", "datasets", "mnist"))
work_dir = os.path.join(here, "..", "..", "bulkandcut_output", str(datetime.now()))

batch_size = 256
img_shape = (1, 28, 28)
train_ds = torchvision.datasets.MNIST(
    root=data_dir,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
    )
valid_ds = torchvision.datasets.MNIST(
    root=data_dir,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False,
    )
train_loader = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=batch_size,
    shuffle=True,
    )
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_ds,
    batch_size=batch_size,
    shuffle=False,
    )
evolution = bnc.Evolution(
    input_shape=img_shape,
    n_classes=len(train_ds.classes),
    work_directory=work_dir,
    train_data_loader=train_loader,
    valid_data_loader=valid_loader,
    debugging=True,
    )
evolution.run(time_budget=3 * 60 * 60)
