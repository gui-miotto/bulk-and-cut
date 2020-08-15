from copy import deepcopy

import numpy as np
import torch


class ConvCell(torch.nn.Module):

    @classmethod
    def NEW(cls, in_channels, rng):
        # sample
        out_channels = rng.integers(low=400, high=800)
        kernel_size = rng.choice([3, 5, 7])
        conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            )
        bnorm = torch.nn.BatchNorm2d(num_features=out_channels)

        return cls(conv_layer=conv, batch_norm=bnorm)


    def __init__(self, conv_layer, batch_norm):
        super(ConvCell, self).__init__()
        self.conv = conv_layer
        self.drop = torch.nn.Dropout2d(p=0.5)
        self.act = torch.nn.ReLU()
        self.bnorm = batch_norm


    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.bnorm(x)
        return x

    def downstream_morphism(self):
        identity_layer = torch.nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

        with torch.no_grad():
            # Initiate weights and biases with the identity function
            torch.nn.init.dirac_(identity_layer.weight)
            torch.nn.init.zeros_(identity_layer.bias)

            # And add some noise to break the symmetry
            identity_layer.weight += torch.rand_like(identity_layer.weight) * 1E-4
            identity_layer.bias += torch.rand_like(identity_layer.bias) * 1E-4

        bnorm = deepcopy(self.bnorm)
        return ConvCell(conv_layer=identity_layer, batch_norm=bnorm)


    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def padding(self):
        return self.conv.padding
