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

        bnorm = deepcopy(self.bnorm)  # TODO: can we do better than this? This is not the real morphism
        return ConvCell(conv_layer=identity_layer, batch_norm=bnorm)

    @torch.no_grad()  # TODO: should I use this decorator here?
    def prune(self, in_select):

        # Number of filters remaining in the pruned layer
        amount = .1
        out_channels = int((1. - amount) * self.out_channels)

        # Filters with the lowest L1 norms will be pruned
        w_l1norm = torch.sum(
            input=torch.abs(self.conv.weight),
            dim=[1,2,3],
        )
        out_select = torch.argsort(w_l1norm)[-out_channels:]
        out_select = torch.sort(out_select).values  #TODO: this shouldn't be necessary

        # Pruning the convolution:
        pruned_conv = torch.nn.Conv2d(
            in_channels=len(in_select),
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        conv_weight = self.conv.weight[out_select][:,in_select]
        conv_bias = self.conv.bias[out_select]
        pruned_conv.weight = torch.nn.Parameter(deepcopy(conv_weight))  # TODO: do I need this deep copy here?
        pruned_conv.bias = torch.nn.Parameter(deepcopy(conv_bias))  # TODO: do I need this deep copy here?

        # Pruning the batch norm:
        pruned_bnorm = torch.nn.BatchNorm2d(num_features=out_channels)
        bnorm_weight = self.bnorm.weight[out_select]
        bnorm_bias = self.bnorm.bias[out_select]
        pruned_bnorm.weight = torch.nn.Parameter(deepcopy(bnorm_weight))  # TODO: do I need this deep copy here?
        pruned_bnorm.bias = torch.nn.Parameter(deepcopy(bnorm_bias))  # TODO: do I need this deep copy here?

        return ConvCell(conv_layer=pruned_conv, batch_norm=pruned_bnorm), out_select


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
