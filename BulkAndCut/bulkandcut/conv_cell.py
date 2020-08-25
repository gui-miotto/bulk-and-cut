from copy import deepcopy

import numpy as np
import torch


class ConvCell(torch.nn.Module):

    @classmethod
    def NEW(cls, in_channels, rng):
        # sample
        out_channels = rng.integers(low=100, high=600)
        kernel_size = rng.choice([3, 5, 7])
        conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            )
        bnorm = torch.nn.BatchNorm2d(num_features=out_channels)

        return cls(conv_layer=conv, batch_norm=bnorm)


    def __init__(self, conv_layer, batch_norm, dropout_p=.5):
        super(ConvCell, self).__init__()
        self.conv = conv_layer
        self.drop = torch.nn.Dropout2d(p=dropout_p)
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
            identity_layer.weight += torch.rand_like(identity_layer.weight) * 1E-5
            identity_layer.bias += torch.rand_like(identity_layer.bias) * 1E-5

        # Batch-norm morphism (is this the best way?):
        bnorm = torch.nn.BatchNorm2d(num_features=self.out_channels)
        bnorm.weight = torch.nn.Parameter(deepcopy(self.bnorm.weight))
        bnorm.running_var = torch.square(deepcopy(self.bnorm.weight).detach()) - self.bnorm.eps
        bnorm.bias = torch.nn.Parameter(deepcopy(self.bnorm.bias))
        bnorm.running_mean = deepcopy(self.bnorm.bias).detach()

        return ConvCell(conv_layer=identity_layer, batch_norm=bnorm)


    @torch.no_grad()
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
        out_select = torch.sort(out_select).values  # this actually not necessary

        # Pruning the convolution:
        pruned_conv = torch.nn.Conv2d(
            in_channels=len(in_select),
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        conv_weight = self.conv.weight[out_select][:,in_select]
        conv_bias = self.conv.bias[out_select]
        pruned_conv.weight = torch.nn.Parameter(deepcopy(conv_weight))
        pruned_conv.bias = torch.nn.Parameter(deepcopy(conv_bias))

        # Pruning the batch norm:
        pruned_bnorm = torch.nn.BatchNorm2d(num_features=out_channels)
        bnorm_weight = self.bnorm.weight[out_select]
        bnorm_bias = self.bnorm.bias[out_select]
        pruned_bnorm.weight = torch.nn.Parameter(deepcopy(bnorm_weight))
        pruned_bnorm.bias = torch.nn.Parameter(deepcopy(bnorm_bias))

        return ConvCell(conv_layer=pruned_conv, batch_norm=pruned_bnorm), out_select


    @torch.no_grad()
    def prune_(self, out_selected, is_input_layer=False):
        amount = .1 #TODO: should be the same used for linear cell as well. Enforce that
        #TODO: improve commentary

        num_out_channels = len(out_selected)
        conv_weight = self.conv.weight[out_selected]
        conv_bias = self.conv.bias[out_selected]

        if is_input_layer:
            num_in_channels = self.in_channels
            in_selected = None  # should be ignored by the calling function
        else:
            num_in_channels = int((1. - amount) * self.in_channels)
            # Upstream filters with the lowest L1 norms will be pruned
            w_l1norm = torch.sum(
                input=torch.abs(self.conv.weight),
                dim=[0,2,3],
            )
            in_selected = torch.argsort(w_l1norm)[-num_in_channels:]
            in_selected = torch.sort(in_selected).values  # This is actually not necessary
            conv_weight = conv_weight[:,in_selected]

        # Pruning the convolution:
        pruned_conv = torch.nn.Conv2d(
            in_channels=num_in_channels,
            out_channels=num_out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        pruned_conv.weight = torch.nn.Parameter(deepcopy(conv_weight))
        pruned_conv.bias = torch.nn.Parameter(deepcopy(conv_bias))

        # Pruning the batch norm:
        pruned_bnorm = torch.nn.BatchNorm2d(num_features=num_out_channels)
        bnorm_weight = self.bnorm.weight[out_selected]
        bnorm_bias = self.bnorm.bias[out_selected]
        pruned_bnorm.weight = torch.nn.Parameter(deepcopy(bnorm_weight))
        pruned_bnorm.bias = torch.nn.Parameter(deepcopy(bnorm_bias))

        # "Pruning" dropout:
        drop_p = self.drop.p * (1. - amount)
        drop_p = drop_p if drop_p > .05 else 0.  # I'll snap this to 0 for small values so that my
                                                 # my search space includes the baseline network
                                                 # beyond any doubt. TODO: remove this after
                                                 # presentation.

        # Wrapping it all up:
        pruned_cell = ConvCell(
            conv_layer=pruned_conv,
            batch_norm=pruned_bnorm,
            dropout_p=drop_p,
            )
        return pruned_cell, in_selected


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
