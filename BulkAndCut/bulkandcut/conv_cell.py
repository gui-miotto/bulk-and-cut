from copy import deepcopy

import numpy as np
import torch

from bulkandcut import rng


class ConvCell(torch.nn.Module):

    @classmethod
    def NEW(cls, in_elements:int):
        # sample
        out_elements = rng.integers(low=100, high=600)
        kernel_size = rng.choice([3, 5, 7])
        conv = torch.nn.Conv2d(
            in_channels=in_elements,
            out_channels=out_elements,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            )
        bnorm = torch.nn.BatchNorm2d(num_features=out_elements)

        return cls(conv_layer=conv, batch_norm=bnorm)


    def __init__(self, conv_layer, batch_norm, dropout_p=.5, is_first_cell:bool = False):
        super(ConvCell, self).__init__()
        self.conv = conv_layer
        self.drop = torch.nn.Dropout2d(p=dropout_p)
        self.act = torch.nn.ReLU()
        self.bnorm = batch_norm
        self.is_first_cell = is_first_cell  # This changes how the cell is pruned


    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.bnorm(x)
        return x


    def downstream_morphism(self):
        identity_layer = torch.nn.Conv2d(
            in_channels=self.out_elements,
            out_channels=self.out_elements,
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
        bnorm = torch.nn.BatchNorm2d(num_features=self.out_elements)
        bnorm.weight = torch.nn.Parameter(deepcopy(self.bnorm.weight))
        bnorm.running_var = torch.square(deepcopy(self.bnorm.weight).detach()) - self.bnorm.eps
        bnorm.bias = torch.nn.Parameter(deepcopy(self.bnorm.bias))
        bnorm.running_mean = deepcopy(self.bnorm.bias).detach()

        return ConvCell(conv_layer=identity_layer, batch_norm=bnorm)


    @torch.no_grad()
    def prune(self, out_selected, amount:float = .1):
        #TODO: improve commentary

        num_out_elements = len(out_selected)
        conv_weight = self.conv.weight[out_selected]
        conv_bias = self.conv.bias[out_selected]

        if self.is_first_cell:
            num_in_elements = self.in_elements
            in_selected = None  # should be ignored by the calling code
        else:
            num_in_elements = int((1. - amount) * self.in_elements)
            # Upstream filters with the lowest L1 norms will be pruned
            w_l1norm = torch.sum(
                input=torch.abs(self.conv.weight),
                dim=[0,2,3],
            )
            in_selected = torch.argsort(w_l1norm)[-num_in_elements:]
            in_selected = torch.sort(in_selected).values  # This is actually not necessary
            conv_weight = conv_weight[:,in_selected]

        # Pruning the convolution:
        pruned_conv = torch.nn.Conv2d(
            in_channels=num_in_elements,
            out_channels=num_out_elements,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        pruned_conv.weight = torch.nn.Parameter(deepcopy(conv_weight))
        pruned_conv.bias = torch.nn.Parameter(deepcopy(conv_bias))

        # Pruning the batch norm:
        bnorm_weight = self.bnorm.weight[out_selected]
        bnorm_bias = self.bnorm.bias[out_selected]
        bnorm_running_var = self.bnorm.running_var[out_selected]
        bnorm_running_mean = self.bnorm.running_mean[out_selected]
        pruned_bnorm = torch.nn.BatchNorm2d(num_features=num_out_elements)
        pruned_bnorm.weight = torch.nn.Parameter(deepcopy(bnorm_weight))
        pruned_bnorm.bias = torch.nn.Parameter(deepcopy(bnorm_bias))
        pruned_bnorm.running_var = deepcopy(bnorm_running_var)
        pruned_bnorm.bnorm_running_mean = deepcopy(bnorm_running_mean)

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
            is_first_cell=self.is_first_cell,
            )
        return pruned_cell, in_selected


    @property
    def in_elements(self):
        return self.conv.in_channels

    @property
    def out_elements(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def padding(self):
        return self.conv.padding
