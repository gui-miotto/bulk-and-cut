from copy import deepcopy

import numpy as np
import torch

from bulkandcut import rng


class LinearCell(torch.nn.Module):

    @classmethod
    def NEW(cls, in_elements):
        # Sample
        out_elements = int(rng.triangular(left=15, right=350, mode=350))
        ll = torch.nn.Linear(in_features=in_elements, out_features=out_elements)
        return cls(linear_layer=ll)

    def __init__(self, linear_layer, dropout_p=.5):
        super(LinearCell, self).__init__()
        self.linear = linear_layer
        self.drop = torch.nn.Dropout(p=dropout_p)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.drop(x)
        x = self.act(x)
        return x

    def downstream_morphism(self):
        identity_layer = torch.nn.Linear(
            in_features=self.out_elements,
            out_features=self.out_elements,
        )

        with torch.no_grad():
            # Initiate weights and biases with the identity function
            torch.nn.init.eye_(identity_layer.weight)
            torch.nn.init.zeros_(identity_layer.bias)

            # And add some noise to break the symmetry
            identity_layer.weight += torch.rand_like(identity_layer.weight) * 1E-5
            identity_layer.bias += torch.rand_like(identity_layer.bias) * 1E-5

        return LinearCell(linear_layer=identity_layer)


    @torch.no_grad()
    def prune(self, out_selected, amount:float):
        #TODO: improve commentary

        num_in_elements = int((1. - amount) * self.in_elements)
        num_out_elements = self.out_elements if out_selected is None else len(out_selected)

        # Upstream units with the lowest L1 norms will be pruned
        w_l1norm = torch.sum(
            input=torch.abs(self.linear.weight),
            dim=0,
        )
        in_selected = torch.argsort(w_l1norm)[-num_in_elements:]
        in_selected = torch.sort(in_selected).values  # this is actually not necessary

        pruned_layer = torch.nn.Linear(
            in_features=num_in_elements,
            out_features=num_out_elements,
            )

        weight = self.linear.weight[:,in_selected]
        bias = self.linear.bias
        if out_selected is not None:
            weight = weight[out_selected]
            bias = bias[out_selected]
        pruned_layer.weight = torch.nn.Parameter(deepcopy(weight))
        pruned_layer.bias = torch.nn.Parameter(deepcopy(bias))

        # "Pruning" dropout:
        drop_p = self.drop.p * (1. - amount)

        # Wrapping it up:
        pruned_cell = LinearCell(
            linear_layer=pruned_layer,
            dropout_p=drop_p,
            )

        return pruned_cell, in_selected


    @property
    def in_elements(self):
        return self.linear.in_features

    @property
    def out_elements(self):
        return self.linear.out_features
