from copy import deepcopy

import numpy as np
import torch


class LinearCell(torch.nn.Module):

    @classmethod
    def NEW(cls, in_features, rng):
        # Sample
        out_features = rng.integers(low=15, high=400)
        ll = torch.nn.Linear(in_features=in_features, out_features=out_features)
        return cls(linear_layer=ll)

    def __init__(self, linear_layer):
        super(LinearCell, self).__init__()
        self.linear = linear_layer
        self.drop = torch.nn.Dropout(p=0.5)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.drop(x)
        x = self.act(x)
        return x

    def downstream_morphism(self):
        identity_layer = torch.nn.Linear(
            in_features=self.out_features,
            out_features=self.out_features,
        )

        with torch.no_grad():  #TODO: why?
            # Initiate weights and biases with the identity function
            torch.nn.init.eye_(identity_layer.weight)
            torch.nn.init.zeros_(identity_layer.bias)

            # And add some noise to break the symmetry
            identity_layer.weight += torch.rand_like(identity_layer.weight) * 1E-4
            identity_layer.bias += torch.rand_like(identity_layer.bias) * 1E-4

        return LinearCell(linear_layer=identity_layer)

    @torch.no_grad()  # TODO: should I use this decorator here?
    def prune(self, in_select):

        # Number of units remaining in the pruned layer
        amount = .1
        out_features = int((1. - amount) * self.out_features)

        # Units with the lowest L1 norms will be pruned
        w_l1norm = torch.sum(
            input=torch.abs(self.linear.weight),
            dim=1,
        )
        out_select = torch.argsort(w_l1norm)[-out_features:]
        out_select = torch.sort(out_select).values  #TODO: this shouldn't be necessary

        # Pruning fully connected layer:
        pruned_layer = torch.nn.Linear(
            in_features=len(in_select),
            out_features=out_features,
        )
        weight = self.linear.weight[out_select][:,in_select]
        bias = self.linear.bias[out_select]
        pruned_layer.weight = torch.nn.Parameter(deepcopy(weight))  # TODO: do I need this deep copy here?
        pruned_layer.bias = torch.nn.Parameter(deepcopy(bias))

        return LinearCell(linear_layer=pruned_layer), out_select

    @torch.no_grad()  # TODO: should I use this decorator here?
    def prune_(self, out_selected):
        #TODO: Adjust the order of the statments in this function to match the one in the conv_cell
        #TODO: improve commentaries in both functions
        amount = .1  #TODO: should be the same used for conv cell as well. Enforce that

        num_in_features = int((1. - amount) * self.in_features)
        num_out_features = self.out_features if out_selected is None else len(out_selected)

        # Upstream units with the lowest L1 norms will be pruned
        w_l1norm = torch.sum(
            input=torch.abs(self.linear.weight),
            dim=0,
        )
        in_selected = torch.argsort(w_l1norm)[-num_in_features:]
        in_selected = torch.sort(in_selected).values  #TODO: this shouldn't be necessary

        pruned_layer = torch.nn.Linear(
            in_features=num_in_features,
            out_features=num_out_features,
            )

        weight = self.linear.weight[:,in_selected]
        bias = self.linear.bias
        if out_selected is not None:
            weight = weight[out_selected]
            bias = bias[out_selected]
        pruned_layer.weight = torch.nn.Parameter(deepcopy(weight))  # TODO: do I need this deep copy here?
        pruned_layer.bias = torch.nn.Parameter(deepcopy(bias))

        return LinearCell(linear_layer=pruned_layer), in_selected


    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features